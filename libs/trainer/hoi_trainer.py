import datetime
import logging
import math
import numpy as np
import sys
import time
import json
import csv
from pyexpat import model
from tqdm import tqdm
import os
import torch
from torch import autograd

from libs.trainer.trainer import BaseTrainer
import libs.utils.misc as utils
from libs.utils.utils import save_checkpoint, write_dict_to_json


class HOITrainer(BaseTrainer):

    def __init__(self,
                 cfg,
                 model,
                 criterion,
                 optimizer,
                 lr_scheduler,
                 postprocessors,
                 log_dir='output',
                 performance_indicator='mAP',
                 last_iter=-1,
                 rank=0,
                 device='cuda',
                 max_norm=0):

        super().__init__(cfg, model, criterion, optimizer, lr_scheduler, 
            log_dir, performance_indicator, last_iter, rank)
        self.postprocessors = postprocessors
        self.device = device
        self.max_norm = max_norm  
        self.pue = cfg.TEST.PUE 
        self.resume_init = True if self.cfg.TRAIN.RESUME else False

    def _read_inputs(self, inputs):
        imgs, targets, filenames = inputs
        imgs = [img.to(self.device) for img in imgs]
        # targets are list type in det tasks
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return imgs, targets

    def _forward(self, data):
        imgs = data[0]
        targets = data[1]
        outputs = self.model(imgs)
        loss_dict = self.criterion(outputs, targets)
        return loss_dict

    def set_dropout(self, model, drop_rate=0.1):
        for name, child in model.named_children():
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            self.set_dropout(child, drop_rate=drop_rate)

    def train(self, train_loader, eval_loader, step):
        if self.epoch == self.cfg.TRAIN.PL_STAGE[step]:
            drop_rate = self.cfg.TRANSFORMER.DROPOUT + 0.05*step
            self.set_dropout(self.model, drop_rate=drop_rate)
            print('Dropout is now set to {}'.format(drop_rate))
            step+=1
            
        if self.resume_init:
            drop_rate = self.cfg.TRANSFORMER.DROPOUT + 0.05*step
            self.set_dropout(self.model, drop_rate=drop_rate)
            print('Dropout is now set to {}'.format(drop_rate))
            #self.resume_init = False

        start_time = time.time()
        self.model.train()
        self.criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('dr', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('rel_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(self.epoch)
        print_freq = self.cfg.TRAIN.PRINT_FREQ

        if self.epoch > self.max_epoch:
            logging.info("Optimization is done !")
            sys.exit(0)

        # if self.epoch == 23 or self.epoch == 41:
        #     if self.epoch < 40:
        #         # TODO: 冻结relation相关的权重
        #         print("冻结参数")
        #         for name, param in self.model.named_parameters():
        #             if 'rel_' in name or 'interaction' in name:  # interaction相关参数匹配
        #                 param.requires_grad = False
        #     else:
        #         for param in self.model.parameters():
        #             param.requires_grad = True
        #         print("===== Frozen Parameters (Epoch 41) =====")
        #         for name, param in self.model.named_parameters():
        #             if not param.requires_grad:
        #                 print(f"[Frozen] {name}")
        #         print("======================================")


        for index, data in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            data = self._read_inputs(data) #把数据解析出来
            loss_dict = self._forward(data)  #前向传递得到loss
            #分步式训练
            # if self.epoch == 25:
            #     print("===== Frozen Parameters (Epoch 0) =====")
            #     for name, param in self.model.named_parameters():
            #         if not param.requires_grad:
            #             print(f"[Frozen] {name}")
            #     print("======================================")
            # if self.epoch < 40:
            #     weight_dict = self.criterion.weight_dict  # dict containing as key the names of the losses and as values their relative weight.
            #     modified_weight_dict = {k: 0 if 'rel' in k else v for k, v in  self.criterion.weight_dict.items()}
            #     losses = sum(loss_dict[k] * modified_weight_dict[k] for k in loss_dict.keys() if k in modified_weight_dict)
            #     #print("weight_dict:{}".format(weight_dict))
            #     #print("modified_weight_dict:{}".format(modified_weight_dict))
            # else:
            weight_dict = self.criterion.weight_dict  # dict containing as key the names of the losses and as values their relative weight.
            #print("weight_dict:{}".format(weight_dict))
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            #每次反向传播之前需要清除之前的梯度
            self.optimizer.zero_grad()
            losses.backward()
            if self.max_norm > 0: #判断是否需要进行梯度裁剪，为了防止梯度过大
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(rel_class_error=loss_dict_reduced['rel_class_error'])
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(dr=self.cfg.TRANSFORMER.DROPOUT if step==0 else drop_rate)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes() #用于同步不同进程之间的指标数据，以便能够正确计算整个训练过程的平均值
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch}
        if self.rank == 0: #用于判断当前进程是否为主进程
            for (key, val) in log_stats.items():
                self.writer.add_scalar(key, val, log_stats['epoch'])
                csv_path = os.path.join(self.log_dir, 'epoch_losses.csv')
                file_exists = os.path.exists(csv_path)
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=log_stats.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(log_stats)
        self.lr_scheduler.step(self.epoch)

        # save checkpoint
        if self.rank == 0 and self.epoch > 0 and self.epoch % self.cfg.TRAIN.SAVE_INTERVAL == 0:
            # evaluation
            if self.cfg.TRAIN.VAL_WHEN_TRAIN:
                self.model.eval()
                performance = self.evaluate(eval_loader)
                self.writer.add_scalar(self.PI, performance, self.epoch)  
                if performance > self.best_performance:
                    self.is_best = True
                    self.best_performance = performance
                else:
                    self.is_best = False
                logging.info(f'Now: best {self.PI} is {self.best_performance}')
            else:
                performance = -1

            # save checkpoint
            try:
                state_dict = self.model.module.state_dict() # remove prefix of multi GPUs
            except AttributeError:
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if self.cfg.TRAIN.SAVE_EVERY_CHECKPOINT:
                    filename = f"{self.model_name}_epoch{self.epoch:03d}_checkpoint.pth"
                else:
                    filename = "checkpoint.pth"
                save_checkpoint(
                    {
                        'epoch': self.epoch,
                        'model': self.model_name,
                        f'performance/{self.PI}': performance,
                        'state_dict': state_dict,
                        'optimizer': self.optimizer.state_dict(),
                    },
                    self.is_best,
                    self.log_dir,
                    filename=f'{self.cfg.OUTPUT_ROOT}_{filename}'
                )
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        self.epoch += 1
        
    def evaluate(self, eval_loader, mode, train_dataset, eval_dataset, rel_topk=100,):
        self.model.eval()

        if self.pue:
            self.model.transformer.decoder.class_embed.train()
            self.model.transformer.decoder.rel_class_embed.train()

        results = []
        count = 0
        print('dataset_size: {}'.format(len(eval_loader.dataset)))
        for data in tqdm(eval_loader):
            if data is None:
                print('dataset is empty')
            imgs, targets, filenames = data
            #print(f'filename:{filenames}')
            imgs = [img.to(self.device) for img in imgs]
            # targets are list type
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            bs = len(imgs)
            target_sizes = targets[0]['size'].expand(bs, 2)
            target_sizes = target_sizes.to(self.device)
            outputs_dict = self.model(imgs)
            file_name = filenames[0]
            pred_out = self.postprocessors(outputs_dict, file_name, target_sizes,
                rel_topk=rel_topk)
            results.append(pred_out)
            count += 1
        
        # save the result
        result_path = f'{self.cfg.OUTPUT_ROOT}/pred.json'
        write_dict_to_json(results, result_path)
        # with open(result_path, 'r') as file:
        #       results = json.load(file)

        # eval
        if mode == 'hico':
            from eval_tools.hico_eval import hico
            eval_tool = hico(annotation_file='data/hico/test_hico.json',
                             train_annotation='data/hico/trainval_hico.json')
            mAP = eval_tool.evalution(results)
        elif mode == 'hoia':
            from eval_tools.hoia_eval import hoia
            eval_tool = hoia(annotation_file='data/hoia/test_hoia.json')
            mAP = eval_tool.evalution(results)

        elif mode == 'vcoco':
            from eval_tools.vcoco_eval import vcoco
            eval_tool = vcoco(annotation_file='data/vcoco/test_vcoco.json')
            mAP = eval_tool.evalution(results)
        elif mode == 'phacoq':
            from eval_tools.hico_eval import hico
            print('mode: phacoq')
            eval_tool = hico(annotation_file='/data/wangyi/wangyi_code/ernet/test.json',
                             train_annotation='/data/wangyi/wangyi_code/ernet/train.json')
            mAP = eval_tool.evalution(results)
        else:
            mAP = 0.0

        return mAP