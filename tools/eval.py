from __future__ import division
from __future__ import print_function
from __future__ import with_statement
import os.path as osp
import argparse
import importlib
import logging
import os
import torch

import _init_paths
from configs import cfg
from configs import update_config
from libs.datasets.collate import collect
from libs.datasets.transform import EvalTransform
from libs.utils.utils import get_model
from libs.utils.utils import list_to_set
torch.backends.cudnn.enabled = False
def parse_args():
    parser = argparse.ArgumentParser(description='HOI Detection Task')
    parser.add_argument(
        '--cfg',
        dest='yaml_file',
        help='experiment configure file name, e.g. configs/hico.yaml',
        required=True,
        type=str)    
    parser.add_argument(
        'opts',
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
          
    return args


def main_per_worker():
    args = parse_args()
    update_config(cfg, args)
    # ngpus_per_node = torch.cuda.device_count()
    device = torch.device(cfg.DEVICE)

    if not os.path.exists(cfg.OUTPUT_ROOT):
        os.makedirs(cfg.OUTPUT_ROOT)
    logging.basicConfig(filename=f'{cfg.OUTPUT_ROOT}/eval.log', level=logging.INFO)
    
    # model
    # model, criterion, postprocessors = getattr(module, 'build_model')(cfg, device)
    model, criterion, postprocessors = get_model(cfg, device)
    # model = torch.jit.script(model)
    #model = torch.nn.DataParallel(model).to(device)
    model.to(device)
    print("model to {}".format(device))
    #model_without_ddp = model.module
    #NOT DATAPARALLEL
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # load model checkpoints
    resume_path = cfg.MODEL.RESUME_PATH
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location='cpu')
        # pretrained_dict = checkpoint['state_dict']
        # model_dict = model.state_dict()
        #
        # allowed_prefixes = ["backbone", "transformer.encoder"]
        # allowed_keys = [
        #     "transformer.level_embed",
        #     "transformer.det_token",
        #     "transformer.rel_det_token",
        #     "transformer.det_pos_embed",
        #     "transformer.rel_det_pos_embed"
        # ]
        #
        # # 筛选出模型中存在且形状匹配，同时满足条件的权重
        # loadable_dict = {
        #     k: v for k, v in pretrained_dict.items()
        #     if k in model_dict and v.size() == model_dict[k].size() and
        #        (any(k.startswith(prefix) for prefix in allowed_prefixes) or k in allowed_keys)
        # }
        # # 记录加载的层数
        # num_loaded = len(loadable_dict)
        # # 更新模型的权重字典，并加载
        # model_dict.update(loadable_dict)
        # model.load_state_dict(model_dict)
        # resume
        if 'state_dict' in checkpoint:
            #model.module.load_state_dict(checkpoint['state_dict'], strict=True)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info(f'==> model pretrained from {resume_path}')
            print("Load successfully from {}".format(resume_path))


    # ONNX
    # ONNX_FILE_PATH = resume_path.replace('pth','onnx')
    # input = torch.randn(1, 3, 800, 1333, device="cuda")
    # torch.onnx.export(model.module, input, ONNX_FILE_PATH, input_names=['input'],
    #               output_names=['output'], export_params=True)

    # get datset
    module = importlib.import_module(cfg.DATASET.FILE)
    Dataset = getattr(module, cfg.DATASET.NAME)
    print(f'dataset:{Dataset}')
    data_root = os.path.join(cfg.DATASET.ROOT, 'test')
    if not os.path.exists(data_root):
        logging.info(f'==> Cannot found data: {data_root}')
        raise FileNotFoundError
    eval_transform = EvalTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        max_size=cfg.DATASET.MAX_SIZE
    )
    logging.info(f'==> load val sub set: {data_root}')
    #eval_dataset = Dataset(cfg, data_root, eval_transform)
    eval_set = []
    if len(eval_set) == 0:
        eval_set = ['.']
    eval_list = []
    for sub_set in eval_set:
        eval_sub_root = osp.join(data_root, sub_set)
        logging.info(f'==> load val sub set: {eval_sub_root}')
        eval_sub_set = Dataset(cfg, eval_sub_root, eval_transform)
        eval_list.append(eval_sub_set)
    eval_dataset = list_to_set(eval_list, 'eval')
    if eval_dataset is not None:
        logging.info(f'==> the size of eval dataset is {len(eval_dataset)}')

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=collect,
        num_workers=0
    )

    # start evaluate in Trainer
    module = importlib.import_module(cfg.TRAINER.FILE)
    Trainer = getattr(module, cfg.TRAINER.NAME)(
        cfg,
        model=model,
        criterion=criterion,
        optimizer=None,
        lr_scheduler=None,
        postprocessors=postprocessors,
        log_dir=cfg.OUTPUT_ROOT+'/output',
        performance_indicator=cfg.PI,
        last_iter=-1,
        rank=0,
        device=device,
        max_norm=None
    )
    logging.info(f'==> start eval...')
    
    assert cfg.TEST.MODE in ['hico', 'hoia', 'vcoco', 'ahoi', 'phacoq']
    Trainer.evaluate(eval_loader, cfg.TEST.MODE, eval_dataset, eval_dataset)
    #Trainer.evaluate(eval_loader, cfg.TEST.MODE)


if __name__ == '__main__':
    main_per_worker()
