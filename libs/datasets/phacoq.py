import json
import numpy as np
import os
import os.path as osp
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import Dataset

VERB_DICT = {'null':0, 'grasp':1, 'loosen':2, 'hold':16, 'pull':3, 'slack':17, 'cut':4, 'inject':5, 'separate':6, 'rotate':7,
             'seal':8, 'tear':9, 'push':10, 'chop':11, 'aspirate':12, 'wipe':13, 'snip':14, 'insert':18, 'polish':15}

INS_DICT = {'forceps':1, 'incision-knife':2, 'cannula':3, 'capsulorhexis-forceps':4, 'phacoemulsifier':5,
            'lens-hook':6, 'cotton-swab':7, 'irrigation-aspiration-handpiece':8, 'aspiration-handpiece':9,
            'irrigation-handpiece':10, 'implant-injector':11, 'needle-holder':13, 'scissors':12}

TIS_DICT = {'null':0, 'conjunctiva':1, 'corneal-limbus':2, 'saline':3, 'viscoelastic':4, 'lens-capsule':5, 'lens-nucleus':6, 'capsule-stain':7,
                'anterior-capsule':8, 'intraocular-lens':9, 'lens-cortex':10, 'eyelid':11, 'wire':14, 'needle':13, 'fibre':12}

IVT_DICT = {
    "forceps_null_null": 1,
    "forceps_grasp_conjunctiva": 2,
    "forceps_loosen_conjunctiva": 3,
    "forceps_hold_needle": 4,
    "forceps_hold_wire": 5,
    "forceps_pull_wire": 6,
    "forceps_slack_wire": 7,
    "incision-knife_null_null": 8,
    "incision-knife_cut_corneal-limbus": 9,
    "cannula_null_null": 10,
    "cannula_inject_saline": 11,
    "cannula_inject_viscoelastic": 12,
    "cannula_inject_capsule-stain": 13,
    "cannula_separate_lens-capsule": 14,
    "cannula_rotate_lens-nucleus": 15,
    "cannula_seal_corneal-limbus": 16,
    "capsulorhexis-forceps_null_null": 17,
    "capsulorhexis-forceps_tear_anterior-capsule": 18,
    "capsulorhexis-forceps_loosen_anterior-capsule": 19,
    "capsulorhexis-forceps_grasp_anterior-capsule": 20,
    "phacoemulsifier_null_null": 21,
    "phacoemulsifier_inject_saline": 22,
    "phacoemulsifier_rotate_lens-nucleus": 23,
    "phacoemulsifier_push_lens-nucleus": 24,
    "phacoemulsifier_chop_lens-nucleus": 25,
    "phacoemulsifier_aspirate_lens-nucleus": 26,
    "lens-hook_null_null": 27,
    "lens-hook_push_lens-nucleus": 28,
    "lens-hook_pull_lens-nucleus": 29,
    "lens-hook_grasp_lens-nucleus": 30,
    "lens-hook_push_intraocular-lens": 31,
    "lens-hook_pull_intraocular-lens": 32,
    "cotton-swab_null_null": 33,
    "cotton-swab_wipe_conjunctiva": 34,
    "cotton-swab_wipe_eyelid": 35,
    "irrigation-aspiration-handpiece_null_null": 36,
    "irrigation-aspiration-handpiece_inject_saline": 37,
    "irrigation-aspiration-handpiece_aspirate_lens-cortex": 38,
    "irrigation-aspiration-handpiece_polish_intraocular-lens": 39,
    "aspiration-handpiece_null_null": 40,
    "aspiration-handpiece_aspirate_lens-cortex": 41,
    "aspiration-handpiece_polish_intraocular-lens": 42,
    "irrigation-handpiece_null_null": 43,
    "irrigation-handpiece_inject_saline": 44,
    "implant-injector_null_null": 45,
    "implant-injector_inject_intraocular-lens": 46,
    "scissors_null_null": 47,
    "scissors_snip_wire": 48,
    "scissors_snip_fibre": 49,
    "needle-holder_null_null": 50,
    "needle-holder_hold_needle": 51,
    "needle-holder_pull_needle": 52,
    "needle-holder_hold_wire": 53,
    "needle-holder_pull_wire": 54,
    "needle-holder_insert_corneal-limbus": 55,
    "needle-holder_slack_wire": 56,
}

OBJ_FICT = {
    'forceps': 0,
    'incision-knife': 1,
    'cannula': 2,
    'capsulorhexis-forceps': 3,
    'phacoemulsifier': 4,
    'lens-hook': 5,
    'cotton-swab': 6,
    'irrigation-aspiration-handpiece': 7,
    'aspiration-handpiece': 8,
    'irrigation-handpiece': 9,
    'implant-injector': 10,
    'scissors': 11,
    'needle-holder': 12,
    'conjunctiva': 13,
    'corneal-limbus': 14,
    'saline': 15,
    'viscoelastic': 16,
    'lens-capsule': 17,
    'lens-nucleus': 18,
    'capsule-stain': 19,
    'anterior-capsule': 20,
    'intraocular-lens': 21,
    'lens-cortex': 22,
    'eyelid': 23,
    'wire': 24,
    'needle': 25,
    'fibre': 26
}


# # 生成relationship array
# def to_interaction():
#     to_interaction = np.zeros((14, 15, 19))
#     for ivt in IVT_DICT:
#         i, v, t = ivt.split('_')
#         if v == 'null':
#             continue
#         if INS_DICT[i] > 13 or TIS_DICT[t] > 14 or VERB_DICT[v] > 18:
#             continue
#         to_interaction[INS_DICT[i], TIS_DICT[t], VERB_DICT[v]] = 1
#     output_file = r'D:\dataset\phacoq_valtest\rel_np.npy'
#     np.save(output_file.npy, to_interaction)
#     print(f"Relational array saved to {output_file}")


class QhacoqDataset(Dataset):
    def __init__(self, cfg, data_root, transform=None, istrain=False):
        self.num_classes_verb = cfg.DATASET.REL_NUM_CLASSES
        self.data_root = data_root
        print("data_root{}".format(data_root))
        self.labels_path = osp.join(osp.abspath(
            self.data_root), 'targets')
        self.transform = transform
        self.change_annot_format(self.labels_path)
        self.ids = []
        for i, hico in enumerate(self.hoi_annotations):
            flag_bad = 0
            if len(hico['annotations']) > cfg.TRANSFORMER.NUM_QUERIES:
                flag_bad = 1
                continue
            for hoi in hico['hoi_annotation']:
                if hoi['subject_id'] >= len(hico['annotations']) or hoi[
                     'object_id'] >= len(hico['annotations']):
                    flag_bad = 1
                    break
            if flag_bad == 0:
                self.ids.append(i)
        self.neg_rel_id = 0

        # Number of images per class
        self.cls_num_list = np.zeros(cfg.DATASET.REL_NUM_CLASSES+1)
        # 记录每个 HOI 类别的频次
        for i in range(len(self.ids)):
            ann_id = self.ids[i]
            hoi_anns = self.hoi_annotations[ann_id]['hoi_annotation']

            hoi_idx = []
            for j in range(len(hoi_anns)):
                hoi_cat = hoi_anns[j]['category_id']
                if isinstance(hoi_cat, list):
                    if len(hoi_cat)>1:
                        for k in range(len(hoi_cat)):
                            hoi_idx.append(hoi_cat[k])
                    else:
                        hoi_idx.append(hoi_cat[0])
                else:
                    hoi_idx.append(hoi_cat)
            hoi_idx = np.unique(np.array(hoi_idx))
            for j in range(len(hoi_idx)):
                self.cls_num_list[hoi_idx[j]]+=1
        self.cls_num_list_path = osp.join(cfg.DATASET.ROOT,
                                'cls_num_list.npy')
        np.save(self.cls_num_list_path,self.cls_num_list)

    # 封装获取类别ID的函数
    def get_category_id(self, mapping_dict, label_list, idx):
        """从映射字典获取类别ID"""
        category_id = next(k for k, v in mapping_dict.items() if v == label_list[idx])
        return OBJ_FICT[category_id]

    def change_annot_format(self, label_dir):
        """
        将自定义数据格式转换为 HICO 数据集的格式
        :param annotation_dir: 存放注释文件的目录
        """
        hico_data = []
        object_category_ids = []
        # 处理标注文件
        for annotation_file in sorted(os.listdir(label_dir)):
            if not annotation_file.endswith('.json'):
                continue

            # 加载标注数据
            with open(os.path.join(label_dir, annotation_file), 'r') as f:
                annotation = json.load(f)

            file_name = annotation_file.replace('.json', '.png')
            hico_item = {
                "file_name": file_name,
                "annotations": [],
                "hoi_annotation": []
            }
            if annotation['boxes_i'] == annotation['boxes_t'] == []:
                continue
            # 处理主体类别的边界框
            for i, bbox in enumerate(annotation['boxes_i']):
                category_id = self.get_category_id(INS_DICT, annotation['labels_i'], i)
                hico_item["annotations"].append({'bbox': bbox, 'category_id': category_id})

            # 处理目标物体类别的边界框
            for i, bbox in enumerate(annotation['boxes_t']):
                category_id = self.get_category_id(TIS_DICT, annotation['labels_t'], i)
                hico_item["annotations"].append({'bbox': bbox, 'category_id': category_id})
                object_category_ids.append(category_id)

            # 处理 HOI 标注
            annotation_map = {anno['category_id']: idx for idx, anno in enumerate(hico_item["annotations"])}
            for i, class_id in enumerate(annotation['labels']):
                ivt = next(k for k, v in IVT_DICT.items() if v == annotation['hoi'][i])
                ins, ver, tis = ivt.split('_')

                # 获取主体和物体的索引
                sub_id = annotation_map.get(OBJ_FICT[ins])
                obj_id = annotation_map.get(OBJ_FICT[tis])

                if sub_id is not None and obj_id is not None:
                    hico_item["hoi_annotation"].append({
                        'subject_id': sub_id,
                        'object_id': obj_id,
                        'category_id': [VERB_DICT[ver]]
                    })

            hico_data.append(hico_item)
        if object_category_ids:
            min_category_id = min(object_category_ids)
            max_category_id = max(object_category_ids)
            print(f"目标物体类别的最小 category_id: {min_category_id}")
            print(f"目标物体类别的最大 category_id: {max_category_id}")
        self.hoi_annotations = hico_data


    def __len__(self):
        return len(self.ids)

    def multi_dense_to_one_hot(self, labels, num_classes):
        num_labels = labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels.ravel()] = 1
        one_hot = np.sum(labels_one_hot, axis=0)[1:]
        in_valid = np.where(one_hot>1)[0]
        one_hot[in_valid] = 1
        return one_hot

    def __getitem__(self, index):
        ann_id = self.ids[index]
        file_name = self.hoi_annotations[ann_id]['file_name']
        img_path = osp.join(osp.join(self.data_root, 'images'), file_name)
        # print(f"image path: {img_path}")

        anns = self.hoi_annotations[ann_id]['annotations']
        hoi_anns = self.hoi_annotations[ann_id]['hoi_annotation']

        if not osp.exists(img_path):
            logging.error("Cannot found image data: " + img_path)
            raise FileNotFoundError
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        num_object = len(anns)
        num_rels = len(hoi_anns)
        boxes = []
        labels = []
        no_object = False
        if num_object == 0:
            # no gt boxes
            no_object = True
            boxes = np.array([]).reshape(-1, 4)
            labels = np.array([]).reshape(-1, )
        else:
            for k in range(num_object):
                ann = anns[k]
                boxes.append(np.asarray(ann['bbox']))
                if isinstance(ann['category_id'], str):
                    ann['category_id'] = int(ann['category_id'].replace('\n', ''))
                cls_id = int(ann['category_id'])
                labels.append(cls_id)
            boxes = np.vstack(boxes)

        boxes = torch.from_numpy(boxes.reshape(-1, 4).astype(np.float32))
        labels = np.array(labels).reshape(-1, )
        target = dict(
            boxes=boxes,
            labels=labels
        )
        if self.transform is not None:
            img, target = self.transform(
                img, target
            )
        target['labels'] = torch.from_numpy(target['labels']).long()
        boxes = target['boxes']

        hoi_labels = []
        hoi_vecs = []
        hoi_boxes = []
        if num_object == 0:
            hoi_vecs = torch.from_numpy(np.array([]).reshape(-1, 4))
            hoi_boxes = torch.from_numpy(np.array([]).reshape(-1, 4))
            hoi_labels = np.array([]).reshape(-1, self.num_classes_verb)
        else:
            for k in range(num_rels):
                hoi = hoi_anns[k]
                if not isinstance(hoi['category_id'], list):
                    hoi['category_id'] = [hoi['category_id']]
                hoi_label_np = np.array(hoi['category_id'])
                if 'vcoco' in self.data_root:
                    hoi_label_np = hoi_label_np + 1
                hoi_labels.append(self.multi_dense_to_one_hot(hoi_label_np,
                                                              self.num_classes_verb + 1))
                # hoi vectors
                sub_ct_coord = boxes[hoi['subject_id']][..., :2]
                obj_ct_coord = boxes[hoi['object_id']][..., :2]
                hoi_vecs.append(torch.cat([sub_ct_coord, obj_ct_coord], dim=-1).reshape(-1, 4))

                # hoi boxes
                sub_wh_coord = boxes[hoi['subject_id']][..., 2:]
                obj_wh_coord = boxes[hoi['object_id']][..., 2:]
                sub_box_xyxy = torch.stack([(sub_ct_coord[0] - 0.5 * sub_wh_coord[0]),
                                            (sub_ct_coord[1] - 0.5 * sub_wh_coord[1]),
                                            (sub_ct_coord[0] + 0.5 * sub_wh_coord[0]),
                                            (sub_ct_coord[1] + 0.5 * sub_wh_coord[1])], dim=-1)
                obj_box_xyxy = torch.stack([(obj_ct_coord[0] - 0.5 * obj_wh_coord[0]),
                                            (obj_ct_coord[1] - 0.5 * obj_wh_coord[1]),
                                            (obj_ct_coord[0] + 0.5 * obj_wh_coord[0]),
                                            (obj_ct_coord[1] + 0.5 * obj_wh_coord[1])], dim=-1)
                hoi_box = torch.Tensor([torch.min(sub_box_xyxy[0], obj_box_xyxy[0]),
                                        torch.min(sub_box_xyxy[1], obj_box_xyxy[1]),
                                        torch.max(sub_box_xyxy[2], obj_box_xyxy[2]),
                                        torch.max(sub_box_xyxy[3], obj_box_xyxy[3])])
                hoi_box = torch.Tensor([(hoi_box[0] + hoi_box[2]) / 2,
                                        (hoi_box[1] + hoi_box[3]) / 2,
                                        (hoi_box[2] - hoi_box[0]),
                                        (hoi_box[3] - hoi_box[1])])
                hoi_boxes.append(hoi_box)
            hoi_labels = np.array(hoi_labels).reshape(-1, self.num_classes_verb)

        target['rel_labels'] = torch.from_numpy(hoi_labels)
        if len(hoi_vecs) == 0:
            target['rel_vecs'] = torch.from_numpy(np.array([]).reshape(-1, 4)).float()
            target['rel_boxes'] = torch.from_numpy(np.array([]).reshape(-1, 4)).float()
        else:
            target['rel_vecs'] = torch.cat(hoi_vecs).reshape(-1, 4).float()
            target['rel_boxes'] = torch.cat(hoi_boxes).reshape(-1, 4).float()

        target['size'] = torch.from_numpy(np.array([h, w]))
        return img, target, file_name

