import mmcv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class phacoq():
    def __init__(self, preds, gts):
        self.gt_anno = mmcv.load(gts)
        self.overlap_iou = 0.5
        self.max_hois = 5
        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []
        self.file_name = []
        self.preds = []
        for gt_i in self.gt_anno:
            self.file_name.append(gt_i['file_name'])
            gt_hoi = gt_i['hoi_annotation']
            gt_bbox = gt_i['annotations']
            ins_boxes = gt_bbox[gt_hoi['subject_id']]['bbox']
            tis_boxes = gt_bbox[gt_hoi['object_id']]['bbox']
            ins_labels = gt_bbox[gt_hoi['subject_id']]['category_id']
            tis_labels = gt_bbox[gt_hoi['object_id']]['category_id']
            verb_labels = gt_hoi['hoi_annotation']['category_id']
            print(ins_boxes, tis_boxes, ins_labels, tis_labels, verb_labels)
        #for pred_i in preds:


eval_tool = phacoq(preds='/data/wangyi/wangyi_code/ernet/test.json', gts='/data/wangyi/wangyi_code/ernet/test.json')