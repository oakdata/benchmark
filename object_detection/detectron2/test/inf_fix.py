import sys  
import os
import os.path as osp


import torch
import cv2
import pickle
import numpy as np 
import random
import logging
from collections import OrderedDict

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import get_lvis_instances_meta
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader
)
from detectron2.structures import BoxMode
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import time
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.config import get_cfg
from detectron2.data.dataset_mapper import DatasetMapper

from roi_heads import Res5ROIHeads
import json
import pickle
import traceback
import argparse
import math
from detectron2.utils.comm import get_world_size, is_main_process
import detectron2.utils.comm as comm

sourcedir = '' 
annodir = ''
config_fp = ''
cat_fp = ''
saved_dir = ''

from eval import voc_eval

def get_test_all():
    testset = []
    count = 0
    for video_name in videos_name:
        steps_name = sorted(os.listdir( osp.join(annodir,video_name)), key = lambda x:int(x.split('_')[1]))
        for step_name in steps_name:
            jsons_name = sorted(os.listdir(osp.join(annodir, video_name, step_name)),key=lambda x: int(x.split('_')[0]))
            for json_name in jsons_name:
                count = count + 1
                if count % 17 == 0:
                    testset.append(video_name + '#' + step_name + '#' +  json_name)
    return testset

def get_data_dicts(files_name, data_map):
    dataset_dicts = []
    for file_name in files_name:
        video_name, step_name, json_name = file_name.split('#')[0], file_name.split('#')[1], file_name.split('#')[2]
        img_path = osp.join( sourcedir,video_name, step_name, json_name[:-5] + '.jpg')

        im = cv2.imread(img_path)
        height, width = im.shape[:2]
        record = {}
        record["file_name"] = img_path
        record["image_id"] = file_name
        record["height"],record["width"] = height,width

        dataset_dicts.append(record)

    return dataset_dicts

def get_trainmap(limitset):
    meta_name = limitset.keys()
    mapping = {i:i for i in range(0,len(limitset.keys()))}

    return meta_name,mapping

def build_fix(cfg,num_oc,wp):
    
    model = build_model(cfg)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_oc
    model.roi_heads = Res5ROIHeads(cfg, model.backbone.output_shape())
    
    w_module = torch.load(wp)
    w = {}
    for key in w_module.keys():
        w[key[7:]] = w_module[key]
    if 'pixel_mean' in w.keys():
        w.pop('pixel_mean')
        w.pop('pixel_std')
    model.load_state_dict(w)

    model.to(torch.device(cfg.MODEL.DEVICE))
    return cfg,model


def returnmap():
    f = open(cat_fp, 'r')
    content = f.read()
    categories = json.loads(content)
    return categories

def process(inputs, outputs, predictions):
    for input, output in zip(inputs, outputs):
        image_id = input["image_id"]
        instances = output["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.detach().numpy()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        for box, score, cls in zip(boxes, scores, classes):
            xmin, ymin, xmax, ymax = box
            # The inverse of data loading logic in `datasets/pascal_voc.py`
            xmin += 1
            ymin += 1
            obj = {}
            obj['image_id'] = image_id
            obj['score'] = score
            obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax'] = xmin,ymin,xmax,ymax
            predictions[cls].append(obj)
    return predictions

from collections import defaultdict

def inference(cfg,model):
    with torch.no_grad():
        model.eval()
        dataset_name = cfg.DATASETS.TEST[0]
        predictions = defaultdict(list)
        data_loader = build_detection_test_loader(cfg, dataset_name,mapper=DatasetMapper(cfg, is_train=False, 
            augmentations=[T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST)]))
          
        for idx, inputs in enumerate(data_loader):
            try:
                outputs = model(inputs)
            except:
                print(idx,inputs)
                print(0/0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            predictions = process(inputs,outputs,predictions)

        all_predictions = comm.gather(predictions, dst=0)
        if not comm.is_main_process():
            return []

        cls_predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                cls_predictions[clsid].extend(lines)
        del all_predictions
        return cls_predictions

videos_name = os.listdir(sourcedir)
videos_name = sorted(videos_name)
all_testset = get_test_all()
limitset = returnmap()
meta_names,train_map = get_trainmap(limitset) # net : annotation
data_map = {value:key for key,value in train_map.items()} ## annotation : net

def main(args):
    method = args.method
    lr = float(args.lr)
    iteration = int(args.iteration)
    batch_size = int(args.batch_size)
    importance = int(args.importance)
    number = int(args.number)

    model_dir = osp.join(saved_dir,'models')
    inf_dir = osp.join(saved_dir,'inf')
    if not osp.isdir( inf_dir ):
        os.makedirs(inf_dir,exist_ok=True)


    files = sorted(os.listdir(model_dir))
    print(files)

    model_names = []
    for name in files:
        count = int(name.split('_')[0])
        if 'model' in name and (count % 6800 == 0 or count == 33830):
            model_names.append(name) 
    
    print(model_names)

    test_set = get_test_all()
    for d in ["test"]:
        DatasetCatalog.register("oak_" + d, lambda d=d: get_data_dicts(test_set,data_map) )

    # model_names: undone
    for name in model_names:
        count = int(name.split('_')[0])
        print(name)

        # load model
        cfg = get_cfg()
        cfg.merge_from_file(config_fp)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.DATASETS.TEST = ('oak_test',)
        cfg.SOLVER.IMS_PER_BATCH = 16
        wp = osp.join(model_dir,name)
        cfg,model = build_fix(cfg,len(train_map),wp)
        default_setup(
            cfg, args
        )
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        outputs = inference(cfg,model)
        comm.synchronize()
        if comm.is_main_process():
            fileObject = open(osp.join(inf_dir, str(count).zfill(5) + '_inf.pkl'), 'wb')
            pickle.dump(outputs,fileObject)
            fileObject.close()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--method', default='',type=str)
    parser.add_argument('--iteration', default=10)
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--batch_size', default= 8)
    parser.add_argument('--checkpoint', default=0)
    parser.add_argument('--importance', default=1000)
    parser.add_argument('--number', default=0)
    args = parser.parse_args()
    launch(
        main,
        4,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

