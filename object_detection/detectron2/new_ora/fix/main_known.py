import sys  
import os
import os.path as osp
import json
import traceback
import argparse
import math
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

from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.config import get_cfg

from plain_train_net import do_train
from util import build_incremental_fix, returnmap


curdir = ''
sourcedir = '' 
annodir = sourcedir
res_dir = ''
config_fp = ''

def get_data_dicts(files_name, data_map):
    dataset_dicts = []
    for file_name in files_name:
        video_name, step_name, json_name = file_name.split('#')[0], file_name.split('#')[1], file_name.split('#')[2]
        img_path = osp.join( sourcedir,video_name, step_name, json_name[:-5] + '.jpg')

        im = cv2.imread(img_path)
        try:
            height, width = im.shape[:2]
        except:
            print(img_path)
            print(file_name)
            print(0/0)
        record = {}
        record["file_name"] = img_path
        record["image_id"] = file_name
        record["height"],record["width"] = height,width

        
        objs = []
        f = osp.join(annodir,video_name,step_name,json_name)
        labels = json.load(open(f,'r'))
        
        for label in labels:
            obj_id = label['id']
            category = label['category']
            box2d = label['box2d']

            if category in limitset.keys():
                obj = {
                    "bbox": [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": data_map[ limitset[category] ],
                    }
                objs.append(obj)
        
        if len(objs) != 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def get_trainmap(limitset):
    meta_name = limitset.keys()
    mapping = {i:i for i in range(0,len(limitset.keys()))}
    return meta_name,mapping

def get_train_set(stop):
    trainset = []
    count = 0
    for video_name in videos_name:
        steps_name = sorted(os.listdir( osp.join(annodir,video_name)), key = lambda x:int(x.split('_')[1]))
        for step_name in steps_name:
            jsons_name = sorted(os.listdir(osp.join(annodir, video_name, step_name)),key=lambda x: int(x.split('_')[0]))
            for json_name in jsons_name:
                count = count + 1
                if count == stop:
                    return trainset
                if count % 17 != 0:
                    trainset.append(video_name + '#' + step_name + '#' +  json_name)
    return trainset

def save_file(count,model,lr,batch_size,iteration):
    saved_dir =  osp.join(res_dir,'new_oracle_fix_' + str(lr) + '_' + str(batch_size) )
    if not osp.isdir( saved_dir ):
        os.makedirs(saved_dir,exist_ok=True)

    model_dir = osp.join(saved_dir,'models')
    if not osp.isdir( model_dir ):
        os.makedirs(model_dir,exist_ok=True)
        
    stat_dir = osp.join(saved_dir,'stats')
    if not osp.isdir( stat_dir ):
        os.makedirs(stat_dir,exist_ok=True)

    torch.save(model.state_dict(), osp.join(model_dir,str(count).zfill(5) +'_model.pth'))

videos_name = os.listdir(sourcedir)
videos_name = sorted(videos_name)
limitset = returnmap()
meta_names,train_map = get_trainmap(limitset) # net : annotation
data_map = {value:key for key,value in train_map.items()} ## annotation : net

from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm


def main(args):
    interval = 3400

    lr = float(args.lr)
    batch_size = int(args.batch_size)
    iteration = int(args.iteration)

    while interval <= 34000:
        cfg = get_cfg()
        cfg.merge_from_file(config_fp)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
        cfg.SOLVER.WARMUP_FACTOR = 0.0
        cfg.SOLVER.WARMUP_ITERS = 0
        cfg.SOLVER.BASE_LR = lr
        cfg.SOLVER.IMS_PER_BATCH = batch_size
        cfg.OUTPUT_DIR = osp.join(curdir,'output')
        cfg.DATASETS.TRAIN = ('oak_train',)
        cfg.DATASETS.TEST = ('oak_test',)
        if not osp.isdir( cfg.OUTPUT_DIR ):
            os.makedirs(cfg.OUTPUT_DIR,exist_ok=True) 
        train_set = get_train_set(interval)

        cfg.SOLVER.MAX_ITER = int( (len(train_set) + batch_size-1)/batch_size ) * iteration

        cfg,model = build_incremental_fix(cfg,len(train_map),0)
        default_setup(
            cfg, args
        )

        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        for d in ["train"]:
            DatasetCatalog.register("oak_" + d, lambda d=d: get_data_dicts(train_set,data_map) )
            do_train(cfg,model)
            if torch.distributed.get_rank() == 0:
                save_file(interval, model, lr, batch_size,iteration)
            comm.synchronize()
        
        for d in ["train"]:
            DatasetCatalog.remove("oak_" + d)
        
        interval += 3400


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--iteration', default=10)
    parser.add_argument('--lr', default=0.05)
    parser.add_argument('--batch_size', default= 16)
    parser.add_argument('--checkpoint', default=0)
    args = parser.parse_args()
    launch(
        main,
        8,
        num_machines=1,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


