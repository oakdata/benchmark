import os
import os.path as osp
import sys
import torch
import cv2
import pickle
import numpy as np 
import random
import logging

from collections import OrderedDict
import json
import traceback
import argparse
import math

import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

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
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.config import get_cfg

from plain_train_net import do_train
from util import build_incremental_idk,returnmap,update_idknetwork

sourcedir = '/grogu/user/jianrenw/data/OAK_FRAME_N/Raw' 
annodir = '/grogu/user/jianrenw/data/OAK_LABEL_N'
curdir = '/grogu/user/jianrenw/baseline/release/inc/idk'
res_dir = '/grogu/user/jianrenw/baseline/release/baseline_res'
config_fp = '/grogu/user/jianrenw/baseline/release/faster_rcnn_R_50_C4.yaml'
steps_name = sorted(os.listdir(sourcedir))



def get_data_dicts(files_name, data_map):
    dataset_dicts = []
    for file_name in files_name:
        step_name, json_name = file_name.split('#')[0], file_name.split('#')[1]
        img_path = osp.join( sourcedir,step_name, json_name[:-5] + '.jpg')

        im = cv2.imread(img_path)
        height, width = im.shape[:2]
        record = {}
        record["file_name"] = img_path
        record["image_id"] = file_name
        record["height"],record["width"] = height,width

        
        objs = []
        f = osp.join(annodir,step_name,json_name)
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
                if box2d['x1'] != box2d['x2'] and box2d['y1'] != box2d['y2']:
                    objs.append(obj)
        
        if len(objs) != 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def save_file(count,model,lr,iteration,batch_size,train_map,tt,it,bt):
    saved_dir =  osp.join(res_dir,'inc_idk_' + str(lr) + '_' + str(iteration) + '_' + str(batch_size) + '_' + str(tt) + '_' + str(it) + '_' + str(bt))
    if not osp.isdir( saved_dir ):
        os.makedirs(saved_dir,exist_ok=True)

    model_dir = osp.join(saved_dir,'models')
    if not osp.isdir( model_dir ):
        os.makedirs(model_dir,exist_ok=True)
    stat_dir = osp.join(saved_dir,'stats')
    if not osp.isdir( stat_dir ):
        os.makedirs(stat_dir,exist_ok=True)

    jsonobj = json.dumps(train_map)
    f = open(osp.join(model_dir, str(count).zfill(5) + '_trainmap.json'),'w')
    f.write(jsonobj)
    f.close()

    torch.save(model.state_dict(), osp.join(model_dir,str(count).zfill(5) +'_model.pth'))
 
 
limitset = returnmap()

def calclass(files_name, train_map, limitset):
    for file_name in files_name:
        step_name, json_name = file_name.split('#')[0], file_name.split('#')[1]
        f = osp.join(annodir,step_name,json_name)
        labels = json.load(open(f,'r'))
        
        for label in labels:
            category = label['category']
            if category in limitset.keys():
                if limitset[category] not in train_map.values():
                    train_map [ len(train_map) ] = limitset[category]
    return train_map
    

def main(args):
    lr = float(args.lr)
    iteration = int(args.iteration)
    batch_size = int(args.batch_size)
    checkpoint = int(args.checkpoint)

    tthresh = 0.1
    idkthresh = 0.1
    bthresh = 0.6

    cfg = get_cfg()
    cfg.merge_from_file(config_fp)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.SOLVER.WARMUP_FACTOR = 0.0
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iteration
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    output_dir=  osp.join(curdir,'output')
    if not osp.isdir( output_dir ):
        os.makedirs(output_dir,exist_ok=True)
    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = ('oak_train',)
    cfg.DATASETS.TEST = ('oak_test',)
    cfg.MODEL.IDKTHRESH = (tthresh,idkthresh,bthresh)

    model = build_incremental_idk(cfg,checkpoint)
    default_setup(
        cfg, args
    )
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,find_unused_parameters=True
        )
    
    train_map = {i:i for i in range(0,20)} # net : annotation
    data_map = {value:key for key,value in train_map.items()} ## annotation : net

    old_num = 20
    count = 0
    trainset = []

    for step_name in steps_name:
        count += 1
        jsons_name = sorted(os.listdir(osp.join(annodir, step_name)),key=lambda x: ('_'.join(x.split('_')[:3]), x.split('_')[3]))
        trainset = [step_name + '#' +  json_name for json_name in jsons_name]
        train_map = calclass(trainset, train_map, limitset)
        data_map = {value:key for key,value in train_map.items()} 
        valid = get_data_dicts( trainset,data_map)
        if len(valid) == 0:
            trainset = []
            print('train set empty at', step_name)
            continue
        cfg, model = update_idknetwork(cfg,model,len(train_map), old_num)        
        old_num = len(train_map)

        for d in ["train"]:
            DatasetCatalog.register("oak_" + d, lambda d=d: get_data_dicts( trainset,data_map))
        do_train(cfg,model)
        for d in ["train"]:
            DatasetCatalog.remove("oak_" + d)
    
        if count % 10 == 0:
            res = []
            if torch.distributed.get_rank() == 0:
                save_file(count, model, lr, iteration,batch_size)
                print('finish ', count)
            comm.synchronize()


               
if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--iteration', default=10)
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