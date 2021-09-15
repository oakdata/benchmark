import os
import os.path as osp
import sys
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
from util import build_icarl2_idk,returnmap,update_idknetwork


curdir = ''
sourcedir = '' 
annodir = sourcedir
res_dir = ''
config_fp = ''

videos_name = os.listdir(sourcedir)
videos_name = sorted(videos_name)
num_pc = 5


def get_cnt(records):
    cur_cnt,mem_cnt = 0,0
    for record in records:
        if 'memory' in record['image_id']:
            mem_cnt += 1
        else:
            cur_cnt += 1
    return cur_cnt, mem_cnt

def get_data_dicts(files_name, data_map,img_memory):
    dataset_dicts = []
    cur_cnt,mem_cnt = 0,0
    for file_name in files_name:
        video_name, step_name, json_name = file_name.split('#')[0], file_name.split('#')[1], file_name.split('#')[2]
        img_path = osp.join( sourcedir,video_name, step_name, json_name[:-5] + '.jpg')

        im = cv2.imread(img_path)
        height, width = im.shape[:2]
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
                if box2d['x1'] != box2d['x2'] and box2d['y1'] != box2d['y2']:
                    objs.append(obj)
        
        if len(objs) != 0:
            record["annotations"] = objs
            dataset_dicts.append(record)

    if len(dataset_dicts) == 0:
        return dataset_dicts
    
    for taski,imgs in enumerate(img_memory):
        if imgs != None:
            imgid = random.randint(0,len(imgs)-1)
            img  = imgs[imgid]
            filename, bbox = img
                
            record = {}
            record['file_name'] = filename
            record['image_id'] = 'memory' + str(taski)
            im = cv2.imread(filename)
            height, width = im.shape[:2]
            record["height"],record["width"] = height,width

            objs = []

            obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": taski,
                    }
            objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    

    return dataset_dicts

def save_file(count,model,lr,iteration,batch_size,img_memory,train_map,tt,it,bt):
    saved_dir =  osp.join(res_dir,'ic2_idk_' + str(lr) + '_' + str(iteration) + '_' + str(batch_size) + '_' + str(tt) + '_' + str(it) + '_' + str(bt))
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

    imgobj = json.dumps(img_memory)
    f = open(osp.join(model_dir, str(count).zfill(5) + '_img.json'),'w')
    f.write(imgobj)
    f.close()

    torch.save(model.state_dict(), osp.join(model_dir,str(count).zfill(5) +'_model.pth'))
 
limitset = returnmap()

def calclass(files_name, train_map, limitset):
    for file_name in files_name:
        video_name, step_name, json_name = file_name.split('#')[0], file_name.split('#')[1], file_name.split('#')[2]
        f = osp.join(annodir,video_name,step_name,json_name)
        labels = json.load(open(f,'r'))
        
        for label in labels:
            category = label['category']
            if category in limitset.keys():
                if limitset[category] not in train_map.values():
                    train_map [ len(train_map) ] = limitset[category]
    return train_map
     
def update_img_memory(valid,img_memory):
    for data in valid:
        if 'memory' not in data['image_id']:
            filename = data['file_name']
            objs = data['annotations']
            for obj in objs:
                class_id = int(obj['category_id'])
                ele = (filename,obj['bbox'])
                if img_memory[class_id] == None:
                    img_memory[class_id] = []
                    img_memory[class_id].append(ele)     
                elif len(img_memory[class_id]) < num_pc:
                    img_memory[class_id].append(ele)          
                else:
                    del_id = random.randint(0,num_pc)
                    if del_id != num_pc:
                        img_memory[class_id][del_id] = ele
    return img_memory


def main(args):
    lr = float(args.lr)
    iteration = int(args.iteration)
    batch_size = int(args.batch_size)
    checkpoint = int(args.checkpoint)

    tthresh = 0.2
    idkthresh = 0.2
    bthresh = 0.6

    cfg = get_cfg()
    cfg.merge_from_file(config_fp)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.SOLVER.WARMUP_FACTOR = 0.0
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = iteration
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    output_dir=  osp.join(curdir,'output')
    if not osp.isdir( output_dir ):
        os.makedirs(output_dir,exist_ok=True)
    cfg.OUTPUT_DIR = output_dir
    cfg.DATASETS.TRAIN = ('kcam_train',)
    cfg.DATASETS.TEST = ('kcam_test',)
    cfg.MODEL.IDKTHRESH = (tthresh,idkthresh,bthresh)

    train_map = {i:i for i in range(0,20)} # net : annotation
    data_map = {value:key for key,value in train_map.items()} ## annotation : net
    img_memory = [None] * len(train_map)
    
    old_num = 20
    count = 0
    trainset = []

    if checkpoint != 0:
        trainmap_pth = osp.join(osp.join(res_dir,'ic2_idk_' + str(lr) + '_' + str(iteration) + '_' + str(batch_size) + '_' + str(tthresh) + '_' + str(idkthresh) + '_' + str(bthresh), 'models',str(checkpoint).zfill(5) + '_trainmap.json'))
        f = open(trainmap_pth,'r')
        train_map = json.load(f)
        data_map = {value:key for key,value in train_map.items()}
        print(len(train_map))
        f.close()
        old_num = len(train_map)

        img_pth = osp.join(osp.join(res_dir,'ic2_idk_' + str(lr) + '_' + str(iteration) + '_' + str(batch_size) + '_' + str(tthresh) + '_' + str(idkthresh) + '_' + str(bthresh),'models',str(checkpoint).zfill(5) + '_img.json'))
        f = open(img_pth,'r')
        img_memory = json.load(f)
        f.close()

    model = build_icarl2_idk(cfg,len(train_map),checkpoint)
    default_setup(
        cfg, args
    )
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,find_unused_parameters=True
        )
    
    for video_name in videos_name:
        steps_name = sorted(os.listdir( osp.join(annodir,video_name)), key = lambda x:int(x.split('_')[1]))
        for step_name in steps_name:
            jsons_name = sorted(os.listdir(osp.join(annodir, video_name, step_name)),key=lambda x: int(x.split('_')[0]))

            for json_name in jsons_name:
                count = count + 1
                if count <= checkpoint:
                    continue
                print(count)
                
                if count % 17 != 0:
                    if len(trainset) != batch_size:
                        trainset.append(video_name + '#' + step_name + '#' +  json_name)
                    if len(trainset) == batch_size:
                        train_map = calclass(trainset, train_map, limitset)
                        data_map = {value:key for key,value in train_map.items()} 
                        valid = get_data_dicts(trainset,data_map,img_memory)

                        cur_cnt,mem_cnt = get_cnt(valid)
                        if len(valid) == 0 or cur_cnt == 0:
                            trainset = []
                            continue

                        cfg, model = update_idknetwork(cfg,model,len(train_map), old_num)    
                        img_memory.extend([None] * (len(train_map)-old_num))    
                        old_num = len(train_map)

                        for d in ["train"]:
                            DatasetCatalog.register("kcam_" + d, lambda d=d: get_data_dicts( trainset,data_map,img_memory))                   
                        
                        batch_num = 1 if int(len(valid)/batch_size) <= 1 else int((len(valid)-1)/batch_size) + 1
                        maxiter = cfg.SOLVER.MAX_ITER * batch_num
                        
                        if mem_cnt == 0:
                            cm_ratio = 0
                        elif cur_cnt / mem_cnt > 1:
                            cm_ratio = 1
                        else:
                            cm_ratio = cur_cnt / mem_cnt  
                            
                        
                        do_train(cfg,model,maxiter,cm_ratio)
                        img_memory = update_img_memory(valid,img_memory)
                                    
                        for d in ["train"]:
                            DatasetCatalog.remove("kcam_" + d)

                        trainset = []
                
                elif count% 170 == 0:
                    res = []
                    if torch.distributed.get_rank() == 0:
                        save_file(count, model,lr,iteration,batch_size,img_memory,train_map,tthresh,idkthresh,bthresh)
                    comm.synchronize()

    torch.save(model.state_dict(), osp.join(curdir,'icarl2_final.pth'))

               
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