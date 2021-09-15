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

from detectron2.solver import build_optimizer, build_lr_scheduler
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import torch.nn as nn
import json

from detectron2.modeling.roi_heads.roi_heads import build_roi_heads


cat_fp = ''
pretrained_fp = ''
loading_fp = ''

def returnmap():
    f = open(cat_fp, 'r')
    content = f.read()
    categories = json.loads(content)
    return categories

def build_incremental_idk(cfg,checkpoint):

    model = build_model(cfg)

    if checkpoint == 0:
        model.roi_heads = build_roi_heads(cfg,model.backbone.output_shape())
        DetectionCheckpointer(model).load(pretrained_fp)
    else:
        path = loading_fp
        trainmap_pth = osp.join(path, str(checkpoint).zfill(5) + '_trainmap.json')
        f = open(trainmap_pth,'r')

        train_map = json.load(f)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_map)
        model.roi_heads = build_roi_heads(cfg,model.backbone.output_shape())

        wp = osp.join(path,str(checkpoint).zfill(5) +'_model.pth')
        w_module = torch.load(wp)
        w = {}
        for key in w_module.keys():
            w[key[7:]] = w_module[key]
        model.load_state_dict(w)

    model.to(torch.device(cfg.MODEL.DEVICE))
    for name,parameters in model.named_parameters():
        if 'backbone' in name:
            parameters.requires_grad = False
        else:
            parameters.requires_grad = True
    return model

def update_idknetwork(cfg, model, num_oc, old_num):


    layer1 = nn.Linear(2048,num_oc-old_num)
    layer2 = nn.Linear(2048,(num_oc-old_num) *4)
    nn.init.normal_(layer1.weight, std=0.01)
    nn.init.normal_(layer2.weight, std=0.001)
    for l in [layer1,layer2]:
        nn.init.constant_(l.bias, 0)

    w = model.state_dict()

    tmpw = w['module.roi_heads.box_predictor.cls_score.weight'].cpu()
    w['module.roi_heads.box_predictor.cls_score.weight'] = torch.cat( ( tmpw[:-1], layer1.weight , tmpw[-1:]), 0)
    tmpw = w['module.roi_heads.box_predictor.cls_score.bias'].cpu()
    w['module.roi_heads.box_predictor.cls_score.bias'] = torch.cat( (  tmpw[:-1] , layer1.bias, tmpw[-1:]), 0)
    tmpw = w['module.roi_heads.box_predictor.bbox_pred.weight'].cpu()
    w['module.roi_heads.box_predictor.bbox_pred.weight'] = torch.cat( ( tmpw[:-4], layer2.weight , tmpw[-4:]) ,0)
    tmpw = w['module.roi_heads.box_predictor.bbox_pred.bias'].cpu()
    w['module.roi_heads.box_predictor.bbox_pred.bias'] = torch.cat( (   tmpw[:-4], layer2.bias , tmpw[-4:] ),0 )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_oc
    model.module.roi_heads =  build_roi_heads(cfg, model.module.backbone.output_shape())
    model.load_state_dict(w)


    model.to(torch.device(cfg.MODEL.DEVICE))
    for name,parameters in model.named_parameters():
        if 'module.backbone' in name:
            parameters.requires_grad = False
        else:
            parameters.requires_grad = True

    return cfg,model
 