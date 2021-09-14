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
from detectron2.modeling.roi_heads.roi_heads import build_roi_heads
import json


cat_fp = ''
pretrained_fp = ''
loading_fp = ''

def returnmap():
    f = open(cat_fp, 'r')
    content = f.read()
    categories = json.loads(content)
    return categories

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.engine import default_argument_parser, default_setup, launch

def build_incremental_fix(cfg,num_oc,checkpoint):
    model = build_model(cfg)
    if checkpoint == 0:
        DetectionCheckpointer(model).load(pretrained_fp)
        
        layer1 = nn.Linear(2048,num_oc-20)
        layer2 = nn.Linear(2048,(num_oc-20) *4)
        nn.init.normal_(layer1.weight, std=0.01)
        nn.init.normal_(layer2.weight, std=0.001)
        for l in [layer1,layer2]:
            nn.init.constant_(l.bias, 0)

        w = model.state_dict()
        print(w['roi_heads.box_predictor.cls_score.weight'].size())
        print(layer1.weight.size())

        tmpw = w['roi_heads.box_predictor.cls_score.weight'].cpu()
        w['roi_heads.box_predictor.cls_score.weight'] = torch.cat( ( tmpw[:-1], layer1.weight, tmpw[-1:]), 0)
        tmpw = w['roi_heads.box_predictor.cls_score.bias'].cpu()
        w['roi_heads.box_predictor.cls_score.bias'] = torch.cat( (  tmpw[:-1] , layer1.bias, tmpw[-1:]), 0)
        tmpw = w['roi_heads.box_predictor.bbox_pred.weight'].cpu()
        w['roi_heads.box_predictor.bbox_pred.weight'] = torch.cat( ( tmpw[:-4], layer2.weight, tmpw[-4:]) ,0)
        tmpw = w['roi_heads.box_predictor.bbox_pred.bias'].cpu()
        w['roi_heads.box_predictor.bbox_pred.bias'] = torch.cat( (   tmpw[:-4], layer2.bias, tmpw[-4:] ),0 )

    elif checkpoint != 0:
        path = loading_fp
        wp = osp.join(path,str(checkpoint).zfill(5)+'_model.pth')
        w_module = torch.load(wp)
        w = {}
        for key in w_module.keys():
            w[key[7:]] = w_module[key]
        
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_oc
    model.roi_heads =  build_roi_heads(cfg, model.backbone.output_shape())
    model.load_state_dict(w)
    model.to(torch.device(cfg.MODEL.DEVICE))
    
    for name,parameters in model.named_parameters():
        if 'backbone' in name:
            parameters.requires_grad = False
        else:
            parameters.requires_grad = True
    return cfg,model
    
    

   

       
