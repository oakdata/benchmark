import os
import os.path as osp
import torch
import cv2
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import pickle
import numpy as np
import os.path as osp
import json

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

sourcedir = '' 
annodir = ''

def parse_rec(file_name,idk,classname,limitset,train_map):
    video_name, step_name, json_name = file_name.split('#')[0], file_name.split('#')[1], file_name.split('#')[2]
    f = osp.join(annodir,video_name,step_name,json_name)
    labels = json.load(open(f,'r'))

    bbox = []
    for label in labels:
        obj_id = label['id']
        category = label['category']

        box2d = label['box2d']
        abbox = [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']]

        if idk == False and classname == category:
            bbox.append(abbox)
        elif idk == True:
            # this category belongs to limitset
            if category in limitset.keys():
            # this category not belongs to the trainset
                if limitset[category] not in train_map.values():
                    bbox.append(abbox)

    return bbox


def voc_eval(classname,class_id,superframe,train_map,outputs,limitset,idk =False,ovthresh=0.5,use_07_metric=False):
    
    #print(len(outputs))

    # load annots
    recs = {}
    for frame in superframe:
        recs[frame] = parse_rec(frame,idk,classname,limitset,train_map)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for frame in superframe:
        bbox = np.array(recs[frame])
        det = [False] * len(recs[frame])
        npos = npos + len(recs[frame])
        class_recs[frame] = {"bbox": bbox, "det": det}

    # no positive values
    if npos == 0:
        return -1,-1,-1

    # read predictions
    BB = []
    confidence = []
    image_ids = []

    for obj in outputs:
        image_ids.append(obj['image_id'])
        confidence.append(obj['score'])
        BB.append([obj['xmin'],obj['ymin'],obj['xmax'],obj['ymax']])
        
    confidence = np.array(confidence)
    BB = np.array(BB)

    # no valid predictions
    if confidence.shape[0] == 0:
        return 0, 0, 0

    #print(image_ids)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["det"][jmax]:
                tp[d] = 1.0
                R["det"][jmax] = 1
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


