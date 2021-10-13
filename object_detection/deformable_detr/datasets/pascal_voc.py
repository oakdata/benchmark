# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T

from PIL import Image
import os
import os.path
import json
from pycocotools.coco import COCO
from collections import defaultdict

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union


def get_cat_string_to_id():
    CLASS_NAMES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        )
    
    # Convert to dictionary format, and assign numerical labels to each category
    class_dictionary = {}
    count = 0
    for elem in CLASS_NAMES:
        if elem not in class_dictionary.keys():
            class_dictionary[elem] = count
            count += 1
        else:
            continue
    print('-----------------------------------------------------------------------------------------')
    print('Class Dictionary', class_dictionary, count)
    return class_dictionary
        
def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Adapted from Detectron2 Pascal VOC dataloader

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with open(os.path.join(dirname, 'VOC2012', "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    
    with open(os.path.join(dirname, 'VOC2007', "ImageSets", "Main", split + ".txt")) as f:
        fileids_2 = np.loadtxt(f, dtype=np.str)

    fileids = fileids + fileids_2
    # with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
    #     fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = os.path.join(dirname, "Annotations/")

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

class PascalDetection(COCO):
    def __init__(self, dirname, transforms, return_masks, selection_set, cache_mode=False, local_rank=0, local_size=1,selection_index=-1):
        
        super().__init__()
        # Get the names of fromes from selection_set
        # Load the json file
        
        

        # with open(os.path.join(dirname, "ImageSets", "Main", selection_set + ".txt")) as f:
        
        #     fileids = np.loadtxt(f, dtype=np.str)
        if selection_set == 'train':
            with open(os.path.join(dirname, 'VOC2012', "ImageSets", "Main", 'trainval' + ".txt")) as f:
                fileids = np.loadtxt(f, dtype=np.str)
            
            with open(os.path.join(dirname, 'VOC2007', "ImageSets", "Main", 'trainval' + ".txt")) as f:
                fileids_2 = np.loadtxt(f, dtype=np.str)
            
            print('*'*40,type(fileids))
            fileids = fileids.tolist() + fileids_2.tolist()
        if selection_set == 'val':
            # with open(os.path.join(dirname, 'VOC2012', "ImageSets", "Main", 'trainval' + ".txt")) as f:
            #     fileids = np.loadtxt(f, dtype=np.str)
            
            with open(os.path.join(dirname, 'VOC2007', "ImageSets", "Main", 'test' + ".txt")) as f:
                fileids = np.loadtxt(f, dtype=np.str)
        

        

        annotation_dirname = os.path.join(dirname, 'VOC2012', "Annotations/")
        annotation_dirname_2 = os.path.join(dirname, 'VOC2007', "Annotations/")

        self.data_length = len(fileids) #+ len(fileids_2)

        assert self.data_length > 0, 'Error: Data length is 0!'

        # dicts = []

        if selection_index != -1:
            try:
                # if selection_index >= len(fileids):
                #     fileids_2 = fileids_2[selection_index-len(fileids):selection_index-len(fileids) + 16]
                # else:
                fileids = fileids[selection_index:selection_index+16]
            except: 
                fileids = fileids[selection_index:]
        
        self.list_of_img_paths = list()
        self.list_of_annotation_paths = list()
        self.ids = list()

        for fileid in fileids:
            if len(fileid) <=6:
                self.list_of_annotation_paths.append(os.path.join(annotation_dirname_2, fileid + ".xml"))
                self.list_of_img_paths.append(os.path.join(dirname,'VOC2007', "JPEGImages", fileid + ".jpg"))
                self.ids.append(fileid)
            else:
                self.list_of_annotation_paths.append(os.path.join(annotation_dirname, fileid + ".xml"))
                self.list_of_img_paths.append(os.path.join(dirname, 'VOC2012',"JPEGImages", fileid + ".jpg"))
                self.ids.append(fileid)
        
        # for fileid in fileids_2:
        #     self.list_of_annotation_paths.append(os.path.join(annotation_dirname_2, fileid + ".xml"))
        #     self.list_of_img_paths.append(os.path.join(dirname, 'VOC2007', "JPEGImages", fileid + ".jpg"))
        #     self.ids.append(fileid)

            # jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

            # with open(anno_file) as f:
            #     tree = ET.parse(f)

            # r = {
            #     "file_name": jpeg_file,
            #     "image_id": fileid,
            #     "height": int(tree.findall("./size/height")[0].text),
            #     "width": int(tree.findall("./size/width")[0].text),
            # }
            # instances = []

            # for obj in tree.findall("object"):
            #     cls = obj.find("name").text
            #     # We include "difficult" samples in training.
            #     # Based on limited experiments, they don't hurt accuracy.
            #     # difficult = int(obj.find("difficult").text)
            #     # if difficult == 1:
            #     # continue
            #     bbox = obj.find("bndbox")
            #     bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            #     # Original annotations are integers in the range [1, W or H]
            #     # Assuming they mean 1-based pixel indices (inclusive),
            #     # a box with annotation (xmin=1, xmax=W) covers the whole image.
            #     # In coordinate space this is represented by (xmin=0, xmax=W)
            #     bbox[0] -= 1.0
            #     bbox[1] -= 1.0
            #     instances.append(
            #         {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            #     )
            # r["annotations"] = instances
            # dicts.append(r)


        assert len(self.list_of_img_paths) > 0, f'no images found at {os.path.join(dirname, "JPEGImages")}'
        assert len(self.list_of_annotation_paths) > 0, f'no images found at {os.path.join(annotation_dirname)}'


        self.cat_to_id_dict = get_cat_string_to_id()

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.createIndex()





        # ----------------------------

        # f = open(selection_set,)
        # data = json.load(f)
        # f.close()
        # self.data_length = len(data)
        # if selection_index != -1:
        #     data = data[selection_index:selection_index+1]

        # # Get paths to images
        # assert len(data) != 0, f'Error loading, selected files is {data}'
        # self.list_of_img_paths = list()
        # self.list_of_annotation_paths = list()
        # self.ids = list()
        # for frame in data:
        #     # Parse the frame string to get the video name
        #     video_name = frame.split('.')[0]
        #     # Get the path to the image
        #     self.list_of_img_paths.append(os.path.join(img_folder, video_name,'源文件', frame[:-4]+'jpg'))
        #     self.ids.append(os.path.join(img_folder, video_name,'源文件', frame[:-4]+'jpg'))
        #     self.list_of_annotation_paths.append(os.path.join(ann_file, video_name,'标注结果', frame[:-4]+'json'))
        
        # assert len(self.list_of_img_paths) > 0, f'no images found at {img_folder}'
        # assert len(self.list_of_annotation_paths) > 0, f'no images found at {ann_file}'




        # # Get annotations to images
        # self.cat_to_id_dict = get_train_test_label_overlap()



        # # super(CocoDetection, self).__init__(img_folder, ann_file,
        # #                                     cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        # self._transforms = transforms
        # self.prepare = ConvertCocoPolysToMask(return_masks)
        # self.createIndex()
    
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        self.dataset = defaultdict(list)

        # Get annotations
        label_count = 0


        for ann_id, ann_path in enumerate(self.list_of_annotation_paths):

            with open(ann_path) as f:
                tree = ET.parse(f)
            
            # r = {
            #     "file_name": self.list_of_img_paths[ann_id],
            #     "image_id": fileid,
            #     "height": int(tree.findall("./size/height")[0].text),
            #     "width": int(tree.findall("./size/width")[0].text),
            # }

            objs = []
            for obj in tree.findall("object"):
                cls = obj.find("name").text

                # We include "difficult" samples in training.
                # Based on limited experiments, they don't hurt accuracy.
                # difficult = int(obj.find("difficult").text)
                # if difficult == 1:
                # continue
                bbox = obj.find("bndbox")

                if cls in self.cat_to_id_dict.keys():
                    bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                    # Original annotations are integers in the range [1, W or H]
                    # Assuming they mean 1-based pixel indices (inclusive),
                    # a box with annotation (xmin=1, xmax=W) covers the whole image.
                    # In coordinate space this is represented by (xmin=0, xmax=W)
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0
                    small_x = min(bbox[0],bbox[2])
                    small_y = min(bbox[1],bbox[3])
                    obj = {
                        "bbox": [small_x, small_y, abs(bbox[0]-bbox[2]), abs(bbox[1]-bbox[3])],
                        # "bbox": [box2d['x1'], box2d['y1'], box2d['x2']-box2d['x1'], box2d['y2']-box2d['y1']],
                        #"bbox_mode": BoxMode.XYXY_ABS,
                        # "category_id": data_map[ limitset[category] ],
                        "area": abs(bbox[0]-bbox[2])*abs(bbox[1]-bbox[3]),
                        "iscrowd": 0,
                        "category_id": self.cat_to_id_dict[cls],#label['id'],
                        "image_id": ann_id,
                        "id": label_count
                        }
                    objs.append(obj)
                    imgToAnns[ann_id].append(obj)
                    anns[label_count] = obj
                    catToImgs[self.cat_to_id_dict[cls]].append(ann_id)
                    self.dataset['annotations'].append(obj)
                    label_count +=1
            
            # Set up images
            imgs[ann_id] = {
                            "file_name": self.list_of_img_paths[ann_id],
                            "id": ann_id
            }
            self.dataset['images'].append(imgs[ann_id])
        
        # Set up categories
        for cat in self.cat_to_id_dict.keys():
            cats[self.cat_to_id_dict[cat]] = {
                                            'id': self.cat_to_id_dict[cat],
                                            "name": cat}
            self.dataset['categories'].append(cats[self.cat_to_id_dict[cat]])


        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
        # Recreate the dataset format
    
    def __len__(self):
        return len(self.list_of_img_paths)
    
    def __getitem__(self, idx):
        img_path = self.list_of_img_paths[idx]
        ann_path = self.list_of_annotation_paths[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        record = {}
        ## record["file_name"] = img_path
        record["image_id"] = idx
        # record["image_id"] = img_path
        # video_name + '/' + json_name.split('.')[0]
        ## record["height"],record["width"] = height,width

        with open(ann_path) as f:
            tree = ET.parse(f)

        
        objs = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")

            if cls in self.cat_to_id_dict.keys():
                
                
                bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
                # print(bbox)
                # print('after', bbox)
                if len(bbox) != 4:
                    print(f'Error, invalid bounding box{bbox}')
                    continue
                # assert len(bbox) == 4, f'Error, invalid bounding box{bbox}'
                # Original annotations are integers in the range [1, W or H]
                # Assuming they mean 1-based pixel indices (inclusive),
                # a box with annotation (xmin=1, xmax=W) covers the whole image.
                # In coordinate space this is represented by (xmin=0, xmax=W)
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                small_x = min(bbox[0],bbox[2])
                small_y = min(bbox[1],bbox[3])
                obj = {
                    "bbox": [small_x, small_y, abs(bbox[0]-bbox[2]), abs(bbox[1]-bbox[3])],
                    # "bbox": [box2d['x1'], box2d['y1'], box2d['x2']-box2d['x1'], box2d['y2']-box2d['y1']],
                    #"bbox_mode": BoxMode.XYXY_ABS,
                    # "category_id": data_map[ limitset[category] ],
                    # "area": abs(bbox[0]-bbox[2])*abs(bbox[1]-bbox[3]),
                    # "iscrowd": 0,
                    "category_id": self.cat_to_id_dict[cls],#label['id'],
                    # "image_id": ann_id,
                    # "id": label_count
                    }
                objs.append(obj)
                # imgToAnns[ann_id].append(obj)
                # anns[label_count] = obj
                # catToImgs[self.cat_to_id_dict[cls]].append(ann_id)
                # self.dataset['annotations'].append(obj)
                # label_count +=1
        record["annotations"] = objs
        target = record
        # target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        # print('before transform', target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w-1)
        boxes[:, 1::2].clamp_(min=0, max=h-1)
        assert (boxes[:, 2:] >= boxes[:, :2]).all(), f"boxes{boxes}"
        classes = [int(obj["category_id"]) for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # "bbox": [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']]
        # for conversion to coco api
        # area = torch.tensor([(obj['bbox'][2] -obj["bbox"][0])*(obj['bbox'][3] -obj["bbox"][1]) for obj in anno])
        area = torch.tensor([(obj['bbox'][2])*(obj['bbox'][3]) for obj in anno])
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.pascal_path)

    # Pick out the training and validation paths

    # Use the paths to get the corresponding image + annotation
    assert root.exists(), f'provided PASCAL path {root} does not exist'
    # mode = 'instances'

    # PATHS = {
    #     "train": (os.path.join(root, 'debug', 'relabel', 'train_frame.json')),
    #     "val": (os.path.join(root, 'debug', 'relabel' , 'test_frame.json'))
    #     }
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    # }


    # path_to_images = os.path.join(root, 'new_data')

    # path_to_annotations = os.path.join(root, 'new_anno')

    # selection_set = PATHS[image_set]
    selection_index = -1
    if args.train_mode == 'incremental':
    
        if image_set == 'val':
            selection_index = -1
        elif image_set =='train':
            try:
                selection_index = args.selection_index
            except:
                raise ValueError('args.selection_index does not exist, make sure it is in main')
        else:
            raise ValueError('Incorrect set provided')

    dataset = PascalDetection(dirname = root, selection_set = image_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),selection_index = selection_index)
    return dataset
