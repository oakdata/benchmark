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


def get_train_test_label_overlap(test_class_path = '/project_data/held/jianrenw/debug/relabel/train_cat.json', train_class_path = '/project_data/held/jianrenw/debug/relabel/train_cat.json'):
    f = open(test_class_path,)
    test_data = json.load(f)
    f.close()
    
    f = open(train_class_path,)
    train_data = json.load(f)
    f.close()

    intersection_set = set.intersection(set(test_data), set(train_data))
    intersection_list = list(intersection_set)

    # Sort the intersection set so that it's reproducible
    intersection_list_sorted = sorted(intersection_list)

    # Convert to dictionary format, and assign numerical labels to each category
    class_dictionary = {}
    count = 0
    for elem in intersection_list_sorted:
        if elem not in class_dictionary.keys():
            class_dictionary[elem] = count
            count += 1
        else:
            continue
    return class_dictionary
        

class OAKDetection(COCO):
# class OAKDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, selection_set, cache_mode=False, local_rank=0, local_size=1,selection_index=-1):
        
        super().__init__()
        # Get the names of fromes from selection_set
        # Load the json file
        f = open(selection_set,)
        data = json.load(f)
        f.close()
        self.data_length = len(data)
        if selection_index != -1:
            data = data[selection_index:selection_index+1]

        # Get paths to images
        assert len(data) != 0, f'Error loading, selected files is {data}'
        self.list_of_img_paths = list()
        self.list_of_annotation_paths = list()
        self.ids = list()
        for frame in data:
            # Parse the frame string to get the video name
            video_name = frame.split('.')[0]
            # Get the path to the image
            self.list_of_img_paths.append(os.path.join(img_folder, video_name,'源文件', frame[:-4]+'jpg'))
            self.ids.append(os.path.join(img_folder, video_name,'源文件', frame[:-4]+'jpg'))
            self.list_of_annotation_paths.append(os.path.join(ann_file, video_name,'标注结果', frame[:-4]+'json'))
        
        assert len(self.list_of_img_paths) > 0, f'no images found at {img_folder}'
        assert len(self.list_of_annotation_paths) > 0, f'no images found at {ann_file}'




        # Get annotations to images
        self.cat_to_id_dict = get_train_test_label_overlap()



        # super(CocoDetection, self).__init__(img_folder, ann_file,
        #                                     cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.createIndex()
    
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        self.dataset = defaultdict(list)

        # Get annotations
        label_count = 0
        for ann_id, ann_path in enumerate(self.list_of_annotation_paths):

            # Extract annotations
            f = open(ann_path,)
            data = json.load(f)
            f.close()
            try:
            # labels = data[0]['DataList'][0]['labels']
                labels = data[0]['labels']
            except: 
                labels = data['DataList'][0]['labels']
            # imgToAnns[]

            objs = []
            for label in labels:
                # obj_id = label['id']
                # # check if obj_id is in list of desired objects 
                # try:
                #     assert int(obj_id) >= 0
                # except:
                #     print('invalid obj_id:', obj_id)
                #     continue
                category = label['category']
                box2d = label['box2d']

                if category in self.cat_to_id_dict.keys():
                    small_x = min(box2d['x1'],box2d['x2'])
                    small_y = min(box2d['y1'],box2d['y2'])
                    obj = {
                        "bbox": [small_x, small_y, abs(box2d['x2']-box2d['x1']), abs(box2d['y2']-box2d['y1'])],
                        # "bbox": [box2d['x1'], box2d['y1'], box2d['x2']-box2d['x1'], box2d['y2']-box2d['y1']],
                        #"bbox_mode": BoxMode.XYXY_ABS,
                        # "category_id": data_map[ limitset[category] ],
                        "area": abs(box2d['x2']-box2d['x1'])*abs(box2d['y2']-box2d['y1']),
                        "iscrowd": 0,
                        "category_id": self.cat_to_id_dict[category],#label['id'],
                        "image_id": ann_id,
                        "id": label_count
                        }
                    objs.append(obj)
                    imgToAnns[ann_id].append(obj)
                    anns[label_count] = obj
                    catToImgs[self.cat_to_id_dict[category]].append(ann_id)
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

        
        # Set up the connection between categories and annotations


                

                


        # if 'annotations' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         imgToAnns[ann['image_id']].append(ann)
        #         anns[ann['id']] = ann

        # if 'images' in self.dataset:
        #     for img in self.dataset['images']:
        #         imgs[img['id']] = img

        # if 'categories' in self.dataset:
        #     for cat in self.dataset['categories']:
        #         cats[cat['id']] = cat

        # if 'annotations' in self.dataset and 'categories' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         catToImgs[ann['category_id']].append(ann['image_id'])

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
        #video_name + '/' + json_name.split('.')[0]
        ## record["height"],record["width"] = height,width

        # Load annotation
        f = open(ann_path,)
        data = json.load(f)
        f.close()
        try:
            # labels = data[0]['DataList'][0]['labels']
            labels = data[0]['labels']
        except: 
            labels = data['DataList'][0]['labels']
        objs = []
        for label in labels:
            obj_id = label['id']
            # check if obj_id is in list of desired objects 
            try:
                assert int(obj_id) >= 0
            except:
                print('invalid obj_id:', obj_id)
                continue
            category = label['category']
            box2d = label['box2d']

            if category in self.cat_to_id_dict.keys():
                small_x = min(box2d['x1'],box2d['x2'])
                small_y = min(box2d['y1'],box2d['y2'])
                obj = {
                    "bbox": [small_x, small_y, abs(box2d['x2']-box2d['x1']), abs(box2d['y2']-box2d['y1'])],
                    # "bbox": [box2d['x1'], box2d['y1'], box2d['x2']-box2d['x1'], box2d['y2']-box2d['y1']],
                    #"bbox_mode": BoxMode.XYXY_ABS,
                    # "category_id": data_map[ limitset[category] ],
                    "category_id": self.cat_to_id_dict[category]#label['id']
                    }
                objs.append(obj)
        
        # if len(objs) != 0:
        record["annotations"] = objs
            # dataset_dicts.append(record)
            
        
        # img, target = super(CocoDetection, self).__getitem__(idx)
        # image_id = self.ids[idx]
        target = record
        # target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
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

# def get_data_dicts(jsons_name, data_map):
#     dataset_dicts = []
#     for json_name in jsons_name:
#         video_name = json_name.split('.')[0]
#         img_path = osp.join( sourcedir, video_name, '源文件', video_name,json_name[:-5] + '.jpg')

#         im = cv2.imread(img_path)
#         height, width = im.shape[:2]
#         record = {}
#         record["file_name"] = img_path
#         record["image_id"] = video_name + '/' + json_name.split('.')[0]
#         record["height"],record["width"] = height,width

        
#         objs = []
#         f = osp.join(sourcedir,video_name,'转换后结果数据',json_name)
#         anno = json.load(open(f,'r'))
#         labels = anno["DataList"][0]["labels"]
        
        
#         for label in labels:
#             obj_id = label['id']
#             category = label['category']
#             box2d = label['box2d']

#             if category in limitset.keys():
#                 obj = {
#                     "bbox": [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']],
#                     "bbox_mode": BoxMode.XYXY_ABS,
#                     "category_id": data_map[ limitset[category] ],
#                     }
#                 objs.append(obj)
        
#         if len(objs) != 0:
#             record["annotations"] = objs
#             dataset_dicts.append(record)

#     return dataset_dicts

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
    root = Path(args.oak_path)

    # Pick out the training and validation paths

    # Use the paths to get the corresponding image + annotation
    assert root.exists(), f'provided OAK path {root} does not exist'
    mode = 'instances'

    PATHS = {
        "train": (os.path.join(root, 'debug', 'relabel', 'train_frame.json')),
        "val": (os.path.join(root, 'debug', 'relabel' , 'test_frame.json'))
        }
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    # }

    path_to_images = os.path.join(root, 'new_data')

    path_to_annotations = os.path.join(root, 'new_anno')

    selection_set = PATHS[image_set]
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

    dataset = OAKDetection(img_folder = path_to_images, ann_file = path_to_annotations, selection_set = selection_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),selection_index = selection_index)
    return dataset
