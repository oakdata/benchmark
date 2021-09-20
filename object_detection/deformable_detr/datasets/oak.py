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

import copy
import random

def get_train_test_label_overlap(test_class_path = '/grogu/user/jianrenw/data/relabel/test_cat.json', train_class_path = '/grogu/user/jianrenw/data/relabel/train_cat.json'):
    # def get_train_test_label_overlap(test_class_path = '/project_data/held/jianrenw/debug/relabel/test_cat.json', train_class_path = '/project_data/held/jianrenw/debug/relabel/train_cat.json'):
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
    print('-----------------------------------------------------------------------------------------')
    print('Class Dictionary', class_dictionary, count)
    return class_dictionary
        

class OAKDetection(COCO):
# class OAKDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, selection_set, cache_mode=False, local_rank=0, local_size=1,selection_index=-1,train_settings = None):
        
        super().__init__()

        # Initialize variables
        self.list_of_img_paths = list()
        self.list_of_annotation_paths = list()
        self.ids = list()

        # If it is either training or validation 
        if selection_set:

            # f = open(selection_set,)
            # data = json.load(f)
            # f.close()
            # self.data_length = len(data)

            # If it is the ewc/ic2/incremental methods
            if selection_index != -1:
                # data = os.listdir(os.path.join(img_folder,f'step_{selection_index.zfill(5)}'))

                # Load the sorted data
                data = [os.path.join(img_folder,f'step_{str(selection_index).zfill(5)}', image_in_step) for image_in_step in sorted(os.listdir(os.path.join(img_folder,f'step_{str(selection_index).zfill(5)}')),key=lambda x: int(x.split('.')[0].split('_')[-1]))]

                # Get matching labels
                data_label = [os.path.join(ann_file,f'step_{str(selection_index).zfill(5)}', image_in_step) for image_in_step in sorted(os.listdir(os.path.join(ann_file,f'step_{str(selection_index).zfill(5)}')),key=lambda x: int(x.split('.')[0].split('_')[-1]))]
            
            # If it is an offline training method
            else:
                # Go through all the steps and get all the images

                # Check if it is the training or test set
                if 'TEST' in img_folder:
                    data = [os.path.join(img_folder, img_in_folder) for img_in_folder in sorted(os.listdir(img_folder),key=lambda x: int(x.split('.')[0].split('_')[-1]))]
                    data_label = [os.path.join(ann_file, img_in_folder) for img_in_folder in sorted(os.listdir(ann_file),key=lambda x: int(x.split('.')[0].split('_')[-1]))]
                
                else:
                    data = []
                    data_label = []
                    for step in sorted(os.listdir(img_folder),key = lambda x: int(x.split('_')[-1])):
                        print(f'Retrieving Images from {step}')
                        data = data + [os.path.join(img_folder,step, image_in_step) for image_in_step in sorted(os.listdir(os.path.join(img_folder,step)),key=lambda x: int(x.split('.')[0].split('_')[-1]))]
                        data_label = data_label + [os.path.join(ann_file,step, image_in_step) for image_in_step in sorted(os.listdir(os.path.join(ann_file,step)),key=lambda x: int(x.split('.')[0].split('_')[-1]))]
                
        
                    

                # try:
                #     data = data[selection_index:selection_index+16]
                # except: 
                #     data = data[selection_index:]
        
            # # Get the train_settings
            # if train_settings:
            #     if train_settings['train_mode'] == 'ic2':
            #         self.ic2_memory = train_settings['memory']
            #         for taski, imgs in enumerate(self.ic2_memory):
            #             if imgs != None:
            #                 imgid = random.randint(0,len(imgs)-1)
            #                 img  = imgs[imgid]
            #                 filename, bbox = img
            #                 # record = {}
            #                 # record['file_name'] = filename
            #                 # record['image_id'] = 'memory' + str(taski)

            # Get paths to images
            assert len(data) != 0, f'Error loading, selected files is {data}'

            # Get matching labels and images

            self.list_of_img_paths = data
            self.ids = data
            self.list_of_annotation_paths = data_label
            
        
        # Get the train_settings
        if train_settings:
            if train_settings['train_mode'] == 'ic2':
                self.ic2_memory = train_settings['memory']
                for taski, imgs in enumerate(self.ic2_memory):
                    if imgs != None:
                        imgid = random.randint(0,len(imgs)-1)

                        # print('image-------------', imgs)
                        img  = imgs[imgid]
                        filename, bbox = img
                        self.list_of_img_paths.append(filename)
                        self.ids.append(filename)

                        # annotation_boundingbox = {'category':,
                        #                         'bbox2d'}
                        self.list_of_annotation_paths.append(bbox)
                        # record = {}
                        # record['file_name'] = filename
                        # record['image_id'] = 'memory' + str(taski)
            if train_settings['train_mode'] == 'memory':
                self.ic2_memory = train_settings['memory']
                for taski, imgs in enumerate(self.ic2_memory):
                    if imgs != None:
                        imgid = random.randint(0,len(imgs)-1)

                        # print('image-------------', imgs)
                        img  = imgs[imgid]
                        filename, bbox = img
                        self.list_of_img_paths.append(filename)
                        self.ids.append(filename)

                        # annotation_boundingbox = {'category':,
                        #                         'bbox2d'}
                        self.list_of_annotation_paths.append(bbox)

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
            if type(ann_path) == str:
                f = open(ann_path,)
                data = json.load(f)
                f.close()

                labels = data
                # try:
                # # labels = data[0]['DataList'][0]['labels']
                #     print(data)
                #     labels = data[0]['labels']
                # except: 
                #     labels = data['DataList'][0]['labels']
                from_memory = False
            
                # objs = []
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
                            "id": label_count,
                            "from_memory": from_memory
                            }
                        # objs.append(obj)
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
            
            else:
                # Not a string, which means that it is memory
                label = ann_path
                from_memory = True
                # for label in labels:
                # category = label['category']
                # box2d = label['box2d']

                # if category in self.cat_to_id_dict.keys():
                # small_x = min(box2d['x1'],box2d['x2'])
                # small_y = min(box2d['y1'],box2d['y2'])

                label['from_memory'] = True
                # obj = copy.deepcopy(label)
                obj = {
                    "bbox": label['bbox'],
                    "area": label['area'],
                    "iscrowd": label['iscrowd'],
                    "category_id": label['category_id'],#label['id'],
                    "image_id": ann_id,
                    "id": label_count,
                    "from_memory": from_memory
                    }
                # obj = {
                #     "bbox": [small_x, small_y, abs(box2d['x2']-box2d['x1']), abs(box2d['y2']-box2d['y1'])],
                #     "area": abs(box2d['x2']-box2d['x1'])*abs(box2d['y2']-box2d['y1']),
                #     "iscrowd": 0,
                #     "category_id": self.cat_to_id_dict[category],#label['id'],
                #     "image_id": ann_id,
                #     "id": label_count,
                #     "from_memory": from_memory
                #     }
                # objs.append(obj)
                imgToAnns[ann_id].append(obj)
                anns[label_count] = obj
                try:
                    catToImgs[self.cat_to_id_dict[category]].append(ann_id)
                except:
                    continue
                
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
        if type(ann_path) == str:
            from_memory = False

            f = open(ann_path,)
            data = json.load(f)
            f.close()
            labels = data
            # try:
            #     # labels = data[0]['DataList'][0]['labels']
            #     labels = data[0]['labels']
            # except: 
            #     labels = data['DataList'][0]['labels']
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
                        "category_id": self.cat_to_id_dict[category],
                        "from_memory": from_memory #label['id']
                        }
                    objs.append(obj)
        else:
            labels = ann_path
            from_memory = True
            objs = []
            # for label in labels:
                # obj_id = label['id']
                # # check if obj_id is in list of desired objects 
                # try:
                #     assert int(obj_id) >= 0
                # except:
                #     print('invalid obj_id:', obj_id)
                #     continue
                # category = label['category']
                # box2d = label['box2d']

                # if category in self.cat_to_id_dict.keys():
                #     small_x = min(box2d['x1'],box2d['x2'])
                #     small_y = min(box2d['y1'],box2d['y2'])

            obj = {
                "bbox": labels['bbox'],
                # "area": label['area'],
                # "iscrowd": label['iscrowd'],
                "category_id": labels['category_id'],#label['id'],
                # "image_id": ann_id,
                # "id": label_count,
                "from_memory": from_memory
                }
                
                # obj = {
                #     "bbox": [small_x, small_y, abs(box2d['x2']-box2d['x1']), abs(box2d['y2']-box2d['y1'])],
                #     # "bbox": [box2d['x1'], box2d['y1'], box2d['x2']-box2d['x1'], box2d['y2']-box2d['y1']],
                #     #"bbox_mode": BoxMode.XYXY_ABS,
                #     # "category_id": data_map[ limitset[category] ],
                #     "category_id": self.cat_to_id_dict[category]#label['id']
                #     }
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

        # Whether a label is from memory
        try:
            from_memory = target['annotations'][0]['from_memory']
        except:
            # print('Seems like list is 0')
            from_memory = 0

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
        target["from_memory"] = torch.as_tensor([int(from_memory == True)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set in ['train', 'memory']:
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

    # PATHS = {
    #     "train": (os.path.join(root, 'debug', 'relabel', 'train_frame.json')),
    #     "val": (os.path.join(root, 'debug', 'relabel' , 'test_frame.json'))
    #     }
    # PATHS = {
    #     "train": (os.path.join(root, 'relabel', 'train_frame.json')),
    #     "val": (os.path.join(root, 'relabel' , 'test_frame.json'))
    #     }

    # path_to_images = os.path.join(root, 'new_data')

    selection_set = True
    if image_set == 'train':
        path_to_images = os.path.join(root, 'OAK_FRAME_N', 'Raw')
        path_to_annotations = os.path.join(root, 'OAK_LABEL_N')
    
    elif image_set == 'val':
        path_to_images = os.path.join(root, 'OAK_TEST', 'Raw')
        path_to_annotations = os.path.join(root, 'OAK_TEST', 'Label')

    else:
        raise ValueError(f'image set cannot be {image_set}')
        selection_set = None
    
    selection_index = -1
    train_settings = None
    if args.train_mode in ['incremental','ewc']:
    
        
        if image_set == 'val':
            selection_index = -1
        elif image_set =='train':
            try:
                selection_index = args.selection_index
            except:
                raise ValueError('args.selection_index does not exist, make sure it is in main')
        elif image_set == 'memory':
            train_settings = {'train_mode': 'memory',
                        'memory': args.ic2_memory}
        else:
            raise ValueError('Incorrect set provided')
    
    if args.train_mode in ['ic2']:
        if image_set == 'val':
            selection_index = -1
        elif image_set =='train':
            try:
                selection_index = args.selection_index
            except:
                raise ValueError('args.selection_index does not exist, make sure it is in main')
        else:
            raise ValueError('Incorrect set provided')
        train_settings = {'train_mode': 'ic2',
                        'memory': args.ic2_memory}

    dataset = OAKDetection(img_folder = path_to_images, ann_file = path_to_annotations, selection_set = selection_set, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(),selection_index = selection_index, train_settings = train_settings)
    return dataset
