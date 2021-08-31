# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import os

from models.ewc import EWC


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--frozen_backbone', type=str, default=None,
                        help="Whether backbone is frozen or not.")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default = 'oak') #default='coco')
    parser.add_argument('--coco_path', default='/project_data/held/jianrenw', type=str) # './data/coco'
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--oak_path', default='/project_data/held/jianrenw', type=str)
    # parser.add_argument('--pascal_path', default='/project_data/held/jianrenw/datasets/VOCdevkit/VOC2012', type=str)
    parser.add_argument('--pascal_path', default='/grogu/user/jianrenw/data/VOCdevkit', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # Selection mode
    parser.add_argument('--train_mode', default='offline', help='Training mode options {offline, incremental, ic2, ewc}')
    parser.add_argument('--selection_index', default=0, help='Index for selection')
    parser.add_argument('--iterations', default=1, help='Number of Iterations for Each Sample')

    # For ic2 setting
    parser.add_argument('--ic2_memory', default=[], help='Icarl2 Memory')
    parser.add_argument('--ic2_memory_size', default=5, help='Icarl2 Memory Size')
    parser.add_argument('--ic2_cmratio', default=0, help='Icarl2 Memory CM Ratio')

    # For ewc setting
    parser.add_argument('--ewc_obj', default=None, help='EWC Object')
    parser.add_argument('--ewc_importance', default=1000, help='EWC Importance')

    
    return parser

def update_ic2_memory(valid,img_memory,num_pc):
    for data_i, data in enumerate(valid.dataset['annotations']):
        
        if not data['from_memory']:
            filename = valid.dataset['images'][int(data['image_id'])]['file_name']
            # filename = data['file_name']
            # objs = data['annotations']
            # for obj in objs:
            class_id = int(data['category_id'])
            ele = (filename,data)
            print(class_id)
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

def get_cmratio(records):
    cur_cnt,mem_cnt = 0,0
    for record in records.dataset['annotations']:
        if record['from_memory']:
            mem_cnt += 1
        else:
            cur_cnt += 1

    if mem_cnt == 0:
        cm_ratio = 0
    elif cur_cnt / mem_cnt > 1:
        cm_ratio = 1
    else:
        cm_ratio = cur_cnt / mem_cnt  
    return cm_ratio

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Make the evaluation folder
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir, 'eval')):
            os.mkdir(os.path.join(args.output_dir, 'eval'))

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.finetune:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            new_state_dict={k:v if v.size()==model_without_ddp.state_dict()[k].size()  else  model_without_ddp.state_dict()[k] for k,v in zip(model_without_ddp.state_dict().keys(), checkpoint['model'].values()) 
                 }
        # missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(new_state_dict, strict=False)
        
    if args.frozen_backbone:
        [x.requires_grad_(False) for x in model_without_ddp.backbone.parameters()]
    
    output_dir = Path(args.output_dir)

    if args.train_mode in ['ic2','ewc']:
        args.ic2_memory = [None] * len(list(dataset_train.cat_to_id_dict.keys()))
        
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
            args.selection_index = checkpoint['args'].selection_index
            if args.train_mode == 'ic2':
                args.ic2_memory = checkpoint['args'].ic2_memory
                args.ic2_cmratio = checkpoint['args'].ic2_cmratio
            if args.train_mode == 'ewc':
                args.ewc_obj = checkpoint['args'].ewc_obj
                args.ewc_importance = checkpoint['args'].ewc_importance
        # check the resumed model
        if not args.eval:
            checkpoint_path, checkpoint_name = os.path.split(args.resume)
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir = os.path.join(args.output_dir, 'eval'),checkpoint_name = f'checkpoint{args.start_epoch-1}.pth' 
            )
    
    if args.eval:
        # Make the output eval directory the same as where the ckpt is stored
        checkpoint_path, checkpoint_name = os.path.split(args.resume)
        
        output_path = os.path.join(checkpoint_path, 'eval')

        if not os.path.exists(output_path):
            os.makedir(output_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, output_dir = output_path, checkpoint_name = f'checkpoint{args.start_epoch-1}.pth' )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()

    if args.train_mode == 'offline':
        end_epoch = args.epochs
    else:
        #elif args.train_mode == 'incremental':
        end_epoch = int(dataset_train.data_length/16) + 1


    
    
    for epoch in range(args.start_epoch, end_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        if args.train_mode == 'ic2':
            args.cmratio = get_cmratio(dataset_train)
        
        if args.train_mode == 'ewc':
            # Initialize the ewc object
            # Check length of memory
            memory_ratio = get_cmratio(dataset_train)
            args.ewc_obj = None
            if memory_ratio > 0:
                args.ewc_obj = EWC(model,data_loader_train,importance = args.ewc_importance, iterations = args.iterations, device = device, criterion = criterion)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, args, sampler_train)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if ((epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0) and (args.train_mode == 'offline'):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            elif ((epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0) and not(args.train_mode == 'offline'):
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        if args.train_mode == 'offline':
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir = os.path.join(args.output_dir, 'eval'), epoch_number = epoch
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
        if args.train_mode == 'incremental':
            if True:
                # if (epoch + 1) % 16 == 0:
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir = os.path.join(args.output_dir, 'eval'), epoch_number = epoch
                )

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

                if args.output_dir and utils.is_main_process():
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                        output_dir / "eval" / name)
            args.selection_index +=16
            dataset_train = build_dataset(image_set='train', args=args)
            # dataset_val = build_dataset(image_set='val', args=args)

            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                    # sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
                    # sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                        pin_memory=True)
        if args.train_mode == 'ic2':
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir = os.path.join(args.output_dir, 'eval'), epoch_number = epoch
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
            args.selection_index +=16

            # Update memory
            args.ic2_memory = update_ic2_memory(dataset_train, args.ic2_memory, args.ic2_memory_size)

            dataset_train = build_dataset(image_set='train', args=args)
            # dataset_val = build_dataset(image_set='val', args=args)

            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                    # sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
                    # sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                        pin_memory=True)
        if args.train_mode == 'ewc':
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, output_dir = os.path.join(args.output_dir, 'eval'), epoch_number = epoch
            )

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
            args.selection_index +=16

            # Update memory
            args.ic2_memory = update_ic2_memory(dataset_train, args.ic2_memory, args.ic2_memory_size)

            dataset_train = build_dataset(image_set='train', args=args)
            # dataset_val = build_dataset(image_set='val', args=args)

            if args.distributed:
                if args.cache_mode:
                    sampler_train = samplers.NodeDistributedSampler(dataset_train)
                    # sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
                else:
                    sampler_train = samplers.DistributedSampler(dataset_train)
                    # sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)

            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                        collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                        pin_memory=True)
        # data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
        #                             drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
        #                             pin_memory=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(par
