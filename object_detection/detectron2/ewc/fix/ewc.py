from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.transforms as T

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    
    def __init__(self, cfg, model: nn.Module, max_iter,importance=1000):
        self.cfg = cfg
        self.model = model
        self.max_iter = max_iter
        self.importance = importance 

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)
        

    def _diag_fisher(self,):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()

        cfg = self.cfg
        max_iter = self.max_iter
        data_loader = build_detection_train_loader(cfg, 
            mapper=DatasetMapper(cfg, is_train=True, 
            augmentations=[T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],cfg.INPUT.MAX_SIZE_TEST)]))
        
        for data, iteration in zip(data_loader, range(0, max_iter)):
            self.model.zero_grad()
            rpn_output, head_output = self.model.module.cal_logit(data)
            label = head_output.max(1)[1].view(-1)
            
            head_loss = F.nll_loss(F.log_softmax(head_output, dim=1), label)

            total_loss = head_loss 
            total_loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / max_iter
       
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices


    
    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'roi_heads.box_predictor.cls_score' in n:
                    self._precision_matrices[n][-1] = 0
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum() 
        return loss

