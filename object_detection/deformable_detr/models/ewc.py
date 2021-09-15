from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

from util.misc import (NestedTensor, nested_tensor_from_tensor_list)
from datasets.data_prefetcher import data_prefetcher


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    # print('print tensor', t)
    if torch.cuda.is_available() and use_cuda:
        # t = t.to('cuda')
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, importance: int,iterations: int,device: torch.device, criterion: torch.nn.Module):

        self.model = model
        self.dataset = deepcopy(dataset)
        self.importance = importance
        self.iterations = iterations
        self.device = device
        self.criterion = deepcopy(criterion)

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        num_iters = self.iterations
        self.model.eval()
        for iteration in range(0,int(num_iters)):

            # for input, iteration in zip(self.dataset, range(0,int(num_iters))):
            prefetcher = data_prefetcher(self.dataset, self.device, prefetch=True)
            input, targets = prefetcher.next()
            if not isinstance(input, NestedTensor):
                input = nested_tensor_from_tensor_list(input)
            self.model.zero_grad()
            input = variable(input.tensors)
            outputs = self.model(input)
            train_setting = None
            loss_dict = self.criterion(outputs, targets, train_setting = train_setting)

            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)



            # output = self.model(input).view(1, -1)
            # label = output.max(1)[1].view(-1)
            # loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        # print('Precision matrix', self._precision_matrices)
        for n, p in model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
