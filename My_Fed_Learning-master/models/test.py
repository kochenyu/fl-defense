#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img_FEMNIST(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, drop_last=True)
    l = len(data_loader)

    for idx, (datas, targets) in enumerate(data_loader):
        if args.gpu != -1:
            datas, targets = datas.cuda(), targets.cuda()
        log_probs = net_g(datas)
        test_loss += F.cross_entropy(log_probs, targets, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    data_loader = DataLoader(datatest, batch_size=args.bs, drop_last=True)
    l = len(data_loader)
    for idx, (datas, targets) in enumerate(data_loader):
        # for j in range(len(datas)):
        # data, target = datas[idx], targets[idx]
        if args.gpu != -1:
             datas, targets = datas.cuda(), targets.cuda()
        log_probs = net_g(datas)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, targets, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(targets.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_img2(net_g, datatest, args):
    net_g.eval()

    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    data_loader = DataLoader(datatest, batch_size=args.bs, drop_last=True)
    l = len(data_loader)
    for idx, (datas, targets) in enumerate(data_loader):
        for j in range(len(datas)):
            data, target = datas[j], targets[j]
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

