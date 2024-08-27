import sys

import torch
from torch import nn, autograd
import numpy as np
import random
from sklearn import metrics
import copy
from time import time
import os
import utils.testFunction as test

class Clients(object):
    def __init__(self, args=None, tdata=None, lmodel=None, title=None, logPath=None):
        self.train_data = tdata
        self.args = args
        self.local_model = lmodel
        self.loss_func = nn.CrossEntropyLoss()
        if self.args is not None:
            self.batch_size = self.args.local_bs
        self.loss = 0
        self.accuracy = 0
        self.title = title
        self.model_path = os.path.join(logPath, f"client_model_{self.title}.pt")

    def train(self, net, client_queue=None, timer=None, use_multiprocessing=True
                                                 , poison_clients=[], client=-1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        # if client in poison_clients:
        #     print("进入标签毒化")
        #     for iter in range(self.args.local_ep):
        #         batch_loss = []
        #         # print(type(self.train_data))
        #         for batch_idx, (images, labels) in enumerate(test.flip_labels_in_loader(self.train_data)):
        #             images, labels = images.to(self.args.device), labels.to(self.args.device)
        #             net.zero_grad()
        #             log_probs = net(images)
        #             loss = self.loss_func(log_probs, labels)
        #             loss.backward()
        #             optimizer.step()
        #             if self.args.verbose and batch_idx % 10 == 0:
        #                 print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                     iter, batch_idx * len(images), len(self.train_data.dataset),
        #                           100. * batch_idx / len(self.train_data), loss.item()))
        #             batch_loss.append(loss.item())
        #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
        # else:
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print(type(self.train_data))
            for batch_idx, (images, labels) in enumerate(self.train_data):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.train_data.dataset),
                              100. * batch_idx / len(self.train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        par = net.state_dict().copy()
        params_cpu = {}
        for key in par.keys():
            # 将数据从gpu转到cpu
            params_cpu[key] = par[key].cpu()
        if client in poison_clients:
            #    如果当前客户端是毒化客户端
            print("---进入模拟毒化----")
            # 这个是客户端完全污染 为 无目标污染
            # poison_fc2_weight = np.random.uniform(low=-2, high=4, size=(10, 512))

            # # 标签反转 模拟30%
            params_cpu_value = copy.deepcopy(params_cpu)

            # 模拟部分污染，进行标签翻转
            params_cpu = test.poison_gradients(params_cpu_value, 'data-attack')

        return params_cpu, sum(epoch_loss) / len(epoch_loss)
        # self.loss = sum(epoch_loss) / len(epoch_loss)
        # if use_multiprocessing:
        #     torch.save(net.state_dict(), self.model_path)
        #     client_queue.put(self.loss)
        # else:
        #     return net.state_dict(), self.loss

    def train2(self, net, client_queue=None, timer=None, use_multiprocessing=True
               , poison_clients=[], client=-1):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        # 标签翻转：
        if client in poison_clients:
            print("进入标签毒化")
            for iteration in range(self.args.local_ep):
                batch_loss = []
                for i, (images, labels) in enumerate(test.flip_labels_in_loader(self.train_data)):
                    for j in range(len(images)):
                        image, label = images[j], labels[j]
                        image, label = image.to(self.args.device), label.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(image)
                        loss = self.loss_func(log_probs, label)
                        loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
        else:
            for iteration in range(self.args.local_ep):
                batch_loss = []
                for i, (images, labels) in enumerate(self.train_data):
                    for j in range(len(images)):
                        image, label = images[j], labels[j]
                        image, label = image.to(self.args.device), label.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(image)
                        loss = self.loss_func(log_probs, label)
                        loss.backward()
                        optimizer.step()
                        # if self.args.verbose and batch_idx % 10 == 0:
                        #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        #         iter, batch_idx * len(images), len(self.train_data.dataset),
                        #               100. * batch_idx / len(self.train_data), loss.item()))
                        batch_loss.append(loss.item())
                # if len(batch_loss) == 0:
                #     for images, labels in self.train_data:
                #         print(images.shape)
                #         break
                # else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        par = net.state_dict().copy()
        params_cpu = {}
        for key in par.keys():
            # 将数据从gpu转到cpu
            params_cpu[key] = par[key].cpu()
        if client in poison_clients:
            #    如果当前客户端是毒化客户端
            print("---进入模拟毒化----")
            # 这个是客户端完全污染 为 无目标污染
            # poison_fc2_weight = np.random.uniform(low=-2, high=4, size=(10, 512))

            # # 标签反转 模拟30%
            params_cpu_value = copy.deepcopy(params_cpu)

            # 模拟部分污染，进行标签翻转
            params_cpu = test.poison_gradients(params_cpu_value, 'data-attack')

        return params_cpu, sum(epoch_loss) / len(epoch_loss)
        # if use_multiprocessing:
        #     torch.save(net.state_dict(), self.model_path)
        #     client_queue.put(self.loss)
        # else:
        #     return net.state_dict(), self.loss
