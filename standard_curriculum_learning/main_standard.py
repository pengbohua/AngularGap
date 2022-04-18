# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import warnings
import json
import collections
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as T
import torch.optim
import torch.utils.data
from torch.utils.data import Subset
from utils import get_dataset, get_model, get_optimizer, get_scheduler
from utils import LossTracker, run_cmd
from torchvision.datasets import CIFAR10
from utils import get_pacing_function, balance_order_val

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data-dir', default='dataset',
                    help='path to dataset')
parser.add_argument('--order-dir', default='angular_gap_order.npy',
                    help='path to train val idx')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset')
parser.add_argument('--printfreq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total')
parser.add_argument('--optimizer', default="sgd", type=str,
                    help='optimizer')
parser.add_argument('--scheduler', default="cosine", type=str,
                    help='lr scheduler')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=1111, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--half', default=False, type=bool,
                    help='training with half precision')
parser.add_argument('--lr_decay', default=0.1, type=float,
                    help='lr decay for milestone scheduler')
# curriculum params
parser.add_argument("--ordering", default="standard", type=str, help="which test case to use. supports: standard")
parser.add_argument('--rand-fraction', default=0., type=float,
                    help='label curruption (default:0)')
args = parser.parse_args()

def main():
    set_seed(args.seed)
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
    ])
    test_transform = T.Compose([
        T.Resize(36),
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=(0.247, 0.243, 0.261))
    ])
    tr_set = CIFAR10('./', train=False, download=True, transform=train_transform)

    # initiate a recorder for saving and loading stats and checkpoints
    if 'hsf' in args.order_dir:
        instance_loss = torch.load(os.path.join('./orders',args.order_dir), map_location=torch.device('cpu'))
        order = [k for k, v in sorted(instance_loss.items(), key=lambda it: it[1])]
    elif 'classification_margin' in args.order_dir:
        angular_gap = torch.load(os.path.join('./orders',args.order_dir), map_location=torch.device('cpu')).numpy()
        ordering = collections.defaultdict(list)
        list(map(lambda a, b: ordering[a].append(b), np.arange(len(angular_gap)), angular_gap))
        order = [k for k, v in sorted(ordering.items(), key=lambda item: -1 * item[1][0])]
    elif 'forgetting_events.pkl' in args.order_dir:
        order_dir = os.path.join('./orders', args.order_dir)
        with open(order_dir, 'rb') as f:
            order_dict = pickle.load(f)
        indices = order_dict['indices']
        forget_counts = order_dict['forgetting counts']
        indices_order = {}
        for ind, count in zip(indices, forget_counts):
            indices_order[int(ind)] = count
        order = [k for k, v in sorted(indices_order.items(), key=lambda it: it[1])]   # forgetting small to large easy to hard
    elif 'angular_gap_order' in args.order_dir:
        order = np.load('orders/angular_gap_order.npy')
    elif 'cscore' in args.order_dir:
        instance_loss = torch.load(args.order_dir, map_location=torch.device('cpu'))
        order = [k for k, v in sorted(instance_loss.items(), key=lambda it: torch.mean(torch.cat(it[1], 0)))]
    else:
        print(
        'Please check if the files %s in your folder -- orders. See ./orders/README.md for instructions on how to create the folder' % (
            args.order_dir))
        raise NotImplementedError
    print('number classes', len(tr_set.classes))
    order,order_val = balance_order_val(order, tr_set, num_classes=len(tr_set.classes), valp=0.0)
    order.extend(order_val)
    print(len(order))

    #check the statistics
    bs = args.batchsize
    N = len(order)
    myiterations = (N//bs+1)*args.epochs

    #initial training
    model = get_model(args.arch, nchannels=3, imsize=32, nclasses=10, args=args)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=myiterations)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],"test_loss": [], "test_acc": [], "iter": [0,] }

    trainsets = Subset(tr_set, order)

    val_set = CIFAR10('./', train=True, download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize*2,
                      shuffle=False, num_workers=args.workers, pin_memory=True)

    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()

    iterations = 0
    history_per_iteration_record = {'train_acc': []}
    for epoch in range(args.epochs):
        tr_loss, tr_acc1, iterations, history_per_iter = standard_train(train_loader, model, criterion, optimizer,scheduler, epoch,iterations)
        print('epoch', epoch, 'lr', optimizer.param_groups[0]['lr'], 'train_loss', tr_loss, 'train_acc_top1', tr_acc1)
        history_per_iteration_record['train_acc'].extend(history_per_iter)
        test_loss, test_acc1 = standard_validate(test_loader, model, criterion)
        # print ("%s epoch %s iterations w/ LEARNING RATE %s"%(epoch, iterations,optimizer.param_groups[0]["lr"]))
        print('epoch', epoch, 'test_acc_top1', test_acc1)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc1.item())
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc1.item())
        history["iter"].append(iterations)
        torch.save(history,"./results/{}_{}.pt".format(args.dataset, args.ordering,args.order_dir[:10]))
        torch.save(history_per_iteration_record, "./results/{}_{}_order{}.pt".format(args.dataset, args.ordering,args.order_dir[:10]))

def standard_train(train_loader, model, criterion, optimizer,scheduler, epoch, iterations):
    # switch to train mode
    model.train()
    history_per_iterations = {'train_acc':[]}
    tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
    for i, (images, target) in enumerate(train_loader):
        iterations += 1
        images, target = cuda_transfer(images, target)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tracker.update(loss, output, target)
        history_per_iterations['train_acc'].append(tracker.top1.avg)
        scheduler.step()
    return tracker.losses.avg, tracker.top1.avg,  iterations, history_per_iterations

def standard_validate(val_loader, model, criterion):
  # switch to evaluate mode
  model.eval()
  with torch.no_grad():
    tracker = LossTracker(len(val_loader), f'val', args.printfreq)
    for i, (images, target) in enumerate(val_loader):
      images, target = cuda_transfer(images, target)
      output = model(images)
      loss = criterion(output, target)
      tracker.update(loss, output, target)
  return tracker.losses.avg, tracker.top1.avg

def set_seed(seed=None):
    if seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    return images, target

if __name__ == '__main__':
    main()

