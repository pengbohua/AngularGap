import collections
import torch
import os
import time
import numpy as np
import json


class LossTracker(object):
    def __init__(self, num, prefix="", print_freq=1):
        self.print_freq = print_freq
        self.batch_time = AverageMeter("Time", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4e")
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")
        self.progress = ProgressMeter(
            num, [self.batch_time, self.losses, self.top1, self.top5], prefix=prefix
        )
        self.end = time.time()

    def update(self, loss, output, target):
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.losses.update(loss.item(), output.size(0))
        self.top1.update(acc1[0], output.size(0))
        self.top5.update(acc5[0], output.size(0))

    def display(self, step):
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        if step % self.print_freq == 0:
            self.progress.display(step)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
