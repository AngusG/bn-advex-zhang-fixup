'''Eval CIFAR10 (PGD) with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import models

from utils import progress_bar

import numpy as np

from advertorch.attacks import L2PGDAttack
from advertorch.attacks import LinfPGDAttack

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument("--resume", default="", type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument('--dataroot', help='path to dataset',
                    default='/scratch/ssd/data')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size per GPU (default=128)')

parser.add_argument('--max_epsilon', default=8, type=int,
                    help='max epsilon to use')
parser.add_argument('--nb_iter', default=20, type=int,
                    help='setps to use if pgd_train is set')

parser.add_argument('--l2', action="store_true",
                    help="do l2 norm PGD variant")

parser.add_argument('--tgt', action="store_true",
                    help="targeted attack objective (default=misclf)")

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True,
                                        download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# Model
if args.resume:
    if os.path.isfile(args.resume):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume)
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
        print("Resuming from epoch %d" % start_epoch)
else:
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch]()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

cel = nn.CrossEntropyLoss()

def test(dataloader, do_awgn=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            if do_awgn:
                inputs += torch.randn_like(inputs) / 16  # awgn
            outputs = net(inputs)
            loss = cel(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            progress_bar(batch_idx, len(dataloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (test_loss / (batch_idx + 1),
                          100. * float(correct) / float(total),
                          correct, total))
        acc = 100. * float(correct) / float(total)
    return (test_loss/batch_idx, acc)


def test_adver(dataloader, adversary, is_targeted):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, orig_targets) in enumerate(dataloader):
        if use_cuda:
            inputs, orig_targets = inputs.cuda(), orig_targets.cuda()
        if is_targeted:
            #targets = (clntarget + 1) % 10
            output = net(inputs)
            targets = output.argsort()[:, 0]  # target least-likely class
            advdata = adversary.perturb(inputs, targets)
        else:
            advdata = adversary.perturb(inputs, orig_targets)
        outputs = net(advdata)
        loss = cel(outputs, orig_targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += orig_targets.size(0)
        correct += predicted.eq(orig_targets.data).cpu().sum()
        progress_bar(batch_idx, len(dataloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                     (test_loss / (batch_idx + 1),
                      100. * float(correct) / float(total),
                      correct, total))
        acc = 100. * float(correct) / float(total)
    return (test_loss / batch_idx, acc)


targeted = True if args.tgt else False
print('targeted ')
print(targeted)
epsilons = np.arange(args.max_epsilon)
stats = np.zeros((len(epsilons), 2))
for i, eps in enumerate(epsilons):
    if i == 0:
        adver_lss, adver_acc = test(testloader)
    else:
        base_epsilon = eps / 255.
        if args.l2:
            l2_epsilon = np.sqrt(3 * 32 * 32) * base_epsilon
            #l2_epsilon = 10 * base_epsilon
            l2_eps_iter = l2_epsilon / (args.nb_iter * 0.75)
            print(l2_epsilon)
            print(l2_eps_iter)
            adversary = L2PGDAttack(
                net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=l2_epsilon, nb_iter=args.nb_iter, eps_iter=l2_eps_iter,
                rand_init=False, clip_min=0., clip_max=1., targeted=targeted)
        else:
            linf_eps_iter = base_epsilon / (args.nb_iter * 0.75)
            print(base_epsilon)
            print(linf_eps_iter)
            adversary = LinfPGDAttack(
                net, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=base_epsilon, nb_iter=args.nb_iter, eps_iter=linf_eps_iter,
                rand_init=False, clip_min=0., clip_max=1., targeted=targeted)

        adver_lss, adver_acc = test_adver(testloader, adversary, targeted)
        print('%d, %f' % (eps, adver_acc))
    stats[i, 0] = adver_lss
    stats[i, 1] = adver_acc

output_file = 'npy/acc_vs_eps'
if args.l2:
    output_file += '_2_'
else:
    output_file += '_inf_'

if targeted:
    output_file += '_tgt_ll'
output_file += args.resume.split('/')[-1].split('.')[0] + '.npy'
print(output_file)
np.save(output_file, stats)
