'''Eval CIFAR10-C with PyTorch.'''
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

import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='fixup_resnet110',
                    choices=model_names, help='model architecture: ' +
                    ' | '.join(model_names) + ' (default: fixup_resnet110)')
parser.add_argument("--resume", default="", type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument('--dataroot', help='path to CIFAR10-C dataset')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--batchsize', default=200, type=int,
                    help='batch size per GPU (default=128)')
parser.add_argument('--sheet_id', help="Google Spreadsheet ID for saving \
                    snap shot of run. There must be a sheet called 'CIFAR-10-C'\
                    for this script to run, otherwise set value for 'SHEET'")


def main():
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
    else:
        print("=> creating model '{}'".format(args.arch))
        net = models.__dict__[args.arch]()

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    tokens = args.resume.split('/')[-1].split('_')
    if 'fixup' in tokens:
        batchnorm = False
    else:
        batchnorm = True
    seed = int(tokens[-1].split('.')[0][0])
    print('got seed %d' % seed)
    label_path = os.path.join(args.dataroot, 'labels.npy')
    labels = np.load(label_path)

    filenames = os.listdir(args.dataroot)
    filenames.sort()

    net.eval()

    cel = nn.CrossEntropyLoss()

    if args.sheet_id is not None:
        # Google Sheets API
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and
        # is created automatically when the authorization flow completes for the
        # first time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        service = build('sheets', 'v4', credentials=creds)
        idx2col = ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V']
        j = 0
        for i in range(len(filenames)):
            if not filenames[i] == 'labels.npy':
                print(j, filenames[i])
                raw_images = np.load(os.path.join(args.dataroot, filenames[i]))
                images = preprocess_images(raw_images)
                test_x = torch.tensor(images, dtype=torch.float)
                test_y = torch.tensor(labels, dtype=torch.long)
                cifar10c_dataset = torch.utils.data.TensorDataset(
                    test_x, test_y)
                cifar10c_loader = torch.utils.data.DataLoader(
                    cifar10c_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=2)
                arr = test_common(net, cifar10c_loader)
                values = [
                    [arr[0]],
                    [arr[1]],
                    [arr[2]],
                    [arr[3]],
                    [arr[4]],
                ]
                body = {
                    'values': values
                }
                SKIP_SIZE = 31
                ROW_OFFSET = 2
                SHEET = 'CIFAR-10-C'
                if batchnorm:
                    ROW_OFFSET += SKIP_SIZE
                ROW_OFFSET += (seed - 1) * (len(values) + 1)
                result = service.spreadsheets().values().update(
                    spreadsheetId=args.sheet_id, range="%s!%s%d:%s%d" %
                    (SHEET, idx2col[j], ROW_OFFSET, idx2col[j],
                     ROW_OFFSET + len(values)),
                    valueInputOption='USER_ENTERED', body=body).execute()
                print('{0} cells updated.'.format(result.get('updatedCells')))
                print(result)
                j += 1


def preprocess_images(img):
    '''
    Preprocess images before converting to tensor
    '''
    img_mean = np.array([0.4914, 0.4822, 0.4465])
    img_var = np.array([0.2023, 0.1994, 0.2010])
    img = np.transpose(img, (0, 3, 1, 2))
    img = img / 255.
    img = img - img_mean[np.newaxis, :, np.newaxis, np.newaxis]
    img = img / img_var[np.newaxis, :, np.newaxis, np.newaxis]
    return img


def test_common(model, dataloader):
    acc = np.zeros(5)
    correct = 0
    total = 0
    j = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_acc = float(correct) / total
            if total % 10000 == 0:
                print(batch_idx, test_acc)
                acc[j] = test_acc
                correct = 0
                total = 0
                j += 1
    return acc


if __name__ == "__main__":
    main()
