from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import sys
import torch.optim as optim
import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main(argv):
    # all possible arguemnts for multigrid
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-data', '--data_set', default='CIFAR10', help='dataset name')
    parser.add_argument('--dataDir', '--data_setDir', default="", help='dataset directory')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model', default="mobileMGV3small")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-a', '--arch', default='1-2-2-2', help='model architecture')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('-channels', '--channels', default='32-64-128-256',
                        help='channels network')

    parser.add_argument('-nzrp', '--nonZeroRP', default=32, type=int,
                        help='number of non zero in P,R matrix in input line')
    parser.add_argument('-fcc', '--mgFcChannel', default=32, type=int,
                        help='threshold for coarse grid level')
    parser.add_argument('-fg', '--fixgroups', default=32, type=int,
                        help='numberOfFixGroup')
    parser.add_argument('-groups', '--dw', default=0, type=float,
                        help='channels network')
    parser.add_argument('--lastChannelConv', default=512, type=int,
                        help='adding1x1Conv')
    parser.add_argument('--lastChannelConvFlag', default=False, type=bool,
                        help='adding1x1Conv')
    parser.add_argument('--changeBatch', default=False, type=bool,
                        help='flag for changing batchSize during training')
    parser.add_argument('--batchSchedule', default='0,32-75,64-150,128-225,256',
                        help='flag for changing batchSize during training')
    parser.add_argument('--ConcatConnector', default=False, type=bool,
                        help='ConcatConnector')
    parser.add_argument('--LeanDoubleNorm', default=False, type=bool,
                        help='ConcatConnector')
    parser.add_argument('--lastChannel', default=512, type=int,
                        help='addingLastLayer')
    parser.add_argument('--resnextMiddleGroupSize', default=32, type=int,
                        help='addingLastLayer')
    parser.add_argument('--resnextMiddlePWGroupSize', default=8, type=int,
                        help='addingLastLayer')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total ')
    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lrdecay', type=str, default='cos',
                        help='mode for learning rate decay')
    parser.add_argument('--mgNstep', type=int, default=4,
                        help='the grid size of multigrid model')
    parser.add_argument('--step', type=int, default=30,
                        help='interval for learning rate decay in step mode')
    parser.add_argument('--schedule', default='0.1-0,0.05-75,0.01-150,0.001-225',
                        help='decrease learning rate at these epochs.')
    parser.add_argument('--stridesNet', default='1-2-2-2',
                        help='strided network')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-c', '--checkpoint', default='./checkpoints/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoints)')
    parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
    parser.add_argument('--input-size', type=int, default=0, help='input resolution')
    parser.add_argument('--numOfCheckpoints', type=int, default=1, help='numOfCheckpoints')

    parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                        help='path to pretrained weight (default: none)')
    parser.add_argument('--outPrint', default='', type=str, help='output filename')
    parser.add_argument('--OpenLayerConvKernel', default=7, type=str, help='openLayerKernelSize')
    parser.add_argument('--OpenLayerStrideFlag', default=False, type=bool, help='openLayerKernelSize')

    parser.add_argument('--deb', '--debug', default=False, type=bool,
                        help='print tempResultsForDebug')
    parser.add_argument('--Runlocaly', '--locally', default=True, type=bool,
                        help='print tempResultsForDebug')
    parser.add_argument('--SameMaskStructure', '--SameMaskStructure', default=True, type=bool,
                        help='SameMaskStructure')
    parser.add_argument('--shuffleChannels', '--shuffleChannels', default=False, type=bool,
                        help='shuffleChannels')
    parser.add_argument('--structuredChannels', '--structuredChannels', default=True, type=bool,
                        help='structuredChannels')
    parser.add_argument('--pwGroup', '--pwGroup', default=True, type=bool,
                        help='pwGroup')
    parser.add_argument('--bottleneck', '--bottleneck', default=False, type=bool,
                        help='bottleneck')
    parser.add_argument('--resNext', '--resNext', default=0, type=int,
                        help='slurm')
    parser.add_argument('--slurm', '--slurm', default=1, type=int,
                        help='slurm')
    now = datetime.datetime.now()
    args = parser.parse_args()
    configArr = [int(it) for it in args.arch.split("-")]
    channelarr = [int(it) for it in args.channels.split("-")]
    stridesarr = [int(it) for it in args.stridesNet.split("-")]
    scheduleDict = {}
    for it in args.schedule.split(","):
        keyVal = it.split('-')
        scheduleDict[int(keyVal[1])] = float(keyVal[0])
    batchScheduleDict = {}
    for it in args.batchSchedule.split("-"):
        keyVal = it.split(',')
        batchScheduleDict[keyVal[0]] = keyVal[1]
    start_epoch = 0
    best_acc = 0
    if not torch.cuda.is_available():
        raise Exception('Error:', 'no cuda')
    device = torch.cuda.current_device()
    lrParam = args.lr
    totalItter = args.epochs
    wd = args.weight_decay
    ###############################################################
    if args.data_set == "CIFAR10":
        if args.input_size > 0:
            nImg = args.input_size
        else:
            nImg = 32
        nClasses = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(nImg, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers)
    elif args.data_set == "CIFAR100":
        if args.input_size > 0:
            nImg = args.input_size
        else:
            nImg = 28
        Root = './data'
        nClasses = 100
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(nImg, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.4065), (0.229, 0.224, 0.225)), ])

        trainset = torchvision.datasets.CIFAR100(root=Root, train=True,
                                                 download=True, transform=transform_train)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR100(root=Root, train=False,
                                                download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=False, num_workers=args.workers)
    elif args.data_set == "IMAGENET":
        data_dir = args.dataDir
        mini_batch_size = args.batch_size
        test_batch = args.batch_size
        scaleNImage = 256
        nImg = 224
        transform = transforms.Compose([
            transforms.RandomResizedCrop(nImg),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        nClasses = 1000
        trainset = torchvision.datasets.ImageFolder(data_dir + 'train/', transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=True,
                                                  num_workers=args.workers)

        transform = transforms.Compose([
            transforms.Scale(scaleNImage),
            transforms.CenterCrop(nImg
                                  ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        testset = torchvision.datasets.ImageFolder(data_dir + '2012Val/', transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False,
                                                 num_workers=args.workers)
    else:
        raise Exception('Error', 'unknown data')
    # initialize net and weights
    myParams = []
    if args.resNext:
        resnextFlag = True
    else:
        resnextFlag = False
    if args.model == "resnetMG":
        from resnetMG import ModelParameters, ResNetMGGroup
        myParams = ModelParameters(nClasses,
                                   channelarr,
                                   configArr,
                                   strideArray=stridesarr,
                                   mgNstep=args.mgNstep,
                                   numberOfCheckpoints=args.numOfCheckpoints,
                                   SameMaskStructure=args.SameMaskStructure,
                                   numberOfFixGroup=args.fixgroups,
                                   mgFcChannel=args.mgFcChannel,
                                   nonZeroRP=args.nonZeroRP,
                                   pwGroup=args.pwGroup,
                                   shuffleChannels=args.shuffleChannels,
                                   structuredChannels=args.structuredChannels,
                                   bottleneck=args.bottleneck,
                                   resNext=resnextFlag
                                   )


    elif args.model == "mobileMGV3" or args.model == "mobileMGV3small":
        from mobilenetv3_MG import ModelParameters, mobileMGV3_full, mobileMGV3_small
        myParams = ModelParameters(nClasses,
                                   channelarr,
                                   configArr,
                                   strideArray=stridesarr,
                                   mgNstep=args.mgNstep,
                                   numberOfCheckpoints=args.numOfCheckpoints,
                                   SameMaskStructure=args.SameMaskStructure,
                                   numberOfFixGroup=args.fixgroups,
                                   mgFcChannel=args.mgFcChannel,
                                   nonZeroRP=args.nonZeroRP,
                                   pwGroup=args.pwGroup,
                                   shuffleChannels=args.shuffleChannels,
                                   structuredChannels=args.structuredChannels,
                                   bottleneck=args.bottleneck,
                                   )

    if args.model == "resnetMG":
        net = ResNetMGGroup(myParams)
    if args.model == "mobileMGV3":
        net = mobileMGV3_full(myParams)
    if args.model == "mobileMGV3small":
        net = mobileMGV3_small(myParams)


    net.to(device)
    sys.stdout.flush()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lrParam, momentum=args.momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train(epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(trainloader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        print('\nEpoch: %d' % epoch)
        net.train()
        end = time.time()
        print("Train")
        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            data_time.update(time.time() - end)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % 50 == 0:
                progress.display(batch_idx)

    def test(epoch, best_acc):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(testloader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
        net.eval()
        print("Test")
        with torch.no_grad():
            end = time.time()
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % 50 == 0:
                    progress.display(batch_idx)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5), flush=True)
        acc = top1.avg
        if acc > best_acc:
            best_acc = acc
        return best_acc

    for epoch in range(start_epoch, start_epoch + totalItter):
        print("Epoch:", epoch)
        print("Current LR:", scheduler.get_lr())
        if args.lrdecay == 'step':
            # lr decay function - step
            if epoch % args.step == 0 and epoch > 0:
                for g in optimizer.param_groups:
                    g['lr'] *= args.gamma
        elif args.lrdecay == 'schedule':
            # lr decay function - schedule
            if epoch in scheduleDict.keys():
                for g in optimizer.param_groups:
                    g['lr'] = scheduleDict[epoch]
        if args.changeBatch:
            if epoch in batchScheduleDict.keys():
                batchSize = batchScheduleDict[epoch]
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True,
                                                          num_workers=args.workers)
        for g in optimizer.param_groups:
            lr = g['lr']
            args.lr = lr
            break
        train(epoch)
        print("test...")
        best_acc = test(epoch, best_acc)
        if args.lrdecay == 'cos':
            scheduler.step()

    print('Finished Training with maximal accuracy', best_acc)


if __name__ == '__main__':
    main(sys.argv)
