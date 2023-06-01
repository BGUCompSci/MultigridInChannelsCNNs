'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
from torch.utils.checkpoint import checkpoint_sequential
from collections import OrderedDict
import math
import copy


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class mgState(enum.Enum):
    FC = 1
    decreasing = 2
    increasing = 3


class ModelParameters:
    def __init__(self, num_classes,
                 channelArray,
                 RepetitionArray,
                 strideArray=[1, 2, 2, 2, 2],
                 mgNstep=2,
                 openLayerKernel=7,
                 openLayerStride=2,
                 flagCatConnector=False,
                 mgFcChannel=16,
                 mgSparseRPFactor=4,
                 mgMaxNonZerosRP=32,
                 FC=False,
                 twoSided=False,
                 nonZeroRP=32,
                 NumberOfpwGroup=32,
                 pwGroup=False,
                 fixGroup=True,
                 numberOfFixGroup=16,
                 numberOfCheckpoints=1,
                 mgParallelMode=False,
                 genMixedRP=False,
                 SameMaskStructure=False,
                 shuffleChannels=False,
                 structuredChannels=True,
                 bottleneck=True,
                 resNext=False,
                 ):
        self.num_classes = num_classes
        self.channelArray = channelArray
        self.RepetitionArray = RepetitionArray
        self.nGridStep = mgNstep
        self.openLayerKernel = openLayerKernel
        self.openLayerStride = openLayerStride
        self.strideArray = strideArray
        self.flagCatConnector = flagCatConnector
        self.mgFcChannel = mgFcChannel
        self.mgSparseRPFactor = mgSparseRPFactor
        self.mgMaxNonZerosRP = mgMaxNonZerosRP
        self.mgParallelMode = mgParallelMode
        self.pwGroup = pwGroup
        self.twoSided = twoSided
        self.NumberOfpwGroup = NumberOfpwGroup
        self.numberOfCheckpoints = numberOfCheckpoints
        self.nonZeroRP = nonZeroRP
        self.fixGroup = fixGroup
        self.numberOfFixGroup = numberOfFixGroup
        self.genMixedRP = genMixedRP
        self.SameMaskStructure = SameMaskStructure

        self.shuffleChannels = shuffleChannels
        self.structuredChannels = structuredChannels
        self.bottleneck = bottleneck
        self.resNext = resNext


class MgBlockDwGroup(nn.Module):
    def __init__(self, in_planes, planes, parameters, stride=1, expansion=1, reuse_PR=False):
        super(MgBlockDwGroup, self).__init__()
        self.parameters = parameters
        self.mgStep = torch.nn.ModuleList()
        self.flagCatConnector = self.parameters.flagCatConnector
        self.connector = False
        self.reuse_PR = False
        self.stride = stride
        self.bottleneck = self.parameters.bottleneck
        if self.bottleneck:
            self.expansion = expansion
        else:
            self.expansion = 1

        if stride != 1 or in_planes != self.expansion * planes:
            self.connector = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=3, padding=1, stride=stride,
                          bias=False, groups=int(in_planes)),
                nn.BatchNorm2d(self.expansion * planes)
            )

            for m in self.shortcut.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            self.mgStep = torch.nn.ModuleList()
            self.mgStep = genTensor4CycleDwGroup(planes, planes, mgState.decreasing, self.mgStep, 1, self.parameters,
                                                 self.expansion, reuse_RP=reuse_PR)
        else:
            self.mgStep = torch.nn.ModuleList()
            self.mgStep = genTensor4CycleDwGroup(planes, planes, mgState.decreasing, self.mgStep, 1, self.parameters,
                                                 self.expansion, reuse_RP=reuse_PR)
        self.reuse_PR = reuse_PR

    def resnetStep(self, x, k1, k2, bn1, bn2, ShortCutFlag=False, stride=1):
        Xstag = k1(x)
        Xstag = bn1(Xstag)
        XstagNorm = F.relu(Xstag, inplace=True)
        Xstag2 = k2(XstagNorm)
        z = bn2(Xstag2)
        if (ShortCutFlag):
            z += x
        return F.relu(z, inplace=True)

    def bottleneckResnetStep(self, x, k1, k2, k3, bn1, bn2, bn3, ShortCutFlag=False, shortcut=None):
        out = k1(x)
        out = bn1(out)
        out = F.relu(out, inplace=True)

        out = k2(out)
        out = bn2(out)
        out = F.relu(out, inplace=True)

        out = k3(out)
        out = bn3(out)

        out += shortcut(x)
        return F.relu(out, inplace=True)

    def MgStepUp(self, x1, x2, xup, P, mgClass):
        # P step
        xdiff = x1 - x2
        z01 = mgClass.bnp1(F.conv2d(xdiff, P, stride=1, padding=(P.shape[3] - 1) // 2, bias=None))
        z02 = xup + z01
        return z02

    def MgStepUp_pwGroup(self, x1, x2, xup, P, mgClass, R=None):
        # P step
        xdiff = x1 - x2
        if self.reuse_PR:
            z01 = F.conv_transpose2d(xdiff, R.weight.data, stride=1,
                                     padding=(int(R.weight.data.shape[3]) - 1) // 2, bias=None, groups=R.groups)
        else:
            z01 = P(xdiff)
        z01 = mgClass.bnp1(z01)
        z02 = xup + z01
        return z02

    def parallelForward(self, startX):
        ind = 0
        indr = 0
        zArr = [startX]
        for step in self.mgStep:
            step.cnt += 1
            if step.mode == mgState.decreasing:
                zArr.append(self.resnetStep(startX, step.k1, step.k2, step.bn1, step.bn2, ShortCutFlag=True))
                currKernellr = torch.mul(step.RMask, step.R)
                startX = F.conv2d(startX, currKernellr, stride=1, bias=None)
                ind += 1
            elif step.mode == mgState.FC:
                zArr.append(F.relu(startX + step.bn2(step.k2(F.relu(step.bn1(step.k1(startX))))), inplace=True))
                ind += 1
                indr = ind
            else:
                for step2 in self.mgStep[indr::]:
                    currKernellP = torch.mul(step2.PMask, step2.P)
                    zArr[ind] = F.conv2d(zArr[ind], currKernellP, bias=None)
                ind -= 1
                indr += 1
        Xparallel = sum(zArr)
        return F.relu(Xparallel, inplace=True)

    def serialForward(self, startX):
        zArr = []
        outForward = []
        R_matrices = []
        Psteps_counter = 0
        for step in self.mgStep:
            step.cnt += 1
            if step.mode == mgState.decreasing:
                if step.twoSided:
                    startX = self.resnetStep(startX, step.k1, step.k2, step.bn1, step.bn2, ShortCutFlag=True)
                if step.pwGroup:
                    outP = step.R(startX)
                else:
                    currKernellr = torch.mul(step.RMask, step.R)
                    outP = F.conv2d(startX, currKernellr, stride=1, bias=None)
                zArr.append(startX)
                outForward.append(outP)
                if self.reuse_PR:
                    R_matrices.append(step.R)
                startX = outP
                cnt2 = len(zArr) - 1
            elif step.mode == mgState.FC:
                if self.parameters.bottleneck:
                    xc = F.relu(step.bn1(step.k1(outP)), inplace=True)
                    xc = F.relu(step.bn2(step.k2(xc)), inplace=True)
                    xc = F.relu(step.bn3(step.k3(xc)), inplace=True)
                    xc = step.shortcut(outP) + xc
                    xc = F.relu(xc, inplace=True)
                else:
                    xc = F.relu(outP + step.bn2(step.k2(F.relu(step.bn1(step.k1(outP))))), inplace=True)

            else:
                if step.pwGroup:
                    if self.reuse_PR:
                        z0 = self.MgStepUp_pwGroup(xc, outForward[cnt2], zArr[cnt2], step.P, step, R_matrices[cnt2])
                    else:
                        z0 = self.MgStepUp_pwGroup(xc, outForward[cnt2], zArr[cnt2], step.P, step)
                else:
                    currKernellP = torch.mul(step.PMask, step.P)
                    z0 = self.MgStepUp(xc, outForward[cnt2], zArr[cnt2], currKernellP, step)
                cnt2 -= 1
                if self.parameters.bottleneck:
                    out = self.bottleneckResnetStep(z0, step.k1, step.k2, step.k3, step.bn1, step.bn2, step.bn3,
                                                    ShortCutFlag=True, shortcut=step.shortcut)
                else:
                    out = self.resnetStep(z0, step.k1, step.k2, step.bn1, step.bn2, ShortCutFlag=True)
                xc = out
                Psteps_counter = Psteps_counter + 1
        return out

    def forward(self, x):
        if self.connector and self.flagCatConnector == False:
            out1 = self.shortcut(x)
            out1 = F.relu(out1, inplace=True)
            x = out1

        startX = x
        if self.parameters.mgParallelMode:
            out = self.parallelForward(startX)
        else:
            out = self.serialForward(startX)
        return out


def genTensor4CycleDwGroup(nInput, nOutput, flagMode, tensorList, stride, params, expansion=1, reuse_RP=False):
    params.useMask = False
    Ng = netMGParamsDWGroup(nInput, nOutput, flagMode, params, stride=stride, expansion=expansion, reuse_RP=reuse_RP)
    decreaseMask = Ng.RMask
    perm = Ng.perm
    tensorList.append(Ng)
    if (nOutput / 2 < params.mgFcChannel):
        Ng = netMGParamsDWGroup(int(nOutput / 2), int(nOutput / 2), mgState.FC, params, expansion=expansion,
                                reuse_RP=reuse_RP)
        tensorList.append(Ng)
    else:
        tensorList = genTensor4CycleDwGroup(int(nOutput / 2), int(nOutput / 2), mgState.decreasing, tensorList, 1,
                                            params, expansion=expansion, reuse_RP=reuse_RP)
    if params.SameMaskStructure:
        params.useMask = True

    Ng = netMGParamsDWGroup(int(nOutput / 2), nOutput, mgState.increasing, params, RP=decreaseMask, perm=perm,
                            expansion=expansion, reuse_RP=reuse_RP)
    tensorList.append(Ng)
    return tensorList


class netMGParamsDWGroup(nn.Module):
    def __init__(self, ch_in, ch_out, flagMode, params, stride=1, RP=[], perm=None, expansion=1, reuse_RP=True):
        super(netMGParamsDWGroup, self).__init__()

        self.shuffleChannels = params.shuffleChannels
        self.structuredChannels = params.structuredChannels
        self.mode = flagMode
        self.stride = stride
        self.shortcut = nn.Sequential()
        self.cnt = 0
        self.pwGroup = params.pwGroup
        self.twoSided = params.twoSided
        self.RMask = []
        self.perm = []
        self.bottleneck = params.bottleneck
        self.resNext = params.resNext
        if self.bottleneck:
            self.expansion = expansion
        else:
            self.expansion = 1
        if params.fixGroup:
            numOfDiag = params.numberOfFixGroup
        groupsChannel = int(np.max([ch_out / numOfDiag, 1]))
        groupsPR = groupsChannel  # groupsChannel
        groupsCNN = groupsChannel
        if flagMode == mgState.FC:
            if self.bottleneck:
                self.k1 = nn.Conv2d(int(ch_out * self.expansion), ch_out, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(ch_out)
                if self.resNext:
                    self.k2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride,
                                        padding=1, bias=False, groups=min(32, ch_out))
                else:
                    self.k2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride,
                                        padding=1, bias=False, groups=1)
                self.bn2 = nn.BatchNorm2d(ch_out)

                self.k3 = nn.Conv2d(ch_out, int(ch_out * self.expansion), kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(int(ch_out * self.expansion))

                self.shortcut = nn.Sequential()
                if ch_in != ch_out:
                    self.shortcut = nn.Sequential(
                        [nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, bias=False, stride=1),
                         nn.BatchNorm2d(ch_out * self.expansion)
                         ])
                else:
                    self.shortcut = nn.Sequential()
            else:
                self.bn1 = nn.BatchNorm2d(ch_out)
                self.bn2 = nn.BatchNorm2d(ch_out)

                self.k1 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
                self.k2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        elif flagMode == mgState.decreasing:
            if params.twoSided:
                if self.bottleneck:
                    self.k1 = nn.Conv2d(ch_out * self.expansion, ch_out, kernel_size=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(ch_out)
                    self.k2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride,
                                        padding=1, bias=False)
                    self.bn2 = nn.BatchNorm2d(ch_out)
                    self.k3 = nn.Conv2d(ch_out, ch_out * self.expansion, kernel_size=1, bias=False)
                    self.bn3 = nn.BatchNorm2d(ch_out * self.expansion)

                    if ch_in != ch_out:
                        self.shortcut = nn.Sequential(
                            [nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, bias=False, stride=1),
                             nn.BatchNorm2d(ch_out * self.expansion)
                             ])
                    else:
                        self.shortcut = nn.Sequential()
                else:
                    self.k1 = nn.Conv2d(int(ch_out), int(ch_out), kernel_size=3, stride=1, groups=groupsChannel,
                                        padding=1,
                                        bias=False)
                    self.k2 = nn.Conv2d(int(ch_out), int(ch_out), kernel_size=3, stride=1, groups=groupsChannel,
                                        padding=1,
                                        bias=False)
            if params.pwGroup:
                if params.shuffleChannels:
                    self.perm = torch.randperm(ch_out * self.expansion)
                elif params.structuredChannels:
                    inverval = groupsChannel
                    perm = torch.arange(0, ch_out * self.expansion, inverval).long()
                    for i in torch.arange(1, groupsChannel):
                        offset = i
                        curr_perm = torch.arange(0, ch_out * self.expansion, inverval).long()
                        curr_perm = (curr_perm + offset).long()
                        perm = torch.cat([perm, curr_perm], dim=0)
                    self.perm = perm
                self.R = nn.Conv2d(int(ch_out) * self.expansion, int(ch_out / 2) * self.expansion, kernel_size=1,
                                   stride=1, groups=groupsPR,
                                   bias=False)
                if ch_in != ch_out:
                    self.shortcut = nn.Sequential(
                        [nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, bias=False, stride=1,
                                   groups=groupsChannel),
                         nn.BatchNorm2d(ch_out * self.expansion)
                         ])
                else:
                    self.shortcut = nn.Sequential()
                self.R.weight.data.uniform_(1.0 / float(numOfDiag), 1.0 / float(numOfDiag))
            elif params.genMixedRP:
                self.R, self.RMask = self.genMixedRP(int(ch_out / 2), int(ch_out), params)
            else:
                self.R, self.RMask = self.genTensorFullOneOneConv2(int(ch_out / 2), int(ch_out), params)
        elif flagMode == mgState.increasing:
            if self.bottleneck:
                self.k1 = nn.Conv2d(int(ch_out * self.expansion), ch_out, kernel_size=1, groups=groupsCNN, stride=1,
                                    bias=False)
                self.bn1 = nn.BatchNorm2d(ch_out)
                self.k2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, groups=groupsCNN, stride=1,
                                    padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(ch_out)
                self.k3 = nn.Conv2d(ch_out, int(ch_out * self.expansion), kernel_size=1, groups=groupsCNN,
                                    stride=1, bias=False)
                self.bn3 = nn.BatchNorm2d(int(ch_out * self.expansion))
                self.shortcut = nn.Sequential()

                nn.init.kaiming_normal_(self.k1.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(self.k2.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(self.k3.weight, mode='fan_out', nonlinearity='relu')

                nn.init.constant_(self.bn1.weight, 1)
                nn.init.constant_(self.bn1.bias, 0)

                nn.init.constant_(self.bn2.weight, 1)
                nn.init.constant_(self.bn2.bias, 0)

                nn.init.constant_(self.bn3.weight, 1)
                nn.init.constant_(self.bn3.bias, 0)

            else:
                self.k1 = nn.Conv2d(int(ch_out), int(ch_out), kernel_size=3, stride=1, groups=groupsChannel, padding=1,
                                    bias=False)
                self.k2 = nn.Conv2d(int(ch_out), int(ch_out), kernel_size=3, stride=1, groups=groupsChannel, padding=1,
                                    bias=False)
                self.bn1 = nn.BatchNorm2d(ch_out)

                self.bn2 = nn.BatchNorm2d(ch_out)

            if params.pwGroup:
                self.perm = perm
                if not reuse_RP:
                    self.P = nn.Conv2d(int(ch_in) * self.expansion, int(ch_out) * self.expansion, kernel_size=1,
                                       stride=1,
                                       groups=groupsPR,
                                       bias=False)
                    self.P.weight.data.uniform_(1.0 / float(numOfDiag), 1.0 / float(numOfDiag))
                else:
                    self.P = None
            elif params.genMixedRP:
                self.P, self.PMask = self.genMixedRP(ch_out, ch_in, params)
            else:
                self.P, self.PMask = self.genTensorFullOneOneConv2(ch_out, ch_in, params, decreaseRMask=RP)
            self.bnp1 = nn.BatchNorm2d(ch_out * self.expansion)
            nn.init.constant_(self.bnp1.weight, 1)
            nn.init.constant_(self.bnp1.bias, 0)

    def genTensorFullOneOneConv2(self, chOut, ChIn, params, decreaseRMask=[]):
        Kesxpand = nn.Parameter(torch.zeros(chOut, ChIn, 1, 1))
        weights = nn.Parameter(torch.Tensor(chOut, ChIn, 1, 1, ))
        KesxpandMask = nn.Parameter(torch.zeros(chOut, ChIn, 1, 1), requires_grad=False)
        weightsMask = nn.Parameter(torch.Tensor(chOut, ChIn, 1, 1), requires_grad=False)
        weights.data.uniform_(0, 1)
        weightsMask.data.uniform_(1, 1)
        r, c, k1, k2 = Kesxpand.shape
        if params.useMask:
            for k in np.arange(r):
                sum = 0.0
                for j in np.arange(c):
                    Kesxpand.data[k][j][0][0] = weights[k][j][0][0] * decreaseRMask[j][k][0][0]
                    sum += weights[k][j][0][0]
                for j in np.arange(c):
                    Kesxpand.data[k][j][0][0] /= sum
                    KesxpandMask.data[k][j][0][0] = decreaseRMask[j][k][0][0]
            return Kesxpand, KesxpandMask
        for k in np.arange(r):
            sum = 0.0
            numOfValues = int(params.nonZeroRP)
            vecArray = np.random.permutation(c)[0:numOfValues]
            for j in vecArray:
                Kesxpand.data[k][j][0][0] = weights[k][j][0][0]
                sum += weights[k][j][0][0]
            for j in vecArray:
                Kesxpand.data[k][j][0][0] /= sum
                KesxpandMask.data[k][j][0][0] = weightsMask[k][j][0][0]
        return Kesxpand, KesxpandMask

    def genMixedRP(self, chOut, ChIn, params):
        step = params.mgMaxNonZerosRP
        numOfDiag = np.min([int(chOut), int(ChIn), params.mgMaxNonZerosRP])
        Kesxpand = nn.Parameter(torch.zeros(chOut, ChIn, 1, 1))
        weights = nn.Parameter(torch.Tensor(chOut, ChIn, 1, 1, ))
        KesxpandMask = nn.Parameter(torch.zeros(chOut, ChIn, 1, 1), requires_grad=False)
        weights.data.uniform_(0, 1)
        r, c, k1, k2 = Kesxpand.shape
        indc = 0
        if chOut < ChIn:
            for k in np.arange(r):
                sumi = 0.0
                for j in np.arange(numOfDiag):
                    indR = (j + indc) % c
                    Kesxpand.data[k][indR][0][0] = weights[k][indR][0][0]
                    KesxpandMask.data[k][indR][0][0] = 1
                    sumi += weights[k][indR][0][0]
                for j in np.arange(c):
                    Kesxpand.data[k][j][0][0] /= sumi
                indc += step
        else:
            for k in np.arange(c):
                sumi = 0.0
                for j in np.arange(numOfDiag):
                    indR = (j + indc) % r
                    Kesxpand.data[indR][k][0][0] = weights[indR][k][0][0]
                    KesxpandMask.data[indR][k][0][0] = 1
                indc += step

            for k in np.arange(r):
                sumi = 0.0
                for j in np.arange(c):
                    sumi += Kesxpand.data[k][j][0][0]
                if sumi > 0:
                    for j in np.arange(c):
                        Kesxpand.data[k][j][0][0] /= sumi
        return Kesxpand, KesxpandMask


class ResNet(nn.Module):
    def __init__(self, block, parameters, lastLayer=True, inputChannels=3):
        super(ResNet, self).__init__()
        self.lastLayer = lastLayer
        self.in_planes = parameters.channelArray[0]
        self.params = parameters
        self.params = parameters
        self.inputChannels = inputChannels
        self.numberOfCheckpoints = parameters.numberOfCheckpoints
        self.openLayer = self.params.openLayerStride == 2
        self.bottleneck = parameters.bottleneck
        if self.bottleneck:
            self.expansion = 4
        else:
            self.expansion = 1
        if self.openLayer:
            self.features = nn.Sequential(OrderedDict([
                ('conv1',
                 nn.Conv2d(self.inputChannels, parameters.channelArray[0], kernel_size=self.params.openLayerKernel,
                           stride=self.params.openLayerStride,
                           padding=int((self.params.openLayerKernel - 1) / 2), bias=False)),
                ('bn1', nn.BatchNorm2d(parameters.channelArray[0])),
                ('relu1', nn.ReLU()),
                ('pooling', nn.MaxPool2d(kernel_size=3, stride=self.params.openLayerStride, padding=1, dilation=1,
                                         ceil_mode=False)), ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv1',
                 nn.Conv2d(self.inputChannels, parameters.channelArray[0], kernel_size=self.params.openLayerKernel,
                           stride=int((self.params.openLayerKernel - 1) / 2), padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(parameters.channelArray[0])),
                ('relu1', nn.ReLU()), ]))
        stage = 1

        for ind in np.arange(len(parameters.channelArray)):
            self._make_layer(block, int(self.params.channelArray[ind]), self.params.RepetitionArray[ind],
                             self.params.strideArray[ind], stage=stage)
            stage += 1
        if self.lastLayer:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.linear = nn.Linear(self.expansion * self.params.channelArray[-1], self.params.num_classes)

        else:
            self.features.add_module('seq', nn.Sequential())

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and (not name.__contains__('R') or not name.__contains__('P')):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, stage=1):
        self.features.add_module(
            'Mg_%d_%d' % (stage, 0),
            block(self.in_planes, planes, self.params, stride, expansion=self.expansion))
        self.in_planes = planes * self.expansion
        for i in range(1, num_blocks):
            self.features.add_module(
                'Mg_%d_%d' % (stage, i),
                block(self.in_planes, planes, self.params, expansion=self.expansion))

    def forward(self, x):
        x.requires_grad = True
        modules = [module for k, module in self._modules.items()][0]
        input_var = checkpoint_sequential(modules, self.numberOfCheckpoints, x)
        if self.lastLayer:
            input_var = self.avgpool(input_var)
            input_var = input_var.view(input_var.size(0), -1)
            out = self.linear(input_var)
        else:
            out = input_var
        return out


def ResNetMGGroup(parameters):
    return ResNet(MgBlockDwGroup, parameters)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, params, stride=1, expansion=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.expansion = expansion
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


def ResNet34(parameters):
    return ResNet(BasicBlock, parameters)


class ASPP(nn.Module):
    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(6 * mult), padding=int(6 * mult),
                          bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(12 * mult), padding=int(12 * mult),
                          bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                          dilation=int(18 * mult), padding=int(18 * mult),
                          bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                          bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self, block, parameters):
        super(DeepLab, self).__init__()
        self.numberOfCheckpoints = parameters.numberOfCheckpoints
        self.conv = nn.Conv2d
        self.resnet = ResNet(block, parameters, lastLayer=False)
        self.aspp = ASPP(parameters.channelArray[-1], 256, parameters.num_classes, conv=self.conv)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        # x.requires_grad = True
        input_var = self.resnet(x)
        out = self.aspp(input_var)
        out = nn.Upsample(size, mode='bilinear', align_corners=True)(out)
        return out


def DeepLabMG(parameters):
    return DeepLab(MgBlockDwGroup, parameters)


#####################################
##########AutoEncoder################
#####################################

class TrnasposedResNet(nn.Module):
    def __init__(self, block, parameters):
        super(TrnasposedResNet, self).__init__()
        self.in_planes = parameters.channelArray[0]
        self.params = parameters
        self.numberOfCheckpoints = parameters.numberOfCheckpoints
        self.openLayer = self.params.openLayerStride == 2

        self.features = nn.Sequential()
        self.Layers = torch.nn.ModuleList()
        stage = 1
        for ind in np.arange(len(parameters.channelArray)):
            upsample = False
            if ind > 0 and ind < len(parameters.channelArray) - 1:
                upsample = True
            self.Layers.append(
                self._make_layer(block, int(self.params.channelArray[ind]), self.params.RepetitionArray[ind],
                                 self.params.strideArray[ind], stage=stage, upsample=upsample))
            stage += 1
        self.features.add_module('seq', nn.Sequential())

    def _make_layer(self, block, planes, num_blocks, stride, stage=1, upsample=False):
        self.features.add_module(
            'Mg_%d_%d' % (stage, 0),
            block(self.in_planes, planes, self.params, stride))
        if upsample:
            self.features.add_module(
                'Upsample_%d_%d' % (stage, 0),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.in_planes = planes
        for i in range(1, num_blocks):
            self.features.add_module(
                'Mg_%d_%d' % (stage, i),
                block(self.in_planes, planes, self.params))

    def forward(self, x):
        modules = [module for k, module in self._modules.items()][0]
        seq = torch.nn.Sequential(modules)
        out = seq(x)
        return out


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class AutoEncoder(nn.Module):
    def __init__(self, block, parameters, inputChannels=3, d_latent=1024):
        super(AutoEncoder, self).__init__()
        self.inputChannels = inputChannels
        self.numberOfCheckpoints = parameters.numberOfCheckpoints
        self.encoder = ResNet(block, parameters, lastLayer=False, inputChannels=self.inputChannels)
        self.dParams = self.decoderParams(parameters)
        self.decoder = TrnasposedResNet(block, self.dParams)
        self.lastLayerSize = parameters.channelArray[-1]
        self.d_latent = d_latent
        self.latent_mapping = nn.Sequential(
            nn.Linear(parameters.channelArray[-1], d_latent, True),
            nn.BatchNorm1d(d_latent),
            nn.Tanh()
        )

        self.latent_mapping_decoder = nn.Sequential(
            nn.Linear(self.d_latent, 4 * 4 * self.lastLayerSize),
            nn.BatchNorm1d(4 * 4 * self.lastLayerSize),
            nn.ReLU()
        )

        self.output_conv = conv3x3(self.dParams.channelArray[-1], self.inputChannels, 1, True)
        self.bnpfinal = nn.BatchNorm2d(self.inputChannels)
        self.finalUpsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output_conv2 = conv3x3(self.inputChannels, self.inputChannels, 1, True)

        self.final_act = nn.Sigmoid()

    def decoderParams(self, encoderParams):
        params = copy.deepcopy(encoderParams)
        params.channelArray = params.channelArray[::-1]
        params.RepetitionArray = params.RepetitionArray[::-1]
        params.strideArray = [[1] * (len(params.strideArray))][0]

        return params

    def forward(self, x):
        ########### Encoder ###############
        x = self.encoder(x)

        ########### Decoder ###############
        x = self.decoder(x)
        x = F.relu(self.bnpfinal(self.output_conv(x)))
        x = self.finalUpsample(x)
        x = self.output_conv2(x)
        x = self.final_act(x)
        return x


def AutoEncoderMG(parameters, inputChannels=3):
    return AutoEncoder(MgBlockDwGroup, parameters, inputChannels)
