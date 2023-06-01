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
import math


class SelectiveSequential(nn.Sequential):
    def __init__(self, to_select, modules_dict):
        super(SelectiveSequential, self).__init__()
        for idx, module in enumerate(modules_dict):
            self.add_module(str(idx), module)
        self._to_select = to_select

    def forward(self, x):
        list = []
        for name, module in self._modules.items():
            x = module(x)
            if name in self._to_select:
                list.append(x)
        return list, x


# Function to find the number closest
# to n and divisible by m
def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if ((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))
        # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1
        # else n2 is the required closest number
    return n2


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


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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
                 bottleneck=False,
                 resNext=False,
                 mbnet=True
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
        self.mbnet = mbnet


class MGBottleneck(nn.Module):
    """ MG bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, mgParams, dw_kernel_size=1,
                 stride=1, act_layer=nn.ReLU, use_se=False,
                 use_hs=False):
        super(MGBottleneck, self).__init__()
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = MgBlockDwGroup(in_chs, mid_chs, out_chs, mgParams, stride=stride, dw_kernel_size=dw_kernel_size,
                                     use_se=use_se,
                                     use_hs=use_hs)

    def forward(self, x):
        x = self.ghost1(x)
        return x


class MgBlockDwGroup(nn.Module):
    def __init__(self, in_planes, mid_planes, planes, parameters, stride=1, dw_kernel_size=1, use_se=False,
                 use_hs=False):
        super(MgBlockDwGroup, self).__init__()
        self.parameters = parameters
        self.mgStep = torch.nn.ModuleList()
        self.flagCatConnector = self.parameters.flagCatConnector
        self.connector = False
        self.stride = stride
        self.bottleneck = self.parameters.bottleneck
        self.expansion = 1
        self.identity = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.connector = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, stride=stride,
                          bias=False, groups=1),
            )

            if self.identity:
                if in_planes == mid_planes:
                    self.conv = nn.Sequential(
                        # dw
                        nn.Conv2d(mid_planes, mid_planes, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=mid_planes,
                                  bias=False),
                        nn.BatchNorm2d(mid_planes),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # Squeeze-and-Excite
                        SELayer(mid_planes) if use_se else nn.Identity(),
                        # pw-linear
                        nn.Conv2d(mid_planes, planes, 1, 1, 0, bias=False, groups=int(np.gcd(mid_planes, planes))),
                        nn.BatchNorm2d(planes),
                    )
                else:
                    self.conv = nn.Sequential(
                        # pw
                        nn.Conv2d(in_planes, mid_planes, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(mid_planes),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # dw
                        nn.Conv2d(mid_planes, mid_planes, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=mid_planes,
                                  bias=False),
                        nn.BatchNorm2d(mid_planes),
                        # Squeeze-and-Excite
                        SELayer(mid_planes) if use_se else nn.Identity(),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # pw-linear
                        nn.Conv2d(mid_planes, planes, 1, 1, 0, bias=False, groups=int(np.gcd(mid_planes, planes))),
                        nn.BatchNorm2d(planes),
                    )
            else:
                self.mgStep = torch.nn.ModuleList()
                self.mgStep = genTensor4CycleDwGroup(planes, mid_planes, planes, mgState.decreasing, self.mgStep, 1,
                                                     self.parameters,
                                                     dw_kernel_size, use_se=use_se, use_hs=use_hs)
        else:
            self.mgStep = torch.nn.ModuleList()
            self.mgStep = genTensor4CycleDwGroup(planes, mid_planes, planes, mgState.decreasing, self.mgStep, 1,
                                                 self.parameters,
                                                 dw_kernel_size, use_se=use_se, use_hs=use_hs)

    def resnetStep(self, x, k1, k2, bn1, bn2, ShortCutFlag=False):
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
        out = F.relu(out)

        out = k2(out)
        out = bn2(out)
        out = F.relu(out)

        out = k3(out)
        out = bn3(out)

        out += shortcut(x)
        return F.relu(out)

    def mbnetStep(self, x, k1, relu=True):
        out = k1(x)
        if relu:
            out = F.relu(out + x)

        return out + x

    def MgStepUp(self, x1, x2, xup, P, mgClass):
        # P step
        xdiff = x1 - x2
        z01 = mgClass.bnp1(F.conv2d(xdiff, P, stride=1, padding=(P.shape[3] - 1) // 2, bias=None))
        z02 = xup + z01
        return z02

    def MgStepUp_pwGroup(self, x1, x2, xup, P, mgClass):
        # P step
        xdiff = x1 - x2
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
        for step in self.mgStep:
            step.cnt += 1
            if step.mode == mgState.decreasing:
                if step.pwGroup:
                    perm = step.perm
                    startX = step.shortcut(startX)
                    outP = step.R(startX)
                    outP = step.bnr(outP)
                else:
                    currKernellr = torch.mul(step.RMask, step.R)
                    outP = F.conv2d(startX, currKernellr, stride=1, bias=None)
                zArr.append(startX)
                outForward.append(outP)
                startX = outP
                cnt2 = len(zArr) - 1
            elif step.mode == mgState.FC:
                if self.parameters.mbnet:
                    if step.relu == False:
                        xc = self.mbnetStep(outP, step.conv, step.relu)
                    else:
                        xc = self.mbnetStep(outP, step.conv)
            else:
                if step.pwGroup:
                    z0 = self.MgStepUp_pwGroup(xc, outForward[cnt2], zArr[cnt2], step.P, step)

                cnt2 -= 1
                out = self.mbnetStep(z0, step.conv)
                xc = out

        return out

    def forward(self, x):
        if self.connector:
            if self.identity:
                x = self.shortcut(x) + self.conv(x)
            else:
                x = self.shortcut(x)

        if self.parameters.mgParallelMode:
            out = self.parallelForward(x)
        else:
            out = self.serialForward(x)

        return out


def genTensor4CycleDwGroup(nInput, nMid, nOutput, flagMode, tensorList, stride, params, dw_kernel_size=1, use_se=False,
                           use_hs=False):
    params.useMask = False
    Ng = netMGParamsDWGroup(nInput, nMid, nOutput, flagMode, params, stride=stride, dw_kernel_size=dw_kernel_size,
                            use_se=use_se, use_hs=use_hs)
    decreaseMask = Ng.RMask
    perm = Ng.perm
    tensorList.append(Ng)
    if (nOutput / 2 < params.mgFcChannel):
        Ng = netMGParamsDWGroup(int(nOutput / 2), int(nMid / 2), int(nOutput / 2), mgState.FC, params,
                                dw_kernel_size=dw_kernel_size, use_se=use_se, use_hs=use_hs)
        tensorList.append(Ng)
    else:
        tensorList = genTensor4CycleDwGroup(int(nOutput / 2), int(nMid / 2), int(nOutput / 2), mgState.decreasing,
                                            tensorList, 1,
                                            params, dw_kernel_size=dw_kernel_size, use_se=use_se, use_hs=use_hs)
    if params.SameMaskStructure:
        params.useMask = True

    Ng = netMGParamsDWGroup(int(nOutput / 2), nMid, nOutput, mgState.increasing, params, RP=decreaseMask, perm=perm,
                            dw_kernel_size=dw_kernel_size, use_se=use_se, use_hs=use_hs)
    tensorList.append(Ng)

    return tensorList


class netMGParamsDWGroup(nn.Module):
    def __init__(self, ch_in, ch_mid, ch_out, flagMode, params, stride=1, RP=[], perm=None, dw_kernel_size=1,
                 use_se=False, use_hs=False):
        super(netMGParamsDWGroup, self).__init__()

        self.shuffleChannels = params.shuffleChannels
        self.structuredChannels = params.structuredChannels
        self.mode = flagMode
        self.stride = stride
        self.shortcut = nn.Sequential()
        self.cnt = 0
        self.RMask = []
        self.perm = []
        self.pwGroup = params.pwGroup
        self.mbnet = params.mbnet
        self.relu = True
        self.expansion = 1
        if params.fixGroup:
            numOfDiag = params.numberOfFixGroup

        groupsChannel = int(np.max([ch_out / numOfDiag, 1]))

        while not (ch_out % groupsChannel == 0 and ch_mid % groupsChannel == 0):
            groupsChannel = groupsChannel - 1
        if flagMode == mgState.FC:
            if self.mbnet:
                if ch_in == ch_mid:
                    self.conv = nn.Sequential(
                        # dw
                        nn.Conv2d(ch_mid, ch_mid, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=ch_mid, bias=False),
                        nn.BatchNorm2d(ch_mid),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # Squeeze-and-Excite
                        SELayer(ch_mid) if use_se else nn.Identity(),
                        # pw-linear
                        nn.Conv2d(ch_mid, ch_out, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(ch_out),
                    )
                else:
                    self.conv = nn.Sequential(
                        # pw
                        nn.Conv2d(ch_in, ch_mid, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(ch_mid),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # dw
                        nn.Conv2d(ch_mid, ch_mid, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=ch_mid, bias=False),
                        nn.BatchNorm2d(ch_mid),
                        # Squeeze-and-Excite
                        SELayer(ch_mid) if use_se else nn.Identity(),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # pw-linear
                        nn.Conv2d(ch_mid, ch_out, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(ch_out),
                    )

        elif flagMode == mgState.decreasing:
            if params.shuffleChannels:
                self.perm = torch.randperm(ch_out * self.expansion)
            elif params.structuredChannels:
                self.perm = None
            self.R = nn.Conv2d(int(ch_out) * self.expansion, int(ch_out / 2) * self.expansion, kernel_size=1,
                               stride=1, groups=groupsChannel,
                               bias=False)
            self.R.weight.data.uniform_(1.0 / float(numOfDiag), 1.0 / float(numOfDiag))
            self.bnr = nn.BatchNorm2d(int(ch_out / 2) * self.expansion)

            if ch_in != ch_out:
                self.shortcut = nn.Sequential(
                    [nn.Conv2d(ch_in, ch_out * self.expansion, kernel_size=1, bias=False, stride=1,
                               groups=groupsChannel),
                     nn.BatchNorm2d(ch_out * self.expansion)
                     ])
        elif flagMode == mgState.increasing:
            if self.mbnet:
                if ch_out == ch_mid:
                    self.conv = nn.Sequential(
                        # dw
                        nn.Conv2d(ch_mid, ch_mid, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=ch_mid, bias=False),
                        nn.BatchNorm2d(ch_mid),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # Squeeze-and-Excite
                        SELayer(ch_mid) if use_se else nn.Identity(),
                        # pw-linear
                        nn.Conv2d(ch_mid, ch_out, 1, 1, 0, bias=False, groups=groupsChannel),
                        nn.BatchNorm2d(ch_out),
                    )
                else:
                    self.conv = nn.Sequential(
                        # pw
                        nn.Conv2d(ch_out, ch_mid, 1, 1, 0, bias=False, groups=groupsChannel),
                        nn.BatchNorm2d(ch_mid),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # dw
                        nn.Conv2d(ch_mid, ch_mid, dw_kernel_size, stride, (dw_kernel_size - 1) // 2,
                                  groups=ch_mid, bias=False),
                        nn.BatchNorm2d(ch_mid),
                        # Squeeze-and-Excite
                        SELayer(ch_mid) if use_se else nn.Identity(),
                        h_swish() if use_hs else nn.ReLU(inplace=True),
                        # pw-linear
                        nn.Conv2d(ch_mid, ch_out, 1, 1, 0, bias=False, groups=groupsChannel),
                        nn.BatchNorm2d(ch_out),
                    )

            if params.pwGroup:
                self.perm = perm
                self.P = nn.Conv2d(int(ch_in) * self.expansion, int(ch_out) * self.expansion, kernel_size=1,
                                   stride=1,
                                   groups=groupsChannel,
                                   bias=False)
            self.P.weight.data.uniform_(1.0 / float(numOfDiag), 1.0 / float(numOfDiag))
            self.bnp1 = nn.BatchNorm2d(ch_out * self.expansion)

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


##########MobileNetV3 ###########

class MGMBV3(nn.Module):
    def __init__(self, cfgs, block, parameters, num_classes=10, width=1.0, dropout=0.0, segOutPuts=False,
                 small=False):
        super(MGMBV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        self.numberOfCheckpoints = parameters.numberOfCheckpoints
        self.segOutPuts = segOutPuts

        output_channel = _make_divisible(16 * width, 8)  # check here

        self.conv_stem = nn.Conv2d(3, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = torch.nn.ModuleList()
        # block = MGBottleneck
        for cfg in self.cfgs:
            layers = []
            # k, t, c, SE, HS, s
            for k, exp_size, c, SE, HS, s in cfg:
                output_channel = _make_divisible(c * width, 8)

                hidden_channel = _make_divisible(input_channel * exp_size * width, 8)
                layers.append(block(input_channel, hidden_channel, output_channel, parameters, k, s,
                                    use_se=SE, use_hs=HS))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)

        output_channel = 1280
        if small:
            output_channel = 1024
        if not self.segOutPuts:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
            self.act2 = nn.ReLU(inplace=True)
            self.classifier = nn.Linear(output_channel, num_classes)

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

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if not self.segOutPuts:
            x = checkpoint_sequential(self.blocks, self.numberOfCheckpoints, x)
            x = self.conv_head(x)
            x = self.act2(x)
            #x = x.view(x.size(0), -1)
            x = F.adaptive_max_pool2d(x, (1, 1)).squeeze()
            if self.dropout > 0.:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.classifier(x)
            return x
        else:
            outlist, finalOut = self.blocks(x)
            x = finalOut
            s2 = outlist[0]
            s4 = outlist[1]
            return s2, s4, x


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output

    return hook


def mobileMGV3_full(parameters, segOutPuts=False, largest=128, **kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [[3, 1, 16, 0, 0, 1]],
        #
        [[3, 4, 24, 0, 0, 2]],
        [[3, 3, 24, 0, 0, 1]],
        #
        [[5, 3, 40, 1, 0, 2]],
        [[5, 3, 40, 1, 0, 1]],
        #
        [[5, 3, 40, 1, 0, 1]],
        [[3, 6, 80, 0, 1, 2],
         [3, 2.5, 80, 0, 1, 1],
         [3, 2.3, 80, 0, 1, 1],
         [3, 2.3, 80, 0, 1, 1]],
        #
        [[3, 6, 112, 1, 1, 1]],
        [[3, 6, 112, 1, 1, 1],
         [5, 6, 160, 1, 1, 2],
         [5, 6, 160, 1, 1, 1],
         [5, 6, 160, 1, 1, 1]]
    ]


    ##just for segmentation, no pooling in 3rd and 5th block;
    if segOutPuts:
        cfgs = [
            # k, t, c, SE, HS, s
            [[3, 1, 64, 0, 0, 1]],
            #
            [[3, 1, 64, 0, 0, 2]],
            [[3, 1.5, 64, 0, 0, 1]],
            #
            [[5, 1.5, 80, 0, 0, 2]],
            [[5, 1.5, 80, 0, 0, 1]],
            #
            [[5, 1.5, 80, 0, 0, 1]],
            [[3, 3, 80, 0, 0, 2],
             [3, 1.25, 80, 0, 1, 1],
             [3, 1.15, 80, 0, 1, 1],
             [3, 1.15, 80, 0, 1, 1]],
            #
            [[3, 3, 112, 0, 1, 1]],
            [[3, 3, 112, 0, 1, 1],
             [5, 3, 160, 0, 1, 1],
             [5, 3, 160, 0, 1, 1],
             [5, 3, 160, 0, 1, 1]]
        ]
    return MGMBV3(cfgs, MGBottleneck, parameters, segOutPuts=segOutPuts, **kwargs)


def mobileMGV3_small(parameters, segOutPuts=False, **kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [[3, 1, 16, 1, 0, 2]],
        [[3, 4.5, 24, 0, 0, 2]],
        [[3, 3.67, 24, 0, 0, 1]],
        [[5, 4, 40, 1, 1, 2],
         [5, 6, 40, 1, 1, 1],
         [5, 6, 40, 1, 1, 1]],
        [[5, 3, 48, 1, 1, 1],
         [5, 3, 48, 1, 1, 1]],
        [[5, 6, 96, 1, 1, 2],
         [5, 6, 96, 1, 1, 1],
         [5, 6, 96, 1, 1, 1]]
    ]
    return MGMBV3(cfgs, MGBottleneck, parameters, segOutPuts=segOutPuts, **kwargs, small=True)


###############Segmentation code, based on code taken from https://github.com/ekzhang/semantic-segmentation ##########


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


all_layers = []


def remove_sequential(network):
    for layer in network.children():
        if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
            remove_sequential(layer)
        if list(layer.children()) == []:  # if leaf node, add it to list
            all_layers.append(layer)


def get_trunk(parameters):
    """Retrieve the pretrained network trunk and channel counts"""

    backbone = mobileMGV3_full(parameters, segOutPuts=True)
    to_select = ['0', '1']
    backbone.blocks = SelectiveSequential(to_select, backbone.blocks)
    backbone.blocks[9][0].conv.dilation = (4, 4)

    s2_ch = 80
    s4_ch = 80
    high_level_ch = 688
    return backbone, s2_ch, s4_ch, high_level_ch


class LRASPP(nn.Module):
    """Lite R-ASPP style segmentation network."""

    def __init__(self, num_classes, params, criterion=None, use_aspp=False, num_filters=128, numberOfCheckpoints=4):
        """Initialize a new segmentation model.
        Keyword arguments:
        num_classes -- number of output classes (e.g., 19 for Cityscapes)
        trunk -- the name of the trunk to use ('mobilenetv3_large', 'mobilenetv3_small')
        use_aspp -- whether to use DeepLabV3+ style ASPP (True) or Lite R-ASPP (False)
            (setting this to True may yield better results, at the cost of latency)
        num_filters -- the number of filters in the segmentation head
        """
        super(LRASPP, self).__init__()
        self.numberOfCheckpoints = numberOfCheckpoints
        self.criterion = criterion
        self.trunk, s2_ch, s4_ch, high_level_ch = get_trunk(params)
        self.old = False
        self.use_aspp = use_aspp
        if self.use_aspp:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=12, padding=12, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv3 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Conv2d(num_filters, num_filters, 3, dilation=36, padding=36, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            aspp_out_ch = num_filters * 4
        else:
            self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
            self.aspp_conv2 = nn.Sequential(
                nn.AvgPool2d(kernel_size=(12, 12), stride=(3, 6)),
                nn.Conv2d(high_level_ch, num_filters, 1, bias=False),
                nn.Sigmoid(),
            )
            aspp_out_ch = num_filters
        if self.old:
            self.convs2 = nn.Conv2d(s2_ch, 32, kernel_size=1, bias=False)
            self.convs4 = nn.Conv2d(s4_ch, 64, kernel_size=1, bias=False)
            self.conv_up1 = ConvBnRelu(aspp_out_ch, num_filters, kernel_size=1)
            self.conv_up2 = ConvBnRelu(num_filters + 64, num_filters, kernel_size=1)
            self.conv_up3 = ConvBnRelu(num_filters + 32, num_filters, kernel_size=1)

        else:
            self.conv_up1 = nn.Conv2d(aspp_out_ch, num_classes, kernel_size=1)
            self.conv_up2 = nn.Conv2d(s4_ch, num_classes, kernel_size=1)

        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        if not self.old:
            self.last = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, inputs, gts=None):
        x = inputs
        s2, s4, final = self.trunk(x)

        if self.use_aspp:
            aspp = torch.cat([
                self.aspp_conv1(final),
                self.aspp_conv2(final),
                self.aspp_conv3(final),
                F.interpolate(self.aspp_pool(final), size=final.shape[2:]),
            ], 1)
        else:
            outconv1 = self.aspp_conv1(final)

            outconv2 = self.aspp_conv2(final)

            outinterp = F.interpolate(
                outconv2,
                final.shape[2:],
                mode='bilinear',
                align_corners=False
            )

            aspp = outconv1 * outinterp

        if self.old:
            y = self.conv_up1(aspp)
            y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

            y = torch.cat([y, self.convs4(s4)], 1)
            y = self.conv_up2(y)
            y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

            y = torch.cat([y, self.convs2(s2)], 1)
            y = self.conv_up3(y)
            y = self.last(y)
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

        else:
            aspp = F.interpolate(aspp, size=s2.shape[2:], mode='bilinear', align_corners=False)
            aspp = self.conv_up1(aspp)
            y_s2 = self.conv_up2(s2)
            y = y_s2 + aspp
            y = self.last(y)
            y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

        return y


class MobileV3MG_Large(LRASPP):
    """MobileNetV3-Large segmentation network."""
    model_name = 'MobileV3MG_Large-lraspp'

    def __init__(self, num_classes, params, **kwargs):
        super(MobileV3MG_Large, self).__init__(
            num_classes, params,
            **kwargs
        )
