import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Human_Stream(nn.Module):
    def __init__(self):
        super(Human_Stream, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_512 = nn.Linear(512, 512)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.conv_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv_1024 = nn.Conv2d(512, 1024, kernel_size=1)
        self.fc_1024_1 = nn.Linear(1536, 1024)
        self.fc_1024_2 = nn.Linear(1024, 1024)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, head, pool5_H, H_num):
        fc7 = self.avgpool(pool5_H)
        fc7 = fc7.view(fc7.size(0), -1)
        fc7_H = self.fc_512(fc7[:H_num, :])
        fc7_H = self.relu(fc7_H)
        fc7_H = self.dropout_1(fc7_H)
        fc7_H = fc7_H.reshape(fc7_H.size(0), fc7_H.size(1), 1, 1)
        head_phi = self.conv_512(head)
        head_phi = self.bn_512(head_phi)
        head_phi = self.relu(head_phi)
        head_g = self.conv_512(head)
        head_g = self.bn_512(head_g)
        head_g = self.relu(head_g)

        att = torch.mean(torch.mul(head_phi, fc7_H), 1, keepdim=True)
        att_shape = att.size()
        att = att.reshape(att_shape[0], att_shape[1], -1)
        att = F.softmax(att, dim=1)
        att = att.reshape(att_shape)

        att_head_H = torch.mul(att, head_g)
        att_head_H = self.conv_1024(att_head_H)
        att_head_H = self.bn_1024(att_head_H)
        att_head_H = self.relu(att_head_H)
        pool5_SH = self.avgpool(att_head_H)
        pool5_SH = pool5_SH.view(pool5_SH.size(0), -1)

        Concat_SH = torch.cat((fc7[:H_num,:], pool5_SH), 1)

        return fc7, Concat_SH


class Object_Stream(nn.Module):
    def __init__(self):
        super(Object_Stream, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_512 = nn.Linear(512, 512)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.conv_512 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv_1024 = nn.Conv2d(512, 1024, kernel_size=1)
        self.fc_1024_1 = nn.Linear(1536, 1024)
        self.fc_1024_2 = nn.Linear(1024, 1024)
        self.bn_512 = nn.BatchNorm2d(512)
        self.bn_1024 = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, head, pool5_O):
        fc7 = self.avgpool(pool5_O)
        fc7 = fc7.view(fc7.size(0), -1)
        fc7_O = self.fc_512(fc7)
        fc7_O = self.relu(fc7_O)
        fc7_O = self.dropout_1(fc7_O)
        fc7_O = fc7_O.reshape(fc7_O.size(0), fc7_O.size(1), 1, 1)

        head_phi = self.conv_512(head)
        head_phi = self.bn_512(head_phi)
        head_phi = self.relu(head_phi)
        head_g = self.conv_512(head)
        head_g = self.bn_512(head_g)
        head_g = self.relu(head_g)

        att = torch.mean(torch.mul(head_phi, fc7_O), 1, keepdim=True)
        att_shape = att.size()
        att = att.reshape(att_shape[0], att_shape[1], -1)
        att = F.softmax(att, dim=1)
        att = att.reshape(att_shape)

        att_head_O = torch.mul(att, head_g)
        att_head_O = self.conv_1024(att_head_O)
        att_head_O = self.bn_1024(att_head_O)
        att_head_O = self.relu(att_head_O)
        pool5_SO = self.avgpool(att_head_O)
        pool5_SO = pool5_SO.view(pool5_SO.size(0), -1)

        Concat_SO = torch.cat((fc7, pool5_SO), 1)

        return Concat_SO

class Pairwise_Stream(nn.Module):
    def __init__(self):
        super(Pairwise_Stream, self).__init__()
        self.conv_64 = nn.Conv2d(2, 64, kernel_size=5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_32 = nn.Conv2d(64, 32, kernel_size=5)
        self.bn_64 = nn.BatchNorm2d(64)
        self.bn_32 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spatial, fc7_H):
        conv1_sp = self.conv_64(spatial)
        conv1_sp = self.bn_64(conv1_sp)
        conv1_sp = self.relu(conv1_sp)
        pool1_sp = self.maxpool(conv1_sp)
        conv2_sp = self.conv_32(pool1_sp)
        conv2_sp = self.bn_32(conv2_sp)
        conv2_sp = self.relu(conv2_sp)
        pool2_sp = self.maxpool(conv2_sp)
        pool2_flat_sp = pool2_sp.reshape(pool2_sp.size(0), -1)

        Concat_SHsp = torch.cat((pool2_flat_sp, fc7_H), 1)

        return Concat_SHsp

class resnet18(nn.Module):

    def __init__(self, layers=None, num_classes=8, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(resnet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if layers is None:
            layers = [2, 2, 2, 2]
        self.layers = layers

        #
        self.num_classes = num_classes
        self.num_fc = 1024

        self.block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        self.human_stream = Human_Stream()
        self.object_stream = Object_Stream()
        self.pairwise_stream = Pairwise_Stream()

        self.fc_1536 = nn.Linear(1536, 1024)
        self.fc_4384 = nn.Linear(5920, 1024)
        self.fc_1024_1 = nn.Linear(1024, 1024)
        self.fc_1024_2 = nn.Linear(1024, 8)
        self.segmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, image, spatial, Hsp_boxes, O_boxes, H_num):


        y = self.conv1(image)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)

        head = self.layer3(y)

        pool5_H, _ = self.crop_pool_layer(head, Hsp_boxes)
        pool5_O, _ = self.crop_pool_layer(head, O_boxes)

        fc7_H = self.layer4(pool5_H)
        fc7_O = self.layer4(pool5_O)

        fc7, Concat_SH = self.human_stream(head, fc7_H, H_num)
        Concat_SO = self.object_stream(head, fc7_O)
        Concat_SHsp = self.pairwise_stream(spatial, fc7)

        Concat_SH = self.fc_1536(Concat_SH)
        Concat_SH = self.relu(Concat_SH)
        Concat_SH = self.dropout(Concat_SH)
        Concat_SH = self.fc_1024_1(Concat_SH)
        Concat_SH = self.relu(Concat_SH)
        Concat_SH = self.dropout(Concat_SH)
        cls_score_H = self.fc_1024_2(Concat_SH)
        cls_score_H = self.segmoid(cls_score_H)

        Concat_SO = self.fc_1536(Concat_SO)
        Concat_SO = self.relu(Concat_SO)
        Concat_SO = self.dropout(Concat_SO)
        Concat_SO = self.fc_1024_1(Concat_SO)
        Concat_SO = self.relu(Concat_SO)
        Concat_SO = self.dropout(Concat_SO)
        cls_score_O = self.fc_1024_2(Concat_SO)
        cls_score_O = self.segmoid(cls_score_O)

        Concat_SHsp = self.fc_4384(Concat_SHsp)
        Concat_SHsp = self.relu(Concat_SHsp)
        Concat_SHsp = self.dropout(Concat_SHsp)
        Concat_SHsp = self.fc_1024_1(Concat_SHsp)
        Concat_SHsp = self.relu(Concat_SHsp)
        Concat_SHsp = self.dropout(Concat_SHsp)
        cls_score_Hsp = self.fc_1024_2(Concat_SHsp)
        cls_score_Hsp = self.segmoid(cls_score_Hsp)

        return cls_score_H, cls_score_O, cls_score_Hsp

    def crop_pool_layer(self, bottom, rois):

        rois = rois.detach()
        batch_size = bottom.size(0)
        D = bottom.size(1)
        H = bottom.size(2)
        W = bottom.size(3)
        roi_per_batch = int(rois.size(0) / batch_size)
        x1 = rois[:, 1::4] / 16.0
        y1 = rois[:, 2::4] / 16.0
        x2 = rois[:, 3::4] / 16.0
        y2 = rois[:, 4::4] / 16.0

        height = bottom.size(2)
        width = bottom.size(3)

        # affine theta
        zero = Variable(rois.data.new(rois.size(0), 1).zero_())
        theta = torch.cat([ \
            (x2 - x1) / (width - 1),
            zero,
            (x1 + x2 - width + 1) / (width - 1),
            zero,
            (y2 - y1) / (height - 1),
            (y1 + y2 - height + 1) / (height - 1)], 1).view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, 7, 7)))
        bottom = bottom.view(1, batch_size, D, H, W).contiguous().expand(roi_per_batch, batch_size, D, H, W) \
            .contiguous().view(-1, D, H, W)
        grid = grid.type_as(bottom)
        crops = F.grid_sample(bottom, grid)

        return crops, grid