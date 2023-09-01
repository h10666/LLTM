import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import torch.nn.functional as F
from torchvision.ops.boxes import box_area

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from opts import parser
from CLIP import clip
import argparse

args, _ = parser.parse_known_args()

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Temporal_head(nn.Module):
    '''
    This class is used to redcue the dim of frame-level features,
    and then concat them followed by a temporal fusion layer or avgpool
    '''
    def __init__(self, frame_fea_dim, video_fea_dim, mode = 'temporal'):
        super(Temporal_head, self).__init__()
        self.T = args.num_segments # 8
        self.mode = args.mode
        self.frame_fea_dim = frame_fea_dim
        self.hidden_dim = video_fea_dim
        if args.two_stream:
            self.dim_reduction = self.build_reduction_layers(self.frame_fea_dim, self.hidden_dim)
        # before_fusion_feas = 4096 ##???
        # in_dim 8*512?  T varies on cases
        if self.mode == 'temporal':
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.temporal_fusion = self.build_temporal_fusion_layers(in_dim=self.T * self.frame_fea_dim, out_dim=self.frame_fea_dim)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))              
    def forward(self, video_fea_map, test_model):
        # frame_feas: <B*T, C, H, W>
        # print('video_fea_map:\t', video_fea_map.size())
        if self.mode == 'temporal':
            if args.arch == 'ViT':
                frame_feas = video_fea_map
            else:
                BT, C, _, _ = video_fea_map.size() # <BT, C, 7, 7>
                frame_feas = self.avgpool(video_fea_map).view(BT, -1) # <BT, C>
            
            # frame_feas = self.dim_reduction(frame_feas) # <B*T, C> --> <B*T, 512>
            frame_feas = frame_feas.view(BT//self.T, -1) # <B*T, 512> --> <B, T*512>
            global_video_fea = self.temporal_fusion(frame_feas) # <B, T*512> --> <B, 512>
        else:
            
            # pooling
            if args.arch == 'ViT':
                frame_feas = video_fea_map
                # print('frame_feas:\t', frame_feas.size())
            else:
                BT, C, _, _ = video_fea_map.size() # <BT, C, 7, 7>
                frame_feas = self.avgpool(video_fea_map).view(BT, -1) # <BT, C>
                # print('frame_feas:\t', frame_feas.size())
                # print('frame_feas:\t', frame_feas)
                
            # print('frame_feas1:\t', frame_feas.size())
             
            if args.do_aug and test_model:
                if args.do_aug_mix:                    
                    frame_feas = frame_feas.view((-1, self.T*5) + frame_feas.size()[1:])
                    # print('frame_feas:\t', frame_feas.size())
                    # print(frame_feas)
                    T = frame_feas.size(1)
                    # print(B)
                    frame_feas1 = frame_feas[:, 0:(T//5), :]
                    frame_feas2 = frame_feas[:, (T//5):(2*T//5), :]
                    # print('feame_feas2:\t', frame_feas2.size())
                    frame_feas3 = frame_feas[:, (2*T//5):(3*T//5), :]
                    frame_feas4 = frame_feas[:, (3*T//5):(4*T//5), :]
                    frame_feas5 = frame_feas[:, (4*T//5):, :]   
                    
                    global_video_fea1 = frame_feas1.mean(dim=1, keepdim=True)
                    # print('global_video_fea1:\t', global_video_fea1)
                    global_video_fea1 = global_video_fea1.squeeze(1)
                    # print('global_video_fea1:\t', global_video_fea1)
                    global_video_fea2 = frame_feas2.mean(dim=1, keepdim=True)
                    # print('global_video_fea2:\t', global_video_fea2)
                    global_video_fea2 = global_video_fea2.squeeze(1)
                    # print('global_video_fea2:\t', global_video_fea2)
                    global_video_fea3 = frame_feas3.mean(dim=1, keepdim=True)
                    # print('global_video_fea3:\t', global_video_fea3.size())
                    global_video_fea3 = global_video_fea3.squeeze(1)
                    # print('global_video_fea3:\t', global_video_fea3.size())
                    global_video_fea4 = frame_feas4.mean(dim=1, keepdim=True)
                    # print('global_video_fea4:\t', global_video_fea4.size())
                    global_video_fea4 = global_video_fea4.squeeze(1)
                    # print('global_video_fea4:\t', global_video_fea4.size())
                    global_video_fea5 = frame_feas5.mean(dim=1, keepdim=True)
                    # print('global_video_fea5:\t', global_video_fea5.size())
                    global_video_fea5 = global_video_fea5.squeeze(1)
                    # print('global_video_fea5:\t', global_video_fea5.size())
                    
                    
                    global_video_fea = torch.cat([global_video_fea1, global_video_fea2, global_video_fea3, global_video_fea4, global_video_fea5], dim=0)
                # print('global_video_fea:\t', global_video_fea.size())
                else:
                    frame_feas = frame_feas.view((-1, self.T*2) + frame_feas.size()[1:])
                    T = frame_feas.size(1)
                    frame_feas1 = frame_feas[:, :(T//2), :]
                    # print('frame_feas1:\t', frame_feas1.size())
                    frame_feas2 = frame_feas[:, (T//2):, :]
                    # print('feame_feas2:\t', frame_feas2.size())
                    global_video_fea1 = frame_feas1.mean(dim=1, keepdim=True)
                    # print('global_video_fea1:\t', global_video_fea1.size())
                    global_video_fea1 = global_video_fea1.squeeze(1)
                    # print('global_video_fea1:\t', global_video_fea1.size())
                    global_video_fea2 = frame_feas2.mean(dim=1, keepdim=True)
                    # print('global_video_fea2:\t', global_video_fea2.size())
                    global_video_fea2 = global_video_fea2.squeeze(1)
                    # print('global_video_fea2:\t', global_video_fea2.size())
                    global_video_fea = torch.cat([global_video_fea1, global_video_fea2], dim=0)

            # exit(0)
            else:
                frame_feas = frame_feas.view((-1, self.T) + frame_feas.size()[1:])
                # print('frame_feas:\t', frame_feas.size())
                global_video_fea = frame_feas.mean(dim=1, keepdim=True)
                global_video_fea = global_video_fea.squeeze(1)
            # exit(0)   
            # print('frame_feas2:\t', frame_feas.size())
            # global_video_fea = frame_feas.mean(dim=1, keepdim=True)
            # print('global_video_fea1:\t', global_video_fea.size())      
            # global_video_fea = global_video_fea.squeeze(1)
            # print('global_video_fea2:\t', global_video_fea.size())
            if args.two_stream:
                hidden_features = self.dim_reduction(global_video_fea)
            else:
                hidden_features = None

        return global_video_fea, frame_feas, hidden_features
    
    def build_reduction_layers(self, in_dim, out_dim=1024):
        if in_dim == 2048:
            net = nn.Sequential(
                    nn.Linear(in_dim, 1024, bias=False),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True)
                )
        else:
            net = nn.Sequential(
                    nn.Linear(in_dim, out_dim, bias=False),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(inplace=True)
                )
        return net
    
    def build_temporal_fusion_layers(self, in_dim, out_dim=2048):
        net = nn.Sequential(
                nn.Linear(in_dim, in_dim//2, bias=False),
                nn.BatchNorm1d(in_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim//2, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            )
        return net

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)