import numpy as np
import torch
from torchvision.ops.roi_align import roi_align

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_region_feas(feature_maps, boxes_list, output_crop_size=[3, 3], img_size=[224, 224]):
    # Building feas for each bounding box by using RoI Align
    # feature_maps:[N,C,H,W], where N=b*T
    IH, IW = img_size
    FH, FW = feature_maps.size()[-2:]  # Feature_H, Feature_W
    region_feas = roi_align(feature_maps, boxes_list, output_crop_size, spatial_scale=float(
        FW)/IW)  # b*T*K, C, S, S; S denotes output_size
    return region_feas.view(region_feas.size(0), -1)  # b*T*K, D*S*S

def box_to_normalized(boxes_tensor, crop_size=[224,224], mode='list'):
    # tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
    new_boxes_tensor = boxes_tensor.clone()
    new_boxes_tensor[...,0] = (boxes_tensor[...,0]-boxes_tensor[...,2]/2.0)*crop_size[0]
    new_boxes_tensor[...,1] = (boxes_tensor[...,1]-boxes_tensor[...,3]/2.0)*crop_size[1]
    new_boxes_tensor[...,2] = (boxes_tensor[...,0]+boxes_tensor[...,2]/2.0)*crop_size[0]
    new_boxes_tensor[...,3] = (boxes_tensor[...,1]+boxes_tensor[...,3]/2.0)*crop_size[1]
    if mode == 'list':
        boxes_list = []
        for boxes in new_boxes_tensor:
            boxes_list.append(boxes)
        return boxes_list
    elif mode == 'tensor':
        return new_boxes_tensor


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k""" 
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_class_names_ZSL(split_mode, phase):
    class_names = []
    with open('dataset/STHELSE/%s/%s_set.txt'%(split_mode, phase),'r') as f:
        lines = f.readlines()
        for line in lines:
            class_names.append(line.split('\n')[0]) 
    return class_names

def modify_target(target, category_list, class_names):
    new_targets = []
    # since = time.time()
    
    for t in target:
        # target_class_name = list(label_json.keys())[list(label_json.values()).index(t)]  # Prints george
        target_class_name = category_list[int(t)]
        new_targets.append(class_names.index(target_class_name))
    # print('time test:\t', time.time()-since)
    return torch.Tensor(new_targets).long()