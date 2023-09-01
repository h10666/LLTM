import numpy as np
import torch
import os
import pickle
import math
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
# from archs.detr_support.detr_matcher import build_matcher
from archs.detr_support import detr_box_ops

from opts import parser
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def adjust_learning_rate1(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    # print('lr:\t', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        try:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = decay * param_group['decay_mult']
        except:
            pass
        
def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        
def loss_boxes(box_outs, box_targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # assert 'pred_boxes' in outputs
        
        idx = _get_src_permutation_idx(indices)
        src_boxes = box_outs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(box_targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(detr_box_ops.generalized_box_iou(
            detr_box_ops.box_cxcywh_to_xyxy(src_boxes),
            detr_box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return loss_bbox
    
# def get_indices(box_outs, box_targets):
#     matcher = build_matcher(args)
#     indices = matcher(box_outs, box_targets)
#     return indices

def build_box_targets(box_tensors_var, box_categories_var):
    """
    box_tensors_var: a set of normalized (cx, cy, w, h); [B, T, K, 4]
    box_categories_var: [B, T, K, 1]
    # num_target_boxes = K = 4???
    """

    # targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
    #         "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
    #                 objects in the target) containing the class labels
    #         "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    # print(box_tensors_var.size(), box_categories_var.size()) # <4, 8, 4, 4> <4, 8, 4>
    # exit(0)
    b, t, _, _ = box_tensors_var.size() # get b*t
    bt = b*t 
    # view()  # BT, K, 4
    box_categories_var = box_categories_var.view(bt, *box_categories_var.size()[2:]).to(device) # <32, 4, 4>
    box_tensors_var = box_tensors_var.view(bt, *box_tensors_var.size()[2:]).to(device)# <32, 4>
    # print(box_tensors_var.size(), box_categories_var.size())
    # exit(0)
    box_targets = []
    for i in range(bt):
        labels = box_categories_var[i]  # [K, 1]
        boxes = box_tensors_var[i] # [K, 4]
        box_targets.append({'labels':labels.long(),'boxes':boxes})
    # print(box_targets)
    # exit(0)
    return box_targets

def optimizer_redefine(model):
    if args.arch == 'region_coord_tsm':
        optimizer = torch.optim.SGD(model.parameters(),
                                        args.visual_lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.optimizer_choice == 'AdamW':
        visual_backbone_params = list(map(id, model.module.base_model.visual_backbone.parameters()))
        rest_net = filter(lambda p: id(p) not in visual_backbone_params, model.parameters())
        optimizer = torch.optim.AdamW([{'params': rest_net},
                                {'params': model.module.base_model.visual_backbone.parameters(), 'lr': args.lr*args.ratio}],
                            betas=(0.9, 0.98), lr=args.lr, eps=1e-8,
                            weight_decay=args.weight_decay)
    elif args.optimizer_choice == 'SGD':
        if args.two_stream or args.joint:
            text_backbone_params = list(map(id, model.module.base_model.text_backbone.parameters()))
            rest_net = filter(lambda p: id(p) not in text_backbone_params, model.parameters())
            
            optimizer = torch.optim.SGD([{'params': rest_net},
                                    {'params': model.module.base_model.text_backbone.parameters(), 
                                    'lr': args.trans_lr}],
                                lr=args.visual_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        args.visual_lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    else:
        pass       
    
    return optimizer
def lr_scheduler_redefine(optimizer, epochs, warmup_epochs):
    lr_scheduler = WarmupCosineAnnealingLR(
            optimizer,
            args.epochs,
            # warmup_epochs=args.lr_warmup_step
            warmup_epochs = int(args.epochs*0.1)
        )
    return lr_scheduler  
#############################################################################
def to_tuple(x, L):
    if type(x) in (int, float):
        return [x] * L
    if type(x) in (list, tuple):
        if len(x) != L:
            raise ValueError('length of {} ({}) != {}'.format(x, len(x), L))
        return tuple(x)
    raise ValueError('input {} has unkown type {}'.format(x, type(x)))

class WarmupLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.num_groups = len(optimizer.param_groups)
        self.warmup_epochs = to_tuple(warmup_epochs, self.num_groups)
        self.warmup_powers = to_tuple(warmup_powers, self.num_groups)
        self.warmup_lrs = to_tuple(warmup_lrs, self.num_groups)
        super(WarmupLR, self).__init__(optimizer, last_epoch)
        assert self.num_groups == len(self.base_lrs)

    def get_lr(self):
        curr_lrs = []
        for group_index in range(self.num_groups):
            if self.last_epoch < self.warmup_epochs[group_index]:
                progress = self.last_epoch / self.warmup_epochs[group_index]
                factor = progress ** self.warmup_powers[group_index]
                lr_gap = self.base_lrs[group_index] - self.warmup_lrs[group_index]
                curr_lrs.append(factor * lr_gap + self.warmup_lrs[group_index])
            else:
                curr_lrs.append(self.get_single_lr_after_warmup(group_index))
        return curr_lrs

    def get_single_lr_after_warmup(self, group_index):
        raise NotImplementedError

class WarmupCosineAnnealingLR(WarmupLR):
    
    def __init__(self,
                 optimizer,
                 total_epoch,
                 final_factor=0,
                 warmup_epochs=0,
                 warmup_powers=1,
                 warmup_lrs=0,
                 last_epoch=-1):
        self.total_epoch = total_epoch
        self.final_factor = final_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer,
                                                      warmup_epochs,
                                                      warmup_powers,
                                                      warmup_lrs,
                                                      last_epoch)

    def get_single_lr_after_warmup(self, group_index):
        warmup_epoch = self.warmup_epochs[group_index]
        progress = (self.last_epoch - warmup_epoch) / (self.total_epoch - warmup_epoch)
        progress = min(progress, 1.0)
        cosine_progress = (math.cos(math.pi * progress) + 1) / 2
        factor = cosine_progress * (1 - self.final_factor) + self.final_factor
        return self.base_lrs[group_index] * factor
#################################################################################################
def save_results(logits_matrix, targets_list,
                 class_to_idx, args):
    """
    Saves the predicted logits matrix, true labels, sample ids and class
    dictionary for further analysis of results
    """
    print("Saving inference results ...")
    path_to_save = os.path.join(
        args.root_model, args.root_results, args.suffix + '_' "test_results.pkl")

    with open(path_to_save, "wb") as f:
        pickle.dump([logits_matrix, targets_list,
                     class_to_idx], f)  

def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)

        
    

    