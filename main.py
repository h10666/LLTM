# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
import time
import torch
import shutil
import json
import pickle
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import ops.utils as utils
import pandas as pd

from ops.dataset import TSNDataSet
# from data_utils.data_loader_frames import VideoFolder
from ops.models import TSN
# from ops.models_simple import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
# from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
# from train_helper import loss_boxes, build_box_targets
from archs.detr_support.detr_matcher import build_matcher
from archs.model_builder import tokenizer_build
from trainer import train
from validater import validate
from train_helper import optimizer_redefine
from train_helper import lr_scheduler_redefine
from archs.Text_Prompt import *
from archs.KLLoss import KLLoss



from tensorboardX import SummaryWriter

best_video_prec1 = 0
best_fusion_prec1 = 0
is_cuda = torch.cuda.is_available() # added by Mr. H
print('is_cuda:', is_cuda)

print('####################################################################################################')

def main():
    
    global args, best_video_prec1, best_fusion_prec1
    args = parser.parse_args()

    num_class, label_text, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    # print('label_text:\t', label_text)
    # print(num_class, args.train_list, args.val_list, args.root_path, prefix)
    # exit(0)
    full_arch_name = args.arch
    # print(full_arch_name)
    # exit(0)
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    # args.store_name = '_'.join(
    #     ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
    #      'e{}'.format(args.epochs)])
    args.store_name = '_'.join(
        [args.dataset, full_arch_name])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)
    # exit(0)
    check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local,
                clip_pretrained_name = args.clip_vis_name,
                two_stream = args.two_stream, 
                joint = args.joint,
                do_aug = args.do_aug,
                do_attention = args.do_attention)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    
    # # policies = None
    # if 'i3d' not in args.arch:
    #     policies = model.get_optim_policies()
    # else:
    #     policies = None

    train_augmentation = model.get_augmentation(flip=False if 'STHELSE' in args.dataset or 'something' in args.dataset or 'epic55' in args.dataset else True)
    print(args)
    print(model)
    # exit(0)
    # torch.load(model, map_location='cpu')
    model = torch.nn.DataParallel(model)
    
    # print(model.module.base_model.visual_backbone)
    # exit(0)
    if is_cuda:
        model = model.cuda()
    
    if args.optimizer_choice == 'AdamW':
        optimizer = optimizer_redefine(model)
        scheduler = lr_scheduler_redefine(optimizer, args.epochs, args.lr_warmup_step)
    elif args.optimizer_choice == 'SGD':
        optimizer = optimizer_redefine(model)

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             args.visual_lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    print('optimizer:\t', optimizer)
    # exit(0)
    '''
    policies = None
    if 'ZSL' not in args.split_mode:
        optimizer = torch.optim.SGD(policies if policies else model.parameters(),
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9,0.98), eps=1e-6, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    '''    
    matcher = build_matcher(args)
    
    # # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if args.arch == 'region_coord_tsm':
        tokenizer = None
    elif args.joint:
        tokenizer = tokenizer_build(args.embed_way)
    else:
        tokenizer = None
    # print('tokenizer:\t', tokenizer)
    # exit(0)
    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> fine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from, map_location='cpu')
        sd = sd['state_dict']
        ###################################
        # items = []
        # for item in sd.keys():
            # print('item:\t', item)
            # item = item.replace('base_model.', 'base_model.visual_backbone.')
            # item = item.replace('visual_backbone.conv1', 'visual_backbone.0')
            # item = item.replace('visual_backbone.bn1', 'visual_backbone.1')
            # item = item.replace('layer1', '4')
            # item = item.replace('layer2', '5')
            # item = item.replace('layer3', '6')
            # item = item.replace('layer4', '7')          
            # item_new = item.replace('base_model.', 'base_model.visual_backbone.')
            # sd[item_new]  = sd[item] 
            # del sd[item]
            # print('item:\t', sd[item_new])
            # items.append(item)
        ###################################
        model_dict = model.state_dict()
        ####################################
        # # items = []
        # for item in model_dict.keys():
        #     print('item:\t', item)
        # #     # items.append(item)
        # # print('keys:\t', type(items), items)
        # print(len(model_dict.keys()))
        # # exit(0)
        #####################################
        replace_dict = []
        for k, v in sd.items():
            # print(k)
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        # print(replace_dict)
        # exit(0)
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))
        # print(replace_dict)
        # exit(0)
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        # exit(0)
        model.load_state_dict(model_dict)
        # exit(0)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
    # normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    #####################################################
    # if args.do_aug:
    if args.tracked_boxes:
        print('... Loading box annotations might take a minute ...')
        since = time.time()
        try:
            with open(args.tracked_boxes, 'rb') as f:
                box_annotations = pickle.load(f)
        except:
            with open(args.tracked_boxes, 'r') as f:
                box_annotations = json.load(f)
        print('load box anno takes ', time.time()-since)
    else:
        box_annotations = None
    # print(type(box_annotations))
    # print(len(box_annotations))
    # print(box_annotations)
    # exit(0)
    #################################################################
    if args.caption_full:
        print('... Loading caption_full might take a minute ...')
        since = time.time()
        with open(args.caption_full, 'r') as k:
            captions = json.load(k)
        print('load full caption input takes ', time.time()-since)
    else:
        captions = None
    #################################################################
    if args.label_full:
        print('... Loading label_full might take a minute ...')
        since = time.time()
        with open(args.label_full, 'r') as k:
            label_captions = json.load(k)
        print('load full label input takes ', time.time()-since)
    else:
        label_captions = None
    #################################################################
    # print(captions)
    # print(label_captions)
    # exit(0)
    # if 'epic55' in args.dataset:

    #################################################################
    if not args.evaluate:
        dataset_train = TSNDataSet(args.root_path, pd.read_pickle(args.train_list) if 'epic55' in args.dataset else args.train_list,
                                    json_input = args.json_data_train, 
                                    json_labels = args.json_file_labels,
                                    num_segments=args.num_segments,
                                    new_length=data_length,
                                    modality=args.modality,
                                    image_tmpl=prefix,
                                    anno=box_annotations,
                                    captions = captions,
                                    label_captions = label_captions,
                                    label_text = label_text,
                                    num_boxes = args.num_boxes,
                                    random_crop = train_augmentation,
                                    transform=torchvision.transforms.Compose([
                                        # train_augmentation,
                                        Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                        ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                        normalize,
                                    ]), dense_sample=args.dense_sample)
        train_loader = torch.utils.data.DataLoader(
                            dataset_train,
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True,
                            drop_last=True, prefetch_factor = 2)
    # exit(0)
    dataset_test = TSNDataSet(args.root_path, pd.read_pickle(args.val_list) if 'epic55' in args.dataset else args.val_list,
                                json_input = args.json_data_val, 
                                json_labels = args.json_file_labels,
                                num_segments=args.num_segments,
                                new_length=data_length,
                                modality=args.modality,
                                image_tmpl=prefix,
                                random_shift=False,
                                anno=box_annotations,
                                label_captions = label_captions,
                                captions = captions,
                                label_text = label_text,
                                num_boxes = args.num_boxes,
                                random_crop = torchvision.transforms.Compose([
                                    GroupCenterCrop(crop_size)
                                    ]),
                                transform=torchvision.transforms.Compose([
                                    # GroupScale(int(scale_size)),
                                    # GroupCenterCrop(crop_size),
                                    Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                                    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                                    normalize,
                                ]), dense_sample=args.dense_sample)
    
    
    # print(dataset_test.classes_dict)
    # exit(0)
    val_loader = torch.utils.data.DataLoader(
                    dataset_test,
                    batch_size=args.batch_size, shuffle=False, drop_last=True,
                    num_workers=args.workers, pin_memory=True, prefetch_factor = 2)
    
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda() if is_cuda else torch.nn.CrossEntropyLoss()
        KLcriterion = KLLoss().cuda() if is_cuda else KLLoss()
    else:
        raise ValueError("Unknown loss type")
    ###############################################################################################
    # if 'ZSL' in args.split_mode:
    #     global criterion_MSE
    #     criterion_MSE = torch.nn.MSELoss().cuda() if torch.cuda.is_available() else torch.nn.MSELoss()
    ###############################################################################################
    # if policies:
    #     for group in policies:
    #         print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
    #             group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
    #####################
    # text_dict, classes = text_prompt(label_text)
    text_dict = []
    classes = []
    #####################
    # exit(0)
    if args.evaluate:
        validate(val_loader, model, criterion, KLcriterion, 0, matcher, tokenizer, text_dict, 
                                classes, class_to_idx = dataset_test.classes_dict)
        return

    log_training = open(os.path.join(args.root_model, args.root_results, args.store_name, 'log'), 'a+')
    with open(os.path.join(args.root_model, args.root_results, args.store_name, 'log'), 'a+') as f:
        f.write(str(args)+'\n')
        f.write(str(model))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_model, args.root_results, args.store_name)) 
    #################################TRAIN################################
    for epoch in range(args.start_epoch, args.epochs):
        # if 'ZSL' in args.split_mode:
        #     scheduler.step()
        # else:
        #     adjust_learning_rate1(optimizer, epoch, args.lr_steps)
        if args.optimizer_choice == 'SGD':
            adjust_learning_rate1(optimizer, epoch, args.lr_steps)
        else:
            pass       
        # train for one epoch
        train(train_loader, model, criterion, KLcriterion, optimizer, epoch, log_training, tf_writer, matcher, tokenizer, text_dict, classes)
        # exit(0)
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            if args.two_stream:
                video_prec1, fusion_prec1 = validate(val_loader, model, criterion, KLcriterion, epoch, matcher, tokenizer, text_dict,
                                                    classes, log_training, tf_writer, class_to_idx = None)
                # remember best prec@1 and save checkpoint
                is_best = video_prec1 > best_video_prec1
                best_video_prec1 = max(video_prec1, best_video_prec1)
                tf_writer.add_scalar('acc/test_v_top1_best', best_video_prec1, epoch)

                output_best1 = 'Best Video Prec@1: %.3f' % (best_video_prec1)
                ######################################################################
                best_fusion_prec1 = max(fusion_prec1, best_fusion_prec1)
                tf_writer.add_scalar('acc/test_f_top1_best', best_fusion_prec1, epoch)

                output_best2 = 'Best Fusion Prec@1: %.3f\n' % (best_fusion_prec1)
                ######################################################################
                print(output_best1, output_best2)
            else:
                video_prec1 = validate(val_loader, model, criterion, KLcriterion, epoch, matcher, tokenizer, text_dict, classes, log_training, tf_writer)
                is_best = video_prec1 > best_video_prec1
                best_video_prec1 = max(video_prec1, best_video_prec1)
                tf_writer.add_scalar('acc/test_v_top1_best', best_video_prec1, epoch)

                output_best1 = 'Best Video Prec@1: %.3f\n' % (best_video_prec1)
                print(output_best1)
            
            if log_training is not None:
                log_training.write(output_best1 + '\n')
                log_training.flush()
                
            best_prec1 = output_best1

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
            
        if args.optimizer_choice == 'AdamW':
            scheduler.step()
        else:
            pass

def save_checkpoint(state, is_best):
    filename = '%s/%s/%s/ckpt.pth.tar' % (args.root_model, args.root_results, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

def adjust_learning_rate1(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    visual_lr = args.visual_lr * decay
    trans_lr = args.trans_lr * decay
    if args.arch=='region_coord_tsm':
        optimizer.param_groups[0]['lr'] = visual_lr
    elif args.two_stream or args.joint:
        optimizer.param_groups[0]['lr'] = visual_lr
        optimizer.param_groups[-1]['lr'] = trans_lr
    else:
        optimizer.param_groups[0]['lr'] = visual_lr        
    # print('lr:\t', lr)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

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

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [os.path.join(args.root_model, args.root_results),
                    os.path.join(args.root_model, args.root_results, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
