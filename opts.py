# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu


import argparse
from ops.utils import str2bool


parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', type=str, default = 'STHELSE')
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'], default = 'RGB')
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--basic_arch', type=str, default='resnet50')

parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr_warmup_step', default=5, type=int, metavar='N',
                    help='')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--trans_lr', default=5e-6, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--visual_lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--ratio', default=0.1, type=float, metavar='M',
                    help='ratio')
parser.add_argument('--mask_ratio', default=0, type=float, metavar='M',
                    help='mask_ratio')
parser.add_argument('--optimizer_choice', type=str, default='', choices=['SGD', 'AdamW', ''])
parser.add_argument('--loss_select', type=str, default='', choices=['KL', 'cost', ''])
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--delta', default=1, type=float, help='delta: weight of box_loss')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_results',type=str, default='STHELSE', choices=['STH_ZSL', 'STHELSE'])
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='ckpt')
parser.add_argument('--video_source_root', type=str, default='/home/10102007/TSM_NEW_CHECK3/dataset/video')
parser.add_argument('--download_root', type=str, default='pretrained_weights')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

######################################################################################################################
parser.add_argument('--json_data_train', type=str, help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, help='path to the json file with ground truth labels')
parser.add_argument('--tracked_boxes', type=str, help='choose tracked boxes')
# parser.add_argument('--texture_input_train', type=str, help='choose texture_input_train')
# parser.add_argument('--texture_input_val', type=str, help='choose texture_input_val')
parser.add_argument('--caption_full', type=str, help='choose caption')
parser.add_argument('--label_full', type=str, help='choose caption')

parser.add_argument('--num_boxes', default=4, type=int, help='num of boxes for each image')

parser.add_argument('--if_augment', default=False, action="store_true", help='use augment for models')
parser.add_argument('--not_preresize', default=False, action="store_true")
parser.add_argument('--not_custom_aug', default=False, action="store_true")

########################################################
parser.add_argument('--joint', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_KL', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_only_c2a', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_only_a2c', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_aug', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_aug_mix', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_tag', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_attention', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_frame', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_region', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_coord_cate', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_cate', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_prompt', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_SEattn', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--do_cross_attn', type=str2bool,
                        nargs='?', const=True, default=False)
parser.add_argument('--vis_info', type=str2bool,
                    nargs='?', const=True, default=True)
parser.add_argument('--hidden_feature_dim', type=int, default=512)
parser.add_argument('--coord_feature_dim', type=int, default=512)
parser.add_argument('--cls_hidden_dim', type=int, default=2048)
parser.add_argument('--attn_input_dim', type=int, default=512)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_class', type=int, default=174)
parser.add_argument('--embed_way', type=str, default = '', choices=['RoBERTa', 'DistilBERT', 'CLIPModel', 'DistilRoBERTa'])
parser.add_argument('--visio_mode', type=str, default = 'clip_based', choices=['clip_based', 'tsm_based', 'developed_based'])
parser.add_argument('--split_mode', type=str, default='', choices=['ZSL/hard', 'ZSL/random', 'COM'])
parser.add_argument('--visual_head_type', type=str, default='pool', choices=['pool', 'TRN'])
parser.add_argument('--frozen_blk', type=str, default='', choices=['visual', 'text', 'visual_text', ''])


parser.add_argument('--clip_vis_name', type=str, default = '', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B-32', 'ViT-B-16'])
parser.add_argument('--jit', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--weights_from_clip', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--read_from_video', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--mode',type=str, default='',choices=['temporal', 'no_temporal'])
parser.add_argument('--two_stream', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--text_select',type=str, default='',choices=['label_text', 'captions'])
parser.add_argument('--mask_mode',type=str, default='',choices=['total_dark', 'partial_dark', 'blur', 'light', 'color'])

parser.add_argument('--debug', type=str, default='no')

parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )
parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )
parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )
parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
parser.add_argument(
        "--position_embedding",
        default="learned",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
parser.add_argument(
        "--hidden_dim",
        default=512,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )

parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")

