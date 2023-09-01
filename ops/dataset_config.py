# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os
from opts import parser
from re import split
args, _ = parser.parse_known_args()

ROOT_DATASET = 'dataset/'  # '/data/jilin/'


def return_ucf101(modality):
    filename_categories = 'UCF101/labels/classInd.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_rgb_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'UCF101/jpg'
        filename_imglist_train = 'UCF101/file_list/ucf101_flow_train_split_1.txt'
        filename_imglist_val = 'UCF101/file_list/ucf101_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_hmdb51(modality):
    filename_categories = 51
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_rgb_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_rgb_val_split_1.txt'
        prefix = 'img_{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'HMDB51/images'
        filename_imglist_train = 'HMDB51/splits/hmdb51_flow_train_split_1.txt'
        filename_imglist_val = 'HMDB51/splits/hmdb51_flow_val_split_1.txt'
        prefix = 'flow_{}_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_something(modality):
    filename_categories = 'something/v1/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1'
        filename_imglist_train = 'something/v1/train_videofolder.txt'
        filename_imglist_val = 'something/v1/val_videofolder.txt'
        prefix = '{:05d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v1/20bn-something-something-v1-flow'
        filename_imglist_train = 'something/v1/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v1/val_videofolder_flow.txt'
        prefix = '{:06d}-{}_{:05d}.jpg'
    else:
        print('no such modality:'+modality)
        raise NotImplementedError
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_epic_55_noCOM(modality):
    # filename_categories = [125, 352]
    filename_categories = 125

    if modality == 'RGB':
        root_data = ROOT_DATASET + 'epic_55/image'
        filename_imglist_train = 'epic_55/annotations/EPIC_55_new_org_train2.pkl'
        filename_imglist_val = 'epic_55/annotations/EPIC_55_new_org_val2.pkl'
        prefix = 'frame_{:010d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_epic_55_random(modality):
    filename_categories = 121
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'epic_55/image'
        filename_imglist_train = 'epic_55/annotations/EPIC_55_new_random_train2.pkl'
        filename_imglist_val = 'epic_55/annotations/EPIC_55_new_random_val2.pkl'
        prefix = 'frame_{:010d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_epic_55_prac(modality):
    filename_categories = 121
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'epic_55/image'
        filename_imglist_train = 'epic_55/annotations/EPIC_55_new_prac_train3.pkl'
        filename_imglist_val = 'epic_55/annotations/EPIC_55_new_prac_val3.pkl'
        prefix = 'frame_{:010d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_somethingv2(modality):
    filename_categories = 'something/v2/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'something/v2/frames'
        filename_imglist_train = 'something/v2/train_videofolder.txt'
        filename_imglist_val = 'something/v2/val_videofolder.txt'
        prefix = '{:04d}.jpg'
    elif modality == 'Flow':
        root_data = ROOT_DATASET + 'something/v2/20bn-something-something-v2-flow'
        filename_imglist_train = 'something/v2/train_videofolder_flow.txt'
        filename_imglist_val = 'something/v2/val_videofolder_flow.txt'
        prefix = '{:06d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_STHELSE(modality):
    filename_categories = 'STHELSE/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/COM/train_videofolder.txt'
        filename_imglist_val = 'STHELSE/COM/val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_STHELSE_SHUFFLED(modality):
    filename_categories = 'STHELSE/SHUFFLED/category10.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/SHUFFLED/train_videofolder.txt'
        filename_imglist_val = 'STHELSE/SHUFFLED/val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_STHELSE_PRAC(modality):
    filename_categories = 'STHELSE/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/PRAC/train_videofolder.txt'
        filename_imglist_val = 'STHELSE/PRAC/val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
#####################################################################################
def return_STHELSE_ZSL(modality):
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/%s/train_videofolder.txt'%(split_mode)
        filename_imglist_val = 'STHELSE/%s/val_videofolder.txt'%(split_mode)
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    
    if 'ZSL' in args.split_mode:
        train_categories = 'STHELSE/%s/train_set.txt'%(split_mode)
        test_categories = 'STHELSE/%s/val_set.txt'%(split_mode)
        return train_categories, test_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
    else:
        filename_categories = 'STHELSE/category.txt'
        return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix       
#####################################################################################

def return_STHELSE_BASE(modality):
    filename_categories = 'STHELSE/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/FEWSHOT/base_train_videofolder.txt'
        filename_imglist_val = 'STHELSE/FEWSHOT/base_val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_STHELSE_5SHOT(modality):
    filename_categories = 'STHELSE/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/FEWSHOT/finetune_5shot_train_videofolder.txt'
        filename_imglist_val = 'STHELSE/FEWSHOT/finetune_5shot_val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_STHELSE_10SHOT(modality):
    filename_categories = 'STHELSE/category.txt'
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'STHELSE/frames'
        filename_imglist_train = 'STHELSE/FEWSHOT/finetune_5shot_train_videofolder.txt'
        filename_imglist_val = 'STHELSE/FEWSHOT/finetune_5shot_val_videofolder.txt'
        prefix = '{:04d}.jpg'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_jester(modality):
    filename_categories = 'jester/category.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = ROOT_DATASET + 'jester/20bn-jester-v1'
        filename_imglist_train = 'jester/train_videofolder.txt'
        filename_imglist_val = 'jester/val_videofolder.txt'
    else:
        raise NotImplementedError('no such modality:'+modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kinetics(modality):
    filename_categories = 400
    if modality == 'RGB':
        root_data = ROOT_DATASET + 'kinetics/images'
        filename_imglist_train = 'kinetics/labels/train_videofolder.txt'
        filename_imglist_val = 'kinetics/labels/val_videofolder.txt'
        prefix = 'img_{:05d}.jpg'
    else:
        raise NotImplementedError('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    ##########################################################
    global split_mode
    split_mode = args.split_mode
    ##########################################################

    dict_single = {'jester': return_jester, 'something': return_something, 'epic55_nocom': return_epic_55_noCOM, 
                   'epic55_random': return_epic_55_random, 'epic55_prac': return_epic_55_prac, 'somethingv2': return_somethingv2,
                   'ucf101': return_ucf101, 'hmdb51': return_hmdb51, 'STHELSE': return_STHELSE, 
                   'STHELSE_SHUFFLED': return_STHELSE_SHUFFLED, 'STHELSE_PRAC': return_STHELSE_PRAC, 'STHELSE_ZSL': return_STHELSE_ZSL, 'STHELSE_BASE': return_STHELSE_BASE, 
                   'STHELSE_5SHOT': return_STHELSE_5SHOT, 'STHELSE_10SHOT': return_STHELSE_10SHOT, 
                   'kinetics': return_kinetics }
    if dataset in dict_single and 'ZSL' not in args.split_mode:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    elif dataset in dict_single and 'ZSL' in args.split_mode:
        train_categories, test_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    if 'ZSL' not in args.split_mode:
        if isinstance(file_categories, str):
            file_categories = os.path.join(ROOT_DATASET, file_categories)
            with open(file_categories) as f:
                lines = f.readlines()
            categories = [item.rstrip() for item in lines]
            label_text = categories
            n_class = len(categories)
            # print(categories)
        else:  # number of categories
            # categories = [None] * file_categories
            categories = file_categories
            n_class = categories
            label_text = None
        print('{}: {} classes'.format(dataset, n_class))
        # return n_class, file_imglist_train, file_imglist_val, root_data, prefix
    else:
        train_categories = os.path.join(ROOT_DATASET, train_categories)
        with open(train_categories) as f:
            lines1 = f.readlines()
        categories_train = [item.rstrip() for item in lines1]
        n_class1 = len(categories_train)
        print('{}: {} classes'.format('train', n_class1))

        test_categories = os.path.join(ROOT_DATASET, test_categories)
        with open(test_categories) as d:
            lines2 = d.readlines()
        categories_test = [item.rstrip() for item in lines2]
        n_class2 = len(categories_test)
        print('{}: {} classes'.format('test', n_class2))
        # n_class = [n_class1, n_class2]
        n_class = n_class1+n_class2
        # print(type(n_class))
    # exit(0)
    return n_class, label_text, file_imglist_train, file_imglist_val, root_data, prefix
