# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
setting='ZSL' # COM, COM_shuffle, FEWSHOT, LONGTAIL_COM, ZSL, GCZSL
# mode='finetune_10shot' # ['base' 'finetune_5shot', 'finetune_10shot']

if setting=='LONGTAIL_COM':
    # only for LONGTAIL_COM
    LONGTAIL_mode = 'STHELSE/top_obj_cls' # ['most_obj_cls' 'num_obj_cls', 'top3_obj_cls'] 
if setting=='ZSL':
    ZSL_mode = 'random' # 'class_split/new_ours', 'class_split/ours', 'class_split/random', 'class_split/hard'



# label_pth = '%s/labels.json'%(setting)
label_pth = 'tools/anno_STHELSE/%s/labels.json'%(setting)

# label_pth = 'COM/labels.json'


if setting=='FEWSHOT':
    if mode=='base':
        label_pth = 'FEWSHOT/base_labels.json'
    else:
        label_pth = 'FEWSHOT/finetune_labels.json'



if __name__ == '__main__':
    # dataset_name = 'STHELSE'  # 'jester-v1'
    with open(label_pth) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        # print(i, idx)
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category3.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i
    # files_input = ['%s-validation.json' % dataset_name, '%s-train.json' % dataset_name, '%s-test.json' % dataset_name]
    # files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    ###
    if setting=='GCZSL':
        files_input = ['%s/train.json'%(setting), '%s/val.json'%(setting)]
        files_output = ['%s/train_videofolder.txt'%(setting), '%s/val_videofolder.txt'%(setting)]
    elif setting=='ZSL':
        files_input = ['tools/anno_STHELSE/%s/%s/train.json'%(setting, ZSL_mode), 'tools/anno_STHELSE/%s/%s/validation.json'%(setting, ZSL_mode)]
        files_output = ['tools/anno_STHELSE/%s/%s/train_videofolder_plus.txt'%(setting, ZSL_mode), 'tools/anno_STHELSE/%s/%s/val_videofolder_plus.txt'%(setting, ZSL_mode)]
    elif setting=='COM' or setting=='COM_shuffle':
        files_input = ['%s/train.json'%(setting), '%s/val.json'%(setting)]
        files_output = ['%s/train_videofolder.txt'%(setting), '%s/val_videofolder.txt'%(setting)]
    elif setting=='LONGTAIL_COM':
        files_input = ['%s/%s/train.json'%(setting, LONGTAIL_mode), '%s/%s/val.json'%(setting, LONGTAIL_mode)]
        files_output = ['%s/%s/train_videofolder.txt'%(setting, LONGTAIL_mode), '%s/%s/val_videofolder.txt'%(setting, LONGTAIL_mode)]
    elif setting=='FEWSHOT':
        files_input = ['%s/%s_train.json'%(setting, mode), '%s/%s_val.json'%(setting, mode)]
        files_output = ['%s/%s_train_videofolder.txt'%(setting, mode), '%s/%s_val_videofolder.txt'%(setting, mode)]
    ###
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        captions = []
        for item in data:
            folders.append(item['id'])
            captions.append(item['label'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            curCaptions = captions[i]

            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('dataset/STHELSE/frames', curFolder))
            output.append('%s %d %d %s' % (curFolder, len(dir_files), curIDX, curCaptions))
            print(output)
            # exit(0)
            print('%d/%d' % (i, len(folders)))
        # exit(0)
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
