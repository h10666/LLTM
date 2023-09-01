# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
setting='PRAC' # COM、FEWSHOT
mode='base' # ['base' 'finetune_5shot', 'finetune_10shot']
if setting=='FEWSHOT':
    if mode=='base':
        label_pth = 'FEWSHOT/base_labels.json'
    else:
        label_pth = 'FEWSHOT/finetune_labels.json'

# label_pth = '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/%s/labels.json'%(setting)
label_pth = '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/COM/labels.json'


if __name__ == '__main__':
    # dataset_name = 'STHELSE'  # 'jester-v1'
    with open(label_pth) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        # print(i, idx)
        # assert i == int(idx)  # make sure the rank is right
        categories.append(cat)
    print(len(categories))
    # exit(0)

    with open('category5.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i
    # files_input = ['%s-validation.json' % dataset_name, '%s-train.json' % dataset_name, '%s-test.json' % dataset_name]
    # files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    ###
    if setting=='COM':
        files_input = ['COM/train.json', 'COM/validation.json']
        files_output = ['COM/train_videofolder.txt', 'COM/val_videofolder.txt']
    elif setting=='FEWSHOT':
        files_input = ['FEWSHOT/%s_train.json'%(mode), 'FEWSHOT/%s_validation.json'%(mode)]
        files_output = ['FEWSHOT/%s_train_videofolder.txt'%(mode), 'FEWSHOT/%s_val_videofolder.txt'%(mode)]
    else:
        files_input = ['/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/PRAC/train.json', '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/PRAC/val.json']
        files_output = ['/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/PRAC/train_videofolder.txt', '/home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/PRAC/val_videofolder.txt']
    ###
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        print(len(folders))
        output = []
        for i in range(len(folders)):
            # print('i:\t', i)
            curFolder = folders[i]
            # print('curFolder:\t', curFolder)
            curIDX = idx_categories[i]
            # print('curIDX:\t', curIDX)
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join('/home/10102007/frames/frames', curFolder))
            output.append('%s %d %d' % (curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
