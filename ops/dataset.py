# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data
from torchvision import transforms
import decord
import time
import random

from PIL import Image
import torch
import torchvision
import os
import av
import cv2
import copy
import numpy as np
from numpy.random import randint
from ops.box_load import box_deal
from opts import parser
import ops.utils as utils
from ops.data_parser import WebmDataset
from video_records import EpicKitchens55_VideoRecord

args, _ = parser.parse_known_args()

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])



class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, json_input, json_labels, 
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', random_crop=None, transform=None,
                 random_shift=True, test_mode=False, anno=None, captions = None, label_captions = None, 
                 label_text = None, num_boxes=10, remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path #项目目录的根目录地址
        self.list_file = list_file #训练或者测试的列表文件地址
        self.num_segments = num_segments #视频分割的段数
        self.new_length = new_length #根据输入数据类型的不同，new_length取不同的值。输入RGB时为1，输入光流时为5
        self.modality = modality #输入数据类型（RGB，光流，RGB差）
        self.image_tmpl = image_tmpl #加载数据集时的格式
        self.random_crop = random_crop
        self.transform = transform #对数据进行预处理，这里默认为None
        self.random_shift = random_shift #布尔型，当设置为True时对训练集进行采样，设置为False时对验证集进行采样
        self.test_mode = test_mode # 布尔型，默认为False，当设置为True时即对测试集进行采样
        self.remove_missing = remove_missing #布尔型，默认为False。与test_model在同一个判断条件下，对数据进行读取。
        self.dense_sample = dense_sample  # using dense sample as I3D 布尔型，设置为True时进行密集采样
        self.twice_sample = twice_sample  # twice sample for more validation 布尔型，设置为True时进行二次采样。
        self.box_annotations = anno
        self.captions_input = captions
        self.label_captions = label_captions
        self.text_label = label_text
        self.num_boxes = num_boxes
        # self.dataset_object = WebmDataset(json_input, json_labels, root_path, is_test=test_mode)
        # self.classes_dict = self.dataset_object.classes_dict
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
    # def _load_image(self, record, idx):
    
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            if 'epic55' in args.dataset:
                idx_untrimmed = record.start_frame + idx
                try:
                    return [Image.open(os.path.join(self.root_path, record.untrimmed_video_name, self.image_tmpl.format(idx_untrimmed))).convert('RGB')]
                except Exception:
                    print('error loading image:', os.path.join(self.root_path, record.untrimmed_video_name, self.image_tmpl.format(idx_untrimmed)))
                    return [Image.open(os.path.join(self.root_path, record.untrimmed_video_name, self.image_tmpl.format(record.start_frame))).convert('RGB')]
            else:
                try:
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
                except Exception:
                    print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self): # _parse_list函数功能在于读取list文件，储存在video_list中
        if 'epic55' in args.dataset:
            self.video_list = [EpicKitchens55_VideoRecord(tup) for tup in self.list_file.iterrows()]
            print('video number:%d' % (len(self.video_list)))
        else:
            # check the frame number is large >3:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            # print(type(tmp))
            # exit(0)
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
            # print(type(self.video_list))
            if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                for v in self.video_list:
                    v._data[1] = int(v._data[1]) / 2
            print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record): # _sample_indices函数功能在于实现TSN的密集采样或者稀疏采样，返回的是采样的帧数列表
        """

        :param record: VideoRecord
        :return: list
        """
        # print(type(record.num_frames), record.num_frames)
        if self.dense_sample:  # i3d dense sample 密集随即采样
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample 稀疏随机采样
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record): #对验证集进行采样
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record): #对测试集进行采样
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index): #__getitem__函数会在TSNDataSet初始化之后执行，功能在于调用执行采样函数，并且调用get方法，得到TSNDataSet的返回
        record = self.video_list[index] #record变量读取的是video_list的第index个数据，包含该视频所在的文件地址、视频包含的帧数和视频所属的分类。
        # print('#######test')
        # check this is a legit video folder
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        ###########################################################compute box################################################################
        # print(record)
        # exit(0)
        input, target, crop_w, crop_h, offset_w, offset_h, org_size = self.get(record, segment_indices)
        ######################## save input ###################
        # print('### input:\t', input.size())
        # torch.save(input, 'input_test8.pt')
        # exit(0)
        #####################################################################################
        if 'epic55' in args.dataset and args.joint:
            tags=record.tags
            # print('tags:\t', tags)
            if len(tags)==1:
                captions_data = tags[0]
                replace_tags = 'something'
            elif len(tags)==2:
                captions_data = tags[0]+','+tags[1]
                replace_tags = 'something'+','+'something'
            else:
                captions_data = tags[0]+','+tags[1]+','+tags[2]
                replace_tags = 'something'+','+'something'+','+'something'
            # print('captions_data:\t', captions_data)
            
            verbs=record.verb
            label_captions=verbs+' '+replace_tags
            # print('label_captions:\t', label_captions)

        else:
            captions_data = self.captions_input[record.path] if self.captions_input else None 
            # captions_data =[]     
        #####################################################################################
            label_captions = self.label_captions[record.path] if self.label_captions else None 
            # label_captions =[]     
        #####################################################################################
        # if args.do_aug:
        video_data = self.box_annotations[record.path] if self.box_annotations else None
        # print(video_data)
        # exit(0)
        # else:
            # video_data = None
        object_set = set()
        for frame_id in segment_indices:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))
        ######################################################################################       
        # print('org_width, org_heightP:\t', org_width, org_height)

        # scale_resize_w, scale_resize_h = 224 / float(org_width), 224 / float(org_height)
        if self.random_shift:
            scale_resize_w, scale_resize_h = 1, 1
        else:
            org_width, org_height = org_size
            # scale_resize_w, scale_resize_h = 256 / float(org_width), 256 / float(org_height)
            scale_resize_w, scale_resize_h = 1, 1


        scale_crop_w, scale_crop_h = 224 / float(crop_w), 224 / float(crop_h) 
        if 'epic55' in args.dataset:
            box_tensors = self.epic_box_deal(self.num_segments, 
                                        self.num_boxes, 
                                        segment_indices, 
                                        record,
                                        scale_resize_w, 
                                        scale_resize_h,
                                        scale_crop_w,
                                        scale_crop_h,
                                        offset_w,
                                        offset_h, 
                                        crop_w, 
                                        crop_h,
                                        org_size) if self.box_annotations else []
            box_categories = []
        else:
            box_tensors, box_categories = box_deal(self.num_segments, 
                                                        self.num_boxes, 
                                                        segment_indices, 
                                                        video_data, 
                                                        object_set,
                                                        scale_resize_w, 
                                                        scale_resize_h,
                                                        scale_crop_w,
                                                        scale_crop_h,
                                                        offset_w,
                                                        offset_h, 
                                                        crop_w, 
                                                        crop_h
                                                    )
        
        # box_tensors = []
        
        # ######################## save box #####################
        # print('box_tensors:\t', box_tensors.size())
        # print(box_tensors)
        # torch.save(box_tensors, 'box_test8.pt')
        #####################################################
        ##########################################################################################################################################
        # print('input:\t', input.size())
        # print('target:\t', target)
        # print('box_tensors:\t', box_tensors)
        # print('box_categories:\t', box_categories)
        # print('captions_data:\t', captions_data)
        # print('label_captions:\t', label_captions)

        # exit(0)
        return input, target, box_tensors, box_categories, captions_data, label_captions
    

    def read_frames_av_fast(self, video_path, cur_vlen, frame_idxs=None):
    # fast version of loading video
        # print('frame_idxs:\t', frame_idxs, cur_vlen)
        assert cur_vlen>0
        reader = av.open(video_path)
        video_stream = reader.streams.video[0]
        
        ## compute the rate of org_fps/cur_fps
        org_vlen = self.get_video_len(video_path)
        print('org_vlen:', org_vlen, 'cur_vlen\t:', cur_vlen)
        print('frame_idxs_before:', frame_idxs)
        if org_vlen!=cur_vlen:
            frame_idxs = np.floor(frame_idxs * org_vlen / cur_vlen)
        print('frame_idxs_after:\t', frame_idxs)
        frames = []
        try:
            for index in frame_idxs:
                index = int(index)
                # print('index:\t', type(index))
                reader.seek(index-1, any_frame=False, stream=video_stream) # offset=index or index-1?
                frame = (next(reader.decode(video=0))).to_image()
                frame_size = frame.size
                frames.append(frame)
        except:
            frames.append(Image.new('RGB', frame_size, (0, 0, 0)))
        # except (RuntimeError, ZeroDivisionError) as exception:
            import traceback
            traceback.print_exc()
            # print('{}: WEBM reader cannot open {}. Empty '
            #     'list returned.'.format(type(exception).__name__, video_path))
        reader.close()
        return frames

    def get_video_len(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not (cap.isOpened()):
            return False
        vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return vlen
    
    def read_frames_cv2(self, video_path, cur_vlen, frame_idxs=None):
        # print('frame_idxs:\t', frame_idxs, cur_vlen)
        assert cur_vlen>0
        cap = cv2.VideoCapture(video_path) 
        assert (cap.isOpened())
        org_vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print('org_vlen:', org_vlen, 'cur_vlen\t:', cur_vlen)
        # print('frame_idxs_before:', frame_idxs)
        if org_vlen!=cur_vlen:
            frame_idxs = np.floor(frame_idxs * org_vlen / cur_vlen)
        # print('frame_idxs_after:\t', frame_idxs)
        frames = []
        for index in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            ret, frame = cap.read()
            if ret:
                # print('frame_type:\t', type(frame))
                # frame_size = (427, 240)
                h, w, c = frame.shape
                # print(h, w, c)
                frame_size = (h, w)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                # frame = torch.from_numpy(frame)
                # frame = torchvision.transforms.ToPILImage()(frame)                                              
                frames.append(frame)
            else:
                frames.append(Image.new('RGB', frame_size, (0, 0, 0)))
                # pass
                # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
        cap.release()
        return frames

    decord.bridge.set_bridge("torch")   
    def read_frames_decord(self, video_path, cur_vlen, frame_idxs=None):
        assert cur_vlen>0
        video_reader = decord.VideoReader(video_path, num_threads=1)
        org_vlen = len(video_reader)
        # print('cur_vlen:', cur_vlen, 'org_vlen:\t', org_vlen)
        # print('cur_frame_idxs:\t', frame_idxs)
        if org_vlen!=cur_vlen:
            frame_idxs = np.floor(frame_idxs * org_vlen / cur_vlen) - 1 
        frame_idxs = np.where(frame_idxs<0, 0, frame_idxs)
        frame_idxs = np.where(frame_idxs>(org_vlen-1), org_vlen-1, frame_idxs)
        # print('org_frame_idxs:\t', frame_idxs)
        # frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = video_reader.get_batch(frame_idxs)
        # print('frames1:\t', frames.size())
        
        frames = frames.permute(0, 3, 1, 2)
        # print('frames2:\t', frames.size())
        #############################################################################
        # frame_list = []
        # # start = time.time()
        # i = 0
        # for item in frames:           
        #     # print(i.size())
        #     i += 1
        #     frame = transforms.ToPILImage()(item).convert("RGB")
        #     frame.save("debug/frame_%d.jpg" % (i))
            # frame_list.append(frame)
        # exit(0)   
        # return frame_list
        return frames
    
    def get(self, record, indices): #get方法的功能在于读取提取的帧图片，并且对帧图片进行变形操作（角裁减、中心提取等）
        #对提取到的帧序号列表进行遍历，找到每一个帧对应的图片，添加到images列表中。之后对提到的images进行数据集变形，返回变形后的数据集和对应的类型标签
        # print('######## test')
        if args.read_from_video:
            video_path = '%s/%s.webm' % (args.video_source_root, record.path)
            # print(video_path)
            # video_path = '%s/99178.webm' % (args.video_source_root)
            # print(video_path)
            # exit(0)
            num_frames = record.num_frames
            # images = self.read_frames_av_fast(video_path, num_frames, indices)
            # images = self.read_frames_cv2(video_path, num_frames, indices)
            images = self.read_frames_decord(video_path, num_frames, indices)
            # print('images:\t', images.size())
            
        # exit(0)
        # # ##################################################
        else:
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    # seg_imgs = self._load_image(record, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1
        # print('images1:\t', type(images), images)
        # ow, oh = images[0].size
        # # print(ow, oh)
        # ###########################################################
        # if args.arch == 'resnet50':
        #     boxes = []
        # else:
        #     boxes = self.get_epic_box_coord(record, indices, ow, oh, self.num_segments, self.num_boxes)
        
        # categorys=[]
        # print(boxes)
        # print(boxes.size())
        # exit(0)
        ## need to get boxes (<T, num_boxes, 4>)
        boxes, categorys = self.get_box_coord(record, indices, self.num_segments, self.num_boxes)
        # print('boxes:\t', boxes.size(), boxes)
        # print('categorys:\t', categorys.size(), categorys)      
        #########################################
        if args.do_aug:
            # frame_list=[]
            # for i in images:           
            #     # print(i.size())
            #     frame = transforms.ToTensor()(i)
            #     frame_list.append(frame)
            # images = frame_list
            if not self.random_shift:
                if_test = True
                images_org = list(images)
                # print('images_org:\t', len(images_org))
                if args.do_aug_mix:
                    images_mask0, images_mask1, images_mask2, images_mask3 = self.mask_aug(images, boxes, categorys, if_test)
                    images = images_org + images_mask0 + images_mask1 + images_mask2 + images_mask3
                else:
                    images_mask = self.mask_aug(images, boxes, categorys, if_test)
                    images = images_org + images_mask
                # print('images_mask:\t', len(images_mask))
                # images = images_org + images_mask0 + images_mask1 + images_mask2 + images_mask3
                # print('images:\t', len(images))
                # exit(0)
            else:
                if_test = False
                if self.decision(args.mask_ratio):
                    images = self.mask_aug(images, boxes, categorys, if_test)
                # print('images:\t', len(images))
        # exit(0)
        # if args.do_aug:
        #     frame_list = []
        #     for i in images:           
        #         # print(type(i))
        #         frame = transforms.ToPILImage()(i).convert("RGB")
        #         frame_list.append(frame)
        #     images = frame_list
        # for i in images:           
        #     # print(i.size())
        #     frame = transforms.ToPILImage()(i).convert("RGB")
        #     frame_list.append(frame)
        # images = frame_list
        # # images = self.mask_aug(images, boxes, categorys)
        # # print(type(images))
        # # exit(0)
        # # print('images:\t', len(images))
        # # exit(0)
        # ################# save original img ###############
        # # print('images_length:\t', len(images))
        # # print('images_size:', images[0].size)
        # # torch.save(images, 'test_images_8.pt')
        # #################################################

        # # trans_GroupMultiScaleCrop = self.transform.transforms[0]
        # # trans_others = torchvision.transforms.Compose(self.transform.transforms[1:])
        # # process_data, crop_w, crop_h, offset_w, offset_h = trans_GroupMultiScaleCrop(images)
        # # process_data = trans_others(process_data)
        

        org_size = images[0].size
        # exit(0)

        if self.random_shift:
            process_data, crop_w, crop_h, offset_w, offset_h = self.random_crop(images)
        else:
            process_data = self.random_crop(images)
            crop_w = crop_h = self.random_crop.transforms[0].size
            # crop_w, crop_h = 224, 224
            org_width, org_height = org_size
            offset_w, offset_h = (org_width-crop_w)//2, (org_height-crop_h)//2
        #     crop_w, crop_h = 256, 256
        #     offset_w, offset_h = (256-crop_w)//2, (256-crop_h)//2
        process_data = self.transform(process_data)

        # print('process_data:\t', process_data.size())
        # exit(0)
        
        
        return process_data, record.label, crop_w, crop_h, offset_w, offset_h, org_size

    def epic_box_deal(self, coord_nr_frames, num_boxes, coord_frame_list, record, 
                            scale_resize_w, scale_resize_h, scale_crop_w, scale_crop_h,
                            offset_w, offset_h, crop_w, crop_h, org_size):
        org_w, org_h = org_size
        idx = record.start_frame + coord_frame_list
        video_data=self.box_annotations[record.untrimmed_video_name] if self.box_annotations else None

        box_tensors = torch.zeros((coord_nr_frames, num_boxes, 4), dtype=torch.float32)
        for frame_index, frame_id in enumerate(idx):
            frame_data = video_data[frame_id]
            obj_coords = frame_data["obj_coords"]
            if obj_coords==[]:
                pass
            else:
                for item in range(len(obj_coords)):
                    # print('item:\t', item)
                    # print(obj_coords[item])
                    left = obj_coords[item][0]*org_w
                    top = obj_coords[item][1]*org_h
                    right = obj_coords[item][2]*org_w
                    bottom = obj_coords[item][3]*org_h
                    
                    left, right = left * scale_resize_w, right * scale_resize_w
                    top, bottom = top * scale_resize_h, bottom * scale_resize_h
                    
                    left, right = left - offset_w, right - offset_w
                    top, bottom = top - offset_h, bottom - offset_h

                    left, right = np.clip([left, right], a_min=0, a_max=crop_w-1)
                    top, bottom = np.clip([top, bottom], a_min=0, a_max=crop_h-1)

                    left, right = left * scale_crop_w, right * scale_crop_w
                    top, bottom = top * scale_crop_h, bottom * scale_crop_h

                    left, right = np.clip([left, right], a_min=0, a_max=223)
                    top, bottom = np.clip([top, bottom], a_min=0, a_max=223)

                    # print('left:\t', type(left), left)
                    gt_box = np.array([(left + right) / 2., (top + bottom) / 2., right - left, bottom - top], dtype=np.float32)
                    gt_box /= 224
                    try:
                        box_tensors[frame_index, item]=torch.tensor(gt_box).float()
                    except:
                        pass
            hands_coords = frame_data["hands_coords"]
            if hands_coords==[]:
                pass
            else:
                for jtem in range(len(hands_coords)):
                    left = hands_coords[jtem][0]*org_w
                    top = hands_coords[jtem][1]*org_h
                    right = hands_coords[jtem][2]*org_w
                    bottom = hands_coords[jtem][3]*org_h
                    
                    left, right = left * scale_resize_w, right * scale_resize_w
                    top, bottom = top * scale_resize_h, bottom * scale_resize_h
                    
                    left, right = left - offset_w, right - offset_w
                    top, bottom = top - offset_h, bottom - offset_h

                    left, right = np.clip([left, right], a_min=0, a_max=crop_w-1)
                    top, bottom = np.clip([top, bottom], a_min=0, a_max=crop_h-1)

                    left, right = left * scale_crop_w, right * scale_crop_w
                    top, bottom = top * scale_crop_h, bottom * scale_crop_h

                    left, right = np.clip([left, right], a_min=0, a_max=223)
                    top, bottom = np.clip([top, bottom], a_min=0, a_max=223)

                    # print('left:\t', type(left), left)
                    gt_box = np.array([(left + right) / 2., (top + bottom) / 2., right - left, bottom - top], dtype=np.float32)
                    gt_box /= 224
                    try:
                        box_tensors[frame_index, jtem+4]=torch.tensor(gt_box).float()
                    except:
                        pass
        return  box_tensors  

    def get_epic_box_coord(self, record, indices, org_w, org_h, coord_nr_frames, num_boxes):

        # print('indices:\t', indices)
        # print('video_name:\t', record.untrimmed_video_name)
        idx = record.start_frame + indices
        # print('idx:\t', idx)
        video_data=self.box_annotations[record.untrimmed_video_name] if self.box_annotations else []
        # print(len(video_data))
        box_tensors = torch.zeros((coord_nr_frames, num_boxes, 4), dtype=torch.float32)

        for frame_index, frame_id in enumerate(idx):
            # print('frame_index:\t', frame_index)
            # print('frame_id:\t', frame_id)
            frame_data = video_data[frame_id]
            # print('frame_data:\t', frame_data)
            obj_coords = frame_data["obj_coords"]
            # print(box_tensors.size())
            # print(box_tensors[0,0].size())
            if obj_coords==[]:
                pass
            else:
                for item in range(len(obj_coords)):
                    # print('item:\t', item)
                    # print(obj_coords[item])
                    left = obj_coords[item][0]
                    top = obj_coords[item][1]
                    right = obj_coords[item][2]
                    bottom = obj_coords[item][3]
                    # print('left:\t', type(left), left)
                    gt_box = np.array([(left + right) * org_w / 2., (top + bottom) * org_h / 2., (right - left) * org_w, (bottom - top) * org_h], dtype=np.float32)
                    try:
                        # print('test####')
                        box_tensors[frame_index, item]=torch.tensor(gt_box).float()
                        # print('test####')
                    except:
                        pass 
            hands_coords = frame_data["hands_coords"]
            if hands_coords==[]:
                pass
            else:
                 for jtem in range(len(hands_coords)):
                    left = hands_coords[jtem][0]
                    top = hands_coords[jtem][1]
                    right = hands_coords[jtem][2]
                    bottom = hands_coords[jtem][3]
                    gt_box = np.array([(left + right) * org_w / 2., (top + bottom) * org_h / 2., (right - left) * org_w, (bottom - top) * org_h], dtype=np.float32)
                    try:
                        box_tensors[frame_index, jtem+4]=torch.tensor(gt_box).float()
                    except:
                        pass          
            # break
        # print(box_tensors)
        # exit(0)


        return box_tensors
    
    def get_box_coord(self, record, indices, coord_nr_frames, num_boxes):
        # print(record.path)
        # exit(0)
        video_data = self.box_annotations[record.path] if self.box_annotations else None
        # print('video_data:\t', video_data)
        object_set = set()
        for frame_id in indices:
            try:
                frame_data = video_data[frame_id]
            except:
                frame_data = {'labels': []}
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                object_set.add(standard_category)
        object_set = sorted(list(object_set))
        # print(object_set)
        box_tensors = torch.zeros((coord_nr_frames, num_boxes, 4), dtype=torch.float32)  # (cx, cy, w, h)
        box_categories = torch.zeros((coord_nr_frames, num_boxes))
        
        for frame_index, frame_id in enumerate(indices):
            # print('frame_id:\t', frame_id)
            try:
                frame_data = video_data[frame_id]
                # print('frame_data:\t', frame_data)
            except:
                frame_data = {'labels': []}
            # exit(0)
            for box_data in frame_data['labels']:
                standard_category = box_data['standard_category']
                global_box_id = object_set.index(standard_category)

                box_coord = box_data['box2d']
                x0, y0, x1, y1 = box_coord['x1'], box_coord['y1'], box_coord['x2'], box_coord['y2']
                gt_box = np.array([(x0 + x1) / 2., (y0 + y1) / 2., x1 - x0, y1 - y0], dtype=np.float32)
                # print(gt_box)
                # exit(0)
                try:
                    box_tensors[frame_index, global_box_id] = torch.tensor(gt_box).float()
                except:
                    pass
                # load box category
                try:
                    box_categories[frame_index, global_box_id] = 1 if box_data['standard_category'] in ['hand', 'person'] else 2 # 0 is for none
                except:
                    pass

                # load image into tensor
                x0, y0, x1, y1 = list(map(int, [x0, y0, x1, y1]))
        return box_tensors, box_categories
    
    def decision(self, probability):
        return random.random() < probability
        
    def mask_aug(self, images, boxes, category, if_test):
        """
        Given a list of images from one video and the associate boxes.
        images[T, C, H, W] boxes[T, num_boxes, 4]
        """
        
        # print('boxes:\t', boxes.size(), boxes)
        # print('category:\t', category.size(), category)
        # frame_list=[]
        # for i in images:           
        #     # print(i.size())
        #     frame = transforms.ToTensor()(i)
        #     frame_list.append(frame)
        # images = frame_list
        # print('images:\t', images[0].size())
        # exit(0)
        if if_test and args.do_aug_mix:
            aug_images0 = []
            aug_images1 = []
            aug_images2 = []
            aug_images3 = []
        else:
            aug_images = []
        
        num_frames, num_boxes, _ = boxes.size()
        # b1, b2, b3, b4
        # find a box randomly which is not a hand
        # category # [0,2,1]
        # mask_box_id = randint(num_boxes)
        # print('mask_box_id:\t', mask_box_id)       
        # while category[mask_box_id]==2: # is not hand
        #     mask_box_id = randint(num_boxes)
        for item in torch.arange(num_frames):
            # print(item)
            #######
            if 'epic55' in args.dataset:
                for i in range(4):
                    mask_box_id = 5
                    if boxes[item][i].sum() != 0:
                        mask_box_id = i
                        break
                    else:
                        continue
            else:
                try:
                    mask_box_id = list(category[item]).index(2)           
                except:
                    mask_box_id = 5
            # print('mask_box_id:\t', mask_box_id)
            if mask_box_id != 5:
                selected_box = boxes[item, mask_box_id, :]
                # print('selected_box:\t', selected_box)
                if if_test and args.do_aug_mix:
                    aug_img0, aug_img1, aug_img2, aug_img3 = self.do_mask(images[item], selected_box, if_test)                              
                else:
                    aug_img = self.do_mask(images[item], selected_box, if_test)                              
            else:
                if if_test and args.do_aug_mix:
                    aug_img0 = images[item]
                    aug_img1 = images[item]
                    aug_img2 = images[item]
                    aug_img3 = images[item]
                else:    
                    aug_img = images[item]
                # aug_img = transforms.ToPILImage()(aug_img).convert("RGB")
            # print('aug_img:\t', aug_img)  
            if if_test and args.do_aug_mix:
                aug_images0.append(aug_img0)
                aug_images1.append(aug_img1)
                aug_images2.append(aug_img2)
                aug_images3.append(aug_img3)
            else:
                aug_images.append(aug_img)
            # print('aug_images:\t', len(aug_images))   
        # exit(0)  
        # selected box is not overlaping with hand  IoU 
        if if_test and args.do_aug_mix:
            return aug_images0, aug_images1, aug_images2, aug_images3
        else:
            return aug_images


    def do_mask(self, img, box, if_test):
        # mask region by dark or noise or hazy ...
        mask_value = 0
        x, y, w, h = box
        x, y, w, h  = list(map(int, [x, y, w, h]))
        if not if_test:
            d_value = random.randint(0,5)
            # if d_value == 0:
            # # ## total box mask
            # # if args.mask_mode == 'total_dark':
            #     img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = mask_value
            # ## partial box mask
            if d_value == 0 or d_value == 1:
            # elif args.mask_mode == 'partial_dark':
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = mask_value
            elif d_value == 2:
            # elif args.mask_mode == 'blur':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]
                # print('mask_erea:\t', mask_erea.size())
                transf1 = transforms.GaussianBlur(11, 5) # kernel_size=15, sigma=(5.0, 15.0)
                # transf1 = transforms.GaussianBlur(kernel_size=(101, 100), sigma=(0.1, 5))               
                # img = transf1(img)
                try:
                    img_mask_erea = transf1(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # print('img_mask_erea:\t', img_mask_erea.size())
                # mask_erea = transf1(img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)])
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea
            elif d_value == 3 or d_value == 4:
            # elif args.mask_mode == 'light':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] 
                transf = transforms.ColorJitter(brightness=(2,2), contrast=(2,2), saturation=(2,2))
                ##liangdu
                # transf1 = transforms.ColorJitter(brightness=(2,2))
                # ##duibidu
                # transf2 = transforms.ColorJitter(contrast=(2,2))
                # ##baohedu
                # transf3 = transforms.ColorJitter(saturation=(2,2))
                try:
                    img_mask_erea = transf(mask_erea)
                    # img_mask_erea = transf1(mask_erea)
                    # img_mask_erea = transf2(mask_erea)
                    # img_mask_erea = transf3(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea 
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea
            elif d_value == 5:   
            # elif args.mask_mode == 'color':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]              
                transf = transforms.ColorJitter(hue=0.5)
                try:
                    img_mask_erea = transf(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea
                
            return img
        else:
            if args.do_aug_mix:
            ## dark
                img0 = copy.deepcopy(img)
                img0[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = mask_value
                ## blur
                img1 = copy.deepcopy(img)
                mask_erea1 = img1[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]
                transf1 = transforms.GaussianBlur(11, 5) # kernel_size=15, sigma=(5.0, 15.0)
                try:
                    img_mask_erea1 = transf1(mask_erea1)
                except:
                    img_mask_erea1 = mask_erea1
                img1[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea1
                ## color
                img2 = copy.deepcopy(img)
                mask_erea2 = img2[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] 
                transf2 = transforms.ColorJitter(brightness=(2,2), contrast=(2,2), saturation=(2,2))
                try:
                    img_mask_erea2 = transf2(mask_erea2)
                except:
                    img_mask_erea2 = mask_erea2
                img2[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea2
                ## light
                img3 = copy.deepcopy(img)
                mask_erea3 = img3[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]              
                transf3 = transforms.ColorJitter(hue=0.5)
                try:
                    img_mask_erea3 = transf3(mask_erea3)
                except:
                    img_mask_erea3 = mask_erea3
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea
                img3[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea3 
                return img0, img1, img2, img3
            elif args.mask_mode == 'partial_dark':
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = mask_value
            elif args.mask_mode == 'blur':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]
                # print('mask_erea:\t', mask_erea.size())
                transf1 = transforms.GaussianBlur(11, 5) # kernel_size=15, sigma=(5.0, 15.0)
                # transf1 = transforms.GaussianBlur(kernel_size=(101, 100), sigma=(0.1, 5))               
                # img = transf1(img)
                try:
                    img_mask_erea = transf1(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # print('img_mask_erea:\t', img_mask_erea.size())
                # mask_erea = transf1(img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)])
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea
            elif args.mask_mode == 'light':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] 
                transf = transforms.ColorJitter(brightness=(2,2), contrast=(2,2), saturation=(2,2))
                ##liangdu
                # transf1 = transforms.ColorJitter(brightness=(2,2))
                # ##duibidu
                # transf2 = transforms.ColorJitter(contrast=(2,2))
                # ##baohedu
                # transf3 = transforms.ColorJitter(saturation=(2,2))
                try:
                    img_mask_erea = transf(mask_erea)
                    # img_mask_erea = transf1(mask_erea)
                    # img_mask_erea = transf2(mask_erea)
                    # img_mask_erea = transf3(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea 
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea  
            elif args.mask_mode == 'color':
                # mask_erea = img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)]
                mask_erea = img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)]              
                transf = transforms.ColorJitter(hue=0.5)
                try:
                    img_mask_erea = transf(mask_erea)
                except:
                    img_mask_erea = mask_erea
                # img[:, (y-h//2):(y+h//2), (x-w//2):(x+w//2)] = img_mask_erea
                img[:, (y-h//4):(y+h//4), (x-w//4):(x+w//4)] = img_mask_erea
                
            return img
                
            # print('img:\t', img.size())
        # img = transforms.ToPILImage()(img).convert("RGB")

        # mask_value = noise(w,h)
        # naive mask
        # <3, 240, 427>
        # img_org = img.copy()
        # img_org = img
        # img.save("debug/img.jpg")
        # exit(0)
         
        # print('img:\t', img.size(), img)
                
        # img.save("debug/img.jpg")
        
        
        # exit(0)
        # img[:, :, x:x+w, y:y+h] = mask_value

        # center-crop mask
        # img[:, :, x+(w/4):x+(3*w/4), y+(h/4):y+(3*h/4)] = mask_value

        # 

        # return img

    def __len__(self):
        return len(self.video_list)
    
# if __name__ == '__main__':
    
#     demo_video_path = '..\2.webm'
#     num_frames = 54
    
#     def read_frames_cv2(video_path, cur_vlen, frame_idxs=None):
#         assert cur_vlen>0
#         cap = cv2.VideoCapture(video_path)
#         assert (cap.isOpened())
#         org_vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         print(org_vlen)
#         if org_vlen!=cur_vlen:
#             frame_idxs = np.around(frame_idxs * org_vlen / cur_vlen)

#         frames = []
#         for index in frame_idxs:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
#             ret, frame = cap.read()
#             if ret:
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 frame = torch.from_numpy(frame)
#                 # (H x W x C) to (C x H x W)
#                 frame = frame.permute(2, 0, 1)
#                 frames.append(frame)
#             else:
#                 pass
#                 # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')
#         cap.release()
#         return frames

#     indices = None
    
    # demo_image = read_frames_cv2(demo_video_path, num_frames, indices)
    
