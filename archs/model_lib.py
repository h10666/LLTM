import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from archs.resnet3d_xl import Net
from archs.resnet_TSM import resnet_TSM
from archs.SEAttention import SEAttention
from archs.TCR import TCR
from archs.text2embed import text2embed
from torch.nn.init import normal_, constant_
from opts import parser
import argparse
from ops.utils import box_to_normalized, build_region_feas


args, _ = parser.parse_known_args()

class VideoRegionModel(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, 
                 ):
        super(VideoRegionModel, self).__init__()

        # self.extract_features = extract_features
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        
        self.i3D = Net(self.nr_actions, extract_features=True,
                       loss_type='softmax')
        # self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        # self.fc = nn.Linear(512, self.nr_actions)
        self.fc = nn.Linear(2048, self.nr_actions)

        self.crop_size1 = [3, 3]

        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # self.crit = nn.CrossEntropyLoss()

        # if opt.fine_tune:
        #     self.fine_tune(opt.fine_tune)

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'
    # def save_demo(input):
    #     path_to_save = '../'

    def forward(self, input, target, box_input, box_categories):
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        # print('input_size:\t', input.size())
        # print('box_input:\t', box_input)
        # exit(0)


        V, TC, _, _ = input.size() # [V, T*C, H, W] (4, 24, 224, 224)
        # demo_input = input.view((V, TC//3, 3) + input.size()[-2:])
        # demo_img = demo_input[0, 0] # (3, 224, 224)
        # print('demo_img:\t', demo_img.size())
        # torch.save(demo_img, 'demo_img.pt')

        # _, T, _, _ = box_input.size() 
        sample_len = 3

        i3d_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # [V, T, C, H, W] ([4, 8, 3, 224, 224])
        # print('i3d_input_size1:\t', i3d_input.size())

        i3d_input = i3d_input.permute(0,2,1,3,4) # [V, C, T, H, W] ([4, 3, 8, 224, 224])
        # print('i3d_input_size2:\t', i3d_input.size())
        # org_features - [V x 2048 x T / 2 x 14 x 14]

        _, org_feas = self.i3D(i3d_input) # [V x 2048 x T / 2 x 14 x 14] (4, 2048, 4, 14, 14)
        # print('org_feas_size:\t', org_feas.size())

        T = org_feas.size(2) # T = 4 
        # Reduce dimension video_features - [V x 512 x T / 2 x 14 x 14]

        conv_fea_maps = self.conv(org_feas) # [V x 256 x T / 2 x 14 x 14] ([4, 256, 4, 14, 14])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size()) # [4, 256, 4, 14, 14]

        global_vis_feas = self.avgpool(conv_fea_maps).view(V, -1) # [4, 1024]
        # print('global_vis_feas_size:\t', global_vis_feas.size())

        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # [V x T x d x 14 x 14] ([4, 4, 256, 14, 14])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())     

        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # [V * T x d x 14 x 14]  ([16, 256, 14, 14])
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())

        box_tensor = box_input[:, ::(args.num_segments//T), :, :].contiguous() # [V, T, P, 4] ([4, 4, 4, 4])

        # exit(0)
        # print('box_tensor_size1:\t', box_tensor.size())   

        box_tensor = box_tensor.view(-1, *box_tensor.size()[2:]) # [V*T, P, 4] # ([16, 4, 4]) 
        # print('box_tensor_size2:\t', box_tensor.size())

        ### convert tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = i3d_input.size()[-2:]
        ### get region feas via RoIAlign
        region_vis_feas = build_region_feas(conv_fea_maps, 
                                            boxes_list, self.crop_size1, img_size) #[V*T*P x C], where C= 3*3*d (64, 2304)
        # print('region_vis_feas_size1:\t', region_vis_feas.size())

        region_vis_feas = region_vis_feas.view(V, T, self.nr_boxes, region_vis_feas.size(-1)) #[V x T x P x C] ([4, 4, 4, 2304])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())

        ### reduce dim of region_vis_feas
        region_vis_feas = self.region_vis_embed(region_vis_feas) #[V x T x P x D_vis] ([4, 4, 4, 512])
        # print('region_vis_feas_size3:\t', region_vis_feas.size())

        return region_vis_feas


        ####


        # Get global features - [V x 512]
        # global_features = self.avgpool(videos_features).squeeze()
        # global_features = self.dropout(global_features)

        # cls_output = self.fc(global_features)



        # return cls_output

class VideoRegionModel_TSM(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionModel_TSM, self).__init__()

        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        # self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        # self.fusion_mode = 'late_fusion'  # 'early_fusion'
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv3d(1024, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        self.fc = nn.Linear(hidden_feature_dim, self.nr_actions)
        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        ###################################################################################################
        # self.dropout = nn.Dropout(0.5)  #changed 0.3 to 0.5
        # self.dropout = nn.Dropout(0.5)
        # if 'early' in self.fusion_mode:
        #     self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #     # self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        #     self.fc = nn.Linear(2048, self.nr_actions)  # changed 512 to 2048
        # elif 'late' in self.fusion_mode:
        #     # changed by HP at 2020/03/25/10:20
        #     # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #     self.avgpool = nn.AdaptiveAvgPool2d(1)
        #     self.fc = nn.Linear(self.backbone.feature_dim, self.nr_actions)  # changed 512 to 2048
        #     std = 0.001
        #     if hasattr(self.fc, 'weight'):
        #         normal_(self.fc.weight, 0, std)
        #         constant_(self.fc.bias, 0)

        # self.crit = nn.CrossEntropyLoss()

        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size1:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # exit(0)
        org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        # print('org_feas_size2:\t', org_feas.size())
        org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        # print('org_feas_size3:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        global_vis_feas = self.avgpool(conv_fea_maps).view(V, -1) # ([4, 2048])
        # print('global_vis_feas_size:\t', global_vis_feas.size())
        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 256, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 256, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        # box_tensor = box_input[:, ::(2), :, :].contiguous() #  
        # print('box_tensor_size1:\t', box_tensor.size())
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 2304])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 512)
        # print('region_vis_feas_size3:\t', region_vis_feas.size())
    
        return region_vis_feas
        #########################################################################################################################

class VideoRegionAndCoordModel_TSM(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        # self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        # self.fusion_mode = 'late_fusion'  # 'early_fusion'
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv3d(1024, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim + self.hidden_feature_dim, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 5,
                      hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim,
                      hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU()
        )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 4
            else:
                spatial_fea_dim = self.hidden_feature_dim * 2

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )

        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size1:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # exit(0)
        org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        # print('org_feas_size2:\t', org_feas.size())
        org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        # print('org_feas_size3:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        # #############################################################################
        # video_features = conv_fea_maps.mean(-1).mean(-1).permute(0, 2, 1).contiguous()
        # # print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # # print('video_features_size2:\t', video_features.size())
        # #############################################################################
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        global_vis_feas = self.avgpool(conv_fea_maps).view(V, -1) # ([4, 2048])
        # print('global_vis_feas_size:\t', global_vis_feas.size())
        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 256, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 256, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())

        global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())

        # box_tensor = box_input[:, ::(2), :, :].contiguous() #  
        # print('box_tensor_size1:\t', box_tensor.size())
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 2304])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 512)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0)
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)

        bf = self.coord_to_feature(box_input) # ([128, 512])
        bf = torch.cat([bf, box_category_embeddings], dim=1)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)

        bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])

        #################################################################################################
        # spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
        # spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
        # bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
        # bf_spatial = self.spatial_node_fusion(
        #     bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
        # # print('bf_spatial_size1:\t', bf_spatial.size())
        # bf_spatial = bf_spatial.view(
        #     V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
        # # print('bf_spatial_size2:\t', bf_spatial.size())
        # bf_temporal_input = bf_spatial.view(
        #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim)
        # # print('bf_temporal_input_size:\t', bf_temporal_input.size())
        # box_features = self.box_feature_fusion(bf_temporal_input.view(
        #     V*self.nr_boxes, -1))
        # # print('box_features_size1:\t', box_features.size())
        # box_features = torch.mean(box_features.view(
        #     V, self.nr_boxes, -1), dim=1)
        # print('box_features_size2:\t', box_features.size()) # ([4, 512])
        
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size()) #
        #########################################################################################
        
        # mix_features = torch.cat([global_vis_feas, region_features, box_features], dim=-1)
        # mix_features = torch.cat([region_features, box_features], dim=-1)

        # print('mix_features_size:\t', mix_features.size()) #
        # cls_output = self.classifier(mix_features)
        # print('cls_output_size:\t', cls_output.size()) #
        # exit(0)

        # return cls_output
        # return mix_features
        # # #################################################################################################
        if self.joint:
            coord_feas = bf.permute(0, 2, 1, 3).contiguous()
        else:
            spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
            spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
            bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
            bf_spatial = self.spatial_node_fusion(
                bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
            bf_spatial = bf_spatial.view(
                V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
            coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            fea_dict = {}
        
            fea_dict['vis'] = region_vis_feas
            fea_dict['coord'] = coord_feas

            global_feas = self.late_fusion(fea_dict)
            # print('global_feas_size:\t', global_feas.size())
        else:
            global_feas = region_vis_feas
            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 

        mix_feas = self.STIN(global_feas)
        output = torch.cat([mix_feas, global_vis_feas], dim = 1)

        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
        #######################################################################################################################
    def late_fusion(self, fea_dict):

        fea_list = []
        vis_feas = fea_dict['vis']
        fea_list.append(vis_feas)
        cor_feas = fea_dict['coord']
        fea_list.append(cor_feas)

        global_feas = torch.cat(fea_list, dim=-1) # [V x T x P x (D_vis+D_coord)]
    
        return global_feas
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())

        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())

        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # print('in_spatial_feas size:', in_spatial_feas.size())
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]

        # print('S_feas_size:\t', S_feas.size())

        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())

        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)

        # print('node_feas_size:\t', node_feas.size())

        ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        # exit(0)

        return ST_feas

class VideoRegionAndCoordModel_TSM2(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM2, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        # self.i3D = Net(self.nr_actions, extract_features=True, loss_type='softmax')
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        # self.fusion_mode = 'late_fusion'  # 'early_fusion'
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv3d(1024, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim + self.hidden_feature_dim, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 5,
                      hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim,
                      hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU()
        )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 4
            else:
                spatial_fea_dim = self.hidden_feature_dim * 2

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )

        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size1:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # exit(0)
        org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        # print('org_feas_size2:\t', org_feas.size())
        org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        # print('org_feas_size3:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        # #############################################################################
        # video_features = conv_fea_maps.mean(-1).mean(-1).permute(0, 2, 1).contiguous()
        # # print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # # print('video_features_size2:\t', video_features.size())
        # #############################################################################
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        global_vis_feas = self.avgpool(conv_fea_maps).view(V, -1) # ([4, 2048])
        # print('global_vis_feas_size:\t', global_vis_feas.size())
        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 256, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 256, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())

        global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())

        # box_tensor = box_input[:, ::(2), :, :].contiguous() #  
        # print('box_tensor_size1:\t', box_tensor.size())
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 2304])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 512)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0)
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        box_category_embeddings = self.category_embed_layer(box_categories)

        bf = self.coord_to_feature(box_input) # ([128, 512])
        bf = torch.cat([bf, box_category_embeddings], dim=1)
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)

        bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])

        #################################################################################################
        # spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
        # spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
        # bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
        # bf_spatial = self.spatial_node_fusion(
        #     bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
        # # print('bf_spatial_size1:\t', bf_spatial.size())
        # bf_spatial = bf_spatial.view(
        #     V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
        # # print('bf_spatial_size2:\t', bf_spatial.size())
        # bf_temporal_input = bf_spatial.view(
        #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim)
        # # print('bf_temporal_input_size:\t', bf_temporal_input.size())
        # box_features = self.box_feature_fusion(bf_temporal_input.view(
        #     V*self.nr_boxes, -1))
        # # print('box_features_size1:\t', box_features.size())
        # box_features = torch.mean(box_features.view(
        #     V, self.nr_boxes, -1), dim=1)
        # print('box_features_size2:\t', box_features.size()) # ([4, 512])
        
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size()) #
        #########################################################################################
        
        # mix_features = torch.cat([global_vis_feas, region_features, box_features], dim=-1)
        # mix_features = torch.cat([region_features, box_features], dim=-1)

        # print('mix_features_size:\t', mix_features.size()) #
        # cls_output = self.classifier(mix_features)
        # print('cls_output_size:\t', cls_output.size()) #
        # exit(0)

        # return cls_output
        # return mix_features
        # # #################################################################################################
        if self.joint:
            coord_feas = bf.permute(0, 2, 1, 3).contiguous()
        else:
            spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
            spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
            bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
            bf_spatial = self.spatial_node_fusion(
                bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
            bf_spatial = bf_spatial.view(
                V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
            coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            fea_dict = {}
        
            fea_dict['vis'] = region_vis_feas
            fea_dict['coord'] = coord_feas

            global_feas = self.late_fusion(fea_dict)
            # print('global_feas_size:\t', global_feas.size())
        else:
            global_feas = region_vis_feas
            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 

        mix_feas = self.STIN(global_feas)
        output = torch.cat([mix_feas, global_vis_feas], dim = 1)

        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
        #######################################################################################################################
    def late_fusion(self, fea_dict):

        fea_list = []
        vis_feas = fea_dict['vis']
        fea_list.append(vis_feas)
        cor_feas = fea_dict['coord']
        fea_list.append(cor_feas)

        global_feas = torch.cat(fea_list, dim=-1) # [V x T x P x (D_vis+D_coord)]
    
        return global_feas
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())

        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())

        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # print('in_spatial_feas size:', in_spatial_feas.size())
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]

        # print('S_feas_size:\t', S_feas.size())

        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())

        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)

        # print('node_feas_size:\t', node_feas.size())

        ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        # exit(0)

        return ST_feas

class VideoRegionAndCoordModel_TSM3(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM3, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim * 1, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim + self.hidden_feature_dim, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            # nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.Linear(512 * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 8,
                      hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_feature_dim,
            #           hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU()
        )
        self.global_vis_feas_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 8, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 2

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim * 4
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V*T, 512, h, w]) ([32, 512, 7, 7])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        conv_fea_maps = conv_fea_maps.view(V, -1, *conv_fea_maps.size()[1:]) # ([4, 8, 512, 7, 7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # exit(0)
        #############################################################################
        video_features = org_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        video_features = conv_fea_maps.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # print('video_features_size2:\t', video_features.size())
        video_features = video_features.view(V, -1)
        # print('video_features_size2:\t', video_features.size())
        video_features = self.global_vis_feas_fusion(video_features)
        # print('video_features_size3:\t', video_features.size())
        #############################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]) # (4, 8, 2048, 7, 7)
        # global_vis_feas = self.avgpool(org_feas).view(V, TC//sample_len, -1) # (4, 8, 2048)
        # # global_vis_feas = self.avgpool(conv_fea_maps).view(V, TC//sample_len, -1)
        # #.mean(-1).mean(-1) # ([4, 8, 512])
        # # .view(V, -1) # 
        # # print('global_vis_feas_size1:\t', global_vis_feas.size())
        # global_vis_feas = global_vis_feas.view(V, -1)
        # # print('global_vis_feas_size2:\t', global_vis_feas.size())
        # global_vis_feas = self.global_vis_feas_fusion(global_vis_feas)
        # # print('global_vis_feas_size3:\t', global_vis_feas.size())
        #############################################################################
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        ##########################################################################################################
        # global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        # global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())
        # exit(0)
        ##########################################################################################################
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # exit(0)
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0)
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        # print('box_categories_size1:\t', box_categories.size())
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        # print('box_categories_size2:\t', box_categories.size())
        box_category_embeddings = self.category_embed_layer(box_categories)
        # print('box_category_embeddings_size:\t', box_category_embeddings.size())
        bf = self.coord_to_feature(box_input) # ([128, 512])
        # print('bf_size1:\t', bf.size())
        bf = torch.cat([bf, box_category_embeddings], dim=1)
        # bf = bf + box_category_embeddings
        # print('bf_size2:\t', bf.size())
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        # print('bf_size3:\t', bf.size())
        bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])
        # print('bf_size4:\t', bf.size())
        # exit(0)
        #################################################################################################
        if self.joint:
            coord_feas = bf.permute(0, 2, 1, 3).contiguous()
            # print('coord_feas_size:\t', coord_feas.size())
        else:
            spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
            spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
            bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
            bf_spatial = self.spatial_node_fusion(
                bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
            bf_spatial = bf_spatial.view(
                V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
            coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            global_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
            # print('global_feas_size:\t', global_feas.size())
        else:
            global_feas = region_vis_feas
            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 
        # exit(0)
        mix_feas = self.STIN(global_feas)
        # print('mix_feas_size:\t', mix_feas.size())       
        output = torch.cat([mix_feas, video_features], dim = 1)
        # output = mix_feas + video_features
        # print('output_size:\t', output.size())
        # output = mix_feas
        # print('mix_feature_size:\t', mix_feature.size())       
        # exit(0)
        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())
        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())
        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        # print('S_feas_size1:\t', S_feas.size())
        # ############################################################
        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())
        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)
        # print('node_feas_size1:\t', node_feas.size())
        node_feas = node_feas.view(V, -1, node_feas.size()[-1])
        # print('node_feas_size2:\t', node_feas.size())
        node_feas = node_feas.view(V, -1)
        # print('node_feas_size3:\t', node_feas.size())
        ST_feas = self.node_fusion(node_feas)
        # print('ST_feas_size:\t', ST_feas.size())

        ############################################################
        # ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        ############################################################
        # S_feas = S_feas.view(V, P, T, -1)
        # # print('S_feas_size2:\t', S_feas.size())
        # ST_feas = S_feas.mean(1).mean(1)
        # # print('ST_feas_size:\t', ST_feas.size())
        # exit(0)

        return ST_feas


class VideoRegionAndCoordModel_TSM4(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM4, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        self.vit = SEAttention()
        self.text_embed = text2embed()
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.conv = nn.Conv3d(2048, self.img_feature_dim, kernel_size=(1, 1, 1), stride=1)
        self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim * 1, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim + self.hidden_feature_dim , self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            # nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.Linear(512 * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 8,
                      hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_feature_dim,
            #           hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU()
        )
        self.global_vis_feas_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 8 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 2

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim * 4
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories, texture):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ########
        print(texture)
        # exit(0)
        text_vector = self.text_embed(texture)
        # print(text_vector.pooler_output)
        # print(type(text_vector.pooler_output))
        # print(text_vector.pooler_output.size())
        # exit(0)
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # vt_feas = self.vit(org_feas)
        # print('org_feas_size:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V*T, 512, h, w]) ([32, 512, 7, 7])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        conv_fea_maps = conv_fea_maps.view(V, -1, *conv_fea_maps.size()[1:]) # ([4, 8, 512, 7, 7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # exit(0)
        #############################################################################
        video_features = org_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = vt_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = conv_fea_maps.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # print('video_features_size2:\t', video_features.size())
        video_features = video_features.view(V, -1)
        # print('video_features_size2:\t', video_features.size())
        video_features = self.global_vis_feas_fusion(video_features)    
        # print('video_features_size3:\t', video_features.size())
        #############################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]) # (4, 8, 2048, 7, 7)
        # global_vis_feas = self.avgpool(org_feas).view(V, TC//sample_len, -1) # (4, 8, 2048)
        # # global_vis_feas = self.avgpool(conv_fea_maps).view(V, TC//sample_len, -1)
        # #.mean(-1).mean(-1) # ([4, 8, 512])
        # # .view(V, -1) # 
        # # print('global_vis_feas_size1:\t', global_vis_feas.size())
        # global_vis_feas = global_vis_feas.view(V, -1)
        # # print('global_vis_feas_size2:\t', global_vis_feas.size())
        # global_vis_feas = self.global_vis_feas_fusion(global_vis_feas)
        # # print('global_vis_feas_size3:\t', global_vis_feas.size())
        #############################################################################
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        ##########################################################################################################
        # global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        # global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())
        # exit(0)
        ##########################################################################################################
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # exit(0)
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0)
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        # print('box_categories_size1:\t', box_categories.size())
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        # print('box_categories_size2:\t', box_categories.size())
        box_category_embeddings = self.category_embed_layer(box_categories)
        # print('box_category_embeddings_size:\t', box_category_embeddings.size())
        bf = self.coord_to_feature(box_input) # ([128, 512])
        # print('bf_size1:\t', bf.size())
        bf = torch.cat([bf, box_category_embeddings], dim=1)
        # bf = bf + box_category_embeddings
        # print('bf_size2:\t', bf.size())
        bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        # print('bf_size3:\t', bf.size())
        bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])
        # print('bf_size4:\t', bf.size())
        # exit(0)
        #################################################################################################
        if self.joint:
            coord_feas = bf.permute(0, 2, 1, 3).contiguous()
            # print('coord_feas_size:\t', coord_feas.size())
        else:
            spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
            spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
            bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
            bf_spatial = self.spatial_node_fusion(
                bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
            bf_spatial = bf_spatial.view(
                V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
            coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            global_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
            # print('global_feas_size:\t', global_feas.size())
        else:
            global_feas = region_vis_feas
            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 
        # exit(0)
        mix_feas = self.STIN(global_feas)
        # print('mix_feas_size:\t', mix_feas.size())       
        output = torch.cat([mix_feas, video_features], dim = 1)
        # output = torch.cat([global_feas, video_features], dim = 1)

        # output = mix_feas + video_features
        # print('output_size:\t', output.size())
        # output = mix_feas
        # print('mix_feature_size:\t', mix_feature.size())       
        # exit(0)
        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
    
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())
        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())
        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        # print('S_feas_size1:\t', S_feas.size())
        # ############################################################
        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())
        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)
        # print('node_feas_size1:\t', node_feas.size())
        node_feas = node_feas.view(V, -1, node_feas.size()[-1])
        # print('node_feas_size2:\t', node_feas.size())
        node_feas = node_feas.view(V, -1)
        # print('node_feas_size3:\t', node_feas.size())
        ST_feas = self.node_fusion(node_feas)
        # print('ST_feas_size:\t', ST_feas.size())

        ############################################################
        # ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        ############################################################
        # S_feas = S_feas.view(V, P, T, -1)
        # # print('S_feas_size2:\t', S_feas.size())
        # ST_feas = S_feas.mean(1).mean(1)
        # # print('ST_feas_size:\t', ST_feas.size())
        # exit(0)

        return ST_feas

class VideoRegionAndCoordModel_TSM5(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM5, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        self.vit = SEAttention()
        self.TCR = TCR()
        print(self.TCR)
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim * 1, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim * 3, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            # nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.Linear(512 * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 8,
                      hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_feature_dim,
            #           hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU()
        )
        self.global_vis_feas_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim * 4
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # vt_feas = self.vit(org_feas)
        # print('org_feas_size:\t', org_feas.size())
        #######################################################################################################
        org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        # print('org_feas_size2:\t', org_feas.size())
        org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        # print('org_feas_size3:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        # print('conv_fea_maps_size:\t', conv_fea_maps.size())
        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        #######################################################################################################
        # conv_fea_maps = self.conv(org_feas) # ([V*T, 512, h, w]) ([32, 512, 7, 7])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        # conv_fea_maps = conv_fea_maps.view(V, -1, *conv_fea_maps.size()[1:]) # ([4, 8, 512, 7, 7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # exit(0)
        #############################################################################
        video_features = org_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = vt_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = conv_fea_maps.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # print('video_features_size2:\t', video_features.size())
        # video_features = video_features.view(V, -1)
        # # print('video_features_size2:\t', video_features.size())
        # video_features = self.global_vis_feas_fusion(video_features)
        # # print('video_features_size3:\t', video_features.size())
        ##################################
        video_features = video_features.view(V * self.nr_frames, -1)
        # print('video_features_size2:\t', video_features.size())
        video_features = self.global_vis_feas_fusion(video_features)
        # print('video_features_size3:\t', video_features.size())
        video_features = video_features.view(V, self.nr_frames, -1)
        # print('video_features_size4:\t', video_features.size())
        video_features = self.avgpool(video_features.transpose(2, 1).contiguous()).view(V, -1)
        # print('video_features_size5:\t', video_features.size())

        ##################################
        # exit(0)
        #############################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]) # (4, 8, 2048, 7, 7)
        # global_vis_feas = self.avgpool(org_feas).view(V, TC//sample_len, -1) # (4, 8, 2048)
        # # global_vis_feas = self.avgpool(conv_fea_maps).view(V, TC//sample_len, -1)
        # #.mean(-1).mean(-1) # ([4, 8, 512])
        # # .view(V, -1) # 
        # # print('global_vis_feas_size1:\t', global_vis_feas.size())
        # global_vis_feas = global_vis_feas.view(V, -1)
        # # print('global_vis_feas_size2:\t', global_vis_feas.size())
        # global_vis_feas = self.global_vis_feas_fusion(global_vis_feas)
        # # print('global_vis_feas_size3:\t', global_vis_feas.size())
        #############################################################################
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        ##########################################################################################################
        # global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        # global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())
        # exit(0)
        ##########################################################################################################
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # exit(0)
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0) 
        coord = box_input
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        # print('box_categories_size1:\t', box_categories.size())
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        # print('box_categories_size2:\t', box_categories.size())
        box_category_embeddings = self.category_embed_layer(box_categories)
        # print('box_category_embeddings_size:\t', box_category_embeddings.size())
        ##############
        bc_embed = box_category_embeddings.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bc_embed_size1:\t', bc_embed.size())
        bc_embed = bc_embed.permute(0,3,1,2).contiguous()
        # print('bc_embed_size2:\t', bc_embed.size())
        bc_embed = self.avgpool2(bc_embed).view(bc_embed.size(0),-1)
        # print('bc_embed_size3:\t', bc_embed.size())
        # exit(0)
        ##############
        #####################################################################################
        # print('coord_size1:\t', coord.size())
        coord = coord.view(V, self.nr_frames, self.nr_boxes * 4).permute(0,2,1).contiguous()
        # print('coord_size2:\t', coord.size())
        # print(coord)
        coord_feas = self.TCR(coord)
        # print('coord_feas_size1:\t', coord_feas.size())
        coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
        # print('coord_feas_size2:\t', coord_feas.size())
        ##############################
        coord_feas = torch.cat([coord_feas, bc_embed], dim = 1)
        # print('coord_feas_size3:\t', coord_feas.size())
        box_feas = self.coord_category_fusion(coord_feas)
        # print('box_feas_size:\t', box_feas.size())
        ##############################
        # exit(0)
        #####################################################################################
        # bf = self.coord_to_feature(box_input) # ([128, 512])
        # print('bf_size1:\t', bf.size())
        # bf = torch.cat([bf, box_category_embeddings], dim=1)
        # bf = bf + box_category_embeddings
        # print('bf_size2:\t', bf.size())
        # bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        # print('bf_size3:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])
        # print('bf_size4:\t', bf.size())
        # ###########################################
        # bf = bf.view(V, self.nr_boxes * self.nr_frames, -1).permute(0,2,1).contiguous()
        # print('bf_size5:\t', bf.size())
        # bf = self.TCR(bf)
        # print('bf_size6:\t', bf.size())
        # bf = bf.permute(0,2,1).contiguous()
        # print('bf_size7:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bf_size8:\t', bf.size())
        ###########################################
        # exit(0)
        #################################################################################################
        # if self.joint:
        #     coord_feas = bf.permute(0, 2, 1, 3).contiguous()
        #     # print('coord_feas_size:\t', coord_feas.size())
        # else:
        #     #############################################################
        #     # spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
        #     # spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
        #     # bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
        #     # bf_spatial = self.spatial_node_fusion(
        #     #     bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
        #     # bf_spatial = bf_spatial.view(
        #     #     V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
        #     # coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        #     ##############################################################
        #     coord_feas = bf.permute(0, 3, 1, 2).contiguous()
        #     # print('coord_feas_size1:\t', coord_feas.size())
        #     coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
            # print('coord_feas_size1:\t', coord_feas.size())
            ##############################################################
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            global_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
            # print('global_feas_size:\t', global_feas.size())
        else:
            global_feas = region_vis_feas
            # print('global_feas_size:\t', global_feas.size())

            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 
        # exit(0)
        mix_feas = self.STIN(global_feas)
        # print('mix_feas_size:\t', mix_feas.size())       
        output = torch.cat([mix_feas, box_feas, video_features], dim = -1)
        # output = torch.cat([global_feas, video_features], dim = 1)

        # output = mix_feas + video_features
        # print('output_size:\t', output.size())
        # output = mix_feas
        # print('mix_feature_size:\t', mix_feature.size())       
        # exit(0)
        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())
        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())
        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        # print('S_feas_size1:\t', S_feas.size())
        # ############################################################
        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())
        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)
        # print('node_feas_size1:\t', node_feas.size())
        node_feas = node_feas.view(V, -1, node_feas.size()[-1])
        # print('node_feas_size2:\t', node_feas.size())
        node_feas = node_feas.view(V, -1)
        # print('node_feas_size3:\t', node_feas.size())
        ST_feas = self.node_fusion(node_feas)
        # print('ST_feas_size:\t', ST_feas.size())

        ############################################################
        # ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        ############################################################
        # S_feas = S_feas.view(V, P, T, -1)
        # # print('S_feas_size2:\t', S_feas.size())
        # ST_feas = S_feas.mean(1).mean(1)
        # # print('ST_feas_size:\t', ST_feas.size())
        # exit(0)

        return ST_feas

class VideoRegionAndCoordModel_TSM6(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM6, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        self.vit = SEAttention()
        self.TCR = TCR()
        print(self.TCR)
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim * 1, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim * 2, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            # nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.Linear(512 * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 8,
                      hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_feature_dim,
            #           hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU()
        )
        self.global_vis_feas_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 8, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 9

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim * 4
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        print('tsm_input_size:\t', tsm_input.size())
        org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        print('org_feas_size:\t', org_feas.size())
        # vt_feas = self.vit(org_feas)
        # print('org_feas_size:\t', org_feas.size())
        #######################################################################################################
        org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        print('org_feas_size2:\t', org_feas.size())
        org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        print('org_feas_size3:\t', org_feas.size())
        conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        print('conv_fea_maps_size:\t', conv_fea_maps.size())
        conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        print('conv_fea_maps_size3:\t', conv_fea_maps.size()) 
        #######################################################################################################
        # conv_fea_maps = self.conv(org_feas) # ([V*T, 512, h, w]) ([32, 512, 7, 7])
        # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        # conv_fea_maps = conv_fea_maps.view(V, -1, *conv_fea_maps.size()[1:]) # ([4, 8, 512, 7, 7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # exit(0)
        #############################################################################
        video_features = org_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = vt_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # video_features = conv_fea_maps.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        print('video_features_size1:\t', video_features.size())
        # video_features = video_features.mean(1)
        # print('video_features_size2:\t', video_features.size())
        video_features = video_features.view(V, -1)
        print('video_features_size2:\t', video_features.size())
        video_features = self.global_vis_feas_fusion(video_features)
        print('video_features_size3:\t', video_features.size())
        ##################################
        # video_features = video_features.view(V * self.nr_frames, -1)
        # print('video_features_size2:\t', video_features.size())
        # video_features = self.global_vis_feas_fusion(video_features)
        # print('video_features_size3:\t', video_features.size())
        # video_features = video_features.view(V, self.nr_frames, -1)
        # print('video_features_size4:\t', video_features.size())
        # video_features = self.avgpool(video_features.transpose(2, 1).contiguous()).view(V, -1)
        # print('video_features_size5:\t', video_features.size())

        ##################################
        # exit(0)
        #############################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]) # (4, 8, 2048, 7, 7)
        # global_vis_feas = self.avgpool(org_feas).view(V, TC//sample_len, -1) # (4, 8, 2048)
        # # global_vis_feas = self.avgpool(conv_fea_maps).view(V, TC//sample_len, -1)
        # #.mean(-1).mean(-1) # ([4, 8, 512])
        # # .view(V, -1) # 
        # # print('global_vis_feas_size1:\t', global_vis_feas.size())
        # global_vis_feas = global_vis_feas.view(V, -1)
        # # print('global_vis_feas_size2:\t', global_vis_feas.size())
        # global_vis_feas = self.global_vis_feas_fusion(global_vis_feas)
        # # print('global_vis_feas_size3:\t', global_vis_feas.size())
        #############################################################################
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        # conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        ##########################################################################################################
        # global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        # global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())
        # exit(0)
        ##########################################################################################################
        box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        print('box_tensor_size:\t', box_tensor.size())
        boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        img_size = tsm_input.size()[-2:]
        region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
        print('region_vis_feas_size1:\t', region_vis_feas.size())
        region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        print('region_vis_feas_size2:\t', region_vis_feas.size())
        region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
        print('region_vis_feas_size:\t', region_vis_feas.size())
        # exit(0)
        # region_features = self.STIN(region_vis_feas)
        # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0) 
        coord = box_input
        box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        box_categories = box_categories.long()
        box_categories = box_categories.transpose(2, 1).contiguous()
        print('box_categories_size1:\t', box_categories.size())
        box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        print('box_categories_size2:\t', box_categories.size())
        box_category_embeddings = self.category_embed_layer(box_categories)
        print('box_category_embeddings_size:\t', box_category_embeddings.size())
        bc_embed = box_category_embeddings.view(V, self.nr_boxes, self.nr_frames, -1)
        print('bc_embed_size1:\t', bc_embed.size())
        bc_embed = bc_embed.permute(0,2,1,3).contiguous()
        print('bc_embed_size2:\t', bc_embed.size())
        ##############
        # bc_embed = box_category_embeddings.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bc_embed_size1:\t', bc_embed.size())
        # bc_embed = bc_embed.permute(0,3,1,2).contiguous()
        # print('bc_embed_size2:\t', bc_embed.size())
        # bc_embed = self.avgpool2(bc_embed).view(bc_embed.size(0),-1)
        # print('bc_embed_size3:\t', bc_embed.size())
        # exit(0)
        ##############
        #####################################################################################
        print('coord_size1:\t', coord.size())
        coord = coord.view(V, self.nr_frames, self.nr_boxes * 4).permute(0,2,1).contiguous()
        print('coord_size2:\t', coord.size())
        # print(coord)
        coord_feas = self.TCR(coord)
        print('coord_feas_size1:\t', coord_feas.size())
        coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
        print('coord_feas_size2:\t', coord_feas.size())
        ##############################
        # coord_feas = torch.cat([coord_feas, bc_embed], dim = 1)
        # print('coord_feas_size3:\t', coord_feas.size())
        # box_feas = self.coord_category_fusion(coord_feas)
        # print('box_feas_size:\t', box_feas.size())
        ##############################
        exit(0)
        #####################################################################################
        # bf = self.coord_to_feature(box_input) # ([128, 512])
        # print('bf_size1:\t', bf.size())
        # bf = torch.cat([bf, box_category_embeddings], dim=1)
        # bf = bf + box_category_embeddings
        # print('bf_size2:\t', bf.size())
        # bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        # print('bf_size3:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])
        # print('bf_size4:\t', bf.size())
        # ###########################################
        # bf = bf.view(V, self.nr_boxes * self.nr_frames, -1).permute(0,2,1).contiguous()
        # print('bf_size5:\t', bf.size())
        # bf = self.TCR(bf)
        # print('bf_size6:\t', bf.size())
        # bf = bf.permute(0,2,1).contiguous()
        # print('bf_size7:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bf_size8:\t', bf.size())
        ###########################################
        # exit(0)
        #################################################################################################
        # if self.joint:
        #     coord_feas = bf.permute(0, 2, 1, 3).contiguous()
        #     # print('coord_feas_size:\t', coord_feas.size())
        # else:
        #     #############################################################
        #     # spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
        #     # spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
        #     # bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
        #     # bf_spatial = self.spatial_node_fusion(
        #     #     bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
        #     # bf_spatial = bf_spatial.view(
        #     #     V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
        #     # coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        #     ##############################################################
        #     coord_feas = bf.permute(0, 3, 1, 2).contiguous()
        #     # print('coord_feas_size1:\t', coord_feas.size())
        #     coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
            # print('coord_feas_size1:\t', coord_feas.size())
            ##############################################################
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        if self.joint:
            global_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
            # print('global_feas_size:\t', global_feas.size())
        else:
            # global_feas = region_vis_feas
            global_feas = torch.cat([region_vis_feas, bc_embed], dim = -1)
            # print('global_feas_size:\t', global_feas.size())

            # global_feas = box_features

        # output = global_feas 
        # output = self.classifier(global_feas) 
        # exit(0)
        mix_feas = self.STIN(global_feas)
        # print('mix_feas_size:\t', mix_feas.size())       
        output = torch.cat([mix_feas, video_features], dim = -1)
        # output = torch.cat([global_feas, video_features], dim = 1)

        # output = mix_feas + video_features
        # print('output_size:\t', output.size())
        # output = mix_feas
        # print('mix_feature_size:\t', mix_feature.size())       
        # exit(0)
        # ################################

        # mix_feature = torch.cat([global_features, output], dim = -1)
        # output = self.mix_feature_fusion(mix_feature)
        # ################################
        # print('output_size:\t', output.size())
        # exit(0)
       
        return output
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())
        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())
        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        # print('S_feas_size1:\t', S_feas.size())
        # ############################################################
        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())
        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)
        # print('node_feas_size1:\t', node_feas.size())
        node_feas = node_feas.view(V, -1, node_feas.size()[-1])
        # print('node_feas_size2:\t', node_feas.size())
        node_feas = node_feas.view(V, -1)
        # print('node_feas_size3:\t', node_feas.size())
        ST_feas = self.node_fusion(node_feas)
        # print('ST_feas_size:\t', ST_feas.size())

        ############################################################
        # ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        ############################################################
        # S_feas = S_feas.view(V, P, T, -1)
        # # print('S_feas_size2:\t', S_feas.size())
        # ST_feas = S_feas.mean(1).mean(1)
        # # print('ST_feas_size:\t', ST_feas.size())
        # exit(0)

        return ST_feas
class VideoRegionAndCoordModel_TSM7(nn.Module):
    """
    This model contains only global pooling without any graph.
    """

    def __init__(self, partial_bn=True, fc_lr5=False,
                 ):
        super(VideoRegionAndCoordModel_TSM7, self).__init__()

        self.joint = args.joint
        self.nr_boxes = args.num_boxes
        self.nr_actions = args.num_class
        self.nr_frames = args.num_segments
        self.img_feature_dim = args.img_feature_dim
        self.coord_feature_dim = args.coord_feature_dim
        self.hidden_feature_dim = args.hidden_feature_dim
        self.fc_lr5 = not (args.tune_from and args.dataset in args.tune_from)
        self.backbone = resnet_TSM(args.basic_arch, args.shift, num_segments=8)
        self.vit = SEAttention()
        self.TCR = TCR()
        # print(self.TCR)
        
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv3d(2048, 512, kernel_size=(1, 1, 1), stride=1)
        # self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)

        if self.img_feature_dim == 256:
            hidden_feature_dim = 512
        else:
            hidden_feature_dim = 2048 

        # self.fc = nn.Linear(hidden_feature_dim * 1, self.nr_actions)
        self.fc = nn.Linear(hidden_feature_dim, self.nr_actions)

        self.crop_size1 = [3, 3]

        ## 此处加个判断512还是2048
        self.region_vis_embed = nn.Sequential(
            # nn.Linear(self.img_feature_dim * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim),
            nn.Linear(512 * self.crop_size1[0] * self.crop_size1[1], hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        ############################################# coord ###############################################
        self.category_embed_layer = nn.Embedding(
            3, args.coord_feature_dim // 2, padding_idx=0, scale_grad_by_freq=True)
        self.coord_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_feature_dim//2, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim//2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.coord_category_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim+self.coord_feature_dim //
                      2, self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.spatial_node_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim*2,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.box_feature_fusion = nn.Sequential(
            nn.Linear(self.nr_frames*self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim,
                      self.coord_feature_dim, bias=False),
            nn.BatchNorm1d(self.coord_feature_dim),
            nn.ReLU()
        )
        self.mix_feature_fusion = nn.Sequential(
            nn.Linear(self.coord_feature_dim * 8,
                      hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(hidden_feature_dim,
            #           hidden_feature_dim, bias=False),
            # nn.BatchNorm1d(hidden_feature_dim),
            # nn.ReLU()
        )
        self.global_vis_feas_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 8, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
            )
        ###################################################################################################
        if self.img_feature_dim == 256:
            if self.joint:                
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 9

            spatial_hidden_dim = self.hidden_feature_dim
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim * 4
        else:
            if self.joint:
                spatial_fea_dim = self.hidden_feature_dim * 10
            else:
                spatial_fea_dim = self.hidden_feature_dim * 8

            spatial_hidden_dim = self.hidden_feature_dim * 4
            temp_fea_dim = self.nr_frames * hidden_feature_dim
            temp_hidden_dim = hidden_feature_dim

        self.spatial_node_fusion_list = nn.Sequential(
                    nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                    nn.BatchNorm1d(spatial_hidden_dim),
                    nn.ReLU(inplace=True)
                    )
        self.temporal_node_fusion_list = nn.Sequential(
            nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
            nn.BatchNorm1d(temp_hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_feature_dim * 4 * 4, hidden_feature_dim * 4, bias=False),
            nn.BatchNorm1d(hidden_feature_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_feature_dim * 4, hidden_feature_dim, bias=False),
            nn.BatchNorm1d(hidden_feature_dim),
            nn.ReLU(inplace=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(self.coord_feature_dim, self.coord_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.coord_feature_dim, 512),
            nn.ReLU(inplace=True)
            # nn.Linear(512, self.nr_actions)
        )
        ###################################################################################################
        # if args.fine_tune:
        #     self.fine_tune(opt.fine_tune)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def partialBN(self, enable):
        self._enable_pbn = enable

    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'


    # def forward(self, global_img_input, local_img_input, box_input, video_label, is_inference=False):
    def forward(self, input, target, box_input, box_categories):
    
        """
        V: num of videos
        T: num of frames
        P: num of proposals
        :param videos: [V x 3 x T x 224 x 224]
        :param proposals_t: [V x T] List of BoxList (size of num_boxes each)
        :return:
        """
        ##################################################################################################
        # print('input_size:\t', input.size())  # ([4, 24, 224, 224])
        # print('box_input:\t', box_input.size())
        V, TC, _, _ = input.size() # ([V, T*C, H, W]) 
        sample_len = 3
        tsm_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # ([V, T, C, H, W]) ([4, 8, 3, 224, 224])
        # print('tsm_input_size:\t', tsm_input.size())
        # org_feas = self.backbone(tsm_input.view(-1, *tsm_input.size()[2:])) # (V*T, c, h, w) (32, 2048, 7, 7)
        # print('org_feas_size:\t', org_feas.size())
        # # vt_feas = self.vit(org_feas)
        # # print('org_feas_size:\t', org_feas.size())
        # #######################################################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]).contiguous() # ([V, T, c, h, w]) (4, 8, 2048, 7, 7)
        # print('org_feas_size2:\t', org_feas.size())
        # org_feas = org_feas.transpose(1, 2) # ([V, c, T, h, w]) ([4, 2048, 8, 7, 7])
        # print('org_feas_size3:\t', org_feas.size())
        # conv_fea_maps = self.conv(org_feas) # ([V, 256, T, h, w]) ([4, 256, 8, 7, 7])
        # print('conv_fea_maps_size:\t', conv_fea_maps.size())
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size()) 
        # #######################################################################################################
        # # conv_fea_maps = self.conv(org_feas) # ([V*T, 512, h, w]) ([32, 512, 7, 7])
        # # print('conv_fea_maps_size1:\t', conv_fea_maps.size())
        # # conv_fea_maps = conv_fea_maps.view(V, -1, *conv_fea_maps.size()[1:]) # ([4, 8, 512, 7, 7])
        # # print('conv_fea_maps_size2:\t', conv_fea_maps.size())
        # # exit(0)
        # #############################################################################
        # video_features = org_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # # video_features = vt_feas.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # # video_features = conv_fea_maps.mean(-1).mean(-1).view(V, TC//sample_len, -1)
        # print('video_features_size1:\t', video_features.size())
        # # video_features = video_features.mean(1)
        # # print('video_features_size2:\t', video_features.size())
        # video_features = video_features.view(V, -1)
        # print('video_features_size2:\t', video_features.size())
        # video_features = self.global_vis_feas_fusion(video_features)
        # print('video_features_size3:\t', video_features.size())
        ##################################
        # video_features = video_features.view(V * self.nr_frames, -1)
        # print('video_features_size2:\t', video_features.size())
        # video_features = self.global_vis_feas_fusion(video_features)
        # print('video_features_size3:\t', video_features.size())
        # video_features = video_features.view(V, self.nr_frames, -1)
        # print('video_features_size4:\t', video_features.size())
        # video_features = self.avgpool(video_features.transpose(2, 1).contiguous()).view(V, -1)
        # print('video_features_size5:\t', video_features.size())

        ##################################
        # exit(0)
        #############################################################################
        # org_feas = org_feas.view(V, TC//sample_len, *org_feas.size()[1:]) # (4, 8, 2048, 7, 7)
        # global_vis_feas = self.avgpool(org_feas).view(V, TC//sample_len, -1) # (4, 8, 2048)
        # # global_vis_feas = self.avgpool(conv_fea_maps).view(V, TC//sample_len, -1)
        # #.mean(-1).mean(-1) # ([4, 8, 512])
        # # .view(V, -1) # 
        # # print('global_vis_feas_size1:\t', global_vis_feas.size())
        # global_vis_feas = global_vis_feas.view(V, -1)
        # # print('global_vis_feas_size2:\t', global_vis_feas.size())
        # global_vis_feas = self.global_vis_feas_fusion(global_vis_feas)
        # # print('global_vis_feas_size3:\t', global_vis_feas.size())
        #############################################################################
        # conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # ([V, T, c, h, w]) ([4, 8, 512, 7 ,7])
        # print('conv_fea_maps_size2:\t', conv_fea_maps.size()) 
        # conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # ([V*T, c, h, w]) ([32, 512, 7, 7]) 
        # print('conv_fea_maps_size3:\t', conv_fea_maps.size())
        ##########################################################################################################
        # global_features = conv_fea_maps.mean(-1).mean(-1).view(V, self.nr_frames, self.img_feature_dim)
        # print('global_features_size1:\t', global_features.size())
        # global_features = global_features.mean(1)
        # print('global_features_size2:\t', global_features.size())
        # exit(0)
        ##########################################################################################################
        # box_tensor = box_input.view(-1, *box_input.size()[2:]) # ([V*T, P, 4]) (32, 4, 4)
        # print('box_tensor_size:\t', box_tensor.size())
        # boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        # img_size = tsm_input.size()[-2:]
        # region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
        # print('region_vis_feas_size1:\t', region_vis_feas.size())
        # region_vis_feas = region_vis_feas.view(V, TC//sample_len, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
        # print('region_vis_feas_size2:\t', region_vis_feas.size())
        # region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
        # print('region_vis_feas_size:\t', region_vis_feas.size())
        # # exit(0)
        # # region_features = self.STIN(region_vis_feas)
        # # print('region_features_size:\t', region_features.size())

        ############################################ coord features #############################################
        # # B = box_input.size(0) 
        coord = box_input
        # box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
        # box_input = box_input.view(V * self.nr_boxes * self.nr_frames, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
        
        # box_categories = box_categories.long()
        # box_categories = box_categories.transpose(2, 1).contiguous()
        # print('box_categories_size1:\t', box_categories.size())
        # box_categories = box_categories.view(V*self.nr_boxes*self.nr_frames)
        # print('box_categories_size2:\t', box_categories.size())
        # box_category_embeddings = self.category_embed_layer(box_categories)
        # print('box_category_embeddings_size:\t', box_category_embeddings.size())
        # bc_embed = box_category_embeddings.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bc_embed_size1:\t', bc_embed.size())
        # bc_embed = bc_embed.permute(0,2,1,3).contiguous()
        # print('bc_embed_size2:\t', bc_embed.size())
        # ##############
        # # bc_embed = box_category_embeddings.view(V, self.nr_boxes, self.nr_frames, -1)
        # # print('bc_embed_size1:\t', bc_embed.size())
        # # bc_embed = bc_embed.permute(0,3,1,2).contiguous()
        # # print('bc_embed_size2:\t', bc_embed.size())
        # # bc_embed = self.avgpool2(bc_embed).view(bc_embed.size(0),-1)
        # # print('bc_embed_size3:\t', bc_embed.size())
        # # exit(0)
        ##############
        #####################################################################################
        # print('coord_size1:\t', coord.size())
        coord = coord.view(V, self.nr_frames, self.nr_boxes * 4).permute(0,2,1).contiguous()
        # print('coord_size2:\t', coord.size())
        # print(coord)
        coord_feas = self.TCR(coord)
        # print('coord_feas_size1:\t', coord_feas.size())
        coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
        # print('coord_feas_size2:\t', coord_feas.size())
        output = coord_feas

        ##############################
        # coord_feas = torch.cat([coord_feas, bc_embed], dim = 1)
        # print('coord_feas_size3:\t', coord_feas.size())
        # box_feas = self.coord_category_fusion(coord_feas)
        # print('box_feas_size:\t', box_feas.size())
        ##############################
        # exit(0)
        #####################################################################################
        # bf = self.coord_to_feature(box_input) # ([128, 512])
        # print('bf_size1:\t', bf.size())
        # bf = torch.cat([bf, box_category_embeddings], dim=1)
        # bf = bf + box_category_embeddings
        # print('bf_size2:\t', bf.size())
        # bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
        # print('bf_size3:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8 ,512])
        # print('bf_size4:\t', bf.size())
        # ###########################################
        # bf = bf.view(V, self.nr_boxes * self.nr_frames, -1).permute(0,2,1).contiguous()
        # print('bf_size5:\t', bf.size())
        # bf = self.TCR(bf)
        # print('bf_size6:\t', bf.size())
        # bf = bf.permute(0,2,1).contiguous()
        # print('bf_size7:\t', bf.size())
        # bf = bf.view(V, self.nr_boxes, self.nr_frames, -1)
        # print('bf_size8:\t', bf.size())
        ###########################################
        # exit(0)
        #################################################################################################
        # if self.joint:
        #     coord_feas = bf.permute(0, 2, 1, 3).contiguous()
        #     # print('coord_feas_size:\t', coord_feas.size())
        # else:
        #     #############################################################
        #     # spatial_message = bf.sum(dim=1, keepdim=True) # ([4, 1, 8, 512])
        #     # spatial_message = (spatial_message - bf) / (self.nr_boxes - 1) # ([4, 4, 8, 512])
        #     # bf_and_message = torch.cat([bf, spatial_message], dim=3) # ([4, 4, 8, 1024])
        #     # bf_spatial = self.spatial_node_fusion(
        #     #     bf_and_message.view(V*self.nr_boxes*self.nr_frames, -1)) # (128, 512)
        #     # bf_spatial = bf_spatial.view(
        #     #     V, self.nr_boxes, self.nr_frames, self.coord_feature_dim) # ([4, 4, 8, 512])
        #     # coord_feas = bf_spatial.permute(0, 2, 1, 3)  # (4, 8, 4, 512)
        #     ##############################################################
        #     coord_feas = bf.permute(0, 3, 1, 2).contiguous()
        #     # print('coord_feas_size1:\t', coord_feas.size())
        #     coord_feas = self.avgpool(coord_feas).view(coord_feas.size(0),-1)
            # print('coord_feas_size1:\t', coord_feas.size())
            ##############################################################
        # print('coord_feas_size:\t', coord_feas.size())

        # # bf_temporal_input = bf_spatial.view(
        # #     V, self.nr_boxes, self.nr_frames*self.coord_feature_dim) # (4, 4, 4096)
        # # box_features = self.box_feature_fusion(bf_temporal_input.view(V*self.nr_boxes, -1)) # ([16, 512])
        # # box_features = torch.mean(box_features.view(V, self.nr_boxes, -1), dim=1)  # ([4, 512]) 
        # #######################################################################################      
        # if self.joint:
        #     global_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
        #     # print('global_feas_size:\t', global_feas.size())
        # else:
        #     # global_feas = region_vis_feas
        #     global_feas = torch.cat([region_vis_feas, bc_embed], dim = -1)
        #     # print('global_feas_size:\t', global_feas.size())

        #     # global_feas = box_features

        # # output = global_feas 
        # # output = self.classifier(global_feas) 
        # # exit(0)
        # mix_feas = self.STIN(global_feas)
        # # print('mix_feas_size:\t', mix_feas.size())       
        # output = torch.cat([mix_feas, video_features], dim = -1)
        # # output = torch.cat([global_feas, video_features], dim = 1)

        # # output = mix_feas + video_features
        # # print('output_size:\t', output.size())
        # # output = mix_feas
        # # print('mix_feature_size:\t', mix_feature.size())       
        # # exit(0)
        # # ################################

        # # mix_feature = torch.cat([global_features, output], dim = -1)
        # # output = self.mix_feature_fusion(mix_feature)
        # # ################################
        # # print('output_size:\t', output.size())
        # # exit(0)
       
        return output
        #########################################################################################################################
    def STIN(self, input_feas, layer_num=2):
        V, T, P, D = input_feas.size()
        # print('input_feas_size1:\t', input_feas.size())
        input_feas = input_feas.permute(0, 2, 1, 3).contiguous() # [V x P x T x D]
        # print('input_feas_size2:\t', input_feas.size())
        spatial_mesg = input_feas.sum(dim=1, keepdim=True)  # (V x 1 x T x D)
        # print('spatial_mesg_size1:\t', spatial_mesg.size())
        # message passed should substract itself, and normalize to it as a single feature
        spatial_mesg = (spatial_mesg - input_feas) / (P - 1)  # [V x P x T x D]
        # print('spatial_mesg_size2:\t', spatial_mesg.size())
        in_spatial_feas = torch.cat([input_feas, spatial_mesg], dim=-1)  # (V x P x T x 2*D)
        # print('in_spatial_feas_size:\t', in_spatial_feas.size())
        # fuse the spatial feas into temporal ones
        # S_feas = self.spatial_node_fusion_list[:layer_num](in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        S_feas = self.spatial_node_fusion_list(in_spatial_feas.view(V*P*T, -1)) #[V*P*T x D_fusion]
        # print('S_feas_size1:\t', S_feas.size())
        # ############################################################
        temporal_feas = S_feas.view(V*P, -1) # [V*P x T*D_fusion]
        # print('temporal_feas_size:\t', temporal_feas.size())
        # node_feas = self.temporal_node_fusion_list[:layer_num](temporal_feas)  # (V*P x D_fusion)
        node_feas = self.temporal_node_fusion_list(temporal_feas)  # (V*P x D_fusion)
        # print('node_feas_size1:\t', node_feas.size())
        node_feas = node_feas.view(V, -1, node_feas.size()[-1])
        # print('node_feas_size2:\t', node_feas.size())
        node_feas = node_feas.view(V, -1)
        # print('node_feas_size3:\t', node_feas.size())
        ST_feas = self.node_fusion(node_feas)
        # print('ST_feas_size:\t', ST_feas.size())

        ############################################################
        # ST_feas = torch.mean(node_feas.view(V, P, -1), dim=1)  # (V x D_fusion)
        # print('ST_feas:\t', ST_feas.size())
        ############################################################
        # S_feas = S_feas.view(V, P, T, -1)
        # # print('S_feas_size2:\t', S_feas.size())
        # ST_feas = S_feas.mean(1).mean(1)
        # # print('ST_feas_size:\t', ST_feas.size())
        # exit(0)

        return ST_feas








