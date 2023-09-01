# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

from opts import parser


args, _ = parser.parse_known_args()

# model.py的主要功能是对之后的训练模型进行准备。使用一些经典模型作为基础模型，如resnet101、BNInception等，并且在模型中即插即用式
# 的添加shift操作和non_local模块。针对不同的输入模块，对最后一层全连接层进行修改，得到我们所需的TSN网络模型。
class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8, img_feature_dim=256,
                 crop_num=1, partial_bn=True, print_spec=True, pretrain='imagenet',
                 is_shift=False, shift_div=8, shift_place='blockres', fc_lr5=False,
                 temporal_pool=False, non_local=False):
        super(TSN, self).__init__()
        # num_class是分类后的标签数
        
        self.modality = modality # 输入的模态（RGB、光流、RGB差异）
        self.num_segments = num_segments # 视频分割的段数
        self.reshape = True #
        self.num_class = num_class
        self.before_softmax = before_softmax # 布尔类型，判断是否进行softmax
        self.dropout = dropout #正则化
        self.crop_num = crop_num # 数据集修改的类别
        self.consensus_type = consensus_type # 聚合函数的选择
        self.img_feature_dim = img_feature_dim  # 图片的特征维度 the dimension of the CNN feature to represent each frame
        self.pretrain = pretrain # 布尔型，判断是否加载预训练的模型

        self.is_shift = is_shift #布尔型，判断是否加入shift操作
        self.shift_div = shift_div # 表示shift操作的比例倒数
        self.shift_place = shift_place # 判断标志，当为“blockres”表示嵌入了残差模块，当为“block”时表示嵌入了直链模块。
        self.base_model_name = base_model # base_model 基础模型，之后的TSN模型以此基础来修改，这里默认是resnet101
        self.fc_lr5 = fc_lr5 #学习率优化策略中对第五个全连接进行参数调整的判断标志。
        self.temporal_pool = temporal_pool # 表示是否在时间维度进行池化降维，相应的第2、3、4层num_segment减半。
        self.non_local = non_local # 布尔型，判断是否加入non_local模块。

        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length
        if print_spec:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout, self.img_feature_dim)))
        # print('dropout:\t', self.dropout)
        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class): #此函数的功能在于对已知的basemodel网络结构进行修改，微调最后一层（全连接层）的结构，成为适合该数据集输出的形式。
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features # 获取网络最后一层的输入feature_map的通道数，存储于feature_dim中
        
        # print('feature_dim:\t', feature_dim)
        if self.dropout == 0: # 判断是否有dropout层，如果没有，则添加一个dropout层后再添加一个全连接层，否则直接连接全连接层。
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else: 
            if 'i3d' in self.base_model_name:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=0))
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            ##################### self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1])) ## added by HP ############DELETE########
            # setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，一般可以用来修改网络结构。以上述为例，
            # 输入包含三个值，分别是基础网络、要赋值的属性名、要赋的值，当这个setattr运行结束后，该属性就变成所赋的值。
            self.new_fc = nn.Linear(feature_dim, num_class) # 全连接层的输入为feature_dim,输出为数据集的类别num_class

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
            # getattr获得属性值 
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std) # 对全连接层的网络权重，进行0均值且指定标准差的初始化操作
                constant_(self.new_fc.bias, 0) # 对偏差bias初始化为0
        return feature_dim

    def _prepare_base_model(self, base_model): # 此函数功能在于选择网络结构模型时，要对其是否进行shift操作和添加non_local网络模块，并且对输入数据集进行预处理。
        print('=> base model: {}'.format(base_model))

        if 'clip' in base_model or 'RN' in base_model:
            from archs.My_clip import My_clip
            self.base_model = My_clip(self.num_class)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool = nn.AdaptiveAvgPool1d(1)

        # elif 'clip_vit32' in base_model:
        #     from archs.clip_vit32 import clip_vit32
        #     self.base_model = clip_vit32(self.num_class)
        #     self.base_model.last_layer_name = 'fc'
        #     self.input_size = 224
        #     self.input_mean = [0.485, 0.456, 0.406]
        #     self.input_std = [0.229, 0.224, 0.225]
        #     self.base_model.avgpool = nn.AdaptiveAvgPool1d(1)

        # elif 'clip_RN2TSM' in base_model:
        #     from archs.clip_RN2TSM import clip_RN2TSM
        #     self.base_model = clip_RN2TSM(self.num_class)
        #     self.base_model.last_layer_name = 'fc'
        #     self.input_size = 224
        #     self.input_mean = [0.485, 0.456, 0.406]
        #     self.input_std = [0.229, 0.224, 0.225]
        #     self.base_model.avgpool = nn.AdaptiveAvgPool1d(1)
      
        elif base_model == 'i3d':
            from archs.resnet3d_xl import Net
            self.base_model = Net(self.num_class, extract_features=True, loss_type='softmax')
            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_i3d':
            from archs.model_lib import VideoRegionModel
            self.base_model = VideoRegionModel()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_tsm':
            from archs.model_lib import VideoRegionModel_TSM
            self.base_model = VideoRegionModel_TSM()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_coord_tsm':
            from archs.model_lib import VideoRegionAndCoordModel_TSM4
            self.base_model = VideoRegionAndCoordModel_TSM4()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif 'resnet' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)

            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'mobilenetv2':
            from archs.mobilenet_v2 import mobilenet_v2, InvertedResidual
            self.base_model = mobilenet_v2(True if self.pretrain == 'imagenet' else False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)
            if self.is_shift:
                from ops.temporal_shift import TemporalShift
                for m in self.base_model.modules():
                    if isinstance(m, InvertedResidual) and len(m.conv) == 8 and m.use_res_connect:
                        if self.print_spec:
                            print('Adding temporal shift... {}'.format(m.use_res_connect))
                        m.conv[0] = TemporalShift(m.conv[0], n_segment=self.num_segments, n_div=self.shift_div)
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            from archs.bn_inception import bninception
            self.base_model = bninception(pretrained=self.pretrain)
            self.input_size = self.base_model.input_size
            self.input_mean = self.base_model.mean
            self.input_std = self.base_model.std
            self.base_model.last_layer_name = 'fc'
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.Embedding):
                pass
            elif isinstance(m, torch.nn.modules.normalization.LayerNorm):
                pass
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    # def forward(self, global_img_input, local_img_input, box_input, video_label, no_reshape=False):
    # def forward(self, input, target, box_input, box_categories, texture, no_reshape=False):
    def forward(self, input, text_input, class_name_input, no_reshape=False):
        V, TC, _, _ = input.size()
        # _, T1, nr_boxes, _ = box_input.size()

        if not no_reshape:
            sample_len = 3
            if self.base_model_name == 'i3d':
                i3d_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:])
                i3d_input = i3d_input.permute(0,2,1,3,4)
                _, base_out = self.base_model(i3d_input)
                base_out = self.base_model.avgpool(base_out) 
                base_out = base_out.transpose(1,2).squeeze().contiguous()
                assert base_out.size(1) == self.num_segments//2
                base_out = base_out.view(-1, base_out.size(-1))

            elif self.base_model_name == 'region_i3d':
                base_out = self.base_model(input, target, box_input, box_categories) # [V, T, P, D_vis] ([4, 4, 4, 512])
                # print('base_out_size1:\t', base_out.size())
                base_out = base_out.permute(0, 3, 1, 2).contiguous() # [V, D_vis, T, P] ([4, 512, 4, 4])
                # base_out = base_out.permute(0, 1, 3, 2).contiguous() # [V, T, D_vis, P] ([4, 4, 512, 4])

                # V2, T2, _, P = base_out.size()
                # base_out = base_out.view(V2, -1, P)

                base_out = self.base_model.avgpool2(base_out).view(base_out.size(0),-1)

                # base_out = self.native_temp(base_out)
                # print('base_out_size2:\t', base_out.size())
                # base_out = self.base_model.avgpool(base_out).view(base_out.size(0),-1)

                # B, D_vis, T2, _ = base_out.size()
                # # print('B, D_vis, T2:\t', B, D_vis, T2)
                # base_out = base_out.view((B, D_vis * T2) + base_out.size()[3:]) # ([4, 2048, 4])
                # # print('base_out_size3:\t', base_out.size())
                # avgpool = nn.AdaptiveAvgPool1d(1)
                # base_out = avgpool(base_out) # ([4, 2048, 1])
                # # print('base_out_size4:\t', base_out.size()) 
                # base_out = base_out.view((B, D_vis, T2) + base_out.size()[2:]) # ([4, 512, 4, 1] [V, D_vis, T, 1])
                # base_out = base_out.permute(0, 2, 3, 1) # ([4, 4, 1, 512])
                # # base_out = base_out.view(base_out.size(0),-1) # [] 
                # # print('base_out_size5:\t', base_out.size())
                # base_out = base_out.squeeze().contiguous() # [4, 4, 512]
                # # print('base_out_size6:\t', base_out.size())
            
            elif self.base_model_name == 'region_tsm':
                base_out = self.base_model(input, target, box_input, box_categories) # (4, 8, 4, 512)
                base_out = base_out.permute(0, 3, 1, 2).contiguous() # (4, 512, 8, 4)
                # base_out = base_out.view(base_out.size(0), -1, base_out.size(-1)) # # (4, 512*8, 4)
                # base_out = self.base_model.avgpool1(base_out) # (4, 4096, 1)                
                base_out = self.base_model.avgpool(base_out).view(base_out.size(0),-1)
            elif self.base_model_name == 'region_coord_tsm':
                base_out = self.base_model(input, target, box_input, box_categories, texture)

            elif 'resnet' in self.base_model_name:
                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
                
            elif 'clip' in self.base_model_name or 'RN' in self.base_model_name:
                base_out = self.base_model(input, text_input, class_name_input)
                # print('base_out:\t', base_out.size())
                # base_out = self.base_model(input)


            if self.dropout > 0:
                base_out = self.new_fc(base_out) 
                # print('base_out_size:\t', base_out.size())
                # exit(0)

            if not self.before_softmax:
                base_out = self.softmax(base_out)

            if self.reshape:
                if self.base_model_name == 'region_i3d' or self.base_model_name == 'region_tsm' or self.base_model_name == 'region_coord_tsm' or 'clip' in self.base_model_name or 'RN' in self.base_model_name:
                    output = base_out
                #     base_out = base_out.view((V, (T1//2) * nr_boxes) + base_out.size()[3:]) # [4, 16, 174] 错了！！！！

                    # print('output_size:\t', output.size())
                else:
                    if (self.is_shift and self.temporal_pool) or (self.base_model_name == 'i3d'): # T--->> T//2 for I3D
                        base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                    else:
                        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            if 'region' and 'clip' and 'RN' not in self.base_model_name:
                output = self.consensus(base_out)
                output = output.squeeze(1)
            # print('output_size:\t', output.size())
            return output
        #####################################################################
        # V, TC, _, _ = input.size()
        # _, T, nr_boxes, _ = box_input.size()

        # if not no_reshape:
        #     sample_len = 3
        #     if self.vis_info:
        #         if 'i3d' in self.base_model_name:
        #             i3d_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # [V, T, C, H, W]
        #             i3d_input = i3d_input.permute(0,2,1,3,4) # [V, C, T, H, W]
        #             _, org_feas = self.base_model(i3d_input) # [V, 2048, T//2, 14, 14]                 
        #             T = org_feas.size(2)                   
        #             conv_fea_maps = self.conv(org_feas)
        #             # print('conv_fea_maps_size1:\t', conv_fea_maps.size()) # [4, 256, 4, 14, 14]
        #             global_vis_feas = self.avgpool2d(conv_fea_maps).view(V, -1) 
        #             conv_fea_maps = conv_fea_maps.permute(0,2,1,3,4).contiguous() # [V x T x d x 14 x 14]                    
        #             conv_fea_maps = conv_fea_maps.view(-1, *conv_fea_maps.size()[2:]) # [V * T x d x 14 x 14]
        #             box_tensor = box_input[:, ::(self.num_segments//T), :, :].contiguous() # [V, T, P, 4]                   
        #             box_tensor = box_tensor.view(-1, *box_tensor.size()[2:]) # [V*T/2, P, 4]
        #             ### convert tensor to list, and [cx, cy, w, h] --> [x1, y1, x2, y2]
        #             boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
        #             img_size = i3d_input.size()[-2:]
        #             ### get region feas via RoIAlign
        #             region_vis_feas = build_region_feas(conv_fea_maps, 
        #                                                 boxes_list, self.crop_size1, img_size) #[V*T*P x C], where C= 3*3*d
        #             region_vis_feas = region_vis_feas.view(V, T//2, nr_boxes, region_vis_feas.size(-1)) #[V x T x P x C]
        #             ### reduce dim of region_vis_feas
        #             region_vis_feas = self.region_vis_embed(region_vis_feas) #[V x T x P x D_vis]
        #         # else:
        #             # base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:])) 

        #     if self.dropout > 0:
        #         base_out = self.new_fc(region_vis_feas)

        #     if not self.before_softmax:
        #         base_out = self.softmax(base_out)

        #     # print('before consensus base_out size:\t', base_out.size()) # B, T//2, NUM_CLASS
        #     base_out = base_out.view((V, (T//2) * nr_boxes) + base_out.size()[3:])
        #     output = self.consensus(base_out)

        #     # print('consensus output size:\t', output.size()) # B, T*3, NUM_CLASS
        #     return output.squeeze(1)  ## [V * NUM_CLASS]
        ######################################################################################################
        # input: V, T*C, H, W
        # V, TC , _, _ = input.size() # added by HP       
        # if not no_reshape:
        #     sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        #     # print(input.view((BT, sample_len, CT // sample_len)+input.size()[-2:]))
        #     if self.modality == 'RGBDiff':
        #         sample_len = 3 * self.new_length
        #         input = self._get_diff(input)

        #     if 'i3d' in self.base_model_name:
        #         # print('base_model input size:\t', input.size()) # V, T*C, H, W ###### C = sample_len
        #         i3d_input = input.view((V, TC//sample_len, sample_len) + input.size()[-2:]) # V, T, C, H, W
        #         # print('i3d_input_size:\t', i3d_input.size())
        #         i3d_input = i3d_input.permute(0,2,1,3,4) # V, C, T, H, W
        #         _, base_out = self.base_model(i3d_input) ## added by HP  input: [V x C x T x 224 x 224] -->> [V x C x T/2 x h x w]
        #         base_out = self.base_model.avgpool(base_out) # [V x C x T/2]
        #         base_out = base_out.transpose(1,2).squeeze().contiguous() # [V x T/2 x C]
        #         # print('base_model output size:\t', base_out.size()) # [V x T/2 x C]
        #         # print(base_out.size(1), self.num_segments//2)
        #         assert base_out.size(1) == self.num_segments//2 #T--->> T//2 for I3D
        #         base_out = base_out.view(-1, base_out.size(-1)) # [V*T/2 x C]
                
        #     else:
        #         base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))          
        # else:
        #     base_out = self.base_model(input)
        
        # # print(type(base_out))

        # if self.dropout > 0:
        #     base_out = self.new_fc(base_out)
        #     # print('new_fc output size:\t', base_out.size()) # B, T//2, NUM_CLASS

        # if not self.before_softmax:
        #     base_out = self.softmax(base_out)

        # if self.reshape:  # note
        #     if (self.is_shift and self.temporal_pool) or ('i3d' in self.base_model_name): # T--->> T//2 for I3D
        #         base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
        #     else:
        #         base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        #     # print('before consensus base_out size:\t', base_out.size()) # B, T//2, NUM_CLASS
        #     output = self.consensus(base_out)
        #     # print('consensus output size:\t', output.size()) # B, T*3, NUM_CLASS
        #     return output.squeeze(1)
        ########################################################################################################

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_model_name == 'BNInception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            base_model.load_state_dict(sd)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
                # return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75])])
                # return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1])])

        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])  
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
