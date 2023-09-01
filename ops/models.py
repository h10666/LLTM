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
                 temporal_pool=False, non_local=False, clip_pretrained_name = 'RN50',
                 two_stream = True, joint = False, do_aug = False, do_attention = False):
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
        self.clip_pretrained_name = clip_pretrained_name
        self.two_stream = two_stream
        self.joint = joint
        self.do_KL = args.do_KL
        self.do_aug = do_aug
        self.do_aug_mix = args.do_aug_mix
        self.do_attention = do_attention
        self.do_frame = args.do_frame
        self.do_region = args.do_region
        self.do_coord_cate = args.do_coord_cate
        self.do_cate = args.do_cate
        self.do_SEattn = args.do_SEattn
        self.do_cross_attn = args.do_cross_attn
        self.cls_hidden_dim = args.cls_hidden_dim
        self.attn_input_dim = args.attn_input_dim
        self.num_layers = args.num_layers
        self.embed_way = args.embed_way
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
        
        self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class): #此函数的功能在于对已知的basemodel网络结构进行修改，微调最后一层（全连接层）的结构，成为适合该数据集输出的形式。
        # print(self.base_model.last_layer_name)
        # feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features # 获取网络最后一层的输入feature_map的通道数，存储于feature_dim中
        # if self.do_attention:
        #     feature_dim = 768
        if args.cls_hidden_dim == 1024:   
            feature_dim = 1024
        elif args.cls_hidden_dim == 2048:
            feature_dim = 2048
        elif args.do_coord_cate or args.cls_hidden_dim == 512:   
            feature_dim = 512
        elif args.cls_hidden_dim == 768:
            feature_dim = 768
         
        # print('feature_dim:\t', feature_dim)
        if self.dropout == 0: # 判断是否有dropout层，如果没有，则添加一个dropout层后再添加一个全连接层，否则直接连接全连接层。
            # print('######test1')
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else: 
            if 'i3d' in self.base_model_name:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=0))
            # elif self.base_model_name == 'STIN':
            #     pass
            else:
                setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            ##################### self.base_model = torch.nn.Sequential(*(list(self.base_model.children())[:-1])) ## added by HP ############DELETE########
            # setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，一般可以用来修改网络结构。以上述为例，
            # 输入包含三个值，分别是基础网络、要赋值的属性名、要赋的值，当这个setattr运行结束后，该属性就变成所赋的值。
            
            if isinstance(self.num_class, (list,)) and len(self.num_class) > 1:
                for a, i in enumerate(range(len(self.num_class))):
                    setattr(self, "new_fc%d"%a, nn.Linear(feature_dim, self.num_class[i]))
            else:
                self.new_fc = nn.Linear(feature_dim, num_class) # 全连接层的输入为feature_dim,输出为数据集的类别num_class
        # print('## new_fc1:\t', self.new_fc1)
        # print('## new_fc0:\t', self.new_fc0)

        std = 0.001
        if isinstance(self.num_class, (list,)) and len(self.num_class) > 1:
            if hasattr(self.new_fc1, 'weight'):
                normal_(self.new_fc1.weight, 0, std)
                constant_(self.new_fc1.bias, 0)
            if hasattr(self.new_fc0, 'weight'):
                normal_(self.new_fc0.weight, 0, std)
                constant_(self.new_fc0.bias, 0)
        else:
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
        
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

        # if 'clip' in base_model or 'RN' in base_model:
        #     from archs.My_clip import My_clip
        #     self.base_model = My_clip(self.num_class)
        #     self.base_model.last_layer_name = 'fc'

        #     self.base_model.avgpool = nn.AdaptiveAvgPool1d(1)
            
        # if 'clip' in base_model or 'RN' in base_model:
        #     from archs.two_stream_model import Two_stream
        #     self.base_model = Two_stream(self.num_class)
        #     self.base_model.last_layer_name = 'classifier'

        #     self.base_model.avgpool = nn.AdaptiveAvgPool1d(1)
        if 'ViT' in base_model or 'RN' in base_model:
            from archs.two_stream_new import Two_stream
            self.base_model = Two_stream(self.two_stream, self.joint, self.do_KL, self.base_model_name, self.num_class, self.num_segments, 
                                         self.is_shift, self.clip_pretrained_name, self.do_aug, self.do_aug_mix, self.do_attention,
                                         self.do_frame, self.do_region, self.do_coord_cate, self.do_cate, self.do_SEattn, self.do_cross_attn,
                                         self.cls_hidden_dim, self.attn_input_dim, self.num_layers, self.embed_way)
            self.base_model.last_layer_name = 'classifier'

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
            from other_files.resnet3d_xl import Net
            self.base_model = Net(self.num_class, extract_features=True, loss_type='softmax')
            self.base_model.last_layer_name = 'classifier'
            # self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_i3d':
            from other_files.model_lib import VideoRegionModel
            self.base_model = VideoRegionModel()
            self.base_model.last_layer_name = 'fc'
            # self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)
            
        # elif base_model == 'STIN':
        #     from other_files.model_lib import VideoModelGlobalCoordLatentNL
        #     self.base_model = VideoModelGlobalCoordLatentNL()
        #     self.base_model.last_layer_name = 'classifier'
        #     # self.input_size = 224
        #     # self.input_mean = [0.485, 0.456, 0.406]
        #     # self.input_std = [0.229, 0.224, 0.225]
        #     # self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
        #     # self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_tsm':
            from other_files.model_lib import VideoRegionModel_TSM
            self.base_model = VideoRegionModel_TSM()
            self.base_model.last_layer_name = 'fc'
            # self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif base_model == 'region_coord_tsm':
            from other_files.model_lib import VideoRegionAndCoordModel_TSM
            self.base_model = VideoRegionAndCoordModel_TSM()
            self.base_model.last_layer_name = 'fc'
            # self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]
            self.base_model.avgpool1 = nn.AdaptiveAvgPool1d(1)
            self.base_model.avgpool2 = nn.AdaptiveAvgPool2d(1)

        elif 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(True if self.pretrain == 'imagenet' else False)
            self.base_model = torchvision.models.resnet50(pretrained = False)
            pthfile = r"%s/resnet50-0676ba61.pth" % (args.download_root)       
            self.base_model.load_state_dict(torch.load(pthfile))
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            self.base_model.last_layer_name = 'fc'
            # self.input_size = 224
            # self.input_mean = [0.485, 0.456, 0.406]
            # self.input_std = [0.229, 0.224, 0.225]

            self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)


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
    # def forward(self, input, target, box_input, box_categories, no_reshape=False):
    def forward(self, input, box_input, box_categories, text_input, label_input, class_input, test_model, no_reshape=False):
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
                base_out = self.base_model(input, box_input, box_categories)
                # print('base_out:\t', base_out.size())
            elif 'resnet' in self.base_model_name:
                base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:])) # <BT, C>
                # print('base_out1:\t', base_out.size())
                
            elif 'ViT' in self.base_model_name or 'RN' in self.base_model_name:
                if args.two_stream:
                    base_out, text_features, hidden_video_features, class_features, logit_scale = self.base_model(input, text_input, class_input, test_model)                   
                    # base_out, box_out = self.base_model(input, text_input, class_name_input)
                elif args.do_KL:
                    base_out, global_tags_feas, labels_feas, logit_scale = self.base_model(input, box_input, box_categories, text_input, label_input, class_input, test_model)
                else:
                    base_out = self.base_model(input, box_input, box_categories, text_input, label_input, class_input, test_model)
                # print('base_out:\t', base_out.size())
                # base_out = self.base_model(input)

            if self.dropout > 0:
                # print('base_out1:\t', base_out.size())
                # video_features = base_out
                if isinstance(self.num_class, list):
                    base_out_verb = self.new_fc0(base_out)
                    base_out_noun = self.new_fc1(base_out)
                else:
                    base_out = self.new_fc(base_out)
                # print('base_out2:\t', base_out.size())
                
                # print(base_out.size())
                # exit(0)
                # if 'clip' and 'RN' not in self.base_model_name:
                #     base_out = self.new_fc(base_out)
                #     print('base_out2:\t', base_out.size())                   
                # else:
                #     base_out = base_out
            else:
                pass

            if not self.before_softmax:
                if isinstance(self.num_class, list):
                    base_out_verb = self.softmax(base_out_verb)
                    base_out_noun = self.softmax(base_out_noun)
                else:
                    base_out = self.softmax(base_out)

            if self.reshape:
                if self.base_model_name == 'region_i3d' or self.base_model_name == 'region_tsm' or self.base_model_name == 'region_coord_tsm' or 'ViT' in self.base_model_name or 'RN' in self.base_model_name:
                    if isinstance(self.num_class, list):
                        output_verb=base_out_verb
                        output_noun=base_out_noun
                        b, _ = output_verb.size()
                    else:
                        output = base_out
                        b, _ = output.size()
                    if args.do_aug:
                        # b, _ = output.size()
                        # print('output:\t', output.size())
                        if test_model and args.do_aug_mix:
                            if isinstance(self.num_class, list):
                                output_verb1 = output_verb[:b//5, :]
                                output_verb2 = output_verb[(b//5):(2*b//5), :]
                                output_verb3 = output_verb[(2*b//5):(3*b//5), :]
                                output_verb4 = output_verb[(3*b//5):(4*b//5), :]
                                output_verb5 = output_verb[(4*b//5):, :]
                                output_verb = (output_verb1 + 2*output_verb2 + output_verb3 + 2*output_verb4 + output_verb5)/5
                                output_noun1 = output_noun[:b//5, :]
                                output_noun2 = output_noun[(b//5):(2*b//5), :]
                                output_noun3 = output_noun[(2*b//5):(3*b//5), :]
                                output_noun4 = output_noun[(3*b//5):(4*b//5), :]
                                output_noun5 = output_noun[(4*b//5):, :]
                                output_noun = (output_noun1 + 2*output_noun2 + output_noun3 + 2*output_noun4 + output_noun5)/5
                            else:    
                                output1 = output[:b//5, :]
                                output2 = output[(b//5):(2*b//5), :]
                                output3 = output[(2*b//5):(3*b//5), :]
                                output4 = output[(3*b//5):(4*b//5), :]
                                output5 = output[(4*b//5):, :]
                                output = (output1 + 2*output2 + output3 + 2*output4 + output5)/5
                            if args.do_KL:
                                global_tags_feas1 = global_tags_feas[:b//5, :]
                                global_tags_feas2 = global_tags_feas[(b//5):(2*b//5), :]
                                global_tags_feas3 = global_tags_feas[(2*b//5):(3*b//5), :]
                                global_tags_feas4 = global_tags_feas[(3*b//5):(4*b//5), :]
                                global_tags_feas5 = global_tags_feas[(4*b//5):, :]
                                global_tags_feas = (global_tags_feas1 + 2*global_tags_feas2 + global_tags_feas3 + 2*global_tags_feas4 + global_tags_feas5)/5
                        elif test_model:
                            if isinstance(self.num_class, list):
                                output_verb1 = output_verb[:b//2, :]
                                output_verb2 = output_verb[b//2:, :]
                                output_verb=(output_verb1 + output_verb2)/2
                                output_noun1 = output_noun[:b//2, :]
                                output_noun2 = output_noun[b//2:, :]
                                output_noun=(output_noun1 + output_noun2)/2
                            else:
                                output1 = output[:b//2, :]
                                output2 = output[(b//2):, :]  
                                output = (output1 + output2)/2
                            if args.do_KL:
                                global_tags_feas1 = global_tags_feas[:b//2, :]
                                global_tags_feas2 = global_tags_feas[b//2:, :]
                                global_tags_feas = (global_tags_feas1+global_tags_feas2)/2
                        else:
                            pass   
                        
                    # print('output:\t', output.size())
                #     base_out = base_out.view((V, (T1//2) * nr_boxes) + base_out.size()[3:]) # [4, 16, 174] 错了！！！！

                    # print('output_size:\t', output.size())
                else:
                    if (self.is_shift and self.temporal_pool) or (self.base_model_name == 'i3d'): # T--->> T//2 for I3D
                        base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])
                    else:
                        if isinstance(self.num_class, list):
                            base_out_verb = base_out_verb.view((-1, self.num_segments) + base_out_verb.size()[1:])
                            base_out_noun = base_out_noun.view((-1, self.num_segments) + base_out_noun.size()[1:])
                        else:
                            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
                            # print('base_out3:\t', base_out.size())
                        

            if 'region' not in self.base_model_name and 'ViT' not in self.base_model_name and 'RN' not in self.base_model_name:
                if isinstance(self.num_class, list):
                    output_verb = self.consensus(base_out_verb)
                    output_verb = output_verb.squeeze(1)
                    output_noun = self.consensus(base_out_noun)
                    output_noun = output_noun.squeeze(1)
                else:
                    output = self.consensus(base_out)              
                    output = output.squeeze(1)
            # print('output:\t', output.size())
            # exit(0)
            if 'epic55' in args.dataset and args.do_KL:
                if isinstance(self.num_class, list):
                    return [output_verb, output_noun], global_tags_feas, labels_feas, logit_scale
                else:
                    return output, global_tags_feas, labels_feas, logit_scale
            elif 'epic55' in args.dataset:
                if isinstance(self.num_class, list):
                    return [output_verb, output_noun]
                else:
                    return output
            elif args.two_stream:
                # return output, box_out
                return output, text_features, hidden_video_features, class_features, logit_scale
            elif args.do_KL:
                return output, global_tags_feas, labels_feas, logit_scale
            else:
                return output







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
