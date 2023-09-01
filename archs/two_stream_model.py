import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from transformers import RobertaModel, RobertaTokenizerFast
from ops.temporal_shift import make_temporal_shift
from archs.clip_support.clip_prepare_model import clip_prepare_model
from archs.clip_support.clip_model import build_model
from archs.detr_support.detr_transformer import Transformer
from archs.detr_support.detr_position_encoding import build_position_encoding
# from archs.text2embed import text2embed
from model_helper import Temporal_head, MLP, FeatureResizer
from typing import List
from opts import parser
from CLIP import clip
import argparse
from torch.cuda.amp import autocast

args, _ = parser.parse_known_args()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# --clip --xxx --xx
# --resnet --xx --xx
class Two_stream(nn.Module):
    def __init__(self, num_class, two_stream=True):
        super(Two_stream,self).__init__()
        self.two_stream = args.two_stream
        self.T = args.num_segments
        if args.two_stream:       
            self.visual_backbone, self.clip_base_model, feature_dim = build_backbone(args.arch)  # arch
            if not args.weights_from_clip:
                self.clip_base_model = nn.Identity()
            # self.caption_embed = self.clip_base_model.encode_text
        else:
            self.visual_backbone, feature_dim = build_backbone(args.arch)  # arch
        
        self.classifier= nn.Linear(512, num_class)
        self.hidden_dim = args.hidden_dim 
        self.temporal_head = Temporal_head(feature_dim, self.hidden_dim)

        # visual : resnet, clip (building)
        if self.two_stream:
            box_obj_numclass = 2
            self.DTransformer, self.position_embedding, self.resizer = build_detr_nets()
            self.coord_decoder = MLP(512, 512, 4, 3)
            self.class_embed = nn.Linear(512, box_obj_numclass+1)
            self.input_proj = nn.Conv2d(2048, 512, kernel_size=1)
            self.query_embed = nn.Embedding(8, 512)
            # self.text2embed = text2embed(text_encoder_type="roberta-base")
        
    def video_encode(self, video):
        B, TC, _, _ = video.size()
        video = video.view((B, TC//3, 3) + video.size()[-2:])
        video_input = video.view(-1, *video.size()[2:])
        # feature map
        video_fea_map = self.visual_backbone(video_input)  # <B*T, 2048, 7, 7>
        # temporal fusion       
        global_video_fea = self.get_global_video_fea(video_fea_map)  ## <B*T, 2048, 7, 7> ---> <B, C>
        return video_fea_map, global_video_fea
    
    def get_global_video_fea(self, video_fea_map):
        
        global_video_fea = self.temporal_head(video_fea_map)
        
        return global_video_fea
        
    def text_encode(self, caption_input):
        # use encode_text from clip_model
        # print('caption_input:\t', caption_input.size(), caption_input)      
        # caption_feas = self.caption_embed(caption_input)
        caption_feas = self.clip_base_model.encode_text(caption_input)
        return caption_feas

    def obj_det_branch(self, image_fea_map, caption_fea, position_embedding):
        
        bs, c, h, w = image_fea_map.shape
        src = self.input_proj(image_fea_map)
        src = src.flatten(2).permute(2, 0, 1)
        # device = src.device
        # print('src:\t', src.size())
        caption_fea = caption_fea.repeat(8, 1)
        with autocast():
            caption_fea = self.resizer(caption_fea)
        
        caption_fea = torch.unsqueeze(caption_fea, 0)

        src = torch.cat([src, caption_fea], dim = 0)

        pos_embed = position_embedding.flatten(2).permute(2, 0, 1)
        # print(pos_embed)
        pos_embed = torch.cat((torch.zeros(1, bs, self.hidden_dim, device=device), pos_embed))
               
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        mask = None
        
        hs = self.DTransformer(src, mask, query_embed, pos_embed)[0]

        dbox_class = self.class_embed(hs)
        dbox_pred = self.coord_decoder(hs).sigmoid()
        
        box_out = {'pred_logits': dbox_class[-1], 'pred_boxes': dbox_pred[-1]}
        return box_out

    def forward(self, video, caption_input, class_name_iput):
        # print('caption_input:\t', caption_input.size(), caption_input)
        video_fea_map, global_video_fea = self.video_encode(video)
        
        if self.two_stream:            

            caption_fea = self.text_encode(caption_input)
            # print('caption_fea:\t', caption_fea.size(), caption_fea)
            position_embedding = self.position_embedding(video_fea_map)
           
            box_out = self.obj_det_branch(video_fea_map, caption_fea, position_embedding)
        
        if self.two_stream:
            return global_video_fea, box_out
        else:
            return global_video_fea
        
def build_detr_nets():
    # pred transformer
    DTransformer = Transformer(
            d_model = 512,
            dropout=0.1,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=6,
            num_decoder_layers=6,
            return_intermediate_dec=True
        )
    # pos
    position_embedding = build_position_encoding(args)
    # resize
    if 'vit' in args.clip_vis_name:
        clip_dim = 512
    else:
        clip_dim = 1024
    resizer = FeatureResizer(input_feat_size=clip_dim,
            output_feat_size=512,
            dropout=0.1)
    return DTransformer, position_embedding, resizer
        
def build_backbone(model_name):
    ##################################################################################
    name = args.clip_vis_name
    jit = args.jit
    visio_mode = args.visio_mode
    visual_head_type = args.visual_head_type
    frozen_blk = args.frozen_blk
    download_root = args.download_root
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    state_dict = clip_prepare_model(name, device, jit, download_root)
    print("clip_RN50_state_dict:\t", type(state_dict), len(state_dict))
    #########################################
    clip_items = []
    for item in state_dict.keys():
        print('clip_item:\t', type(item), item)
        clip_items.append(item)
    print('clip_keys:\t', type(clip_items), clip_items)
    # # exit(0)
    # print(state_dict.keys())  
    clip_base_model = build_model(state_dict, visio_mode, visual_head_type, frozen_blk)
    print('clip_RN50:\t', clip_base_model.visual)
    ##################################################################################
    if not args.weights_from_clip:
        visual_base_model = torchvision.models.resnet50(pretrained = False)
        pthfile = r"%s/resnet50-0676ba61.pth" % (args.download_root)       
        visual_base_model.load_state_dict(torch.load(pthfile))        
        feature_dim = getattr(visual_base_model, 'fc').in_features
        ###################################################################
        print('org_RN50:\t', visual_base_model)
        state_dict=visual_base_model.state_dict()
        orgres_items = []
        for item in state_dict.keys():
            print('orgres_item:\t', type(item), item)
            orgres_items.append(item)
        print('orgres_keys:\t', type(orgres_items), orgres_items)
        print('org_RN50_state_dict:\t', type(state_dict), len(state_dict))
        ###################################################################
        if args.shift:
            print('Adding temporal shift...')
            make_temporal_shift(visual_base_model, args.num_segments,
                                args.shift_div, args.shift_place, args.temporal_pool)
    # exit(0)
    ##################################################################################   
    if not args.two_stream:      
        if not args.weights_from_clip:        
            visual_base_model = torch.nn.Sequential(*(list(visual_base_model.children())[:-2]))
            feature_dim = feature_dim
        else:        
            visual_base_model = clip_base_model.visual          
            if 'RN' in model_name and args.shift:
                print('Adding temporal shift...')
                make_temporal_shift(visual_base_model, args.num_segments,
                                    args.shift_div, args.shift_place, args.temporal_pool)            
            if 'vit' in model_name:
                feature_dim = 512
            else:
                feature_dim = 1024
        
        return visual_base_model, feature_dim
    else:
        if not args.weights_from_clip:
           visual_base_model = torch.nn.Sequential(*(list(visual_base_model.children())[:-2]))
           feature_dim = feature_dim
        #    return visual_base_model, clip_base_model, feature_dim
        #    caption_embed = clip_base_model.encode_text
        else:
            visual_base_model = clip_base_model.visual         
            if 'RN' in model_name and args.shift:
                print('Adding temporal shift...')
                make_temporal_shift(visual_base_model, args.num_segments,
                                    args.shift_div, args.shift_place, args.temporal_pool)
            # caption_embed = clip_base_model.encode_text
            if 'vit' in model_name:
                feature_dim = 512
            else:
                feature_dim = 1024
        return visual_base_model, clip_base_model, feature_dim
             