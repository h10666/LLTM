import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# from transformers import RobertaModel, RobertaTokenizerFast
from ops.temporal_shift import make_temporal_shift
from archs.clip_support.clip_prepare_model import clip_prepare_model
from archs.clip_support.clip_model import build_model
from archs.detr_support.detr_transformer import Transformer
from archs.detr_support.detr_position_encoding import build_position_encoding
from archs.model_helper import FeatureResizer
# from CLIP import clip

def build_detr_nets(hidden_dim, arch):
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
    position_embedding = build_position_encoding(hidden_dim)
    # resize
    if 'ViT' in arch:
        clip_dim = 512
    else:
        clip_dim = 1024
    resizer = FeatureResizer(input_feat_size=clip_dim,
            output_feat_size=512,
            dropout=0.1)
    return DTransformer, position_embedding, resizer
                    
def build_visual_backbone(name='RN50', shift=False, clip_pretrained_name = 'RN50', 
                          download_root = 'pretrained_weights', num_segments = 8):
    if name=='RN50':
        visual_backbone = torchvision.models.resnet50(pretrained=False)
        # print(visual_backbone)
        # exit(0)
        
        pthfile = r"%s/resnet50-0676ba61.pth" % (download_root) 
        # pthfile = r"%s/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth" % (download_root)       
              
        visual_backbone.load_state_dict(torch.load(pthfile))        
        feature_dim = getattr(visual_backbone, 'fc').in_features
        ##################### check state_dict #########################
        # print('org_visual_backbone:\t', visual_backbone)
        # state_dict = visual_backbone.state_dict()
        # orgres_items = []
        # for item in state_dict.keys():
        #     print('orgres_item:\t', item, '  ',  state_dict[item].size())
        #     orgres_items.append(item)
        # print('org_RN50_state_dict:\t', type(state_dict), len(state_dict))
        # print('org_RN50_keys:\t', type(orgres_items), orgres_items)
        # exit(0)       
        ################################################################
        if shift:
            print('Adding temporal shift...')
            make_temporal_shift(visual_backbone, num_segments)

        visual_backbone = torch.nn.Sequential(*(list(visual_backbone.children())[:-2]))
        # pthfile = r"%s/resnet50-0676ba61.pth" % (download_root)
        # visual_backbone.load_state_dict(torch.load(pthfile))
        # print(visual_backbone)
        # exit(0)
    elif name=='m_RN50':
        visual_backbone = get_clip(clip_pretrained_name).visual
        visual_backbone.attnpool = nn.Identity()
        # print(visual_backbone)
        # exit(0)
        feature_dim = 2048 #### get it from arch of visual_backbone???
        #################### check state_dict #########################
        # print('clip_RN50:\t', visual_backbone)
        # state_dict = visual_backbone.state_dict()
        # clip_items = []
        # for item in state_dict.keys():
        #     print('clip_item:\t', item, '  ',  state_dict[item].size())
        #     clip_items.append(item)
        # print('clip_RN50_state_dict:\t', type(state_dict), len(state_dict))
        # print('clip_keys:\t', type(clip_items), clip_items)
        # exit(0)
        # # print(state_dict.keys())                 
        ################################################################
        if shift:
            print('Adding temporal shift...')
            make_temporal_shift(visual_backbone, num_segments)
    elif name=='ViT':
        visual_backbone = get_clip(clip_pretrained_name).visual
        # print('visual_backbone:\t', visual_backbone)
        # exit(0)
        feature_dim = 768 #### get it from arch of visual_backbone??
    # print(visual_backbone)
    return visual_backbone, feature_dim

class TextEncoder(nn.Module):
    def __init__(self, clip_pretrained_name):
        super().__init__()
        self.clip = get_clip(clip_pretrained_name)
        self.transformer = self.clip.transformer
        self.token_embedding = self.clip.token_embedding
        self.positional_embedding = self.clip.positional_embedding
        self.ln_final = self.clip.ln_final
        self.text_projection = self.clip.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.out_proj.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # print('x1:\t', x.size())                
        x = x + self.positional_embedding.type(self.dtype)
        # print('x2:\t', x.size())                
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        # print('x3:\t', x.size())                
        
        x = self.transformer(x)
        # print('x4:\t', x.size())                
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        # print('x5:\t', x.size())                
        
        x = self.ln_final(x).type(self.dtype)
        # print('x6:\t', x.size())                
        
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # print('x7:\t', x.size())                
        
        return x

def build_text_backbone(clip_pretrained_name):
    return TextEncoder(clip_pretrained_name)

def get_clip(clip_pretrained_name):
    
    visio_mode = 'clip_based'
    visual_head_type = 'pool'
    frozen_blk = ''
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    state_dict = clip_prepare_model(name = clip_pretrained_name, device = device, jit = False, download_root ='pretrained_weights')
    clip_base_model = build_model(state_dict, visio_mode, visual_head_type, frozen_blk)

    return clip_base_model

from transformers import RobertaTokenizerFast, RobertaModel
from transformers import DistilBertTokenizerFast, DistilBertModel
# from transformers import CLIPTokenizer, CLIPTextModel
# from transformers import DistilRobertaTokenizerFast, DistilRobertaModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def tokenizer_build(x):
    if x == 'RoBERTa':
        PATH = 'pretrained_weights/roberta-base'
        tokenizer = RobertaTokenizerFast.from_pretrained(PATH)
    elif x == 'DistilBERT':   
        PATH = 'pretrained_weights/distilbert-base-uncased'
        tokenizer = DistilBertTokenizerFast.from_pretrained(PATH)
    # elif x == 'CLIPModel':
    #     # PATH = 'pretrained_weights/clip-vit-base-patch32'
    #     # tokenizer = CLIPTokenizer.from_pretrained(PATH)
    #     tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        
    elif x == 'DistilRoBERTa':
        PATH = 'pretrained_weights/distilroberta-base'
        tokenizer = RobertaTokenizerFast.from_pretrained(PATH)

        
           
    # print(tokenizer)
    return tokenizer

def RobertaModel_build(x):
    if x == 'RoBERTa':
        PATH = 'pretrained_weights/roberta-base'
        text_backbone = RobertaModel.from_pretrained(PATH)
    elif x == 'DistilBERT':
        PATH = 'pretrained_weights/distilbert-base-uncased'
        text_backbone = DistilBertModel.from_pretrained(PATH)
    # elif x == 'CLIPModel':
    #     # PATH = 'pretrained_weights/clip-vit-base-patch32'
    #     # text_backbone = CLIPTextModel.from_pretrained(PATH)
    #     text_backbone = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
    elif x == 'DistilRoBERTa':
        PATH = 'pretrained_weights/distilroberta-base'
        text_backbone = RobertaModel.from_pretrained(PATH)
    
    
    for p in text_backbone.parameters():
        p.requires_grad_(False)
    #     # print(p.requires_grad)
    return text_backbone