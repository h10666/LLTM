import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast
from ops.utils import box_to_normalized, build_region_feas
from torch.nn.modules.dropout import Dropout
from archs.model_helper import Temporal_head, MLP
from archs.cross_attention import CrossAttention1, CrossAttention2
from archs.model_builder import build_text_backbone, build_visual_backbone, build_detr_nets, RobertaModel_build
from other_files.SEAttention import SEAttention


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# --clip --xxx --xx
# --resnet --xx --xx
class Two_stream(nn.Module):
    def __init__(self, two_stream, joint, do_KL, arch, num_class, num_segments,
                 is_shift, clip_pretrained_name, do_aug, do_aug_mix, 
                 do_attention, do_frame, do_region, do_coord_cate, do_cate, do_SEattn, do_cross_attn, 
                 cls_hidden_dim, attn_input_dim, num_layers, embed_way):
        super(Two_stream, self).__init__()
        
        self.two_stream = two_stream
        self.joint = joint
        self.do_KL = do_KL
        self.T = num_segments
        self.is_shift = is_shift
        self.num_class = num_class
        self.clip_pretrained_name = clip_pretrained_name
        self.do_aug = do_aug
        self.do_aug_mix = do_aug_mix
        self.do_attention = do_attention
        self.do_frame = do_frame
        self.do_region = do_region
        self.do_coord_cate = do_coord_cate
        self.do_cate = do_cate
        self.do_SEattn = do_SEattn
        self.do_cross_attn = do_cross_attn
        self.hidden_dim = 512
        self.coord_feature_dim = 512
        self.cls_hidden_dim = cls_hidden_dim 
        self.attn_input_dim = attn_input_dim 
        self.num_layers = num_layers
        self.embed_way = embed_way
        self.download_root = 'pretrained_weights'
        
         
        if self.two_stream:           
            self.visual_backbone, feature_dim = build_visual_backbone(arch, self.is_shift, self.clip_pretrained_name, self.download_root, num_segments)
            self.text_backbone = build_text_backbone(self.clip_pretrained_name)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.text_backbone.clip = nn.Identity()
            # self.DTransformer, self.position_embedding, self.resizer = build_detr_nets(self.hidden_dim, arch)
            # box_obj_numclass = 2
            # self.coord_decoder = MLP(512, 512, 4, 3)
            # self.class_embed = nn.Linear(512, box_obj_numclass+1)
            # self.input_proj = nn.Conv2d(2048, 512, kernel_size=1)
            # self.query_embed = nn.Embedding(8, 512)
        else:
            self.visual_backbone, feature_dim = build_visual_backbone(arch, self.is_shift, self.clip_pretrained_name, self.download_root, num_segments)
            if self.do_coord_cate or self.do_region:
                self.conv = nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=1)           
                self.crop_size1 = [3, 3]
                # self.nr_boxes = 4
                self.nr_boxes = 6

                ## region_embed_layer
                self.region_vis_embed = nn.Sequential(
                        nn.Linear(self.hidden_dim * self.crop_size1[0] * self.crop_size1[1], feature_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5)
                    )
                ## category embed
                categ_dim = self.hidden_dim // 2
                self.category_embed_layer = nn.Embedding(
                        3, categ_dim, padding_idx=0, scale_grad_by_freq=True)
                ## coord to feature layer
                self.coord_to_feature = nn.Sequential(
                        nn.Linear(4, self.coord_feature_dim//2, bias=False),
                        nn.BatchNorm1d(self.coord_feature_dim//2),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.coord_feature_dim//2,
                                self.coord_feature_dim, bias=False),
                        nn.BatchNorm1d(self.coord_feature_dim),
                        nn.ReLU(inplace=True)
                    )
                ## coord and category fusion layer
                # if self.do_cate:
                #    cc_fusion_dim = self.coord_feature_dim
                cc_fusion_dim = self.coord_feature_dim+self.coord_feature_dim // 2
                self.coord_category_fusion = nn.Sequential(
                        nn.Linear(cc_fusion_dim, self.coord_feature_dim, bias=False),
                        nn.BatchNorm1d(self.coord_feature_dim),
                        nn.ReLU(inplace=True)
                    )
                ## spatial fusion layer
                if self.do_region and self.do_coord_cate:
                    spatial_fea_dim = self.hidden_dim * 10
                    spatial_hidden_dim = self.hidden_dim * 4
                elif self.do_region:
                    spatial_fea_dim = self.hidden_dim * 8
                    spatial_hidden_dim = self.hidden_dim * 4
                elif self.do_coord_cate:
                    spatial_fea_dim = self.hidden_dim * 2
                    spatial_hidden_dim = self.hidden_dim
                self.spatial_node_fusion_list = nn.Sequential(
                        nn.Linear(spatial_fea_dim, spatial_hidden_dim, bias=False),
                        nn.BatchNorm1d(spatial_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(spatial_hidden_dim, spatial_hidden_dim, bias=False),
                        nn.BatchNorm1d(spatial_hidden_dim),
                        nn.ReLU(inplace=True)
                        )
                ## temporal fusion layer
                if self.do_region:
                    temp_fea_dim = self.T * self.hidden_dim * 4
                    temp_hidden_dim = self.hidden_dim * 4
                elif self.do_coord_cate:
                    temp_fea_dim = self.T * self.hidden_dim
                    temp_hidden_dim = self.hidden_dim
                self.temporal_node_fusion_list = nn.Sequential(
                        nn.Linear(temp_fea_dim, temp_hidden_dim, bias=False),
                        nn.BatchNorm1d(temp_hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(temp_hidden_dim, temp_hidden_dim, bias=False),
                        nn.BatchNorm1d(temp_hidden_dim),
                        nn.ReLU(inplace=True)
                        )
                ## final fusion
                if self.do_region :
                    node_input_dim = feature_dim * 4
                    node_hidden_dim = feature_dim * 2
                    # node_hidden_dim = feature_dim + 1024
                    
                    node_out_dim = feature_dim
                    self.node_fusion = nn.Sequential(
                            nn.Linear(node_input_dim, node_hidden_dim, bias=False),
                            nn.BatchNorm1d(node_hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(node_hidden_dim, node_out_dim, bias=False),
                            nn.BatchNorm1d(node_out_dim),
                            nn.ReLU(inplace=True)
                            )
                elif self.do_coord_cate:
                    node_input_dim = self.hidden_dim * self.nr_boxes
                    # node_hidden_dim = self.hidden_dim * 2
                    node_out_dim = self.hidden_dim
                    self.node_fusion = nn.Sequential(
                                nn.Linear(node_input_dim, node_out_dim, bias=False),
                                nn.BatchNorm1d(node_out_dim),
                                nn.ReLU(inplace=True)
                                )
            if self.joint:
                self.text_backbone = RobertaModel_build(self.embed_way)
                # print(self.text_backbone)
                
                # exit(0)
                # self.text_backbone = build_text_backbone(self.clip_pretrained_name)
                # self.text_backbone.clip = nn.Identity()
                if self.embed_way == 'CLIPModel':
                    cls_input_dim = feature_dim + 512
                else:
                    cls_input_dim = feature_dim + 768
                if self.do_frame and self.do_region:
                    cls_input_dim = feature_dim*2+768
                    self.MLP = nn.Sequential(
                                    nn.Linear(cls_input_dim, self.cls_hidden_dim, bias=False),
                                    nn.BatchNorm1d(self.cls_hidden_dim),
                                    nn.ReLU(inplace=True)
                                )
                elif self.do_frame and self.do_coord_cate:
                    cls_input_dim = feature_dim+self.coord_feature_dim+768
                    self.MLP = nn.Sequential(
                                    nn.Linear(cls_input_dim, self.cls_hidden_dim, bias=False),
                                    nn.BatchNorm1d(self.cls_hidden_dim),
                                    nn.ReLU(inplace=True)
                                )
                else:
                    self.MLP = nn.Sequential(
                                        nn.Linear(cls_input_dim, self.cls_hidden_dim, bias=False),
                                        nn.BatchNorm1d(self.cls_hidden_dim),
                                        nn.ReLU(inplace=True)
                                    )
                # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                   
                if self.do_KL or self.do_cross_attn:
                    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                    if self.do_frame and self.do_coord_cate:
                        in_dim = 3328
                    else:
                        in_dim = 2816
                    self.dim_trans = nn.Sequential(
                                            # nn.Linear(4096, 2048, bias=False),
                                            # nn.BatchNorm1d(2048),
                                            # nn.ReLU(inplace=True),
                                            nn.Linear(in_dim, 768, bias=False),
                                            nn.BatchNorm1d(768),
                                            nn.ReLU(inplace=True)
                                            )
                if self.do_attention:
                    encoder_layer = nn.TransformerEncoderLayer(d_model=self.attn_input_dim, nhead=8)
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                    self.cls_token = nn.Parameter(torch.randn(1, 1, self.attn_input_dim))
                    self.temporal_embed = nn.Parameter(torch.randn(1, self.T + 1, self.attn_input_dim))
                    self.attn_drop = nn.Dropout(0.3) 
                    self.vis_reduce = self.vis_reduction_layer(2048, self.attn_input_dim)
                    self.text_reduce = self.text_reduction_layer(768, self.attn_input_dim)
                    # self.CrossAttention_v2t = CrossAttention1(dim = 768, num_heads=8, qkv_bias=False, qk_scale=None, 
                    #                                     attn_drop=0.5, proj_drop=0.3, initialize='random')
                    # self.CrossAttention_t2v = CrossAttention1(dim = 2048, num_heads=8, qkv_bias=False, qk_scale=None, 
                                                        # attn_drop=0.5, proj_drop=0.3, initialize='random')
                    # self.CrossAttention = CrossAttention2(dim = 2816, heads = 8, dim_head = 64, dropout = 0.5)
                if self.do_cross_attn:
                    self.dim_reduce = nn.Sequential(
                                    nn.Linear(2048, 768, bias=False),
                                    nn.BatchNorm1d(768),
                                    nn.ReLU(inplace=True)
                                    )
                    self.dim_increase = nn.Sequential(
                                    nn.Linear(768, 2048, bias=False),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(inplace=True)
                                    )
                    self.CrossAttention_v2t = CrossAttention1(dim = 768, num_heads=8, qkv_bias=False, qk_scale=None, 
                                                        attn_drop=0.5, proj_drop=0.3, initialize='random')
                    self.CrossAttention_t2v = CrossAttention1(dim = 2048, num_heads=8, qkv_bias=False, qk_scale=None, 
                                                        attn_drop=0.5, proj_drop=0.3, initialize='random')
                if self.do_SEattn:
                    self.SEattn = SEAttention()
                    self.avgpool = nn.AdaptiveAvgPool1d(1)
                    self.vis_reduce = self.vis_reduction_layer(2048, self.attn_input_dim)
                    self.text_reduce = self.text_reduction_layer(768, self.attn_input_dim)                   
            else:
                if self.do_frame and self.do_coord_cate:
                    cls_input_dim = feature_dim+512
                else:
                    cls_input_dim = feature_dim
                self.MLP = nn.Sequential(
                                    nn.Linear(cls_input_dim, self.cls_hidden_dim, bias=False),
                                    nn.BatchNorm1d(self.cls_hidden_dim),
                                    nn.ReLU(inplace=True)
                                )                
        self.temporal_head = Temporal_head(feature_dim, self.hidden_dim)
        if self.joint:
            if self.do_attention:
                feature_dim = 768
                self.classifier= nn.Linear(feature_dim, self.num_class)
            else:
                try:
                    self.classifier= nn.Linear(feature_dim, self.num_class)
                except:
                    self.classifier0=nn.Linear(feature_dim, self.num_class[0])
                    self.classifier1=nn.Linear(feature_dim, self.num_class[1])

        else:
            try:
                self.classifier= nn.Linear(feature_dim, self.num_class)
            except:
                self.classifier0=nn.Linear(feature_dim, self.num_class[0])
                self.classifier1=nn.Linear(feature_dim, self.num_class[1])
            
        
    def video_encode(self, video, test_model):
        B, TC, _, _ = video.size()
        video = video.view((B, TC//3, 3) + video.size()[-2:])
        # print('video:\t', video.size())
        # exit(0)
        video_input = video.view(-1, *video.size()[2:])
        # feature map
        video_fea_map = self.visual_backbone(video_input)  # <B*T, 2048, 7, 7>
        # print('video_fea_map:\t', video_fea_map.size())
        # print('video_fea_map:\t', video_fea_map)
        
        # exit(0)
        # temporal fusion       
        global_video_fea, frame_feas, hidden_video_features = self.get_global_video_fea(video_fea_map, test_model)  ## <B*T, 2048, 7, 7> ---> <B, C>
        return video_fea_map, global_video_fea, frame_feas, hidden_video_features
    
    def get_global_video_fea(self, video_fea_map, test_model):
        
        global_video_fea, frame_feas, hidden_video_features = self.temporal_head(video_fea_map, test_model)
        # print(global_video_fea.size(), hidden_video_features.size())
        # exit(0)
        
        return global_video_fea, frame_feas, hidden_video_features

    def obj_det_branch(self, image_fea_map, caption_fea, position_embedding):
        # print('image_fea_map:\t', image_fea_map.size())
        # print('caption_fea1:\t', caption_fea.size())
        # print('position_embedding1:\t', position_embedding.size())
              
        bs, c, h, w = image_fea_map.shape
        src = self.input_proj(image_fea_map)
        # print('src1:\t', src.size())       
        src = src.flatten(2).permute(2, 0, 1)
        # print('src2:\t', src.size())       
        # device = src.device
        caption_fea = caption_fea.repeat(8, 1)
        # print('caption_fea2:\t', caption_fea.size())
        
        with autocast():
            caption_fea = self.resizer(caption_fea)
        # print('caption_fea3:\t', caption_fea.size())        
        caption_fea = torch.unsqueeze(caption_fea, 0)
        # print('caption_fea4:\t', caption_fea.size())                
        src = torch.cat([src, caption_fea], dim = 0)
        # print('src3:\t', src.size())              
        pos_embed = position_embedding.flatten(2).permute(2, 0, 1)
        # print('pos_embed1:\t', pos_embed.size())                      
        pos_embed = torch.cat((torch.zeros(1, bs, self.hidden_dim, device=device), pos_embed))
        # print('pos_embed2:\t', pos_embed.size())                                          
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # print('query_embed:\t', query_embed.size())                                                 
        mask = None       
        hs = self.DTransformer(src, mask, query_embed, pos_embed)[0]
        # print('hs:\t', query_embed.size())                                                         
        dbox_class = self.class_embed(hs)
        dbox_pred = self.coord_decoder(hs).sigmoid()       
        box_out = {'pred_logits': dbox_class[-1], 'pred_boxes': dbox_pred[-1]}
        return box_out

    def forward(self, video, box_input, box_categories, caption_input, label_input, class_iput, test_model):
        
        # print('video:\t', video.size())
        # print('video:\t', video)
        # print('box_input1:\t', box_input.size())
        video_fea_map, global_video_fea, frame_feas, hidden_video_features = self.video_encode(video, test_model)
        # print('hidden_video_features:\t', hidden_video_features.size())
        # print('video_fea_map:\t', video_fea_map.size())
        # print('frame_feas:\t', frame_feas.size())
        # exit(0)
        # print('global_video_fea:\t', global_video_fea.size())
        # print('global_video_fea:\t', global_video_fea)
        
        
        if self.do_region or self.do_coord_cate:
            rcc_feas = self.get_rcc_feas(video, video_fea_map, box_input, box_categories, test_model)
            # print('rcc_feas:\t', rcc_feas.size())
            
            if self.do_aug and test_model:
                if self.do_aug_mix:
                    bt = rcc_feas.size(1)
                    rcc_feas1 = rcc_feas[:,:bt//5, :, :]
                    rcc_output1 = self.STIN(rcc_feas1)
                    # print('rcc_output1:\t', rcc_output1.size())
                    rcc_feas2 = rcc_feas[:,(bt//5):(2*bt//5), :, :]               
                    rcc_output2 = self.STIN(rcc_feas2)
                    # print('rcc_output2:\t', rcc_output2.size())
                    rcc_feas3 = rcc_feas[:,(2*bt//5):(3*bt//5), :, :]               
                    rcc_output3 = self.STIN(rcc_feas3)
                    # print('rcc_output3:\t', rcc_output3.size())
                    
                    rcc_feas4 = rcc_feas[:,(3*bt//5):(4*bt//5), :, :]               
                    rcc_output4 = self.STIN(rcc_feas4)
                    # print('rcc_output4:\t', rcc_output4.size())
                    
                    rcc_feas5 = rcc_feas[:,(4*bt//5):, :, :]               
                    rcc_output5 = self.STIN(rcc_feas5)
                    # print('rcc_output5:\t', rcc_output5.size())
                    
                    rcc_output = torch.cat([rcc_output1, rcc_output2, rcc_output3, rcc_output4, rcc_output5], dim = 0)
                    # print('rcc_output:\t', rcc_output.size())
                else:
                    bt = rcc_feas.size(1)
                    rcc_feas1 = rcc_feas[:,:bt//2, :, :]
                    rcc_output1 = self.STIN(rcc_feas1)
                    # print('rcc_output1:\t', rcc_output1.size())
                    rcc_feas2 = rcc_feas[:,(bt//2):, :, :]               
                    rcc_output2 = self.STIN(rcc_feas2)
                    rcc_output = torch.cat([rcc_output1, rcc_output2], dim = 0)
            else:   
                rcc_output = self.STIN(rcc_feas)
            # print('rcc_output:\t', rcc_output.size())
        # exit(0)
        ############################################################################################
           
        if self.two_stream:
            text_ouput = self.text_backbone(caption_input)
            class_output = self.text_backbone(class_iput)
            
            logit_scale = self.logit_scale.exp()
        else:
            if self.joint:
                text_input_ids, attention_mask = caption_input                
                tokenize = {"input_ids":text_input_ids, "attention_mask":attention_mask}
                if self.do_KL:
                    logit_scale = self.logit_scale.exp()
                    label_input_ids, label_attention_mask = label_input
                    label_tokenize = {"input_ids":label_input_ids, "attention_mask":label_attention_mask}
                # print(tokenize)
                if self.do_attention or self.do_SEattn:
                    tags_embedding = self.text_backbone(**tokenize).last_hidden_state
                    # print('tags_embedding:\t', tags_embedding.size())  
                    tags_embedding = tags_embedding[:, 1:, :].contiguous()
                    # print('tags_embedding:\t', tags_embedding.size())
                else:
                    if self.embed_way == 'RoBERTa':
                        tags_embedding = self.text_backbone(**tokenize).pooler_output
                        if self.do_KL:
                            labels_embedding = self.text_backbone(**label_tokenize).pooler_output
                    elif self.embed_way == 'DistilBERT':
                        tags_embedding = self.text_backbone(**tokenize).last_hidden_state
                        tags_embedding = tags_embedding[:, 0, :].contiguous()
                        if self.do_KL:
                            labels_embedding = self.text_backbone(**label_tokenize).pooler_output
                    elif self.embed_way == 'CLIPModel':
                        # tags_embedding = self.text_backbone(**tokenize)
                        tags_embedding = self.text_backbone(**tokenize).pooler_output   
                    elif self.embed_way == 'DistilRoBERTa':
                        tags_embedding = self.text_backbone(**tokenize).pooler_output
                        if self.do_KL:
                            labels_embedding = self.text_backbone(**label_tokenize).pooler_output
                    # print(tags_embedding)
                    # exit(0)
                # org_tags_embedding = tags_embedding  
                # print('org_tags_embedding:\t', org_tags_embedding.size())

                if self.do_aug and test_model:
                    if self.do_aug_mix:
                        tags_embedding = torch.cat([tags_embedding, tags_embedding, tags_embedding, tags_embedding, tags_embedding], dim = 0)       
                    else:
                        tags_embedding = torch.cat([tags_embedding, tags_embedding], dim = 0)
                    if self.do_cross_attn:
                        output_feas = self.get_cross_attn_out(global_video_fea, tags_embedding)
                        # print('output_feas:\t', output_feas.size())
                    elif self.do_frame and self.do_coord_cate:
                        visual_feas = torch.cat([global_video_fea, rcc_output, tags_embedding], dim = 1)
                        # print('visual_feas:\t', visual_feas.size())
                        output_feas = self.MLP(visual_feas)
                        # print('output_feas:\t', output_feas.size())
                    else:
                        visual_feas = torch.cat([global_video_fea, tags_embedding], dim = 1)
                        # print('visual_feas:\t', visual_feas.size())
                        output_feas = self.MLP(visual_feas)
                    # print('output_feas:\t', output_feas.size())
                    if self.do_KL:
                        global_tag_feas = self.dim_trans(visual_feas) 
                    # print('global_tag_feas:\t', global_tag_feas.size())                
                else:
                    if self.do_attention:
                        output_feas = self.get_attn_out(frame_feas, tags_embedding)
                    elif self.do_SEattn:
                        output_feas = self.get_SEattn_out(frame_feas, tags_embedding)
                        # print(output_feas.size())
                        output_feas = self.avgpool(output_feas).view(output_feas.size(0), -1)
                        # print(output_feas.size())
                    elif self.do_cross_attn:
                        output_feas = self.get_cross_attn_out(global_video_fea, tags_embedding)
                        # print('output_feas:\t', output_feas.size())
                    else:
                        if self.do_frame and self.do_region:
                            visual_feas = torch.cat([global_video_fea, rcc_output, tags_embedding], dim = 1)
                            # print('visual_feas:\t', visual_feas.size())
                        elif self.do_frame and self.do_coord_cate:
                            visual_feas = torch.cat([global_video_fea, rcc_output, tags_embedding], dim = 1)
                            # print('visual_feas:\t', visual_feas.size())
                        elif self.do_frame:
                            visual_feas = torch.cat([global_video_fea, tags_embedding], dim = 1)
                        else:                                     
                        # visual_feas = torch.cat([global_video_fea, tags_embedding], dim = 1)
                        # visual_feas = torch.cat([global_video_fea, rcc_output, tags_embedding], dim = 1)
                            visual_feas = torch.cat([rcc_output, tags_embedding], dim = 1)                       
                        # print('visual_feas:\t', visual_feas.size())
                        # exit(0)
                        output_feas = self.MLP(visual_feas)
                        # print('output_feas:\t', output_feas.size())
                        if self.do_KL:
                            global_tag_feas = self.dim_trans(visual_feas)
                            # print('global_tag_feas:\t', global_tag_feas.size())
                        
            else:
                # visual_feas = torch.cat([global_video_fea, rcc_output], dim = 1)
                if self.do_region and self.do_coord_cate:
                    visual_feas = rcc_output
                    # print('visual_feas:\t', visual_feas.size())   
                    output_feas = self.MLP(visual_feas)
                    # print('output_feas:\t', output_feas.size())      
                elif self.do_frame and self.do_coord_cate:
                    visual_feas = torch.cat([global_video_fea, rcc_output], dim = 1)
                    # print('visual_feas:\t', visual_feas.size())
                    output_feas = self.MLP(visual_feas)
                    # print('output_feas:\t', output_feas.size())   
                elif self.do_region:
                    visual_feas = rcc_output
                    # print('visual_feas:\t', visual_feas.size())  
                    output_feas = self.MLP(visual_feas)
                    # print('output_feas:\t', output_feas.size())      
                elif self.do_coord_cate:
                    output_feas = rcc_output
                elif self.do_frame:
                    visual_feas = global_video_fea
                    # print('visual_feas:\t', visual_feas.size())
                    output_feas = self.MLP(visual_feas)
                    # print('output_feas:\t', output_feas.size())
                    # print(output_feas)
                # exit(0)  
                # output_feas = global_video_fea

        ## pred box
        # if self.two_stream:            
        #     caption_fea = self.text_backbone(caption_input)
        #     position_embedding = self.position_embedding(video_fea_map)           
        #     box_out = self.obj_det_branch(video_fea_map, caption_fea, position_embedding)
        # exit(0)
        if self.two_stream:
            # return global_video_fea, box_out
            return global_video_fea, text_ouput, hidden_video_features, class_output, logit_scale
        elif self.do_KL:
            return output_feas, global_tag_feas, labels_embedding, logit_scale
        else:
            return output_feas
        
    def vis_reduction_layer(self, in_dim, out_dim=512):
        net = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            )
        return net
    def text_reduction_layer(self, in_dim, out_dim=512):
        net = nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True)
            )
        return net
    
    def get_rcc_feas(self, video, video_fea_map, box_input, box_categories, test_model):
        B, TC, _, _ = video.size()
        # print(video.size())
        ## get region feas
        if self.do_region:
            conv_fea_maps = self.conv(video_fea_map)
            # print('conv_fea_maps:\t', conv_fea_maps.size())
            box_tensor = box_input.view(-1, *box_input.size()[2:])
            # print('box_tensor_size:\t', box_tensor.size())
            boxes_list = box_to_normalized(box_tensor, crop_size=[224,224])
            img_size = video.size()[-2:]
            if self.do_aug and test_model:
                if self.do_aug_mix:
                    conv_fea_maps1 = conv_fea_maps[:B//5,:,:,:]
                    # print('conv_fea_maps1:\t', conv_fea_maps1.size())
                    conv_fea_maps2 = conv_fea_maps[(B//5):(2*B//5),:,:,:]
                    # print('conv_fea_maps2:\t', conv_fea_maps2.size())
                    conv_fea_maps3 = conv_fea_maps[(2*B//5):(3*B//5),:,:,:]
                    conv_fea_maps4 = conv_fea_maps[(3*B//5):(4*B//5),:,:,:]
                    conv_fea_maps5 = conv_fea_maps[(4*B//5):,:,:,:]
                    
                    # exit(0) 
                    region_vis_feas1 = build_region_feas(conv_fea_maps1, boxes_list, self.crop_size1, img_size)
                    region_vis_feas1 = region_vis_feas1.view(-1, self.T, self.nr_boxes, region_vis_feas1.size(-1))
                    region_vis_feas1 = self.region_vis_embed(region_vis_feas1)
                    # print('region_vis_feas_size1:\t', region_vis_feas1.size())
                    
                    region_vis_feas2 = build_region_feas(conv_fea_maps2, boxes_list, self.crop_size1, img_size) 
                    region_vis_feas2 = region_vis_feas2.view(-1, self.T, self.nr_boxes, region_vis_feas2.size(-1))
                    region_vis_feas2 = self.region_vis_embed(region_vis_feas2)
                    # print('region_vis_feas_size2:\t', region_vis_feas2.size())
                    
                    region_vis_feas3 = build_region_feas(conv_fea_maps3, boxes_list, self.crop_size1, img_size)
                    region_vis_feas3 = region_vis_feas3.view(-1, self.T, self.nr_boxes, region_vis_feas3.size(-1))
                    region_vis_feas3 = self.region_vis_embed(region_vis_feas3)
                    
                    region_vis_feas4 = build_region_feas(conv_fea_maps4, boxes_list, self.crop_size1, img_size)
                    region_vis_feas4 = region_vis_feas4.view(-1, self.T, self.nr_boxes, region_vis_feas4.size(-1))
                    region_vis_feas4 = self.region_vis_embed(region_vis_feas4)
                    
                    region_vis_feas5 = build_region_feas(conv_fea_maps5, boxes_list, self.crop_size1, img_size)
                    region_vis_feas5 = region_vis_feas5.view(-1, self.T, self.nr_boxes, region_vis_feas5.size(-1))
                    region_vis_feas5 = self.region_vis_embed(region_vis_feas5)
                    
                    rcc_feas = torch.cat([region_vis_feas1, region_vis_feas2, region_vis_feas3, region_vis_feas4, region_vis_feas5], dim = 1)
                    # print('rcc_feas:\t', rcc_feas.size())
                else:
                    conv_fea_maps1 = conv_fea_maps[:B//2,:,:,:]
                    # print('conv_fea_maps1:\t', conv_fea_maps1.size())
                    conv_fea_maps2 = conv_fea_maps[(B//2):,:,:,:]
                    
                    region_vis_feas1 = build_region_feas(conv_fea_maps1, boxes_list, self.crop_size1, img_size)
                    region_vis_feas1 = region_vis_feas1.view(-1, self.T, self.nr_boxes, region_vis_feas1.size(-1))
                    region_vis_feas1 = self.region_vis_embed(region_vis_feas1)
                    # print('region_vis_feas_size1:\t', region_vis_feas1.size())
                    
                    region_vis_feas2 = build_region_feas(conv_fea_maps2, boxes_list, self.crop_size1, img_size) 
                    region_vis_feas2 = region_vis_feas2.view(-1, self.T, self.nr_boxes, region_vis_feas2.size(-1))
                    region_vis_feas2 = self.region_vis_embed(region_vis_feas2)
                    # print('region_vis_feas_size2:\t', region_vis_feas2.size())
                    rcc_feas = torch.cat([region_vis_feas1, region_vis_feas2], dim = 1)

            else:
                region_vis_feas = build_region_feas(conv_fea_maps, boxes_list, self.crop_size1, img_size) # ([V*T*P, d], where d = 3*3*c) ([128, 4608])
                print('region_vis_feas_size1:\t', region_vis_feas.size())
                region_vis_feas = region_vis_feas.view(-1, TC//3, self.nr_boxes, region_vis_feas.size(-1)) # ([V, T, P, d])
                print('region_vis_feas_size2:\t', region_vis_feas.size())
                region_vis_feas = self.region_vis_embed(region_vis_feas) # ([V, T, P, D_vis]) # (4, 8, 4, 2048)
                print('region_vis_feas_size:\t', region_vis_feas.size())
                rcc_feas = region_vis_feas
        # exit(0)
        ############################################################################################
        ## coord+category feas
        if self.do_coord_cate:
            box_input = box_input.transpose(2, 1).contiguous() # ([V, nr_boxes, T, 4]) ([4, 4, 8, 4])
            # print('box_input:\t', box_input.size())
            box_input = box_input.view(B * self.nr_boxes * self.T, 4) # ([V*nr_boxes*T, 4]) ([128, 4])
            bf = self.coord_to_feature(box_input) # ([128, 512])
                # print('bf_size1:\t', bf.size())
            if self.do_cate:
                box_categories = box_categories.long()
                box_categories = box_categories.transpose(2, 1).contiguous()
                # print('box_categories_size1:\t', box_categories.size())
                box_categories = box_categories.view(B*self.nr_boxes*self.T)
                # print('box_categories_size2:\t', box_categories.size())
                box_category_embeddings = self.category_embed_layer(box_categories)
                # print('box_category_embeddings_size:\t', box_category_embeddings.size()) 
                bf = torch.cat([bf, box_category_embeddings], dim=1)
                # print('bf_size2:\t', bf.size())
                bf = self.coord_category_fusion(bf)  # (b*nr_b*nr_f, coord_feature_dim)
                # print('bf_size3:\t', bf.size())
            else:
                pass
            # exit(0)
            
            bf = bf.view(B, self.nr_boxes, self.T, self.coord_feature_dim) # ([4, 4, 8 ,512])
            # print('bf_size4:\t', bf.size())
            if self.do_aug_mix and test_model:
                bf = torch.cat([bf, bf, bf, bf, bf], dim = 2)
            elif self.do_aug and test_model:
                bf = torch.cat([bf, bf], dim = 2)
            coord_feas = bf.permute(0, 2, 1, 3).contiguous()
            # print('coord_feas:\t', coord_feas.size())
            rcc_feas = coord_feas
            # exit(0)
            
        if self.do_region and self.do_coord_cate:   
            rcc_feas = torch.cat([region_vis_feas, coord_feas], dim = -1)
            # print('rcc_feas:\t', rcc_feas.size())
        # print('rcc_feas:\t', rcc_feas.size())
        # exit(0)
        return rcc_feas
    
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
    
    def get_attn_out(self, frame_feas, tags_embedding):
        B, _, _ = frame_feas.size()
        frame_feas = frame_feas.view(-1, frame_feas.size(-1))
        frame_feas = self.vis_reduce(frame_feas)   
        frame_feas = frame_feas.view(-1, 8, frame_feas.size(-1))
        # print('frame_feas:\t', frame_feas.size())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # print('cls_tokens:\t', cls_tokens.size())
        # exit(0)
        frame_feas = torch.cat([cls_tokens, frame_feas], dim = 1)
        # print('frame_feas:\t', frame_feas.size())
        # exit(0)
        frame_feas += self.temporal_embed
        # print('frame_feas:\t', frame_feas.size())
        # exit(0)
        
        ##
        tags_embedding = tags_embedding.view(-1, tags_embedding.size(-1))
        # print('tags_embedding:\t', tags_embedding.size())
        # exit(0)
        if self.attn_input_dim == 512:
            tags_embedding = self.text_reduce(tags_embedding)
        else:
            pass
        tags_embedding = tags_embedding.view(frame_feas.size(0), -1, tags_embedding.size(-1))    
        visual_feas = torch.cat([tags_embedding, frame_feas], dim = 1)
        attn_input_feas = visual_feas.transpose(1, 0).contiguous()
        # print('attn_input_feas:\t', attn_input_feas.size())
        # exit(0)
        attn_input_feas = self.attn_drop(attn_input_feas)
        attn_output = self.transformer_encoder(attn_input_feas)
        # print('attn_output:\t', attn_output.size())
        # exit(0)
        attn_output = attn_output.transpose(1, 0).contiguous()
        # print('attn_output:\t', attn_output.size())
        output_feas = attn_output[:, 0, :].contiguous()
        # print('output_feas:\t', output_feas.size()) 
        return output_feas
    
    def get_SEattn_out(self, frame_feas, tags_embedding):
        frame_feas = frame_feas.view(-1, frame_feas.size(-1))
        frame_feas = self.vis_reduce(frame_feas)   
        frame_feas = frame_feas.view(-1, self.T, frame_feas.size(-1))
        # print('frame_feas:\t', frame_feas.size())
        tags_embedding = tags_embedding.view(-1, tags_embedding.size(-1))
        # print('tags_embedding:\t', tags_embedding.size())
        # exit(0)
        if self.attn_input_dim == 512:
            tags_embedding = self.text_reduce(tags_embedding)
        else:
            pass
        
        tags_embedding = tags_embedding.view(frame_feas.size(0), -1, tags_embedding.size(-1))    
        attn_input_feas = torch.cat([tags_embedding, frame_feas], dim = 1)
        # print('attn_input_feas:\t', attn_input_feas.size())
        attn_input_feas = attn_input_feas.transpose(2, 1).contiguous()
        # attn_input_feas = self.attn_drop(attn_input_feas)
        attn_output = self.SEattn(attn_input_feas)
        return attn_output
    def get_cross_attn_out(self, global_video_fea, tags_embedding):
        video_feas = self.dim_reduce(global_video_fea)
        video2text = self.CrossAttention_v2t(video_feas, tags_embedding, mask = None)
        text_feas = self.dim_increase(tags_embedding)
        # print(text_feas.size())
        text2video = self.CrossAttention_t2v(text_feas, global_video_fea, mask = None)
        visual_feas = torch.cat([video2text, text2video], dim = 1)
        output_feas = self.MLP(visual_feas)
        return output_feas
         