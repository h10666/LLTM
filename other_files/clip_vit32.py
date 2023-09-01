import torch
from opts import parser
import torch.nn as nn
from archs.clip_prepare_model import clip_prepare_model
from archs.clip_model import build_model


args, _ = parser.parse_known_args()

class clip_vit32(nn.Module):
    def __init__(self, num_class, partial_bn=True):
        super(clip_vit32, self).__init__()  
        
        name = args.clip_vis_name
        jit = args.jit
        visio_mode = args.visio_mode
        visual_head_type = args.visual_head_type
        frozen_blk = args.frozen_blk
        download_root = None
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        state_dict = clip_prepare_model(name, device, jit, download_root)
        self.base_model = build_model(state_dict, visio_mode, visual_head_type, frozen_blk).visual
        # print('model1:\t', self.base_model)
        # exit(0)
        # self.base_model.transformer = nn.Identity()
        # self.base_model.token_embedding = nn.Identity()
        # self.base_model.ln_final = nn.Identity()
        self.fc = nn.Linear(512, num_class)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    
    def forward(self, video, text_input, class_name_input):
        # visual_features=self.base_model.encode_video(video)
        # print('video_size1:\t', video.size())
        B, TC, _, _ = video.size()
        video = video.view((B, TC//3, 3) + video.size()[-2:])
        # print('video_size2:\t', video.size())
        video_input = video.view(-1, *video.size()[2:])
        # print('video_size3:\t', video_input.size())
        visual_features=self.base_model(video_input)
        # print('visual_features1', visual_features.size())
        visual_features = visual_features.view(B, 8, -1)
        # print('visual_features2', visual_features.size())
        visual_features = self.avgpool(visual_features.transpose(2,1).contiguous())
        # print('visual_features3', visual_features.size())
        visual_features = visual_features.view(B, -1)
        # print('visual_features4', visual_features.size())

        return visual_features

    def partialBN(self, enable):
        self._enable_pbn = enable