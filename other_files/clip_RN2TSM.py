# from torch._C import Module
from CLIP.clip.model import *
from CLIP.clip import clip
import os
import warnings

if torch.__version__.split(".") < ["1", "7", "1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

def clip_prepare_model(name, device, jit, download_root = None):
    if name in clip._MODELS:
        # model_path = clip._download(clip._MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_path = "%s/%s.pt" % (download_root, name)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {clip.available_models()}")
    # exit(0)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location= device if jit else "cpu").eval()
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")
    
    return model.state_dict()

              transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 visual_head_type = 'pool'
                 ):
        super(Video_CLIP, self).__init__(embed_dim,
                 # vision
                 image_resolution,
                 vision_layers,
                 vision_width,
                 vision_patch_size,
                 # text
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers)

        self.mode = mode
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.visual_head_type = visual_head_type
        self.input_size = 224
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ## Noted by Mr. YAN
        # self.hidden_size = 512 # It needs a more suitable name. 
        # self.num_frames = 8 ## ??? need pass arg to it
        if 'RN50' in args.clip_vis_name:
            self.hidden_size = args.img_feature_dim * 4
        else:
            self.hidden_size = args.img_feature_dim * 2
        # self.hidden_size = args.img_feature_dim * 2
        self.num_frames = args.num_segments

        if self.mode == 'tsm_based':
            self.visual = nn.Sequential(
                resnet_TSM(args.basic_arch, args.shift, num_segments=8),
                # nn.AdaptiveAvgPool2d((1,1))
            )
            if 'renet50' in args.basic_arch:
                hidden_feas = 2048
            else:
                hidden_feas = 512
            self.global_vis_feas_fusion = nn.Sequential(
                nn.Linear(hidden_feas, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True)
            )
            # self.dtype = float
        # elif self.mode == 'rs50_based':
        #     self.visual = getattr(torchvision.models, 'resnet50')(pretrained=True)
        elif self.mode == 'clip_based':
            pass

        if self.visual_head_type == 'pool':
            self.visual_head = nn.AdaptiveAvgPool1d(1)
        elif self.visual_head_type == 'TRN':
            # 512 --> 512
            self.visual_head = RelationModuleMultiScale(self.hidden_size, self.num_frames, self.hidden_size)

    def encoder_frozen(self, frozen_list = ['visual', 'text']):
        '''
        This function is used to forzen weights of visual or language encoder
        '''
        if 'visual' in frozen_list:
            # frozen visual encoder
            frozen_param(self.visual)
        if 'text' in frozen_list:
            # frozen text encoder
            frozen_param(self.transformer) # transformer is used to encode text in original code of 'CLIP'
        
    
    def forward(self, video, text_input, class_names_input):

        # print('text:\t', text_input.size())
        video_features = self.encode_video(video)

        # text_input = clip.tokenize(text).to(device)
        # print('text_input:\t', text_input.size())
        text_features = self.encode_text(text_input)
        # print('text_features1:\t', text_features.size())
        # exit(0)

        # class_input = clip.tokenize(f'a video about {c}' for c in class_names).to(device)
        # class_input = clip.tokenize(f'{c}' for c in class_names).to(device)
        # print('class_input:\t', class_input.size())
        class_features = self.encode_text(class_names_input)
        # print('class_features:\t', class_features.size())


        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # print('video_features', video_features.size())
        # print('text_features', text_features.size())
        # print('logit_scale', logit_scale)
        # exit(0)
        # return logits_per_video, logits_per_text, video_features.float(), class_features.float()
        return logit_scale.float(), video_features.float(), text_features.float(), class_features.float()

    
    def encode_video(self, video):
        video = video.type(self.dtype)
        B, TC, _, _ = video.size()
        # print('video1:\t', video.size())
        video = video.view((B, TC//3, 3) + video.size()[-2:])
        # print('video2:\t', video.size())
        video_input = video.view(-1, *video.size()[2:])
        # print('video3:\t', video_input.size())


        if self.mode == 'clip_based':
            # if args.arch == 'clip_RN2TSM' and args.shift:
            #     print('Adding temporal shift...')
            #     make_temporal_shift(self.visual, args.num_segments,
            #                         args.shift_div, args.shift_place, args.temporal_pool)
            #     print(self.visual)
            #     visual_features = self.visual(video_input)
            # else:
            #     visual_features = self.encode_image(video_input) # resnet50: [B*T, 1024]
            visual_features = self.encode_image(video_input) # resnet50: [B*T, 1024]
            # print('visual_features1:\t', visual_features.size())
            # exit(0)
            if self.visual_head_type == 'pool':
                # pool
                visual_features = visual_features.view(B, 8, -1)
                # print('visual_features2:\t', visual_features.size())
                visual_features = self.visual_head(visual_features.transpose(2,1).contiguous())
                # print('visual_features3:\t', visual_features.size())
                visual_features = visual_features.view(B, -1)
                # print('visual_features4:\t', visual_features.size())
            elif self.visual_head_type == 'TRN':
                # input:[B, T, C], output:[B, C]
                visual_features = visual_features.view(B, 8, -1)
                visual_features = self.visual_head(visual_features)
                # print('visual_features2:\t', visual_features.size()) 

        else:
            visual_features = self.visual(video_input)
            visual_features = visual_features.mean(-1).mean(-1).view(B, TC//3, -1)
            visual_features = self.avgpool(visual_features.transpose(2,1).contiguous())
            visual_features = visual_features.view(B, -1)
            # print('visual_features:\t', visual_features.size())
            visual_features = self.global_vis_feas_fusion(visual_features)

        return visual_features
    
    

    # @property
    # def dtype(self):
    #     return float
        # return self.visual.conv1.weight.dtype
        # return self.visual.xxxx.weight.dtype


    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        print('#' * 20, 'NO FLIP!!!')
        return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])


# added by Mr. YAN
def frozen_param(net):
    # forzen all param in net
    for param in net.parameters():
        param.requires_grad = False


def build_model(state_dict: dict, mode = 'clip_based', visual_head = 'pool', frozen_blk = ''):
    # print('state_dict:\t', state_dict)
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = Video_CLIP(
        mode, embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, visual_head
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)


    load_part_state_dict(model, state_dict)

    # ## added by hp 
    # for p in model.parameters(): 
    #     p.data = p.data.float() 

    # model.load_state_dict(state_dict)

    # frozen_blk should be list. NOTED by Mr. Yan
    model.encoder_frozen(frozen_blk) # forzen encoder 
    return model
    # return model.eval() 



def load_part_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    ## print(model_dict)
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
