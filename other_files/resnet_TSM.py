import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from CLIP.clip import model
from ops.temporal_shift import make_temporal_shift

class resnet_TSM(nn.Module):
    def __init__(self, basic_arch, shift, num_segments, download_root = None):
        super(resnet_TSM,self).__init__()
        
        model = torchvision.models.resnet50(pretrained=False)
        pthfile = r"%s/resnet50-0676ba61.pth" % (download_root)       
        model.load_state_dict(torch.load(pthfile))
        # model = getattr(torchvision.models, basic_arch)(pretrained=pretrained)
        # print(model)
        
        # 获取网络最后一层的输入特征维度
        self.feature_dim = getattr(model, 'fc').in_features
        # print('feature_dim:\t', self.feature_dim)
        
        if shift:
            make_temporal_shift(model, num_segments)
        self.model = torch.nn.Sequential(*(list(model.children())[:-2])) # drop fc and adaptive_pool

        # self.model[7][0].conv2.stride = (1, 1)

        # self.model[7][0].downsample[0].stride = (1, 1)
        ## print(self.model)
        ## exit(0)
        # # self.features=model.features
        


    def forward(self,x):
        x=self.model(x)
        return x