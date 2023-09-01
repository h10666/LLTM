import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class CrossAttention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.q.weight.data.fill_(0)
            self.q.bias.data.fill_(0)
            self.kv.weight.data.fill_(0)
            self.kv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, mask):
        _, dims = x.size()
        h = self.num_heads
        # project x to q, k, v vaalues
        q = self.q(x)
        # print('q:\t', q.size())
        k, v = self.kv(y).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> b h d', h=h), (q, k, v))

        scale = (dims//self.num_heads) ** -0.5
        # q *= self.scale
        q *= scale
        # splice out CLS token at index 1
        # cls_q = q[:, 0:1]
        # q_ = q[:, 1:]

        # mask = repeat(mask, 'b d -> (b r) d', r=self.num_heads)
        # # let CLS token attend to key / values of all patches across time and space
        # cls_out = attn_mask(cls_q, k, v, mask)

        # # attention
        # out = attn_mask(q_, k, v, mask)
        out = self.attn_mask(q, k, v, mask)
        # concat back the cls token
        # out = torch.cat((cls_out, out), dim=1)
        # print('out1:\t', out.size())
        # merge back the heads
        out = rearrange(out, 'b h d -> b (h d)', h=h)
        # print('out2:\t', out.size())
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        # print(x.size())
        # exit(0)
        return x
    
    def attn_mask(self, q, k, v, mask):
        dots = einsum('b h d, b h d -> b h', q, k) * self.scale
        # print('dots:\t', dots.size())
        attn = dots.softmax(dim=-1)
        # print('attn:\t', attn.size())
        out = einsum('b h, b h d -> b h d', attn, v)
        return out
        
    

class CrossAttention2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.5):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        # print('x_qkv:\t', x_qkv.size())
        b, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b (h d) -> b h d', h = h)
        # print('k:\t', k.size())
        v = self.to_v(x_qkv)
        v = rearrange(v, 'b (h d) -> b h d', h = h)
        # print('v:\t', v.size())

        # q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = self.to_q(x_qkv)
        q = rearrange(q, 'b (h d) -> b h d', h = h)
        # print('q:\t', q.size())
        
        # exit(0)

        dots = einsum('b h d, b h d -> b h', q, k) * self.scale
        # print('dots:\t', dots.size())
        attn = dots.softmax(dim=-1)
        # print('attn:\t', attn.size())
        out = einsum('b h, b h d -> b h d', attn, v)
        
        out = rearrange(out, 'b h d -> b (h d)')
        out =  self.to_out(out)
        # print('out:\t', out.size())
        # exit(0)
        return out