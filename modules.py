import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial

torch.set_printoptions(precision=3,edgeitems=32,linewidth=350)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs)[0] + x,self.fn(x, **kwargs)[1]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x),x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = torch.softmax(dots,dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out,attn[:,0,0]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        b,n,_=x.shape
        ##map is used for visualization
        map=torch.ones(b,n).cuda()

        for attn, ff in self.layers:
            x,attn_map = attn(x, mask=mask)
            x,_ = ff(x)
            ##
            map=map*attn_map
        return x,map



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##resduals and GCA blocks before the dynamic-static fusion module
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bias1 = nn.Parameter(torch.randn(1, planes,1,1))
        self.bias2= nn.Parameter(torch.randn(1, planes, 1, 1))
        self.AUX_fc = nn.Sequential(nn.Linear(planes, planes*2,bias=False),
                                nn.Linear(planes*2, 7, bias=False))
        self.glb = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1,bias=False)# nn.BatchNorm2d(planes),


        ##global convolution
        self.gc = nn.Sequential(nn.Conv2d(in_channels=planes,out_channels=planes,kernel_size=256//planes*7,groups=planes,bias=False),
                                   #nn.BatchNorm2d(planes),
                                   )
        ##dimensionality-reduction ratio r=16
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=planes //16, kernel_size=1,bias=False),
                                   #nn.BatchNorm2d(planes//16),
                                   nn.ReLU(),)
        self.fc2=nn.Sequential(nn.Conv2d(in_channels=planes//16, out_channels=planes , kernel_size=1,bias=False),
                                   nn.Sigmoid(),
                                   )
   

    def forward(self, x):
        x,_=x
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        b, c, h, w = out.shape
        ##GCA block is deployed since layer2, layer1 is the building block of the ResNet18
        if c>64:##if GCA block
            ##global covolution
            weight=self.gc(out)
            ##auxiliary classifier
            aux = self.AUX_fc(torch.mean(weight.view(b // 16, 16, -1), dim=1))
            weight=self.fc1(weight)
            weight =2*(self.fc2(weight))
            ##introduce global temporal information
            weight = torch.sqrt(weight * torch.mean(weight.view(b // 16, 16, c, 1, 1), dim=1).unsqueeze(1).repeat(1, 16, 1, 1, 1).view(-1, c, 1, 1) + 0.00001)
            weight=weight.repeat(1, 1, h, w)
            ##rescale feature map
            out=out*weight
        else:
            aux=out##if residuals
        out += identity
        out = self.relu(out)


        return out,aux
##conv layers used to build the dynamic-static fusion module
class BasicBlock_act(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_act, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)


        return out
class ResNet(nn.Module):   # GCA block after layer1

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 num_patches=7*7, dim=256*4*4, depth=4, heads=4, mlp_dim=512, dim_head=64, dropout=0.):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or"
                             " a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(self.inplanes),
                                   nn.ReLU(),
                                   nn.Conv2d(self.inplanes, self.inplanes,kernel_size=3,stride=1,padding=1,bias=False))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.bias = nn.Parameter(torch.randn(1, 64, 1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ##residuals
        self.layer1 = self._make_layer(block, 64, layers[0])
        ##GCA blocks
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        ##Dynamic-static fusion module
        self.layer4 = self._make_layer(BasicBlock_act, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            )
        self.pos_transformer = Transformer(dim=256, depth=2, heads=4, dim_head=32, mlp_dim=256, dropout=0.)
        self.pos_embedding_pos = nn.Parameter(torch.randn(1, 16, 256))##corresponding to 16 patches of the static branch of the dynamic-static fusion module
        ##temporal_transformer
        self.temporal_transformer = Transformer(dim, 2, 4, 32, 512, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, dim))##corresponding to 16 frames


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 112, 112)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x,logits1 = self.layer1((x,0))##Residual
        x,logits2 = self.layer2((x,0))##GCA block
        x,logits3 = self.layer3((x,0))##GCA block
        ##Dynamic-static fusion module
        b, c, h, w = x.shape
        act=((x.view(b//16,16,c,h,w))[:,1:16]-(x.view(b//16,16,c,h,w))[:,0:15]).view(-1,c,h,w)
        x1 = (x.view(b//16,16,c,h,w))[:,0:15,:,:].contiguous().view(-1, c, 7,7)
        POS = self.patch_embedding(x1)
        POS = POS.view(-1,256,16).transpose(1,2)+self.pos_embedding_pos[:, :16]
        POS =self.pos_transformer(POS)[0].transpose(1,2).view(-1,256,4,4)
        act = self.layer4(act)
        x=POS+act
        ##

        b_l, c, h, w = x.shape

        x = x.reshape((b_l // 15, 15, c * h * w))

        b, n, _ = x.shape

        x = x + self.pos_embedding[:, :(n)]
        x,map = self.temporal_transformer(x)



        return x,logits1,logits2,logits3


def GCA_IAL():
    return ResNet(BasicBlock, [1, 1, 1, 3])


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GCA_IAL()
    model(img)
