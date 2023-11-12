import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.dyna_ch = depth_channel_att(in_channel) if filter else nn.Identity()
        self.sfconv = SFconv(in_channel) if filter else nn.Identity()
    
        self.proj = nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel)
        self.proj_act = nn.GELU()

    def forward(self, x):

        out = self.conv1(x)

        out = self.proj(out)
        out = self.proj_act(out)
        out = self.dyna_ch(out)

        out = self.sfconv(out)
        out = self.conv2(out)
        
        return out + x

class depth_channel_att(nn.Module):
    def __init__(self, dim, kernel=3) -> None:

        super().__init__()
        self.kernel = (1, kernel)
        pad_r = pad_l = kernel // 2
        self.pad = nn.ReflectionPad2d((pad_r, pad_l, 0, 0))
        self.conv = nn.Conv2d(dim, kernel*dim, kernel_size=1, stride=1, bias=False, groups=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.filter_act = nn.Tanh()
        self.filter_bn = nn.BatchNorm2d(kernel*dim)
        self.gamma = nn.Parameter(torch.zeros(dim,1,1))
        self.beta = nn.Parameter(torch.ones(dim,1,1))

    def forward(self, x):

        filter = self.filter_bn(self.conv(self.gap(x))) 
        filter = self.filter_act(filter) 
        b, c, h, w = filter.shape
        filter = filter.view(b, self.kernel[1], c//self.kernel[1], h*w).permute(0, 1, 3, 2).contiguous()
        B, C, H, W = x.shape
        out = x.permute(0, 2, 3, 1).view(B, H*W, C).unsqueeze(1)
        out = F.unfold(self.pad(out), kernel_size=self.kernel, stride=1) 
        out = out.view(B, self.kernel[1], H*W, -1)
        out = torch.sum(out * filter, dim=1, keepdim=True).permute(0,3,1,2).reshape(B,C,H,W)

        return out * self.gamma + x * self.beta

class SFconv(nn.Module):
    def __init__(self, features, M=4, r=2, L=32) -> None:
        super().__init__()
        
        d = max(int(features/r), L)
        self.features = features
        self.convs = nn.ModuleList([])

        self.convh = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
            )

        self.convm = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.convll = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Conv2d(features, features, 1, 1, 0)
        self.gamma = nn.Parameter(torch.zeros((1,features,1,1)), requires_grad=True)

    def forward(self, x):
        lowlow = self.convll(x)
        low = self.convl(lowlow)
        middle = self.convm(low)
        high = self.convh(middle)
        emerge = low+middle+high+lowlow
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        middle_att = self.fcs[1](fea_z)
        low_att = self.fcs[2](fea_z)
        lowlow_att = self.fcs[3](fea_z)

        attention_vectors = torch.cat([high_att, middle_att, low_att, lowlow_att], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        high_att, middle_att, low_att, lowlow_att = torch.chunk(attention_vectors, 4, dim=1)

        fea_high = high * high_att
        fea_middle = middle * middle_att
        fea_low = low * low_att
        fea_lowlow = lowlow * lowlow_att
        out = self.out(fea_high + fea_middle + fea_low + fea_lowlow) 
        return out * self.gamma + x
    


