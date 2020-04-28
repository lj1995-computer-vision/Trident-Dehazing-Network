import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torch.nn.init as init
from torch.nn import Parameter
try:
    import sys
    sys.path.append("DCNv2")
    from dcn_v2 import DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

#   Wide Activation Block
class WAB(nn.Module):
    def __init__(self,n_feats,expand=4):
        super(WAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * expand,3,1,1, bias=True),
            nn.BatchNorm2d(n_feats * expand),
            nn.ReLU(True),
            nn.Conv2d(n_feats* expand, n_feats , 3, 1, 1, bias=True),
            nn.BatchNorm2d(n_feats)
        )

    def forward(self, x):
        res = self.body(x).mul(0.2)+x
        return res

#   codes of UNet are modified from pix2pix
def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=16):
        super(UNet, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        # dlayer6 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        dlayer6 = blockUNet(nf*8, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer5 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer4 = blockUNet(nf*16, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer3 = blockUNet(nf*8, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer2 = blockUNet(nf*4, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv = nn.Conv2d(nf*2, output_nc, 3,padding=1, bias=True)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        dout1 = self.tail_conv(dout1)

        # dout1=torch.sigmoid(dout1)

        return dout1

class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)

#   Deformable Upsampling Block
class DUB(nn.Module):
    def __init__(self, a,b,c):
        super(DUB, self).__init__()
        self.conv3=nn.Sequential(
            nn.BatchNorm2d(a),
            nn.ReLU(inplace=True),
            DCN(a, b, kernel_size=3, stride=1,padding=1)
        )
        self.conv1=nn.Sequential(
            nn.BatchNorm2d(a+b),
            nn.ReLU(inplace=True),
            DCN(a+b, c, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        y=self.conv3(x)
        x=self.conv1(torch.cat([x,y],1))
        return F.upsample_nearest(x, scale_factor=2)

def make_model(pretrained):
    model=Net(pretrained)
    return model

class Net(nn.Module):
    def __init__(self,pretrained=True):
        super(Net, self).__init__()

        if(pretrained==True):
            print ("=> loading checkpoint from pretrained dpn92-5k-1k")
            dpn92 = pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained='imagenet+5k').features
        else:dpn92 = pretrainedmodels.__dict__['dpn92'](num_classes=1000, pretrained=False).features
        
        #   dx: downsample to factor x
        #   ux: upsample to factor x

        #   Haze Density Map Generate sub-Net
        self.d64u1=UNet(input_nc=3,output_nc=3, nf=8)

        #   Encoder Decoder sub-Net
        self.d8=dpn92[:5]       #out608
        self.d16=dpn92[5:9]     #out1096
        self.d32=dpn92[9:29]

        self.u16=DUB(2432,512,256)
        self.u8=DUB(1352,256,128)
        self.u4=DUB(736,128,256)
        self.u2=DUB(256,64,128)
        self.u1=DUB(128,32,16)

        self.in16=nn.InstanceNorm2d(1096,affine=False)
        self.in8=nn.InstanceNorm2d(608,affine=False)

        # Details Refinement sub-Net
        self.d4u1=nn.Sequential(
            nn.Conv2d(3,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            invPixelShuffle(4),
            nn.Conv2d(256,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            nn.Sequential(*[WAB(16) for _ in range(3)]),
            nn.Conv2d(16, 256, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 13, 3, 1, 1, bias=True)
        )

        self.tail = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DCN(32, 3, 3, 1, 1)
        )

    def forward(self,x):
        b,c,h,w=x.shape
        mod1=h%64
        mod2=w%64
        if(mod1):
            down1=64-mod1
            x=F.pad(x,(0,0,0,down1),"reflect")
        if(mod2):
            down2=64-mod2
            x=F.pad(x,(0,down2,0,0),"reflect")

        d8=self.d8(x)
        d16=self.d16(d8)
        d32=self.d32(d16)
        d16=torch.cat(d16,1)
        d8=torch.cat(d8,1)
        d16=self.in16(d16)
        d8=self.in8(d8)

        u16=self.u16(torch.cat(d32,1))
        u8=self.u8(torch.cat([u16,d16],1))
        u4=self.u4(torch.cat([u8,d8],1))
        u2=self.u2(u4)
        u1=self.u1(u2)

        d64u1=self.d64u1(x)
        d4u1=self.d4u1(x)
        x=torch.cat([u1,d64u1,d4u1],1)
        x = self.tail(x)

        if(mod1):x=x[:,:,:-down1,:]
        if(mod2):x=x[:,:,:,:-down2]
        return x.clamp(0,1)
