import torch
import torch.nn as nn

import torch.nn.functional as F
from timm.layers import DropPath


from backbone.ResNet import ResNet34, ResNet50
from PIL import  Image
import  numpy as np

from utils.param_flop import compute_flops_params


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )




def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W)  # [B, C, H, W]
    return out


class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, num_channels)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1)
        return x * y
class CAAM(nn.Module):

    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.se_module = SqueezeExcitation(feat_in)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)

        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_se = self.se_module(x)
        cam = self.conv_cam(self.dropout(x_se))
        cam_res = cam.clone()  # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]

        residual = x  # [B, C, H, W]
        cam = patch_split(cam, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH * rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH * rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3),
                                   x) * bin_confidence  # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats)  # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats)  # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)  # [B, bin_num_h * bin_num_w, K, C]

        query = self.proj_query(x)  # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats)  # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats)  # [B, bin_num_h * bin_num_w, K, C//2]

        aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        out = out.view(B, -1, rH, rW, value.shape[-1])  # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size)  # [B, C, H, W]

        out = residual + self.conv_out(out)

        return out


class SegBlock(nn.Module):
    def __init__(self, dim=512, bin=(2, 2),
                 drop_path=0., norm_layer=nn.BatchNorm2d):
        super().__init__()
    
        self.attn = CAAM(feat_in=dim, num_classes=2, bin_size=bin, norm_layer=nn.BatchNorm2d)

        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=False),  # 深度卷积
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),  # 逐点卷积
            norm_layer(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        u = x.clone()

        x = self.attn(x)
        x = self.conv(torch.cat([u, x], dim=1))


        return x



class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x




class SegDecoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 bin_size=[(4, 4), (4, 4), (4, 4), (4, 4)]
                 ):
        super(SegDecoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        # self.b4 = SegBlock(dim=decode_channels, bin=bin_size[0])
        self.b4 = ConvBNReLU(in_channels=decode_channels,out_channels=decode_channels,kernel_size=3)
        self.p3 = Fusion(decode_channels)
        self.b3 = SegBlock(dim=decode_channels, bin=bin_size[1])

        self.p2 = Fusion(decode_channels)
        self.b2 = SegBlock(dim=decode_channels, bin=bin_size[2])

        self.Conv3 = ConvBN(encode_channels[-3], encode_channels[-4], 1)

        self.p1 = Fusion(encode_channels[-4])
        self.b1 = SegBlock(dim=encode_channels[-4], bin=bin_size[3])


        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)


        x = self.b4(res4)
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.Conv3(x)
        x = self.p1(x, res1)
        x = self.b1(x)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




class PCAMNet(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.,
                 num_classes=2,
                 # bin_size = [(4,4), (4, 4), (4,4), (4,4)]
                 # bin_size = [(2, 2), (2, 2), (2, 2), (2, 2)]
                bin_size = [(8, 8), (8, 8), (8, 8), (8, 8)]
                 ):
        super().__init__()

        self.backbone = ResNet50(pretrained=False)



        self.seg_decoder = SegDecoder(encode_channels, decode_channels,bin_size=bin_size)
        self.seg_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                      nn.Dropout2d(p=dropout, inplace=True),
                                      Conv(encode_channels[-4], num_classes, kernel_size=1))


    def forward(self, x):

        res1, res2, res3, res4 = self.backbone(x)

        feature_seg = self.seg_decoder(res1, res2, res3, res4, x.size()[2], x.size()[3])
        seg = self.seg_head(feature_seg)
        seg = F.interpolate(seg, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        return seg

    def update_conv_layer(self, old_conv, new_in_channels, new_out_channels):
        # 获取旧的权重
        old_weights = old_conv.weight.detach()

        # 创建新的卷积层
        new_conv = nn.Conv2d(new_in_channels, new_out_channels, kernel_size=1, bias=False)

        # 复制旧权重到新权重，需要考虑输入和输出通道的变化
        with torch.no_grad():
            min_in_channels = min(new_in_channels, old_weights.size(0))
            min_out_channels = min(new_out_channels, old_weights.size(1))
            new_conv.weight[:min_out_channels, :min_in_channels, :, :] = old_weights[:min_out_channels,
                                                                         :min_in_channels, :, :]
            # 初始化其余部分的权重
            if new_in_channels > old_weights.size(1):
                torch.nn.init.xavier_uniform_(new_conv.weight[:, old_weights.size(1):, :, :])
            if new_out_channels > old_weights.size(0):
                torch.nn.init.xavier_uniform_(new_conv.weight[old_weights.size(0):, :, :, :])

        return new_conv



if __name__ == '__main__':
    bin_size = [(2, 2), (2, 2), (2, 2), (2, 2)]
    # bin_size = [(4,4),(4,4),(4,4),(4,4)]
    # bin_size = [(8,8),(8,8),(8,8),(8,8)]
    model = PCAMNet(bin_size=bin_size).to('cuda')
    # model.eval()
    x = torch.randn(1, 3, 512, 512).to('cuda')
    y = model(x)
    print(y.size())
    print(compute_flops_params(model, (1, 3, 512, 512)))



