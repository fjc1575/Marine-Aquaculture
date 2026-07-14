import torch
import torch.nn as nn

import torch.nn.functional as F
from utils.visual import visualize_output

from backbone.ResNet import ResNet34, ResNet50
from PIL import  Image
import  numpy as np

from utils.param_flop import compute_flops_params



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





class Expaliner(nn.Module):
    def __init__(self,channel=2048):


        super(Expaliner, self).__init__()
        self.channel_out = 512
        self.conv0 = ConvBNReLU(channel, self.channel_out, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(16, stride=1)

        self.att_conv1 = SeparableConv(self.channel_out, self.channel_out, kernel_size=3)
        self.att_bn1 = nn.BatchNorm2d(self.channel_out)
        self.att_conv2 = SeparableConv(self.channel_out, self.channel_out, kernel_size=3)
        self.att_bn2 = nn.BatchNorm2d(self.channel_out)
        self.att_conv3 = SeparableConv(self.channel_out, self.channel_out, kernel_size=3)
        self.att_bn3 = nn.BatchNorm2d(self.channel_out)
        self.att_conv4 = SeparableConv(self.channel_out, self.channel_out, kernel_size=3)
        self.att_bn4 = nn.BatchNorm2d(self.channel_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, feature):

        feature = self.conv0(feature)

        input_gray = torch.mean(input, dim=1, keepdim=True)
        input_resized = F.interpolate(input_gray, (16, 16), mode='bilinear')

        ex = feature
        fe = feature.clone()  # 使用 .clone() 来确保计算图保持一致
        org = fe.clone()  # 同样，克隆原始 feature 防止修改

        # 获取尺寸
        a1, a2, a3, a4 = fe.size()
        fe = fe.view(a1, a2, -1)

        # 非in-place标准化操作，避免直接修改张量
        fe_min = fe.min(2, keepdim=True)[0]
        fe_max = fe.max(2, keepdim=True)[0]

        fe = (fe - fe_min) / (fe_max - fe_min)
        fe = fe.view(a1, a2, a3, a4)

        # 处理NaN值
        fe[torch.isnan(fe)] = 1
        fe[(org == 0)] = 0

        # 与input_resized的元素级乘法
        new_fe = fe * input_resized

        # attention层的计算
        ax = self.att_conv1(new_fe)
        ax = self.att_bn1(ax)
        ax = self.relu(ax)
        ax = self.att_conv2(ax)
        ax = self.att_bn2(ax)
        ax = self.relu(ax)
        ax = self.att_conv3(ax)
        ax = self.att_bn3(ax)
        ax = self.relu(ax)
        ax = self.att_conv4(ax)
        ax = self.att_bn4(ax)
        ax = self.relu(ax)

        # 平均池化和softmax
        ax = self.avgpool(ax)
        w = F.softmax(ax.view(ax.size(0), -1), dim=1)

        # 计算 saliency map
        b, c, u, v = fe.size()
        score_saliency_map = torch.zeros((b, 1, u, v), device=fe.device)

        # for i in range(c):
        #     saliency_map = torch.unsqueeze(ex[:, i, :, :], 1)
        #     score = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(w[:, i], 1), 1), 1)
        #     score_saliency_map += score * saliency_map
        score_saliency_map = torch.sum(w.view(a1, a2, 1, 1) * ex, dim=1, keepdim=True)
        score_saliency_map = F.relu(score_saliency_map)

        # 克隆org, 防止in-place操作
        org = score_saliency_map.clone()
        a1, a2, a3, a4 = score_saliency_map.size()
        score_saliency_map = score_saliency_map.view(a1, a2, -1)

        # 非in-place标准化操作
        score_min = score_saliency_map.min(2, keepdim=True)[0]
        score_max = score_saliency_map.max(2, keepdim=True)[0]

        score_saliency_map = (score_saliency_map - score_min) / (score_max - score_min)
        score_saliency_map = score_saliency_map.view(a1, a2, a3, a4)

        # 处理NaN值
        score_saliency_map[torch.isnan(score_saliency_map)] = org[torch.isnan(score_saliency_map)]

        att = score_saliency_map

        # # # # attention机制
        # rx = att * ex
        # rx = rx + ex

        return att.squeeze(1)


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
        # self.se_module =  SqueezeExcitation(num_channel)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)#节点（Node）方向
        out = self.relu(out + x)

        #channel
        # out = out.permute(0, 3, 1, 2)
        # out = self.se_module(out)
        # out = out.permute(0, 2, 3, 1)


        out = self.conv2(out)#通道（Channel）方向
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_se = self.se_module(x)
        cam = self.conv_cam(self.dropout(x_se))
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
    def __init__(self, dim=512, bin=(2, 2),norm_layer=nn.BatchNorm2d):
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

#
# class EdgeBlock(nn.Module):
#     def __init__(self, dim, fc_ratio):
#         super(EdgeBlock, self).__init__()
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(dim, dim // fc_ratio, 1, 1),
#             nn.ReLU6(),
#             nn.Conv2d(dim // fc_ratio, dim, 1, 1),
#             nn.Sigmoid()
#         )
#
#         self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         u = x.clone()
#
#         c_attn = self.avg_pool(x)
#         c_attn = self.fc(c_attn)
#         c_attn = u * c_attn
#
#         s_max_out, _ = torch.max(x, dim=1, keepdim=True)
#         s_avg_out = torch.mean(x, dim=1, keepdim=True)
#         s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
#         s_attn = self.s_conv(s_attn)
#         s_attn = self.sigmoid(s_attn)
#         s_attn = u * s_attn
#
#         return c_attn + s_attn

class EdgeBlock(nn.Module):
    def __init__(self, dim, fc_ratio):
        super(EdgeBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim // fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()


        # self.conv_edge = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_edge = SeparableConv(dim,dim,kernel_size=3)
        self.bn = nn.BatchNorm2d(dim)
        self.sobel_x1, self.sobel_y1 = self.get_sobel(dim, 1)

    def get_sobel(self,in_chan, out_chan):
        filter_x = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).astype(np.float32)
        filter_y = np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]).astype(np.float32)

        filter_x = filter_x.reshape((1, 1, 3, 3))
        filter_x = np.repeat(filter_x, in_chan, axis=1)
        filter_x = np.repeat(filter_x, out_chan, axis=0)

        filter_y = filter_y.reshape((1, 1, 3, 3))
        filter_y = np.repeat(filter_y, in_chan, axis=1)
        filter_y = np.repeat(filter_y, out_chan, axis=0)

        filter_x = torch.from_numpy(filter_x)
        filter_y = torch.from_numpy(filter_y)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        filter_y = nn.Parameter(filter_y, requires_grad=False)
        conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y.weight = filter_y
        sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
        sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))

        return sobel_x, sobel_y

    def run_sobel(self,conv_x, conv_y, input):
        g_x = conv_x(input)
        g_y = conv_y(input)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
        return torch.sigmoid(g) * input

    def forward(self, x):
        u = x.clone()

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        sobel_attn = self.run_sobel(self.sobel_x1, self.sobel_y1, x)
        sobel_attn = F.relu(self.bn(sobel_attn))
        sobel_attn = self.conv_edge(sobel_attn)
        sobel_attn = u + sobel_attn

        return c_attn + s_attn + sobel_attn


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


class EdgeDecoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 fc_ratio=4):
        super(EdgeDecoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)

        self.b4 = EdgeBlock(dim=decode_channels, fc_ratio=fc_ratio)

        self.p3 = Fusion(decode_channels)
        self.b3 = EdgeBlock(dim=decode_channels, fc_ratio=fc_ratio)

        self.p2 = Fusion(decode_channels)
        self.b2 = EdgeBlock(dim=decode_channels, fc_ratio=fc_ratio)

        self.Conv3 = ConvBN(encode_channels[-3], encode_channels[-4], 1)

        self.p1 = Fusion(encode_channels[-4])
        self.b1 = EdgeBlock(dim=encode_channels[-4], fc_ratio=fc_ratio)


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


class SegDecoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 bin_size=[(4, 4), (4, 4), (4, 4), (4, 4)]
                 ):
        super(SegDecoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.b4 = SegBlock(dim=decode_channels, bin=bin_size[0])

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


class ReciproCam:
    def __init__(self, device):
        self.device = device
        self.feature = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.gaussian = torch.tensor(
            [
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
                [1 / 8.0, 1 / 4.0, 1 / 8.0],
                [1 / 16.0, 1 / 8.0, 1 / 16.0],
            ]
        ).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)

        self.conv1 = nn.Conv2d(2048, 2, kernel_size=1).to(device)

    def _mosaic_feature(self, feature_map, is_gaussian=False):
        _, num_channel, height, width = feature_map.shape
        new_features = torch.zeros(height * width, num_channel, height, width).to(
            self.device
        )
        if is_gaussian is False:
            for k in range(height * width):
                for i in range(height):
                    for j in range(width):
                        if k == i * width + j:
                            new_features[k, :, i, j] = feature_map[0, :, i, j]
        else:
            for k in range(height * width):
                for i in range(height):
                    kx_s, kx_e = max(i - 1, 0), min(i + 1, height - 1)
                    sx_s = 1 if i == 0 else 0
                    sx_e = 1 if i == height - 1 else 2
                    for j in range(width):
                        ky_s, ky_e = max(j - 1, 0), min(j + 1, width - 1)
                        sy_s = 1 if j == 0 else 0
                        sy_e = 1 if j == width - 1 else 2
                        if k == i * width + j:
                            r_feature_map = (
                                feature_map[0, :, i, j]
                                .reshape(num_channel, 1, 1)
                                .repeat(1, self.gaussian.shape[0], self.gaussian.shape[1])
                            )
                            score_map = r_feature_map * self.gaussian.repeat(
                                num_channel, 1, 1
                            )
                            new_features[
                                k, :, kx_s : kx_e + 1, ky_s : ky_e + 1
                            ] = score_map[:, sx_s : sx_e + 1, sy_s : sy_e + 1]

        return new_features

    def weight_accum(self, mosaic_predict, height, width):


        cam = torch.zeros(height, width).to(self.device)
        for i in range(height):
            for j in range(width):
                cam[i, j] = self.avgpool(mosaic_predict[i * width + j].unsqueeze(0)).view(-1)
        return cam

    def __call__(self, feature_map, class_index=1,class_num=2):


        b, c, height, weight = feature_map.shape

        spatial_masked_feature_maps = []
        for i in range(b):
            spatial_masked_feature_map = self._mosaic_feature(
                feature_map[i].unsqueeze(0), is_gaussian=False
            )
            spatial_masked_feature_maps.append(spatial_masked_feature_map)

        spatial_masked_feature_map = torch.stack(spatial_masked_feature_maps, dim=0).to(self.device)

        class_masked_feature_map = self.conv1(spatial_masked_feature_map.view(-1, c, height, weight)).view(b, height * weight, class_num,height, weight)

        class_masked_feature_map = F.softmax(class_masked_feature_map, dim=2)

        target_masked_feature_map = class_masked_feature_map[:, :, class_index, :, :]

        cam_maps = []
        for i in range(b):


            cam_map = self.weight_accum(target_masked_feature_map[i], height, weight)
            cam_maps.append(cam_map)

        cam_maps = torch.stack(cam_maps, dim=0)

        #标准化
        cam_maps = cam_maps - cam_maps.min()
        cam_maps = cam_maps / cam_maps.max()

        return cam_maps

class BPCAMNET(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.,
                 num_classes=2,
                 mode = 'out'
                 ):
        super().__init__()
        self.mode = mode
        self.backbone = ResNet50(pretrained=False)

        self.edge_decoder = EdgeDecoder(encode_channels, decode_channels)
        self.edge_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                       nn.Dropout2d(p=dropout, inplace=True),
                                       Conv(encode_channels[-4], 1, kernel_size=1))

        # self.seg_decoder = SegDecoder(encode_channels, decode_channels,bin_size=[(2, 2), (2, 2), (2, 2), (2, 2)])
        self.seg_decoder = SegDecoder(encode_channels, decode_channels, bin_size=[(4, 4), (4, 4), (4, 4), (4, 4)])
        self.seg_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                      nn.Dropout2d(p=dropout, inplace=True),
                                      Conv(encode_channels[-4], num_classes, kernel_size=1))

        self.fuse_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                       nn.Dropout2d(p=dropout, inplace=True),
                                       Conv(encode_channels[-4], num_classes, kernel_size=1))

        # self.explainer = Expaliner(2048)

        self.init_weight()
    def forward(self, x):

        res1, res2, res3, res4 = self.backbone(x)


        # attn = self.explainer(x,res4)

        feature_seg = self.seg_decoder(res1, res2, res3, res4, x.size()[2], x.size()[3])
        seg = self.seg_head(feature_seg)
        seg = F.interpolate(seg, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        feature_edge = self.edge_decoder(res1, res2, res3, res4, x.size()[2], x.size()[3])
        edge = self.edge_head(feature_edge)
        edge = F.interpolate(edge, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        out = self.fuse_head(feature_seg + feature_edge)
        out = F.interpolate(out, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        if self.training:
            return seg, edge, out, res4
        else:
            if self.mode == 'edge':
                return edge
            elif self.mode == 'seg':
                return seg
            else:
                return out,torch.sum(res4,dim=1)
                # return out
    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
    model = BPCAMNET(mode='edge').to('cuda')
    model.eval()
    x = torch.randn(6, 3, 512, 512).to('cuda')
    y = model(x)
    print(y.size())
    print(compute_flops_params(model, (1, 3, 512, 512)))



