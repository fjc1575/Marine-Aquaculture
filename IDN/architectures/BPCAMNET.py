import torch
import torch.nn as nn

import torch.nn.functional as F


from utils.visual import visualize_output

from backbone.ResNet import ResNet34, ResNet50, ResNet101
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

############正常###############
        x = self.b4(res4)
        x = self.p3(x, res3)
        x = self.b3(x)
        x = self.p2(x, res2)
        x = self.b2(x)
        x = self.Conv3(x)
        x = self.p1(x, res1)
        x = self.b1(x)
##############消融实验###############
        # x = res4
        # x = self.p3(x, res3)
        # x = self.p2(x, res2)
        # x = self.Conv3(x)
        # x = self.p1(x, res1)
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




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
        # self.seg_decoder = SegDecoder(encode_channels, decode_channels, bin_size=[(8, 8), (8, 8), (8, 8), (8, 8)])

        # self.seg_decoder = SegDecoder(encode_channels, decode_channels, bin_size=[(1, 1), (1, 1), (1, 1), (1, 1)])
        self.seg_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                      nn.Dropout2d(p=dropout, inplace=True),
                                      Conv(encode_channels[-4], num_classes, kernel_size=1))

        self.fuse_head = nn.Sequential(SeparableConvBNReLU(encode_channels[-4], encode_channels[-4], kernel_size=3),
                                       nn.Dropout2d(p=dropout, inplace=True),
                                       Conv(encode_channels[-4], num_classes, kernel_size=1))


    def forward(self, x):

        res1, res2, res3, res4 = self.backbone(x)
        feature_seg = self.seg_decoder(res1, res2, res3, res4, x.size()[2], x.size()[3])
        seg = self.seg_head(feature_seg)
        seg = F.interpolate(seg, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        feature_edge = self.edge_decoder(res1, res2, res3, res4, x.size()[2], x.size()[3])
        edge = self.edge_head(feature_edge)
        edge = F.interpolate(edge, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        out = self.fuse_head(feature_seg + feature_edge)
        out = F.interpolate(out, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=False)

        if self.training:
            return seg, edge, out
        else:
            if self.mode == 'edge':
                return edge
            elif self.mode == 'seg':
                return seg
            else:
                return out

    # def load_pretrained_weights(self,model, weight_path, strict=False):
    #     """
    #     加载预训练权重到模型中，处理潜在的键不匹配和大小不匹配问题。
    #
    #     参数:
    #     - model: 需要加载权重的模型对象。
    #     - weight_path: 预训练权重的文件路径。
    #     - strict: 是否严格匹配所有参数。如果为 False，将忽略不匹配的层。
    #
    #     返回:
    #     - model: 加载了预训练权重的模型。
    #     """
    #
    #     # 加载预训练的 state_dict
    #     pretrained_dict = torch.load(weight_path)
    #
    #     # 获取当前模型的 state_dict
    #     model_dict = model.state_dict()
    #
    #     # 过滤掉预训练的 state_dict 中不匹配的部分
    #     if not strict:
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                            k in model_dict and model_dict[k].size() == v.size()}
    #
    #     # 更新当前模型的 state_dict
    #     model_dict.update(pretrained_dict)
    #
    #     # 加载更新后的 state_dict
    #     model.load_state_dict(model_dict)
    #
    #     # 处理可能的大小不匹配问题
    #     for name, param in model.named_parameters():
    #         if name not in pretrained_dict:
    #             if "seg_head" in name or "fuse_head" in name:
    #                 if isinstance(param, nn.Conv2d):
    #                     nn.init.kaiming_normal_(param.weight, mode='fan_out', nonlinearity='relu')
    #                 elif isinstance(param, nn.BatchNorm2d):
    #                     nn.init.constant_(param.weight, 1)
    #                     nn.init.constant_(param.bias, 0)
    #                 elif isinstance(param, nn.Linear):
    #                     nn.init.normal_(param.weight, 0, 0.01)
    #                     nn.init.constant_(param.bias, 0)
    #
    #     return model


if __name__ == '__main__':
    model = BPCAMNET(mode='out').to('cuda')
    model.eval()
    x = torch.randn(1, 3, 512, 512).to('cuda')
    y = model(x)
    print(y.size())
    print(compute_flops_params(model, (1, 3, 512, 512)))



