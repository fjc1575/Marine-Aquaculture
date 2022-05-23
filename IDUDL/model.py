import torch
from torch import nn
from torch.nn import functional as F

number = 3

class FEN(nn.Module):

    def __init__(self, input_dim, channel, out_channel):
        super(FEN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, channel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(number):
            self.conv2.append(nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append(nn.BatchNorm2d(channel))
        self.conv3 = nn.Conv2d(channel, out_channel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(number):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super().__init__()
        self.doubleconv= nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU()
        )
    def forward(self,x):
        return self.doubleconv(x)


class FCSSN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.input=DoubleConv(in_channels,64)

        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64,128)

        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128,256)

        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256,512)

        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512,1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):

        c1 = self.input(x)

        p1 = self.pool1(c1)
        c2 = self.conv2(p1)

        p2 = self.pool2(c2)
        c3 = self.conv3(p2)

        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        p4 = self.pool4(c4)
        c5 = self.conv5(p4)


        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        out = nn.ReLU()(c10)

        return out





def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)