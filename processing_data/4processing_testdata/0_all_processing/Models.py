from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
 #           nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
  #          nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class modelF1(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=4, out_ch=1):
        super(modelF1, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        #print('x size', x.size())
        e1 = self.Conv1(x) # in_ch=4 input channels
        #print('e1 size', e1.size())

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #print('e2 size', e2.size())
        
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        #print('e3 size', e3.size())
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        #print('e4 size', e4.size())
        
        d4 = self.Up4(e4)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        #print('d4 size', d4.size())

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        #print('d3 size', d3.size())

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2) #64 output channels
        #print('d2 size', d2.size())

        #d1 = self.Conv(d2)
        
        return d2

class modelP1(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super(modelP1, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        return self.Conv(x)


class modelB1(nn.Module):
    #b_width = 16 is as in the br-net code https://github.com/QingyuZhao/BR-Net/blob/master/BR_net_CF_net_toy_example.ipynb
    def __init__(self, in_ch=4, out_ch=1, b_width = 16, n_confounders=3):
        super(modelB1, self).__init__()

        n1 = 64
        in_feat = 512*24*20
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        
        
        self.b_pred = nn.Sequential(nn.Flatten(start_dim=1), 
                                  nn.Linear(in_features = in_feat, out_features= b_width, bias=True),
                                  nn.ReLU(), 
                                  nn.Linear(in_features = b_width, out_features = n_confounders, bias=True),
                                  nn.ReLU()
        )

        
    def forward(self, x):
        
        x = self.Maxpool1(x) # 64 input
        x = self.Conv2(x)
        
        x = self.Maxpool2(x)
        x = self.Conv3(x)
        
        x = self.Maxpool3(x)
        x = self.Conv4(x)
        
        return self.b_pred(x)