import torch.nn as nn
from attention_mechanism import Self_Attn, Local_Self_Attn
import torch.nn.functional as F
import torch

# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, 256, 4, 1, 0) #4
        self.deconv1_1_bn = nn.BatchNorm2d(256)
        self.deconv1_2 = nn.ConvTranspose2d(2, 256, 4, 1, 0) #4
        self.deconv1_2_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1) #8
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1) #16
        self.deconv3_bn = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 2) #30
        self.deconv4_bn = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 4, 2, 2) #58
        self.deconv5_bn = nn.BatchNorm2d(32)

        #self.deconv5_5 = nn.ConvTranspose2d(96, 32, 1, 1, 0)  # 58
        #self.deconv5_5bn = nn.BatchNorm2d(32)

        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1) #116

        self.attn4 = Self_Attn(64, 'relu')
        self.attn5 = Self_Attn(32, 'relu')
        #self.gamma = nn.Parameter(torch.zeros(1))
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label,epoch):#[32,100,1,1] #[32,2,1,1]
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))#[32,256,4,4]
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))#[32,256,4,4]
        #print(y.shape)
        x = torch.cat([x, y], 1) #[32,512,4,4]
        #print(x.shape)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        #print(x.shape) torch.Size([32, 64, 30, 30])
        if(epoch<500):
            return  x      #torch.Size([32, 64, 30, 30])
        else:
            m_batchsize, c, width, height = x.size()
            patch_x = x.view(m_batchsize, c, 5, 6, 5, 6)
            patch_x = patch_x.permute(0, 2, 4, 1, 3, 5).contiguous().view(m_batchsize, 25, c, 6, 6)
            patch_x = patch_x.contiguous().view(m_batchsize * 25, c, 6, 6)
            # print(patch_x.shape)

            x, lp4 = self.attn4(patch_x)

            y = x.view(m_batchsize, 25, c, 6, 6)
            y = y.view(m_batchsize, 5, 5, c, 6, 6)
            y = y.permute(0, 3, 1, 4, 2, 5)
            # print(y.shape)
            y = y.reshape(m_batchsize, c, 30, 30)

            x = F.relu(self.deconv5_bn(self.deconv5(y)))

            x, p5 = self.attn5(x)

            # x = torch.cat([x, y], 1)
            # x = F.tanh(self.deconv6(x))

            x = F.tanh(self.deconv6(x))

            # x = F.relu(self.deconv5_5bn(self.deconv5_5(x)))

            # x = F.relu(self.deconv4_bn(self.deconv4(x)))
            # x = F.tanh(self.deconv5(x))

            return x


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv0   =  nn.Conv2d(in_channels=2, out_channels=64, kernel_size=1)
        self.conv1_1 = nn.Conv2d(1, 32, 4, 2, 1)  #bsx32x58x58
        self.conv1_2 = nn.Conv2d(2, 32, 4, 2, 1)  #bsx32x58x58
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 2)   #bsx64x58x58-->bsx128x30x30
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 2)   #128 30 30 、、、 256 16 16
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)   #256 16  16 、、、、512 8 8
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)  #512 8 8 、、、、1024 4 4
        self.conv5_bn = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(1024, 1, 4, 2, 0)    #1024 4 4 、、、、、1 1 1
        self.attn2 = Self_Attn(128, 'relu')
        #self.attn3 = Self_Attn(512, 'relu')



    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label,epoch,full):
        if full:
            # print(input.shape)
            x = F.leaky_relu(self.conv1_1(input.to(torch.float32)), 0.2)
            y = F.leaky_relu(self.conv1_2(label), 0.2)
            # print(x.shape)
            # print(y.shape)
            x = torch.cat([x, y], 1)
            x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
            x, p2 = self.attn2(x)
            x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
            # x, p3 = self.attn3(x)
            x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
            x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
            x = F.sigmoid(self.conv6(x))
            return x
        else:
            if(epoch<500):
                label = F.leaky_relu(self.conv0(label), 0.2) #torch.Size([32, 64, 30, 30])
                x = torch.cat([input, label], 1)
                x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
                # x, p3 = self.attn3(x)
                x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
                x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
                x = F.sigmoid(self.conv6(x))
                return x
            else:
                # print(input.shape)
                x = F.leaky_relu(self.conv1_1(input.to(torch.float32)), 0.2)
                y = F.leaky_relu(self.conv1_2(label), 0.2)
                # print(x.shape)
                # print(y.shape)
                x = torch.cat([x, y], 1)
                x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
                x, p2 = self.attn2(x)
                x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
                # x, p3 = self.attn3(x)
                x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
                x = F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2)
                x = F.sigmoid(self.conv6(x))
                return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()