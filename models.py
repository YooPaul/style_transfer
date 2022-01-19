import torch
import torch.nn as nn
from torchvision import transforms

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(c, c, 3, 1)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = nn.Conv2d(c, c, 3, 1)
        self.bn2 = nn.BatchNorm2d(c)
        
        self.activation = nn.ReLU(True)

    def forward(self, x):
        identity = x

        x = self.bn1(self.conv1(x))
        x = self.activation(x)

        x = self.bn2(self.conv2(x))
        return x + transforms.CenterCrop(x.shape[-2:])(identity)


class StyleTransfer(nn.Module):
    def __init__(self):
        super(StyleTransfer, self).__init__()

        self.padding = nn.ReflectionPad2d(40)

        conv_package = lambda in_c, out_c, kernel, s : [nn.Conv2d(in_c, out_c, kernel, s, (kernel - 1) // 2), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        deconv_package = lambda in_c, out_c, kernel, s : [nn.ConvTranspose2d(in_c, out_c, kernel, s, padding=1, output_padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)]
        
        
        self.conv_block = nn.Sequential(  *(conv_package(3, 32, 9, 1) + conv_package(32, 64, 3, 2) + conv_package(64, 128, 3, 2)) )

        self.res_blocks = nn.Sequential( *[ResidualBlock(128) for _ in range(5)] )

        self.deconv_block = nn.Sequential( *(deconv_package(128, 64, 3, 2) + deconv_package(64, 32, 3, 2)) )

        self.output_layer = nn.Conv2d(32, 3, 9, 1, 'same')
    
    def forward(self, x):
        x = self.padding(x)

        x = self.conv_block(x)
        x = self.res_blocks(x)
        x = self.deconv_block(x)
        out = self.output_layer(x)
        return (torch.tanh(out) + 1) / 2 # normalize output to [0, 1]
