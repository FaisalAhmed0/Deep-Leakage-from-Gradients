import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
  '''
  This is a bottleneck layer for building ResNet 50 with sigmoid activations
  '''
  def __init__(self, in_channels, out_channels):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
    self.bn1 =  nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    

    nn.init.uniform_(self.conv1.weight, a=-0.5 , b=0.5)
    nn.init.uniform_(self.conv2.weight, a=-0.5 , b=0.5)



    self.shortcut_projection = None
    if in_channels != out_channels*4:
      self.shortcut_projection = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
          nn.BatchNorm2d(out_channels)   
      )

  def forward(self, x):
    output = torch.sigmoid(self.bn1(self.conv1(x)))
    output = self.bn2(self.conv2(output))
    # if input and output are not the same shape, perform a projection before adding the input to the output
    if self.shortcut_projection:
      output += self.shortcut_projection(x)
    else:
      output += x
    return torch.sigmoid(output)


class ResNet18(nn.Module):
  '''
  Resnet50 with sigmoid activations and strides of 1
  '''
  def __init__(self, in_channels, out_channels):
    super(ResNet18, self).__init__()
    self.blocks = [2, 2, 2, 2]
    self.block_input = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=1, padding=(3, 3), bias=False)
    self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    self.layer1 = self._build_layer(self.blocks[0], 64)
    self.layer2 = self._build_layer(self.blocks[1], 128)
    self.layer3 = self._build_layer(self.blocks[2], 256)
    self.layer4 = self._build_layer(self.blocks[3], 512)
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.linear = nn.Linear(512, out_channels)

    nn.init.uniform_(self.conv1.weight, a=-0.5 , b=0.5)
    # nn.init.uniform_(self.conv1.bias, a=-0.5 , b=0.5)
    nn.init.uniform_(self.linear.weight, a=-0.5 , b=0.5)
    nn.init.uniform_(self.linear.bias, a=-0.5 , b=0.5)

  def _build_layer(self, blocks, block_output):
    layers = []
    for block in range(blocks):
      layers.append(BasicBlock(self.block_input, block_output))
      self.block_input = block_output * 1
    return nn.Sequential(*layers)

  def forward(self, x):
    x =  torch.sigmoid(self.bn1(self.conv1(x)))
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = self.linear(x.view(-1, 512))
    return x


class LeNet(nn.Module):
  def __init__(self, in_channels, out_channels, img_size, device):
      super(LeNet, self).__init__()
      self.conv = nn.Sequential(
          nn.Conv2d(in_channels, 12, kernel_size=5, padding=5//2, stride=2),
          nn.Sigmoid(),
          nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
          nn.Sigmoid(),
          nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
          nn.Sigmoid(),
      )
      x_dummy = torch.randn(1, in_channels, img_size, img_size).to(device)
      # print((self.body(x_dummy)).shape)
      out_size = torch.prod( torch.tensor((self.conv(x_dummy)).shape))
      # print(out_size)
      self.linear = nn.Sequential(
          nn.Linear(out_size, out_channels)
      )
      for layer in self.conv:
        if type(layer) != nn.Sigmoid:
          nn.init.uniform_(layer.weight, a=-0.5 , b=0.5)
          nn.init.uniform_(layer.bias, a=-0.5 , b=0.5)
      for layer in self.linear:
        nn.init.uniform_(layer.weight, a=-0.5 , b=0.5)
        nn.init.uniform_(layer.bias, a=-0.5 , b=0.5)

  def forward(self, x):
    x = self.conv(x)
    x = x.view(x.shape[0], -1)
    x = self.linear(x)
    return x