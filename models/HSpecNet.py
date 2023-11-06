from math import ceil
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
class HSpecNet(nn.Module):
    def __init__(self,data_in_channel):
        super(HSpecNet, self).__init__()
        self.in_channels    = data_in_channel
        self.out_channels   = data_in_channel
        self.factor         = 4

        self.mid_channels_1 = 16
        self.mid_channels_2 = 64
        self.pan_conv1 = nn.Conv2d(in_channels=1, out_channels=self.mid_channels_1, kernel_size=3,padding=1,stride=1)
        self.pan_conv2 = nn.Conv2d(in_channels=self.mid_channels_1, out_channels=self.mid_channels_1, kernel_size=3,padding=1,stride=1)
        
        self.hs_conv1 = nn.Conv2d(in_channels=self.in_channels , out_channels=self.mid_channels_2, kernel_size=1)
        self.hs_conv2 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=1)
        
        self.fusion_conv1 = nn.Conv2d(in_channels=self.mid_channels_1+self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=3,padding=1,stride=1)
        self.fusion_conv2 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=3,padding=1,stride=1)
        self.fusion_conv3 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=3,padding=1,stride=1)
        self.fusion_conv4 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=3,padding=1,stride=1)
        
        self.restore_conv1 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.mid_channels_2, kernel_size=1)
        self.restore_conv2 = nn.Conv2d(in_channels=self.mid_channels_2, out_channels=self.out_channels , kernel_size=1)
        

    def forward(self, X_PAN, X_MS_UP): 
      
        
        pan_feature =  F.relu(self.pan_conv2(F.relu(self.pan_conv1(X_PAN))))
        pan_feature =   X_PAN - pan_feature
        
        HS_feature1 =F.relu(self.hs_conv1(X_MS_UP))
        HS_feature2 =F.relu(self.hs_conv2(HS_feature1))
        
        feature = torch.cat((pan_feature,HS_feature2),dim=1)
        feature = F.relu(self.fusion_conv4(F.relu(self.fusion_conv3(F.relu(self.fusion_conv2(F.relu(self.fusion_conv1(feature))))))))
        
        feature = HS_feature2+feature
        restore1 = F.relu(self.restore_conv1(feature))
        restore2 = self.restore_conv2(restore1+HS_feature1)
        output = restore2+X_MS_UP
        return output