import torch
from torch import nn
class super_loss(nn.Module):
    def __init__(self,loss_type=None):
        super(super_loss, self).__init__()
        self.loss=loss_type
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        
        


    def forward(self, pansharpening,target):
        if (self.loss == 'L1'):
            return self.L1(pansharpening,target)
        elif(self.loss == 'L2'):
            return self.L2(target,pansharpening) 

        # elif(self.loss == 'SAM'):
        #     SAM_loss = SAM_torch(target, pansharpening)
        #     return 0.8*self.L1(pansharpening,target)+0.1*SAM_loss
        # elif(self.loss == 'SSIM'):
        #     ssim_module =SSIM(data_range=1, size_average=True, channel=4)
        #     return self.L1(pansharpening,target)+0.1*(1-ssim_module(target,pansharpening))
        # elif(self.loss == 'SSIM+SAM'):
        #     SAM_loss = SAM_torch(target, pansharpening)
        #     ssim_module =SSIM(data_range=1, size_average=True, channel=4)
        #     return 1.0*self.L1(pansharpening,target)+0.1*(1-ssim_module(target,pansharpening))+0.1*SAM_loss