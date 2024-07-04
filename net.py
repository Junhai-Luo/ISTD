from torch import nn

import os
from loss import SoftIoULoss, ISNetLoss
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode='test'):
        super(Net, self).__init__()

        self.model_name = model_name
        self.cal_loss = SoftIoULoss()

        if model_name == 'U_Net':
            self.model = U_Net()
        elif model_name == 'U_Net4':
            self.model = U_Net4()
        elif model_name == 'R2U_Net':
            self.model = R2U_Net()
        elif model_name == 'AttU_Net':
            self.model = AttU_Net()
        elif model_name == 'R2AttU_Net':
            self.model = R2AttU_Net()
        elif model_name == 'NestedUNet':
            self.model = NestedUNet()
        elif model_name == 'ResU_Net4':
            self.model = ResU_Net4()
        elif model_name == 'U_Net4DC':
            self.model = U_Net4DC()
        elif model_name == 'ResU_Net4DC':
            self.model = ResU_Net4DC()
        elif model_name == 'U_Net4DCio':
            self.model = U_Net4DCio()
        elif model_name == 'ResU_Net4DCio':
            self.model = ResU_Net4DCio()
        elif model_name == 'U_Net4_fuse':
            self.model = U_Net4_fuse()
        elif model_name == 'U_Net4DC_fuse':
            self.model = U_Net4DC_fuse()
        elif model_name == 'ResU_Net4_fuse':
            self.model = ResU_Net4_fuse()
        elif model_name == 'ResU_Net4DC_fuse':
            self.model = ResU_Net4DC_fuse()
        elif model_name == 'U_Net4DCio_fuse':
            self.model = U_Net4DCio_fuse()
        elif model_name == 'ResU_Net4DCio_fuse':
            self.model = ResU_Net4DCio_fuse()



    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
