import torch as th
import torch.nn as nn
from dual_path_transformer import DPTNet    #dptnet
from conf import dpt_conf
from dccrn import DCCRN                     #dccrn
from conf import dccrn_conf
from conv_tas_net import ConvTasNet    #convtasnet
from conf import skip_conf

class TENET(nn.Module):
    def __init__(
        self,
        model="DPTNet",
    ):
        super(TENET, self).__init__()

        if model == "DPTNet":
            self.model = DPTNet(**dpt_conf)
        elif model == "DCCRN":
            self.model = DCCRN(**dccrn_conf)
        elif model == "TCN-skip":
            self.model = ConvTasNet(**skip_conf)

    def forward(self, x1, x2=None, eval_mode=False):
        if eval_mode:
            est, feat = self.model(x1)
            return est
        else:
            est1, feat1 = self.model(x1)
            est2, feat2 = self.model(x2)
            return [est1, est2], [], []
