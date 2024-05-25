import torch.nn.functional as F
from torchaudio.models import ConvTasNet

from losses import ScaleInvariantSDRLoss
from models import UNet, UNetDNP, TransUNet

default_params = {
    "UNet": {
        "n_channels": 1,
        "n_class": 2,
        "unet_depth": 6, 
        "unet_scale_factor": 16,
    },
    "UNetDNP": {
        "n_channels": 1,
        "n_class": 2,
        "unet_depth": 6,
        "n_filters": 16,
    },
    "ConvTasNet": {
        "num_sources": 2,
        "enc_kernel_size": 16,
        "enc_num_feats": 128,
        "msk_kernel_size": 3,
        "msk_num_feats": 32,
        "msk_num_hidden_feats": 128,
        "msk_num_layers": 8,
        "msk_num_stacks": 3,
    },
    "TransUNet": {
        "img_dim": 256,
        "in_channels": 1,
        "classes": 2,
        "vit_blocks": 6, # 12
        "vit_heads": 4,
        "vit_dim_linear_mhsa_block": 128, # 1024
        "apply_masks": True
    },
    "SepFormer": {},
}

# 적어도 이 프로그램에서 parameters를 지정한 내용은 없다.
def get_model(name, parameters=None):
    if not parameters:
        parameters = default_params[name]

    if name == "UNet":
        model = UNet(**parameters)
        data_mode = "amplitude"
        # mse_loss는 평균 제곱 오차 손실 함수이다. f_mse_loss_example.py에 예시 있음.
        loss_fn = F.mse_loss
        loss_mode = "min"

    if name == "UNetDNP":
        model = UNetDNP(**parameters)
        data_mode = "time"
        # 클래스 간 불균형이 클 때 사용된다고 한다.
        loss_fn = ScaleInvariantSDRLoss
        loss_mode = "max"

    if name == "ConvTasNet":
        model = ConvTasNet(**parameters)
        data_mode = "time"
        loss_fn = ScaleInvariantSDRLoss
        loss_mode = "max"

    if name == "TransUNet":
        model = TransUNet(**parameters)
        data_mode = "amplitude"
        loss_fn = F.mse_loss
        loss_mode = "min"

    # if name == "SepFormer":
    #     model = Sepformer(**parameters)
    #     data_mode = "time"
    #     loss_fn = ScaleInvariantSDRLoss
    #     loss_mode = "max"

    return {
        "model": model,
        "data_mode": data_mode,
        "loss_fn": loss_fn,
        "loss_mode": loss_mode,
    }
