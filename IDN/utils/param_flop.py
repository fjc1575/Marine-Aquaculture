import torch


def compute_flops_params(model, input_shape):
    from thop import profile, clever_format
    input = torch.randn(input_shape).to('cuda')
    model = model.to('cuda')
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params