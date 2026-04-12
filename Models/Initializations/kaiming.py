import math

import torch


def _calculate_fan(tensor: torch.Tensor):
    ndim = tensor.dim()
    if ndim < 2:
        raise ValueError("Fan cannot be computed for tensors with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = tensor[0][0].numel() if ndim > 2 else 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def kaiming_normal_(tensor: torch.Tensor, mode: str = "fan_in") -> torch.Tensor:
    """He normal initialization for layers followed by ReLU.

    std = sqrt(2 / fan)
    """
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = math.sqrt(2.0 / fan)
    with torch.no_grad():
        tensor.normal_(0.0, std)
    return tensor


def kaiming_uniform_(tensor: torch.Tensor, mode: str = "fan_in") -> torch.Tensor:
    """He uniform initialization for layers followed by ReLU.

    bound = sqrt(3) * std,  std = sqrt(2 / fan)
    """
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    std = math.sqrt(2.0 / fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor
