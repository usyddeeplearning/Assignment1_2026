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


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Glorot normal initialization.

    std = gain * sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0.0, std)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Glorot uniform initialization.

    bound = sqrt(3) * std,  std = gain * sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    return tensor
