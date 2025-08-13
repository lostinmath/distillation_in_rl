def to_device(nested_dict, device="cpu"):
    """
    Converts nested dict with tensors to device
    """
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            to_device(v, device)
        else:
            nested_dict[k] = v.to(device=device)
