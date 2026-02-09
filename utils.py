from einops import rearrange
from torch import Tensor


def split_img_into_windows(img_tensor: Tensor, window_size: int) -> Tensor:
    return rearrange(
        img_tensor,
        "b (nw_h ws_h) (nw_w ws_w) c -> (b nw_h nw_w) ws_h ws_w c",
        ws_h=window_size,
        ws_w=window_size,
    )


def combine_windows_into_img(windows_tensor: Tensor, img_height: int, img_width: int) -> Tensor:
    return rearrange(
        windows_tensor,
        "(b nw_h nw_w) ws_h ws_w c -> b (nw_h ws_h) (nw_w ws_w) c",
        nw_h=img_height // windows_tensor.shape[1],
        nw_w=img_width // windows_tensor.shape[2],
    )
