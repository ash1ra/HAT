import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Literal, Optional, Self, overload

import torch
from einops import einsum, rearrange, repeat
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def create_logger(
    log_level: str,
    log_file_name: str,
    max_log_file_size: int = 5 * 1024 * 1024,
    backup_count: int = 10,
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S",
    )

    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    Path("logs").mkdir(parents=True, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_file_name = f"logs/{log_file_name}_{current_date}.log"

    file_handler = RotatingFileHandler(
        filename=log_file_name,
        maxBytes=max_log_file_size,
        backupCount=backup_count,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = create_logger(log_level="INFO", log_file_name="HAT")


class Timer:
    def __init__(self) -> None:
        self.global_start_time = 0.0
        self.start_time = 0.0
        self.last_iter_duration = 0.0

    def start(self) -> None:
        if self.global_start_time == 0.0:
            self.global_start_time = perf_counter()

        self.start_time = perf_counter()

    def stop(self) -> float:
        return perf_counter() - self.start_time

    def get_elapsed_time(self) -> float:
        if self.global_start_time == 0.0:
            return 0.0
        else:
            return perf_counter() - self.global_start_time

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_iter_duration += self.stop()


class InfiniteDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, repeats: int = 1, *args, **kwargs):
        if repeats > 1:
            dataset = ConcatDataset([dataset] * repeats)

        super().__init__(dataset, *args, **kwargs)

        self._iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)  # type: ignore
        except StopIteration:
            logger.info("[DataLoader] Restarting InfiniteDataLoader...")
            self._iterator = super().__iter__()
            batch = next(self._iterator)
        return batch


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


def rgb2y(img_tensor: Tensor) -> Tensor:
    weights = torch.tensor(
        [65.481, 128.553, 24.966],
        device=img_tensor.device,
        dtype=img_tensor.dtype,
    )
    bias = 16.0

    return (einsum(img_tensor, weights, "b c h w, c -> b h w").unsqueeze(1) + bias) / 255


def calculate_psnr(sr_img_tensor: Tensor, hr_img_tensor: Tensor, crop_border: int) -> float:
    if sr_img_tensor.dim() == 3:
        sr_img_tensor = rearrange(sr_img_tensor, "c h w -> 1 c h w")
        hr_img_tensor = rearrange(hr_img_tensor, "c h w -> 1 c h w")

    sr_img_tensor = torch.round(rgb2y(sr_img_tensor) * 255.0)
    hr_img_tensor = torch.round(rgb2y(hr_img_tensor) * 255.0)

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    mse = torch.mean((sr_img_tensor - hr_img_tensor) ** 2)

    if mse == 0:
        return float("inf")
    else:
        return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()


def create_window_for_ssim_metric(window_size: int, sigma: float, num_channels: int) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    window_1d = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma**2))
    window_1d /= window_1d.sum()

    window_2d = einsum(window_1d, window_1d, "i, j -> i j")

    return repeat(window_2d, "h w -> c 1 h w", c=num_channels)


@overload
def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    return_map: Literal[False],
    window: Optional[Tensor] = None,
) -> float: ...


@overload
def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    return_map: Literal[True],
    window: Optional[Tensor] = None,
) -> tuple[float, Tensor]: ...


def ssim_metric(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    window_size: int,
    window: Optional[Tensor] = None,
    return_map: bool = False,
) -> float | tuple[float, Tensor]:
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    if window is None:
        window = create_window_for_ssim_metric(
            window_size=window_size,
            sigma=1.5,
            num_channels=sr_img_tensor.size(1),
        ).to(device=sr_img_tensor.device, dtype=sr_img_tensor.dtype)

    mu_input = torch.cat([sr_img_tensor, hr_img_tensor], dim=0)
    mu_output = F.conv2d(
        input=mu_input,
        weight=window,
        stride=1,
        padding=0,
        groups=sr_img_tensor.size(1),
    )

    mu1, mu2 = mu_output.chunk(2, dim=0)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma_input = torch.cat(
        [
            sr_img_tensor * sr_img_tensor,
            hr_img_tensor * hr_img_tensor,
            sr_img_tensor * hr_img_tensor,
        ],
        dim=0,
    )
    sigma_output = F.conv2d(
        input=sigma_input,
        weight=window,
        stride=1,
        padding=0,
        groups=sr_img_tensor.size(1),
    )

    conv_sr_sq, conv_hr_sq, conv_sr_hr = sigma_output.chunk(3, dim=0)

    sigma1_sq = conv_sr_sq - mu1_sq
    sigma2_sq = conv_hr_sq - mu2_sq
    sigma12 = conv_sr_hr - mu1_mu2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    mean_metric = (2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)

    ssim_map = mean_metric * contrast_metric
    ssim_value = ssim_map.mean().item()

    if return_map:
        return ssim_value, ssim_map
    else:
        return ssim_value


@overload
def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: Literal[False] = False,
) -> float: ...


@overload
def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: Literal[True] = True,
) -> tuple[float, Tensor]: ...


def calculate_ssim(
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    crop_border: int,
    window_size: int = 11,
    return_map: bool = False,
) -> float | tuple[float, Tensor | None]:
    if sr_img_tensor.dim() == 3:
        sr_img_tensor = rearrange(sr_img_tensor, "c h w -> 1 c h w")
        hr_img_tensor = rearrange(hr_img_tensor, "c h w -> 1 c h w")

    sr_img_tensor = rgb2y(sr_img_tensor) * 255.0
    hr_img_tensor = rgb2y(hr_img_tensor) * 255.0

    if crop_border > 0:
        sr_img_tensor = sr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]
        hr_img_tensor = hr_img_tensor[..., crop_border:-crop_border, crop_border:-crop_border]

    if sr_img_tensor.size(-1) < window_size or sr_img_tensor.size(-2) < window_size:
        if return_map:
            return 0.0, None
        else:
            return 0.0

    return ssim_metric(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        window_size=window_size,
        return_map=return_map,
    )


def format_time(total_seconds: float) -> str:
    total_seconds = 0 if total_seconds < 0 else total_seconds

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
