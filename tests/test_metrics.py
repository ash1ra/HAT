import pytest
import torch
from torch import Tensor

import utils

TEST_PSNR_IMGS = [
    # batch_size, num_channels, img_height, img_width, crop_border
    (1, 3, 64, 64, 4),
    (8, 3, 64, 64, 0),
    (2, 3, 37, 53, 2),
    (1, 1, 64, 64, 4),
]

TEST_SSIM_IMGS = [
    # batch_size, num_channels, img_height, img_width, crop_border, window_size
    (1, 3, 64, 64, 4, 8),
    (4, 3, 64, 64, 0, 11),
    (2, 3, 63, 63, 2, 11),
    (1, 1, 48, 48, 4, 8),
]


def test_rgb2y_shape() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    input_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)

    output_img_tensor = utils.rgb2y(input_img_tensor)

    assert output_img_tensor.shape == (batch_size, 1, img_height, img_width)


def test_rgb2y_black_image() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    input_img_tensor = torch.zeros(batch_size, num_channels, img_height, img_width)

    expected_value = 16.0 / 255.0
    expected_output_img_tensor = torch.full((batch_size, 1, img_height, img_width), fill_value=expected_value)

    output_img_tensor = utils.rgb2y(input_img_tensor)

    assert torch.allclose(output_img_tensor, expected_output_img_tensor, rtol=1e-6)


def test_rgb2y_white_image() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    input_img_tensor = torch.ones(batch_size, num_channels, img_height, img_width)

    expected_value = (65.481 + 128.553 + 24.966 + 16.0) / 255.0
    expected_output_img_tensor = torch.full((batch_size, 1, img_height, img_width), fill_value=expected_value)

    output_img_tensor = utils.rgb2y(img_tensor=input_img_tensor)

    assert torch.allclose(output_img_tensor, expected_output_img_tensor, rtol=1e-6)


@pytest.mark.parametrize("batch_size, num_channels, img_height, img_width, crop_border", TEST_PSNR_IMGS)
def test_calculate_psnr(
    batch_size: int,
    num_channels: int,
    img_height: int,
    img_width: int,
    crop_border: int,
) -> None:
    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)

    psnr_value = utils.calculate_psnr(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=crop_border,
    )

    assert psnr_value > 0.0


def test_calculate_psnr_identical() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = sr_img_tensor.clone()

    psnr_value = utils.calculate_psnr(sr_img_tensor=sr_img_tensor, hr_img_tensor=hr_img_tensor, crop_border=0)

    assert psnr_value == float("inf")


def test_calculate_psnr_opposite() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    sr_img_tensor = torch.zeros(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.ones(batch_size, num_channels, img_height, img_width)

    psnr_value = utils.calculate_psnr(sr_img_tensor=sr_img_tensor, hr_img_tensor=hr_img_tensor, crop_border=0)

    assert psnr_value == pytest.approx(1.321921, rel=1e-5)


@pytest.mark.parametrize("batch_size, num_channels, img_height, img_width, crop_border, window_size", TEST_SSIM_IMGS)
def test_calculate_ssim(
    batch_size: int,
    num_channels: int,
    img_height: int,
    img_width: int,
    crop_border: int,
    window_size: int,
) -> None:
    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)

    ssim_value = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=crop_border,
        window_size=window_size,
        return_map=False,
    )

    assert -1.0 <= ssim_value <= 1.0


def test_calculate_ssim_identical() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64

    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = sr_img_tensor.clone()

    ssim_value = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=4,
        window_size=11,
        return_map=False,
    )

    assert ssim_value == pytest.approx(1.0, rel=1e-5)


def test_calculate_ssim_opposite() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64

    sr_img_tensor = torch.zeros(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.ones(batch_size, num_channels, img_height, img_width)

    ssim_value = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=4,
        window_size=11,
        return_map=False,
    )

    assert ssim_value == pytest.approx(0.135, abs=1e-2)


def test_calculate_ssim_small_img() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 5, 5

    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)

    ssim_value = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=4,
        window_size=11,
        return_map=False,
    )

    assert ssim_value == 0.0


def test_calculate_ssim_map() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    crop_border = 4
    window_size = 11

    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)
    hr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width)

    ssim_value, ssim_map = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=crop_border,
        window_size=window_size,
        return_map=True,
    )

    assert isinstance(ssim_value, float)
    assert isinstance(ssim_map, Tensor)

    expected_img_height = img_height - 2 * crop_border - window_size + 1
    expected_img_width = img_width - 2 * crop_border - window_size + 1

    assert ssim_map.shape == (batch_size, 1, expected_img_height, expected_img_width)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ssim_device_compatibility() -> None:
    batch_size, num_channels, img_height, img_width = 1, 3, 64, 64
    device = "cuda"
    dtype = torch.bfloat16

    sr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width).to(device=device, dtype=dtype)
    hr_img_tensor = torch.rand(batch_size, num_channels, img_height, img_width).to(device=device, dtype=dtype)

    ssim_value, ssim_map = utils.calculate_ssim(
        sr_img_tensor=sr_img_tensor,
        hr_img_tensor=hr_img_tensor,
        crop_border=4,
        window_size=11,
        return_map=True,
    )

    assert ssim_map.device.type == device
    assert ssim_map.dtype == dtype
