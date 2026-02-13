from typing import Callable
import torch
import pytest

import models


TEST_IMGS = [
    # batch_size, img_height, img_width, num_channels, scaling_factor
    (1, 64, 64, 3, 2),  # Single square image with scaling_factor = 2
    (1, 64, 64, 3, 3),  # Single square image with scaling_factor = 3
    (1, 64, 64, 3, 4),  # Single square image with scaling_factor = 4
    (4, 64, 64, 3, 2),  # Multiple square images with scaling_factor = 2
    (4, 64, 64, 3, 3),  # Multiple square images with scaling_factor = 3
    (4, 64, 64, 3, 4),  # Multiple square images with scaling_factor = 4
    (1, 48, 96, 3, 2),  # Single rectangular image
    (4, 48, 96, 3, 2),  # Multiuple rectangular images
    (1, 64, 64, 1, 2),  # Single grayscale image
    (4, 64, 64, 1, 2),  # Multiple grayscale images
    (1, 63, 63, 3, 2),  # Single non-divisible by 8 (window_size) square image
    (1, 31, 35, 3, 4),  # Single non-divisible by 8 (window_size) rectangular image
    (1, 7, 7, 3, 2),  # Single image with size < window_size
]


@pytest.fixture()
def model_factory() -> Callable:
    def _create_model(**kwargs) -> torch.nn.Module:
        model_parameters = dict(
            in_channels=3,
            num_rhag_blocks=1,
            num_hab_blocks=1,
            num_channels=32,
            compress_ratio=3,
            squeeze_factor=4,
            window_size=8,
            num_heads=4,
            cab_scale=0.01,
            train_img_size=(64, 64),
            mlp_ratio=4,
            overlap_ratio=0.5,
            oca_ratio=0.5,
            scaling_factor=4,
            drop_path_prob=0.0,
        )

        model_parameters.update(kwargs)

        return models.HAT(**model_parameters)  # type: ignore

    return _create_model


@pytest.mark.parametrize("batch_size, img_height, img_width, num_channels, scaling_factor", TEST_IMGS)
def test_hat(
    model_factory: Callable,
    batch_size: int,
    img_height: int,
    img_width: int,
    num_channels: int,
    scaling_factor: int,
) -> None:
    model = model_factory(in_channels=num_channels, scaling_factor=scaling_factor)

    input_img_tensor = torch.randn(batch_size, num_channels, img_height, img_width)

    output_img_tensor = model(input_img_tensor)

    assert input_img_tensor.shape[1] == output_img_tensor.shape[1], "Wrong number of channels"
    assert input_img_tensor.shape[2] * scaling_factor == output_img_tensor.shape[2], "Wrong height of the output image"
    assert input_img_tensor.shape[3] * scaling_factor == output_img_tensor.shape[3], "Wrong width of the output image"


def test_hat_backward_pass(model_factory: Callable) -> None:
    scaling_factor = 4

    model = model_factory(scaling_factor=scaling_factor)

    input_img_tensor = torch.randn(1, 3, 64, 64)
    target_img_tensor = torch.randn(1, 3, 64 * scaling_factor, 64 * scaling_factor)

    output_img_tensor = model(input_img_tensor)

    loss = torch.nn.functional.l1_loss(input=output_img_tensor, target=target_img_tensor)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient!"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hat_device_compatibility(model_factory: Callable) -> None:
    scaling_factor = 4
    device = "cuda"

    model = model_factory(scaling_factor=scaling_factor).to(device)

    input_img_tensor = torch.randn(1, 3, 64, 64).to(device)

    output_img_tensor = model(input_img_tensor)

    assert output_img_tensor.device.type == "cuda"


def test_hat_scaling_factor_error(model_factory: Callable) -> None:
    scaling_factor = 6  # not 2^n or 3

    with pytest.raises(ValueError):
        _ = model_factory(scaling_factor=scaling_factor)
