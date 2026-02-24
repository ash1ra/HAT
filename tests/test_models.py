from typing import Callable

import pytest
import torch

import models

HAB_IMGS = [
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

RHAG_CONFIGS = [
    # window_size, num_heads, overlap_ratio
    (8, 4, 0.5),
    (4, 2, 0.5),
    (8, 4, 0.25),
]

OCAB_CONFIGS = [
    # window_size, num_heads, overlap_ratio
    (8, 4, 0.5),
    (4, 2, 0.5),
    (8, 4, 0.25),
]

HAB_CONFIGS = [
    # window_size, num_heads
    (8, 4),
    (4, 2),
]

WMSA_CONFIGS = [
    # window_size, num_heads
    (8, 4),
    (4, 2),
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
            scaling_factor=4,
            use_gradient_checkpointing=True,
        )

        model_parameters.update(kwargs)

        return models.HAT(**model_parameters)  # type: ignore

    return _create_model


@pytest.mark.parametrize("batch_size, img_height, img_width, num_channels, scaling_factor", HAB_IMGS)
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


def test_hat_calculate_rpi_sa(model_factory: Callable) -> None:
    model = model_factory()

    window_size = model.window_size
    rpi_sa = model._calculate_rpi_sa()

    assert rpi_sa.shape == (window_size**2, window_size**2)

    max_val = (2 * window_size - 1) ** 2 - 1

    assert rpi_sa.min() >= 0
    assert rpi_sa.max() <= max_val


def test_hat_calculate_rpi_oca(model_factory: Callable) -> None:
    model = model_factory()

    window_size = model.window_size
    overlapped_window_size = window_size + int(model.overlap_ratio * window_size)
    rpi_oca = model._calculate_rpi_oca()

    assert rpi_oca.shape == (window_size**2, overlapped_window_size**2)

    max_val = (window_size + overlapped_window_size - 1) ** 2

    assert rpi_oca.min() >= 0
    assert rpi_oca.max() <= max_val


def test_hat_calculate_attention_mask(model_factory: Callable) -> None:
    model = model_factory()

    window_size = model.window_size
    img_height, img_width = 64, 64

    attention_mask = model._calculate_attention_mask(x_size=(img_height, img_width))

    num_pixels = window_size**2

    assert attention_mask.shape[-2] == num_pixels
    assert attention_mask.shape[-3] == num_pixels

    assert torch.equal(torch.unique(attention_mask), torch.tensor([-100.0, 0.0]))


@pytest.mark.parametrize("window_size, num_heads, overlap_ratio", RHAG_CONFIGS)
def test_rhag(
    window_size: int,
    num_heads: int,
    overlap_ratio: float,
) -> None:
    batch_size, img_height, img_width, num_channels = 1, 64, 64, 32
    overlapped_window_size = window_size + int(overlap_ratio * window_size)

    rpi_sa = torch.randint(0, ((2 * window_size - 1) ** 2), (window_size**2, window_size**2))
    rpi_oca = torch.randint(
        0, ((window_size + overlapped_window_size - 1) ** 2), (window_size**2, overlapped_window_size**2)
    )
    attention_mask = torch.zeros(
        ((img_height // window_size) * (img_width // window_size), window_size**2, window_size**2)
    )

    rhag = models.RHAG(
        num_hab_blocks=1,
        num_channels=num_channels,
        compress_ratio=3,
        squeeze_factor=4,
        window_size=window_size,
        num_heads=num_heads,
        cab_scale=0.01,
        train_img_size=(img_height, img_width),
        mlp_ratio=4,
        overlap_ratio=overlap_ratio,
    )

    input_img_tensor = torch.randn(batch_size, img_height * img_width, num_channels)

    output_img_tensor = rhag(
        input_img_tensor,
        x_size=(img_height, img_width),
        rpi_sa=rpi_sa,
        rpi_oca=rpi_oca,
        attention_mask=attention_mask,
    )

    assert input_img_tensor.shape == output_img_tensor.shape


@pytest.mark.parametrize("window_size, num_heads, overlap_ratio", OCAB_CONFIGS)
def test_ocab(
    window_size: int,
    num_heads: int,
    overlap_ratio: float,
) -> None:
    batch_size, img_height, img_width, num_channels = 1, 64, 64, 32
    overlapped_window_size = window_size + int(overlap_ratio * window_size)

    rpi_oca = torch.randint(
        0, ((window_size + overlapped_window_size - 1) ** 2), (window_size**2, overlapped_window_size**2)
    )

    ocab = models.OCAB(
        num_channels=num_channels,
        num_heads=num_heads,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        mlp_ratio=2,
    )

    input_img_tensor = torch.randn(batch_size, img_height * img_width, num_channels)

    output_img_tensor = ocab(
        input_img_tensor,
        x_size=(img_height, img_width),
        rpi_oca=rpi_oca,
    )

    assert input_img_tensor.shape == output_img_tensor.shape


@pytest.mark.parametrize("window_size, num_heads", HAB_CONFIGS)
def test_hab(
    window_size: int,
    num_heads: int,
) -> None:
    batch_size, img_height, img_width, num_channels = 1, 64, 64, 32

    rpi_sa = torch.randint(0, ((2 * window_size - 1) ** 2), (window_size**2, window_size**2))
    attention_mask = torch.zeros(
        ((img_height // window_size) * (img_width // window_size), window_size**2, window_size**2)
    )

    hat = models.HAB(
        num_channels=num_channels,
        compress_ratio=3,
        squeeze_factor=4,
        window_size=window_size,
        num_heads=num_heads,
        cab_scale=0.01,
        train_img_size=(img_height, img_width),
        shift_size=0,
        mlp_ratio=2,
    )

    input_img_tensor = torch.randn(batch_size, img_height * img_width, num_channels)

    output_img_tensor = hat(
        input_img_tensor,
        x_size=(img_height, img_width),
        rpi_sa=rpi_sa,
        attention_mask=attention_mask,
    )

    assert input_img_tensor.shape == output_img_tensor.shape


def test_mlp() -> None:
    batch_size, img_height, img_width, num_channels = 1, 64, 64, 32

    mlp = models.MLP(
        in_features=num_channels,
        hidden_features=num_channels * 2,
        out_features=num_channels,
    )

    input_img_tensor = torch.randn(batch_size, img_height, img_width, num_channels)

    output_img_tensor = mlp(input_img_tensor)

    assert input_img_tensor.shape == output_img_tensor.shape


@pytest.mark.parametrize("window_size, num_heads", HAB_CONFIGS)
def test_wmsa(
    window_size: int,
    num_heads: int,
) -> None:
    num_windows, num_pixels, num_channels = 16, window_size**2, 32

    rpi_sa = torch.randint(0, ((2 * window_size - 1) ** 2), (window_size**2, window_size**2))

    wmsa = models.WMSA(
        num_channels=num_channels,
        window_size=window_size,
        num_heads=num_heads,
    )

    input_img_tensor = torch.randn(num_windows, num_pixels, num_channels)

    output_img_tensor = wmsa(input_img_tensor, rpi_sa=rpi_sa)

    assert input_img_tensor.shape == output_img_tensor.shape


def test_cab() -> None:
    batch_size, img_height, img_width, num_channels = 1, 64, 64, 32

    cab = models.CAB(
        num_channels=num_channels,
        compress_ratio=3,
        squeeze_factor=4,
    )

    input_img_tensor = torch.randn(batch_size, num_channels, img_height, img_width)

    output_img_tensor = cab(input_img_tensor)

    assert input_img_tensor.shape == output_img_tensor.shape
