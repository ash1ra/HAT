import pytest
import torch
from torch.utils.data import TensorDataset

import utils

TEST_CASES = [
    # (batch_size, img_height, img_width, num_channels, window_size)
    (1, 64, 64, 3, 8),  # Single square image
    (1, 64, 64, 3, 4),  # Single square image with window_size = 4
    (1, 64, 64, 3, 16),  # Single square image with window_size = 16
    (4, 64, 64, 3, 8),  # Multiple square images
    (4, 64, 64, 96, 8),  # Multiple square images with num_channels > 3 (internal features)
    (1, 48, 96, 3, 8),  # Single rectangular image
    (1, 48, 96, 3, 12),  # Single rectangular image with window_size = 12
    (4, 48, 96, 3, 8),  # Multiuple rectangular images
    (1, 64, 64, 1, 8),  # Single grayscale image
    (4, 64, 64, 1, 8),  # Multiple grayscale images
]

TOLERANCE = 1e-8


def test_infinite_dataloader_repeats() -> None:
    dataset_size = 5
    data_loader_repeats = 3

    dataset = TensorDataset(torch.arange(dataset_size))

    data_loader = utils.InfiniteDataLoader(dataset=dataset, repeats=data_loader_repeats)

    assert len(data_loader) == dataset_size * data_loader_repeats


def test_infinite_dataloader_loops_infinitely() -> None:
    dataset = TensorDataset(torch.arange(3))

    dataloader = utils.InfiniteDataLoader(dataset=dataset, batch_size=2, repeats=1)

    batch_1 = next(dataloader)
    batch_2 = next(dataloader)
    batch_3 = next(dataloader)

    assert torch.equal(batch_1[0], torch.tensor([0, 1]))
    assert torch.equal(batch_2[0], torch.tensor([2]))
    assert torch.equal(batch_3[0], torch.tensor([0, 1]))


@pytest.mark.parametrize("batch_size, img_height, img_width, num_channels, window_size", TEST_CASES)
def test_window_cycle_consistency(
    batch_size: int,
    img_height: int,
    img_width: int,
    num_channels: int,
    window_size: int,
) -> None:
    input_img_tensor = torch.randn(batch_size, img_height, img_width, num_channels)

    output_windows_tensor = utils.split_img_into_windows(img_tensor=input_img_tensor, window_size=window_size)
    restored_input_img_tensor = utils.combine_windows_into_img(
        windows_tensor=output_windows_tensor, img_height=img_height, img_width=img_width
    )

    assert output_windows_tensor.shape == (
        batch_size * (img_height // window_size) * (img_width // window_size),
        window_size,
        window_size,
        num_channels,
    )

    assert input_img_tensor.shape == restored_input_img_tensor.shape

    assert torch.allclose(input_img_tensor, restored_input_img_tensor, rtol=TOLERANCE)


def test_split_img_into_windows_error() -> None:
    input_img_tensor = torch.randn(1, 63, 63, 3)
    window_size = 8

    with pytest.raises(Exception):
        utils.split_img_into_windows(img_tensor=input_img_tensor, window_size=window_size)


def test_combine_windows_into_img_error() -> None:
    batch_size, img_height, img_width, num_channels = 16, 64, 64, 3
    window_size = 8

    input_windows_tensor = torch.randn(batch_size * 63 * 63, window_size, window_size, num_channels)

    with pytest.raises(Exception):
        utils.combine_windows_into_img(windows_tensor=input_windows_tensor, img_height=img_height, img_width=img_width)
