from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from config import DeviceType
from test import test as func_test


class DummyDataset(Dataset):
    def __init__(self, size: int = 10, img_size: int = 32, scaling_factor: int = 2) -> None:
        self.size = size
        self.img_size = img_size
        self.scaling_factor = scaling_factor

    def __len__(self):
        return self.size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "lr": torch.rand(3, self.img_size // self.scaling_factor, self.img_size // self.scaling_factor),
            "hr": torch.rand(3, self.img_size, self.img_size),
        }


class DummyModel(nn.Module):
    def __init__(self, scaling_factor: int = 2) -> None:
        super().__init__()

        self.scaling_factor = scaling_factor

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=self.scaling_factor, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(self.conv(x))


@patch("test.calculate_psnr")
@patch("test.calculate_ssim")
@pytest.mark.parametrize(
    "batch_size, device",
    [
        (1, "cpu"),
        (2, "cpu"),
        pytest.param(1, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        pytest.param(2, "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ],
)
def test_test(mock_calculate_ssim, mock_calculate_psnr, batch_size: int, device: DeviceType) -> None:

    mock_calculate_psnr.return_value = 35.0
    mock_calculate_ssim.return_value = 0.95

    scaling_factor = 2

    model = DummyModel(scaling_factor=scaling_factor).to(device)

    dataset = DummyDataset(size=10, scaling_factor=scaling_factor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.L1Loss()

    test_loss, test_psnr, test_ssim = func_test(
        model=model,
        dataloader=dataloader,
        dataset_name="MockDataset",
        loss_fn=loss_fn,
        scaling_factor=scaling_factor,
        device=device,
        dtype=torch.float16,
    )

    assert test_loss > 0.0

    assert test_psnr == 35.0
    assert test_ssim == 0.95

    assert mock_calculate_psnr.call_count == 10
    assert mock_calculate_ssim.call_count == 10
