from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from inference import _tiled_inference, create_lr_hr_pair, inference

TEST_LR_HR_PAIR_PARAMETERS = [
    # num_channels, input_img_height, input_img_width
    [3, 64, 64],
    [1, 64, 64],
    [3, 61, 57],
]

TEST_INFERENCE_PARAMETERS = [
    # num_channels, input_img_height, input_img_width, scaling_factor
    [3, 64, 64, 2],
    [3, 64, 64, 3],
    [3, 64, 64, 4],
    [3, 61, 53, 2],
]


class DummyModel(nn.Module):
    def __init__(self, scaling_factor: int = 2) -> None:
        super().__init__()

        self.scaling_factor = scaling_factor

        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=self.scaling_factor, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample(self.conv(x))


@pytest.mark.parametrize("num_channels, input_img_height, input_img_width", TEST_LR_HR_PAIR_PARAMETERS)
def test_create_lr_hr_pair(num_channels: int, input_img_height: int, input_img_width: int) -> None:
    input_img_tensor = torch.rand(num_channels, input_img_height, input_img_width)
    scaling_factor = 2

    lr_img_tensor, hr_img_tensor = create_lr_hr_pair(
        input_img_tensor=input_img_tensor,
        scaling_factor=scaling_factor,
    )

    expected_lr_img_height = input_img_height // scaling_factor
    expected_lr_img_width = input_img_width // scaling_factor

    expected_hr_img_height = expected_lr_img_height * scaling_factor
    expected_hr_img_width = expected_lr_img_width * scaling_factor

    expected_lr_img_tensor = torch.rand(num_channels, expected_lr_img_height, expected_lr_img_width)
    expected_hr_img_tensor = torch.rand(num_channels, expected_hr_img_height, expected_hr_img_width)

    assert expected_lr_img_tensor.shape == lr_img_tensor.shape
    assert expected_hr_img_tensor.shape == hr_img_tensor.shape

    assert lr_img_tensor.shape[1] * scaling_factor == hr_img_tensor.shape[1]
    assert lr_img_tensor.shape[2] * scaling_factor == hr_img_tensor.shape[2]


@pytest.mark.parametrize("num_channels, input_img_height, input_img_width, scaling_factor", TEST_INFERENCE_PARAMETERS)
def test_inference(
    num_channels: int,
    input_img_height: int,
    input_img_width: int,
    scaling_factor: int,
    tmp_path: Path,
) -> None:
    Image.new("RGB", (input_img_width, input_img_height), color="red").save(tmp_path / "input_img.png")

    model = DummyModel(scaling_factor=scaling_factor)

    output_img_path = Path(tmp_path / "output_img.png")

    inference(
        model=model,
        input_img_path=Path(tmp_path / "input_img.png"),
        output_img_path=output_img_path,
        scaling_factor=scaling_factor,
        tile_size=None,
        create_comparison=False,
        device="cpu",
        dtype=torch.float16,
    )

    assert output_img_path.exists()
    assert output_img_path.is_file()

    with Image.open(output_img_path) as output_img:
        assert output_img.mode == "RGB"

        expected_width = input_img_width * scaling_factor
        expected_height = input_img_height * scaling_factor

        assert output_img.size == (expected_width, expected_height)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("num_channels, input_img_height, input_img_width, scaling_factor", TEST_INFERENCE_PARAMETERS)
def test_inference_cuda(
    num_channels: int,
    input_img_height: int,
    input_img_width: int,
    scaling_factor: int,
    tmp_path: Path,
) -> None:
    Image.new("RGB", (input_img_width, input_img_height), color="green").save(tmp_path / "input_img.png")

    model = DummyModel(scaling_factor=scaling_factor).to("cuda")

    output_img_path = Path(tmp_path / "output_img_cuda.png")

    inference(
        model=model,
        input_img_path=Path(tmp_path / "input_img.png"),
        output_img_path=output_img_path,
        scaling_factor=scaling_factor,
        tile_size=None,
        create_comparison=False,
        device="cuda",
        dtype=torch.float16,
    )

    assert output_img_path.exists()
    assert output_img_path.is_file()

    with Image.open(output_img_path) as output_img:
        assert output_img.mode == "RGB"

        expected_width = input_img_width * scaling_factor
        expected_height = input_img_height * scaling_factor

        assert output_img.size == (expected_width, expected_height)


@pytest.mark.parametrize("num_channels, input_img_height, input_img_width, scaling_factor", TEST_INFERENCE_PARAMETERS)
def test_tiled_inference(
    num_channels: int,
    input_img_height: int,
    input_img_width: int,
    scaling_factor: int,
) -> None:
    tile_size = 128
    tile_overlap = 32

    lr_img_tensor = torch.rand(num_channels, input_img_height, input_img_width)
    model = DummyModel(scaling_factor=scaling_factor)

    sr_img_tensor = _tiled_inference(
        model=model,
        lr_img_tensor=lr_img_tensor,
        scaling_factor=scaling_factor,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        device="cpu",
        dtype=torch.float16,
    )

    expected_height = input_img_height * scaling_factor
    expected_width = input_img_width * scaling_factor

    assert sr_img_tensor.shape == (num_channels, expected_height, expected_width)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("num_channels, input_img_height, input_img_width, scaling_factor", TEST_INFERENCE_PARAMETERS)
def test_tiled_inference_cuda(
    num_channels: int,
    input_img_height: int,
    input_img_width: int,
    scaling_factor: int,
) -> None:
    tile_size = 128
    tile_overlap = 32

    lr_img_tensor = torch.rand(num_channels, input_img_height, input_img_width).to("cuda")
    model = DummyModel(scaling_factor=scaling_factor).to("cuda")

    sr_img_tensor = _tiled_inference(
        model=model,
        lr_img_tensor=lr_img_tensor,
        scaling_factor=scaling_factor,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        device="cuda",
        dtype=torch.float16,
    )

    expected_height = input_img_height * scaling_factor
    expected_width = input_img_width * scaling_factor

    assert sr_img_tensor.shape == (num_channels, expected_height, expected_width)
