# Ported from BasicSR (matlab_functions.py) to ensure academic reproducibility.
# Implements MATLAB-like bicubic interpolation required for standard SR benchmarks (Set5, Set14).
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/matlab_functions.py
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import config


def cubic(x: np.ndarray) -> np.ndarray:
    abs_x = np.abs(x)
    abs_x2 = abs_x**2
    abs_x3 = abs_x**3

    return (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * ((abs_x <= 1).astype(type(abs_x))) + (
        -0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2
    ) * (((abs_x > 1) * (abs_x <= 2)).astype(type(abs_x)))


def calculate_weights_indices(
    in_length: int,
    out_length: int,
    scale: float,
    kernel_width: int,
    antialiasing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if (scale < 1) and antialiasing:
        kernel_width: int | float = kernel_width / scale

    x = np.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    p = int(np.ceil(kernel_width)) + 2

    indices = left.reshape(int(out_length), 1) + np.linspace(0, p - 1, p).reshape(1, int(p))

    distance_to_center = u.reshape(int(out_length), 1) - indices

    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    weights_sum = np.sum(weights, 1).reshape(int(out_length), 1)
    weights /= weights_sum

    weights_zero_idx = np.where(weights_sum == 0)
    if len(weights_zero_idx[0]) > 0:
        weights[weights_zero_idx, 0] = 1

    padded_indices = indices.astype(int)
    padded_indices -= 1

    padded_indices = np.abs(padded_indices)
    padded_indices = np.where(padded_indices < in_length, padded_indices, 2 * in_length - 1 - padded_indices)
    padded_indices = np.clip(padded_indices, 0, in_length - 1)

    return weights, padded_indices


def imresize(img: np.ndarray, scale: float, antialiasing: bool = True) -> np.ndarray:
    if scale == 1:
        return img

    if len(img.shape) == 3:
        input_img_height, input_img_width, input_img_num_channels = img.shape
    else:
        input_img_height, input_img_width = img.shape
        input_img_num_channels = 1

    output_img_height = int(np.ceil(input_img_height * scale))
    output_img_width = int(np.ceil(input_img_width * scale))

    kernel_width = 4

    height_weights, height_indices = calculate_weights_indices(
        in_length=input_img_height,
        out_length=output_img_height,
        scale=scale,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    width_weights, width_indices = calculate_weights_indices(
        in_length=input_img_width,
        out_length=output_img_width,
        scale=scale,
        kernel_width=kernel_width,
        antialiasing=antialiasing,
    )

    img_aug = np.zeros((output_img_height, input_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img[:, :, channel] if input_img_num_channels > 1 else img
        pixels = channel_data[height_indices]
        img_aug[:, :, channel] = np.sum(height_weights[:, :, None] * pixels, axis=1)

    output_img = np.zeros((output_img_height, output_img_width, input_img_num_channels), dtype=np.float32)

    for channel in range(input_img_num_channels):
        channel_data = img_aug[:, :, channel]
        pixels = channel_data[:, width_indices]
        output_img[:, :, channel] = np.sum(width_weights[None, :, :] * pixels, axis=2)

    output_img = np.clip(output_img, 0, 255)

    return np.round(output_img).astype(np.uint8)


def process_single_img(
    img_path: Path,
    hr_dir: Path,
    lr_dir: Path,
    scaling_factor: Literal[2, 4, 8],
) -> None:
    img_name = f"{img_path.stem}.png"
    hr_img_path = hr_dir / img_name
    lr_img_path = lr_dir / img_name

    if hr_img_path.exists() and lr_img_path.exists():
        if hr_img_path.stat().st_size > 0 and lr_img_path.stat().st_size > 0:
            return

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")

            img_width, img_height = img.size

            remainder_w = img_width % scaling_factor
            remainder_h = img_height % scaling_factor

            if remainder_w != 0 or remainder_h != 0:
                img = img.crop((0, 0, img_width - remainder_w, img_height - remainder_h))

            img.save(hr_img_path, compress_level=1)

            img_np = np.array(img)

            lr_img_np = imresize(img_np, scale=1 / scaling_factor)

            lr_img = Image.fromarray(lr_img_np)

            lr_img.save(lr_img_path, compress_level=1)
    except Exception as e:
        print(f"[Error] Failed to process '{img_path.name}': {e}")


def prepare_data(
    input_data_path: Path,
    output_data_path: Path,
    scaling_factor: Literal[2, 4, 8],
    num_workers: Optional[int] = None,
) -> None:
    print(f"[Data] Preparing data from '{input_data_path}'")

    output_data_path.mkdir(parents=True, exist_ok=True)

    hr_dir_output_path = output_data_path / "HR"
    hr_dir_output_path.mkdir(parents=True, exist_ok=True)

    lr_dir_output_path = output_data_path / f"LR_x{scaling_factor}"
    lr_dir_output_path.mkdir(parents=True, exist_ok=True)

    if input_data_path.exists():
        if input_data_path.is_dir():
            img_paths = sorted([p for p in input_data_path.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        elif input_data_path.is_file():
            with open(input_data_path, "r") as f:
                img_paths = sorted([Path(line.strip()) for line in f if line.strip()])
    else:
        raise FileNotFoundError(f"[Error] Input path '{input_data_path}' not found.")

    print(f"[Data] Found {len(img_paths)} images. Processing...")

    worker = partial(
        process_single_img, hr_dir=hr_dir_output_path, lr_dir=lr_dir_output_path, scaling_factor=scaling_factor
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(worker, img_paths), total=len(img_paths), desc="Processing images..."))

    print(f"[Data] Processing completed. Output saved to '{output_data_path}'.\n")


if __name__ == "__main__":
    prepare_data(
        input_data_path=config.TRAIN_DATASET_PATH.with_suffix(".txt"),
        output_data_path=config.TRAIN_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
    )

    prepare_data(
        input_data_path=config.VAL_DATASET_PATH.with_suffix(".txt"),
        output_data_path=config.VAL_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
    )

    for test_dataset_path in config.TEST_DATASET_PATHS:
        prepare_data(
            input_data_path=test_dataset_path.with_suffix(".txt"),
            output_data_path=test_dataset_path,
            scaling_factor=config.SCALING_FACTOR,
        )
