import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

import config
from models import HAT
from utils import logger


def create_lr_and_hr_imgs(input_img_tensor: Tensor, scaling_factor: int) -> tuple[Tensor, Tensor]:
    num_channels, input_img_height, input_img_width = input_img_tensor.shape

    lr_img_height = input_img_height // scaling_factor
    lr_img_width = input_img_width // scaling_factor

    hr_img_height = lr_img_height * scaling_factor
    hr_img_width = lr_img_width * scaling_factor

    hr_img_tensor = input_img_tensor[:, :hr_img_height, :hr_img_width]

    lr_img_tensor = transforms.Resize(
        (lr_img_height, lr_img_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )(hr_img_tensor)

    return lr_img_tensor, hr_img_tensor


def save_comparison_img(
    lr_upscaled_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    output_path: Path,
    vertical_comparison: bool = False,
) -> None:
    comparison_img_path = output_path.parent / f"{output_path.stem}_comparison.png"

    logger.info("Creating comparison image...")

    img_tensors = [lr_upscaled_img_tensor, sr_img_tensor, hr_img_tensor]
    img_labels = ["Bicubic", "HAT", "Original"]

    imgs = [transforms.ToPILImage()(img_tensor.float().cpu().clamp(0, 1)) for img_tensor in img_tensors]

    img_width, img_height = imgs[0].size
    header_height = 50

    if vertical_comparison:
        canvas_width = img_width
        canvas_height = (img_height + header_height) * 3
    else:
        canvas_width = img_width * 3
        canvas_height = img_height + header_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(imgs, img_labels)):
        label_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = label_bbox[2] - label_bbox[0]
        text_height = label_bbox[3] - label_bbox[1]

        if vertical_comparison:
            x_offset = 0
            y_offset = i * (img_height + header_height)

            text_x = (img_width // 2) - (text_width // 2)
            text_y = y_offset + (header_height - text_height) // 2 - 5

            img_y_pos = y_offset + header_height
        else:
            x_offset = i * img_width
            y_offset = 0

            text_x = x_offset + (img_width // 2) - (text_width // 2)
            text_y = (header_height - text_height) // 2 - 5

            img_y_pos = header_height

        canvas.paste(img, (x_offset, img_y_pos))
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    logger.info(f"Saving comparison result to '{comparison_img_path}'...")
    canvas.save(comparison_img_path)


def tiled_inference(
    model: nn.Module,
    lr_img_tensor: Tensor,
    scaling_factor: int,
    tile_size: int,
    tile_overlap: int,
    device: config.DeviceType,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_channels, lr_img_height, lr_img_width = lr_img_tensor.shape

    sr_img_height = lr_img_height * scaling_factor
    sr_img_width = lr_img_width * scaling_factor
    sr_img_shape = (num_channels, sr_img_height, sr_img_width)

    sr_accumulated_values = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")
    sr_weight_map = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")

    stride = tile_size - tile_overlap
    height_steps = list(range(0, lr_img_height - tile_size, stride)) + [lr_img_height - tile_size]
    width_steps = list(range(0, lr_img_width - tile_size, stride)) + [lr_img_width - tile_size]

    if lr_img_height < tile_size:
        height_steps = [0]

    if lr_img_width < tile_size:
        width_steps = [0]

    pbar = tqdm(total=len(height_steps) * len(width_steps), desc="Processing tiles", leave=False)

    for height_step in height_steps:
        for width_step in width_steps:
            lr_height_end = min(height_step + tile_size, lr_img_height)
            lr_width_end = min(width_step + tile_size, lr_img_width)
            lr_height_start = max(0, lr_height_end - tile_size)
            lr_width_start = max(0, lr_width_end - tile_size)

            lr_img_patch = lr_img_tensor[:, lr_height_start:lr_height_end, lr_width_start:lr_width_end]
            lr_img_patch = lr_img_patch.to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
                sr_img_patch = model(lr_img_patch.unsqueeze(dim=0)).squeeze(dim=0).cpu()

            sr_height_end = lr_height_end * scaling_factor
            sr_width_end = lr_width_end * scaling_factor
            sr_height_start = lr_height_start * scaling_factor
            sr_width_start = lr_width_start * scaling_factor

            sr_accumulated_values[:, sr_height_start:sr_height_end, sr_width_start:sr_width_end] += sr_img_patch
            sr_weight_map[:, sr_height_start:sr_height_end, sr_width_start:sr_width_end] += 1.0

            pbar.update(1)
    pbar.close()

    return sr_accumulated_values.div_(sr_weight_map)


@torch.inference_mode()
def inference(
    model: nn.Module,
    input_img_path: Path,
    output_img_path: Path,
    scaling_factor: int,
    device: config.DeviceType,
    dtype: torch.dtype = torch.float32,
    tile_size: int | None = None,
    tile_overlap: int = 32,
    create_comparison: bool = False,
    vertical_comparison: bool = False,
) -> None:
    model.eval()

    if not input_img_path.exists():
        raise FileNotFoundError(f"File '{input_img_path.name}' not found.")

    if not input_img_path.is_file():
        raise ValueError("Not a file")

    if input_img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        raise ValueError("Image must be in PNG or JPG format.")

    logger.info(f"Loading image from '{input_img_path}'...")

    input_img_tensor = decode_image(str(input_img_path), mode=ImageReadMode.RGB)
    input_img_tensor = transforms.ToDtype(dtype=torch.float32, scale=True)(input_img_tensor)

    logger.info(f"Image loaded ({input_img_tensor.shape[2]}x{input_img_tensor.shape[1]}).")

    if create_comparison:
        lr_img_tensor, hr_img_tensor = create_lr_and_hr_imgs(input_img_tensor, scaling_factor=scaling_factor)
    else:
        lr_img_tensor = input_img_tensor

    if tile_size:
        logger.info(f"Mode: Tiling Inference | Tile size: {tile_size} | Overlap: {tile_overlap}.")

        sr_img_tensor = tiled_inference(
            model=model,
            lr_img_tensor=lr_img_tensor,
            scaling_factor=scaling_factor,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device,
            dtype=dtype,
        )
    else:
        logger.info("Mode: Full Image Inference (No Tiling).")

        lr_img_tensor = lr_img_tensor.to(device)
        lr_img_height, lr_img_width = lr_img_tensor.shape[1], lr_img_tensor.shape[2]

        padding_right = (config.WINDOW_SIZE - (lr_img_width % config.WINDOW_SIZE)) % config.WINDOW_SIZE
        padding_bottom = (config.WINDOW_SIZE - (lr_img_height % config.WINDOW_SIZE)) % config.WINDOW_SIZE

        lr_img_tensor_padded = F.pad(lr_img_tensor, pad=(0, padding_right, 0, padding_bottom), mode="reflect")

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
            sr_img_tensor_padded = model(lr_img_tensor_padded.unsqueeze(dim=0)).squeeze(dim=0)

        sr_img_tensor = sr_img_tensor_padded[:, : lr_img_height * scaling_factor, : lr_img_width * scaling_factor]

    sr_img_tensor.clamp_(0, 1)

    logger.info(f"Saving result to '{output_img_path}'...")
    save_image(sr_img_tensor, output_img_path, format="PNG")

    if create_comparison:
        _, hr_img_height, hr_img_width = hr_img_tensor.shape

        lr_upscaled_img_tensor = transforms.Resize(
            (hr_img_height, hr_img_width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )(lr_img_tensor)

        save_comparison_img(
            lr_upscaled_img_tensor=lr_upscaled_img_tensor,
            sr_img_tensor=sr_img_tensor,
            hr_img_tensor=hr_img_tensor,
            output_path=output_img_path,
            vertical_comparison=vertical_comparison,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image Super-Resolution inference using the HAT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path where the SR image will be savbed")
    parser.add_argument("-s", "--scaling-factor", type=int, required=True, help="Upscaling factor (e.g., 2, 3 or 4)")
    parser.add_argument(
        "-ts",
        "--tile-size",
        type=int,
        default=None,
        help=f"Size of the processing tiles for memory efficiency (e.g., 256 or 512). Must be divisible by window_size ({config.WINDOW_SIZE}). If omitted, processes the full image at once",
    )
    parser.add_argument(
        "-to",
        "--tile-overlap",
        type=int,
        default=32,
        help="Number of overlapping pixels between adjacent tiles to prevent edge blending artifacts",
    )
    parser.add_argument(
        "-c",
        "--comparison",
        action="store_true",
        help="Generate an additional image comparing the Bicubic baseline, HAT output, and original image",
    )
    parser.add_argument(
        "-v",
        "--vertical",
        action="store_true",
        help="Stack comparison images vertically",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Floating-point precision used for Automatic Mixed Precision (AMP) during model inference",
    )

    args = parser.parse_args()

    if args.scaling_factor <= 1:
        raise ValueError(f"Scaling factor must be >=2 (e.g., 2, 3, 4), you passed: {args.scaling_factor}")

    if args.tile_size:
        if args.tile_size < 0 and args.tile_size % config.WINDOW_SIZE != 0:
            raise ValueError(
                f"Tile size must be >0 and be divisible by {config.WINDOW_SIZE}, you passed: {args.tile_size}"
            )

    if args.tile_overlap < 0:
        raise ValueError(f"Tile overlap must be >0, you passed: {args.tile_overlap}")

    torch.backends.fp32_precision = "tf32"  # type: ignore
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore
    torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HAT(
        in_channels=3,
        num_rhag_blocks=config.NUM_RHAG_BLOCKS,
        num_hab_blocks=config.NUM_HAB_BLOCKS,
        num_channels=config.NUM_CHANNELS,
        compress_ratio=config.COMPRESS_RATIO,
        squeeze_factor=config.SQUEEZE_FACTOR,
        window_size=config.WINDOW_SIZE,
        num_heads=config.NUM_HEADS,
        cab_scale=config.CAB_SCALE,
        train_img_size=(config.PATCH_SIZE, config.PATCH_SIZE),
        mlp_ratio=config.MLP_RATIO,
        overlap_ratio=config.OVERLAP_RATIO,
        scaling_factor=args.scaling_factor,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    ).to(device=device, memory_format=torch.channels_last)  # type: ignore

    if config.LOAD_BEST_CHECKPOINT and config.BEST_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.BEST_CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    elif config.LOAD_CHECKPOINT and config.CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    else:
        raise FileNotFoundError(
            "Failed to load model weights. Please verify that the checkpoint paths in 'config.py' exist and are valid."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    inference(
        model=model,
        input_img_path=args.input,
        output_img_path=args.output,
        scaling_factor=args.scaling_factor,
        device=device,
        dtype=getattr(torch, args.dtype, torch.float32),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        create_comparison=args.comparison,
        vertical_comparison=args.vertical,
    )


if __name__ == "__main__":
    main()
