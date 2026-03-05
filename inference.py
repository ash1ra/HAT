import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms
from torchvision.utils import save_image
from tqdm import tqdm

import config
from models import HAT
from utils import logger


def create_lr_hr_pair(input_img_tensor: Tensor, scaling_factor: int) -> tuple[Tensor, Tensor]:
    num_channels, input_img_height, input_img_width = input_img_tensor.shape

    lr_img_height = input_img_height // scaling_factor
    lr_img_width = input_img_width // scaling_factor

    hr_img_height = lr_img_height * scaling_factor
    hr_img_width = lr_img_width * scaling_factor

    hr_img_tensor = input_img_tensor[:, :hr_img_height, :hr_img_width]

    lr_img_tensor = transforms.Resize(
        size=(lr_img_height, lr_img_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )(hr_img_tensor)

    return lr_img_tensor, hr_img_tensor


def save_comparison_img(
    lr_upscaled_img_tensor: Tensor,
    sr_img_tensor: Tensor,
    hr_img_tensor: Tensor,
    output_path: Path,
    crop_box: tuple[int, int, int, int] | None = None,
) -> None:
    comparison_img_path = output_path.parent / f"{output_path.stem}_comparison.png"
    logger.info("Generating visual comparison canvas...")

    img_tensors = [lr_upscaled_img_tensor, sr_img_tensor, hr_img_tensor]
    img_labels = ["Bicubic", "HAT", "Original"]

    imgs = [transforms.ToPILImage()(img_tensor.float().clamp(0, 1).cpu()) for img_tensor in img_tensors]

    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf", 32)
    except OSError:
        font = ImageFont.load_default()

    footer_height = 60
    spacing = 20

    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    if crop_box is not None:
        x_start, y_start, x_end, y_end = crop_box
        w, h = imgs[0].size

        x_start, x_end = max(0, x_start), min(w, x_end)
        y_start, y_end = max(0, y_start), min(h, y_end)

        patches = [img.crop((x_start, y_start, x_end, y_end)) for img in imgs]
        crop_w, crop_h = patches[0].size

        full_img_with_box = imgs[2].copy()
        draw_full = ImageDraw.Draw(full_img_with_box)
        draw_full.rectangle([x_start, y_start, x_end, y_end], outline="red", width=5)

        aspect_ratio = w / h
        new_full_w = int(crop_h * aspect_ratio)
        resample_filter = Image.Resampling.BICUBIC
        full_img_resized = full_img_with_box.resize((new_full_w, crop_h), resample=resample_filter)

        row_images = [full_img_resized] + patches
        row_labels = ["Full"] + img_labels
        raw_image_widths = [new_full_w, crop_w, crop_w, crop_w]

        col_widths = []
        for img_w, label in zip(raw_image_widths, row_labels):
            label_bbox = dummy_draw.textbbox((0, 0), label, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            col_widths.append(max(img_w, text_width + 10))

        canvas_width = sum(col_widths) + spacing * (len(col_widths) - 1)
        canvas_height = crop_h + footer_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        current_x = 0
        for img, label, col_w, img_w in zip(row_images, row_labels, col_widths, raw_image_widths):
            img_x = current_x + (col_w // 2) - (img_w // 2)
            canvas.paste(img, (img_x, 0))

            label_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]

            text_x = current_x + (col_w // 2) - (text_width // 2)
            text_y = crop_h + (footer_height - text_height) // 2 - 5

            draw.text((text_x, text_y), label, fill="black", font=font)

            current_x += col_w + spacing

    else:
        img_width, img_height = imgs[0].size

        col_widths = []
        for label in img_labels:
            label_bbox = dummy_draw.textbbox((0, 0), label, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            col_widths.append(max(img_width, text_width + 10))

        canvas_width = sum(col_widths) + (spacing * 2)
        canvas_height = img_height + footer_height

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        draw = ImageDraw.Draw(canvas)

        current_x = 0
        for img, label, col_w in zip(imgs, img_labels, col_widths):
            img_x = current_x + (col_w // 2) - (img_width // 2)
            canvas.paste(im=img, box=(img_x, 0))

            label_bbox = draw.textbbox(xy=(0, 0), text=label, font=font)
            text_width = label_bbox[2] - label_bbox[0]
            text_height = label_bbox[3] - label_bbox[1]

            text_x = current_x + (col_w // 2) - (text_width // 2)
            text_y = img_height + (footer_height - text_height) // 2 - 5

            draw.text(xy=(text_x, text_y), text=label, fill=(0, 0, 0), font=font)

            current_x += col_w + spacing

    logger.info(f"Saving comparison image to: '{comparison_img_path}'")
    canvas.save(comparison_img_path)


def _tiled_inference(
    model: nn.Module,
    lr_img_tensor: Tensor,
    scaling_factor: int,
    tile_size: int,
    tile_overlap: int,
    device: config.DeviceType = "cpu",
    dtype: torch.dtype = torch.float16,
):
    num_channels, lr_img_height, lr_img_width = lr_img_tensor.shape

    sr_img_height = lr_img_height * scaling_factor
    sr_img_width = lr_img_width * scaling_factor
    sr_img_shape = (num_channels, sr_img_height, sr_img_width)

    sr_accumulated_values = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")
    sr_weight_map = torch.zeros(sr_img_shape, dtype=torch.float32, device="cpu")

    stride = tile_size - tile_overlap

    height_steps = [0]
    width_steps = [0]

    if lr_img_height >= tile_size:
        height_steps = list(range(0, lr_img_height - tile_size, stride)) + [lr_img_height - tile_size]

    if lr_img_width >= tile_size:
        width_steps = list(range(0, lr_img_width - tile_size, stride)) + [lr_img_width - tile_size]

    pbar = tqdm(total=len(height_steps) * len(width_steps), desc="Processing tiles", leave=False)

    for height_step in height_steps:
        for width_step in width_steps:
            lr_height_start = height_step
            lr_width_start = width_step

            lr_height_end = min(height_step + tile_size, lr_img_height)
            lr_width_end = min(width_step + tile_size, lr_img_width)

            lr_img_patch = lr_img_tensor[:, lr_height_start:lr_height_end, lr_width_start:lr_width_end]
            lr_img_patch = lr_img_patch.to(device=device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
                sr_img_patch = model(lr_img_patch.unsqueeze(0)).squeeze(0).cpu()

            sr_height_start = lr_height_start * scaling_factor
            sr_width_start = lr_width_start * scaling_factor

            sr_height_end = lr_height_end * scaling_factor
            sr_width_end = lr_width_end * scaling_factor

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
    tile_size: int | None = None,
    tile_overlap: int = 32,
    create_comparison: bool = False,
    crop_box: tuple[int, int, int, int] | None = None,
    device: config.DeviceType = "cpu",
    dtype: torch.dtype = torch.float16,
) -> None:
    if not input_img_path.exists():
        raise FileNotFoundError(f"Input image file at '{input_img_path}' not found.")

    if not input_img_path.is_file():
        raise ValueError(f"Input image should be a file, passed '{input_img_path}'.")

    if input_img_path.suffix.lower() not in config.IMG_EXTENSIONS:
        raise ValueError(f"Input file should be an image with, you passed '{input_img_path.suffix}'.")

    logger.info(f"Loading input image from '{input_img_path}'.")

    input_img_tensor = decode_image(input=str(input_img_path), mode=ImageReadMode.RGB)
    input_img_tensor = transforms.ToDtype(dtype=torch.float32, scale=True)(input_img_tensor)

    lr_img_tensor = input_img_tensor

    logger.info(
        f"Successfully loaded image with dimensions: {input_img_tensor.shape[2]}x{input_img_tensor.shape[1]} (WxH)."
    )

    if create_comparison:
        lr_img_tensor, hr_img_tensor = create_lr_hr_pair(input_img_tensor, scaling_factor)

    if tile_size:
        logger.info(f"Inference Mode: Tiled Processing (Tile Size: {tile_size}, Overlap: {tile_overlap}).")

        sr_img_tensor = _tiled_inference(
            model=model,
            lr_img_tensor=lr_img_tensor,
            scaling_factor=scaling_factor,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            device=device,
            dtype=dtype,
        )
    else:
        logger.info("Inference Mode: Full Image Processing (No Tiling).")

        lr_img_tensor = lr_img_tensor.to(device=device)

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
            sr_img_tensor = model(lr_img_tensor.unsqueeze(0)).squeeze(0)

    sr_img_tensor.clamp_(0, 1)

    logger.info(f"Saving result to '{output_img_path}'...")
    save_image(sr_img_tensor, output_img_path, format="PNG")

    if create_comparison:
        _, hr_img_height, hr_img_width = hr_img_tensor.shape

        lr_upscaled_img_tensor = transforms.Resize(
            size=(hr_img_height, hr_img_width),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )(lr_img_tensor)

        save_comparison_img(
            lr_upscaled_img_tensor=lr_upscaled_img_tensor,
            sr_img_tensor=sr_img_tensor,
            hr_img_tensor=hr_img_tensor,
            output_path=output_img_path,
            crop_box=crop_box,
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
        "-cb",
        "--crop-box",
        type=int,
        nargs=4,
        metavar=("X_START", "Y_START", "X_END", "Y_END"),
        default=None,
        help="Coordinates for a specific crop box in the comparison image (e.g., --crop 100 100 612 612). Coordinates are mapped to the HR image space.",
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

    if args.tile_size < 0 and args.tile_size % config.WINDOW_SIZE != 0:
        raise ValueError(f"Tile size must be >0 and be divisible by {config.WINDOW_SIZE}, you passed: {args.tile_size}")

    if args.tile_overlap < 0:
        raise ValueError(f"Tile overlap must be >0, you passed: {args.tile_overlap}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

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

    inference(
        model=model,
        input_img_path=args.input,
        output_img_path=args.output,
        scaling_factor=args.scaling_factor,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        create_comparison=args.comparison,
        crop_box=tuple(args.crop_box) if args.crop_box else None,
        device=device,
        dtype=getattr(torch, args.dtype, torch.float16),
    )


if __name__ == "__main__":
    main()
