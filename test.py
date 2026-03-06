import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from config import DeviceType
from datasets import StaticPairDataset
from models import HAT
from utils import calculate_psnr, calculate_ssim, logger


@torch.inference_mode()
def test(
    model: nn.Module,
    dataloader: DataLoader,
    dataset_name: str,
    loss_fn: nn.Module,
    scaling_factor: int,
    device: DeviceType,
    dtype: torch.dtype,
) -> tuple[float, float, float]:
    model.eval()
    torch.cuda.empty_cache()

    avg_loss, avg_psnr, avg_ssim = 0.0, 0.0, 0.0

    for batch in tqdm(dataloader, desc=dataset_name, leave=False):
        lr_img_tensor = batch["lr"].to(device=device, non_blocking=True)
        hr_img_tensor = batch["hr"].to(device=device, non_blocking=True)

        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=True):
            sr_img_tensor = model(lr_img_tensor)
            loss = loss_fn(sr_img_tensor, hr_img_tensor)
            avg_loss += loss.item()

        batch_psnr, batch_ssim = 0.0, 0.0
        batch_size = sr_img_tensor.size(0)

        for j in range(batch_size):
            sr_img_tensor = sr_img_tensor[j].float()
            hr_img_tensor = hr_img_tensor[j].float()

            batch_psnr += calculate_psnr(
                sr_img_tensor=sr_img_tensor,
                hr_img_tensor=hr_img_tensor,
                crop_border=scaling_factor,
            )

            batch_ssim += calculate_ssim(
                sr_img_tensor=sr_img_tensor,
                hr_img_tensor=hr_img_tensor,
                crop_border=scaling_factor,
                return_map=False,
            )

        avg_psnr += batch_psnr / batch_size
        avg_ssim += batch_ssim / batch_size

    torch.cuda.empty_cache()

    avg_loss /= len(dataloader)
    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)

    return avg_loss, avg_psnr, avg_ssim


def main() -> None:
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
        scaling_factor=config.SCALING_FACTOR,
        use_gradient_checkpointing=config.USE_GRADIENT_CHECKPOINTING,
    ).to(device=device, memory_format=torch.channels_last)  # type: ignore

    loss_fn = nn.L1Loss()

    if config.LOAD_BEST_CHECKPOINT and config.BEST_CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.BEST_CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    elif config.LOAD_CHECKPOINT and config.CHECKPOINT_DIR_PATH.exists():
        model.load_state_dict(load_file(config.CHECKPOINT_DIR_PATH / "model.safetensors", device=device))
    else:
        raise FileNotFoundError(
            "Failed to load model weights. Please verify that the checkpoint paths in 'config.py' exist and are valid."
        )

    for test_dataset_path in config.TEST_DATASET_PATHS:
        dataset = StaticPairDataset(
            data_path=test_dataset_path,
            scaling_factor=config.SCALING_FACTOR,
            patch_size=config.PATCH_SIZE,
            test_mode=True,
            dev_mode=False,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.VAL_NUM_WORKERS,
            pin_memory=True if device == "cuda" else False,
            prefetch_factor=config.VAL_PREFETCH_FACTOR,
            persistent_workers=True if config.VAL_NUM_WORKERS > 0 else False,
        )

        test_loss, test_psnr, test_ssim = test(
            model=model,
            dataloader=dataloader,
            dataset_name=test_dataset_path.name,
            loss_fn=loss_fn,
            scaling_factor=config.SCALING_FACTOR,
            device=device,
            dtype=torch.bfloat16,
        )

        logger.info(
            f"Dataset: {test_dataset_path.name} | PSNR: {test_psnr:.2f} | SSIM: {test_ssim:.4f} | Loss: {test_loss:.4f}"
        )


if __name__ == "__main__":
    main()
