import math
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import config
from datasets import DynamicPairDataset, StaticPairDataset
from models import HAT
from trainer import Trainer
from utils import InfiniteDataLoader, logger


def main():
    torch.backends.fp32_precision = "tf32"  # type: ignore
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"  # type: ignore
    torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.CURRENT_PHASE == "pretraining":
        train_dataset = DynamicPairDataset(
            data_path=config.PRETRAIN_DATASET_PATH,
            scaling_factor=config.SCALING_FACTOR,
            patch_size=config.PATCH_SIZE,
            test_mode=False,
            dev_mode=False,
        )
    elif config.CURRENT_PHASE == "fine-tuning":
        train_dataset = StaticPairDataset(
            data_path=config.TRAIN_DATASET_PATH,
            scaling_factor=config.SCALING_FACTOR,
            patch_size=config.PATCH_SIZE,
            test_mode=False,
            dev_mode=False,
        )

    val_dataset = StaticPairDataset(
        data_path=config.VAL_DATASET_PATH,
        scaling_factor=config.SCALING_FACTOR,
        patch_size=config.PATCH_SIZE,
        test_mode=True,
        dev_mode=False,
    )

    train_dataloader = InfiniteDataLoader(
        dataset=train_dataset,
        repeats=math.ceil(config.NUM_ITERATIONS * config.BATCH_SIZE / len(train_dataset)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN_NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.TRAIN_PREFETCH_FACTOR,
        persistent_workers=True if config.TRAIN_NUM_WORKERS > 0 else False,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.VAL_NUM_WORKERS,
        pin_memory=True if device == "cuda" else False,
        prefetch_factor=config.VAL_PREFETCH_FACTOR,
        persistent_workers=True if config.VAL_NUM_WORKERS > 0 else False,
    )

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
    ).to(memory_format=torch.channels_last)  # type: ignore

    torch._dynamo.config.suppress_errors = True

    model = torch.compile(model)

    loss_fn = nn.L1Loss()

    optimizer = Adam(
        params=model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
    )

    scheduler = MultiStepLR(
        optimizer=optimizer,
        milestones=config.SCHEDULER_MILESTONES,
        gamma=config.SCHEDULER_GAMMA,
    )

    wandb_id = None
    target_checkpoint_path = None

    if config.LOAD_BEST_CHECKPOINT and config.BEST_CHECKPOINT_DIR_PATH.exists():
        target_checkpoint_path = config.BEST_CHECKPOINT_DIR_PATH
    elif config.LOAD_CHECKPOINT and config.CHECKPOINT_DIR_PATH.exists():
        target_checkpoint_path = config.CHECKPOINT_DIR_PATH

    if target_checkpoint_path:
        state_path = target_checkpoint_path / "state.pth"

        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            wandb_id = state_dict.get("wandb_id", None)

    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT_NAME,
            name=f"HAT_x{config.SCALING_FACTOR}_{config.CURRENT_PHASE}",
            id=wandb_id,
            config=config.WANDB_CONFIG,
            tags=[f"x{config.SCALING_FACTOR}", config.CURRENT_PHASE],
            resume="allow",
        )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaling_factor=config.SCALING_FACTOR,
        num_iters=config.NUM_ITERATIONS,
        log_freq=config.LOG_FREQ,
        val_freq=config.VAL_FREQ,
        save_freq=config.SAVE_CHECKPOINT_FREQ,
        root_dir_path=Path(""),
        gradient_clipping_norm=config.GRADIENT_CLIPPING_NORM,
        accumulation_steps=config.EFFECTIVE_BATCH_SIZE // config.BATCH_SIZE,
        scheduler=scheduler,
        device=device,
        dtype=torch.bfloat16,
        use_wandb=config.USE_WANDB,
    )

    if target_checkpoint_path:
        trainer.load_checkpoint(target_checkpoint_path)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by used. Saving last state...")
        trainer.save_checkpoint(is_best=False)
    finally:
        if config.USE_WANDB:
            wandb.finish()


if __name__ == "__main__":
    main()
