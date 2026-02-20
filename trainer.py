from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from thop import profile
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import wandb
from utils import Timer, calculate_psnr, calculate_ssim, format_time, logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scaling_factor: int,
        num_iters: int,
        log_freq: int,
        val_freq: int,
        save_freq: int,
        root_dir_path: Path,
        gradient_clipping_norm: float,
        accumulation_steps: int = 1,
        scheduler: Optional[LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        device: config.DeviceType = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaling_factor = scaling_factor
        self.num_iters = num_iters
        self.log_freq = log_freq
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.gradient_clipping_norm = gradient_clipping_norm
        self.accumulation_steps = accumulation_steps
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.dtype = dtype

        self.root_dir_path = root_dir_path
        self.checkpoints_dir_path = self.root_dir_path / "checkpoints"
        self.logs_dir_path = self.root_dir_path / "logs"
        self.imgs_dir_path = self.root_dir_path / "images"

        for dir in [self.checkpoints_dir_path, self.logs_dir_path, self.imgs_dir_path]:
            dir.mkdir(parents=True, exist_ok=True)

        self.timer = Timer()
        self.avg_iter_time = 0.0

        self.current_iter = 0
        self.best_psnr = float("-inf")

    def _update_avg_time(self) -> None:
        time_scaler = 0.1

        iter_duration = self.timer.last_iter_duration

        if self.avg_iter_time == 0.0:
            self.avg_iter_time = iter_duration
        else:
            self.avg_iter_time = (1 - time_scaler) * self.avg_iter_time + time_scaler * iter_duration

        self.timer.last_iter_duration = 0.0

    def _log_model_info(self) -> None:
        dummy_input = torch.randn(1, 3, config.PATCH_SIZE, config.PATCH_SIZE).to(self.device)

        self.model.eval()
        flops, _ = profile(model=self.model, inputs=(dummy_input,), verbose=False)
        self.model.train()

        for module in self.model.modules():
            if hasattr(module, "total_ops"):
                del module.total_ops
            if hasattr(module, "total_params"):
                del module.total_params

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        dash_line = "-" * 60

        logger.info(dash_line)
        logger.info("Model & Training Information")
        logger.info(dash_line)
        logger.info(f"Model Architecture: {self.model.__class__.__name__}")
        logger.info(f"Total Parameters: {total_params / 1e6:.2f} M")
        logger.info(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
        logger.info(f"GFLOPs (per patch): {flops / 1e9:.4f} G")

        if self.device == "cuda":
            num_gpus = torch.cuda.device_count()
            logger.info(f"Device: CUDA ({num_gpus} GPUs active)")

            for i in range(num_gpus):
                gpu = torch.cuda.get_device_properties(i)
                total_memory = gpu.total_memory / (1024**3)
                logger.info(f"   [{i}] {gpu.name} ({total_memory:.1f} GB VRAM)")
        else:
            logger.info("Device: CPU")

        logger.info(dash_line)
        logger.info("Architecture Settings")
        logger.info(dash_line)
        logger.info(f"Number of Channels: {config.NUM_CHANNELS}")
        logger.info(f"RHAG Blocks: {config.NUM_RHAG_BLOCKS}")
        logger.info(f"HAB Blocks: {config.NUM_HAB_BLOCKS}")
        logger.info(f"Attention Heads: {config.NUM_HEADS}")
        logger.info(f"Window Size: {config.WINDOW_SIZE}")
        logger.info(f"Squeeze Factor: {config.SQUEEZE_FACTOR}")
        logger.info(f"Compress Ratio: {config.COMPRESS_RATIO}")
        logger.info(f"CAB Scale: {config.CAB_SCALE}")
        logger.info(f"Overlap Ratio: {config.OVERLAP_RATIO}")
        logger.info(f"MLP Ratio: {config.MLP_RATIO}")

        logger.info(dash_line)
        logger.info("Hyperparameters")
        logger.info(dash_line)
        logger.info(f"Current phase: {config.CURRENT_PHASE}")
        logger.info(f"Scaling Factor: x{self.scaling_factor}")
        logger.info(f"Patch Size: {config.PATCH_SIZE}")

        if config.EFFECTIVE_BATCH_SIZE == config.BATCH_SIZE:
            logger.info(f"Batch Size: {config.BATCH_SIZE}")
        else:
            logger.info(
                f"Effective Batch Size: {config.EFFECTIVE_BATCH_SIZE} ({config.BATCH_SIZE} * {config.EFFECTIVE_BATCH_SIZE // config.BATCH_SIZE})"
            )

        logger.info(f"Total iteration: {self.num_iters:,}")
        logger.info(f"Loss Function: {self.loss_fn.__class__.__name__}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")

        if isinstance(self.optimizer, torch.optim.Adam):
            logger.info(f"  - Betas: {config.ADAM_BETAS}")
            logger.info(f"  - Epsilon: {config.ADAM_EPS}")

        logger.info(f"Initial Learning Rate: {config.LEARNING_RATE}")
        logger.info(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")

        if self.scheduler:
            logger.info(f"  - Milestones: {config.SCHEDULER_MILESTONES}")
            logger.info(f"  - Gamma: {config.SCHEDULER_GAMMA}")

        logger.info(f"Scaler: {'Enabled' if self.scaler else 'Disabled'}")
        logger.info(f"Precision (Data Type): {self.dtype}")
        logger.info(f"Gradient Clipping: {config.GRADIENT_CLIPPING_NORM}")
        logger.info(f"Gradient Checkpointing: {config.USE_GRADIENT_CHECKPOINTING}")

        logger.info(dash_line)
        logger.info("Data Processing")
        logger.info(dash_line)
        logger.info(f"Training Workers: {config.TRAIN_NUM_WORKERS} (Prefetch: {config.TRAIN_PREFETCH_FACTOR})")
        logger.info(f"Val Workers: {config.VAL_NUM_WORKERS} (Prefetch: {config.VAL_PREFETCH_FACTOR})")
        logger.info(dash_line)

    def _log_imgs(self, lr_img_tensor: Tensor, sr_img_tensor: Tensor, hr_img_tensor: Tensor) -> None:
        lr_img_tensor = lr_img_tensor.float().cpu().clamp(0, 1)
        sr_img_tensor = sr_img_tensor.float().cpu().clamp(0, 1)
        hr_img_tensor = hr_img_tensor.float().cpu().clamp(0, 1)

        lr_img_tensor_resized = torch.nn.functional.interpolate(
            input=lr_img_tensor.unsqueeze(0),
            size=(hr_img_tensor.shape[1], hr_img_tensor.shape[2]),
            mode="nearest",
        ).squeeze(0)

        combined_img_tensor = torch.cat([lr_img_tensor_resized, sr_img_tensor, hr_img_tensor], dim=2)

        wandb.log(
            {
                "val/visual_results": wandb.Image(
                    data_or_path=combined_img_tensor,
                    caption=f"Iter {self.current_iter}: LR (Nearest) | SR ({self.model.__class__.__name__}) | HR(Truth)",
                )
            },
            step=self.current_iter,
        )

    def _log_train_progress(self, loss: float) -> None:
        elapsed_time = format_time(self.timer.get_elapsed_time())
        remaining_time = format_time((self.num_iters - self.current_iter) * self.avg_iter_time)

        current_lr = self.optimizer.param_groups[0]["lr"]

        logger.info(
            f"Iter: [{self.current_iter:>6d}/{self.num_iters}] "
            f"({format_time(self.avg_iter_time)} / {elapsed_time} / {remaining_time}) | Loss: {loss:.4f} | LR: {current_lr:.2e}"
        )

        if config.USE_WANDB:
            wandb.log(
                {
                    "train/loss": loss,
                    "train/lr": current_lr,
                    "train/iteration": self.current_iter,
                },
                step=self.current_iter,
            )

    def _log_val_progress(
        self,
        avg_loss: float,
        avg_psnr: float,
        avg_ssim: float,
        ssim_map: Tensor,
        lr_img_tensor: Tensor,
        sr_img_tensor: Tensor,
        hr_img_tensor: Tensor,
    ) -> None:
        logger.info(
            f"Validation | Iter: {self.current_iter} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}."
        )

        if config.USE_WANDB:
            self._log_imgs(
                lr_img_tensor=lr_img_tensor[0],
                sr_img_tensor=sr_img_tensor[0],
                hr_img_tensor=hr_img_tensor[0],
            )
            wandb.log(
                {
                    "val/loss": avg_loss,
                    "val/psnr": avg_psnr,
                    "val/ssim": avg_ssim,
                    "val/iteration": self.current_iter,
                    "val/ssim_map": wandb.Image(
                        data_or_path=torch.clamp(ssim_map.squeeze(0), 0.0, 1.0),
                        caption=f"Iter {self.current_iter}: SSIM Map (brighter = better)",
                    ),
                },
                step=self.current_iter,
            )

    def _train_step(self, batch: dict[str, Tensor], is_accumulating: bool = False) -> Tensor:
        lr_img_tensor = batch["lr"].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)
        hr_img_tensor = batch["hr"].to(device=self.device, memory_format=torch.channels_last, non_blocking=True)

        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=True):
            sr_img_tensor = self.model(lr_img_tensor)
            loss = self.loss_fn(sr_img_tensor, hr_img_tensor)

        if self.accumulation_steps > 1:
            loss /= self.accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not is_accumulating:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clipping_norm)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

        return loss.detach() * self.accumulation_steps

    @torch.inference_mode()
    def validate(self) -> None:
        self.model.eval()
        torch.cuda.empty_cache()

        avg_loss, avg_psnr, avg_ssim = 0.0, 0.0, 0.0

        for batch in tqdm(self.val_dataloader, desc="Validating...", total=len(self.val_dataloader), leave=False):
            lr_img_tensor = batch["lr"].to(device=self.device, non_blocking=True)
            hr_img_tensor = batch["hr"].to(device=self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype, enabled=True):
                sr_img_tensor = self.model(lr_img_tensor)
                loss = self.loss_fn(sr_img_tensor, hr_img_tensor)
                avg_loss += loss.item()

            batch_psnr, batch_ssim = 0.0, 0.0
            batch_size = sr_img_tensor.size(0)

            for j in range(batch_size):
                sr_img_item = sr_img_tensor[j].float()
                hr_img_item = hr_img_tensor[j].float()

                batch_psnr += calculate_psnr(
                    sr_img_tensor=sr_img_item,
                    hr_img_tensor=hr_img_item,
                    crop_border=self.scaling_factor,
                )

                ssim_value, ssim_map = calculate_ssim(
                    sr_img_tensor=sr_img_item,
                    hr_img_tensor=hr_img_item,
                    crop_border=self.scaling_factor,
                    return_map=True,
                )

                batch_ssim += ssim_value

            avg_psnr += batch_psnr / batch_size
            avg_ssim += batch_ssim / batch_size

        self.model.train()
        torch.cuda.empty_cache()

        avg_loss /= len(self.val_dataloader)
        avg_psnr /= len(self.val_dataloader)
        avg_ssim /= len(self.val_dataloader)

        self._log_val_progress(
            avg_loss=avg_loss,
            avg_psnr=avg_psnr,
            avg_ssim=avg_ssim,
            ssim_map=ssim_map,
            lr_img_tensor=lr_img_tensor,
            sr_img_tensor=sr_img_tensor,
            hr_img_tensor=hr_img_tensor,
        )

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(is_best=True)

    def train(self) -> None:
        self._log_model_info()

        if self.current_iter > 0:
            logger.info(
                f"Resuming training on {self.device.upper()} from iteration {self.current_iter:,} / {self.num_iters:,}."
            )
        else:
            logger.info(f"Starting training on {self.device.upper()} for {self.num_iters:,} iterations.")

        self.model.train()

        batch_counter = 0

        for batch in self.train_dataloader:
            with self.timer:
                batch_counter += 1
                is_accumulating = batch_counter % self.accumulation_steps != 0

                loss_tensor = self._train_step(batch, is_accumulating)

                if not is_accumulating:
                    self.current_iter += 1
                    self._update_avg_time()

                    if self.current_iter % self.log_freq == 0:
                        self._log_train_progress(loss_tensor.item())

                    if self.current_iter % self.val_freq == 0 and self.current_iter != 0:
                        logger.info("Validation started (it may take a few minutes)...")
                        self.validate()

                    if self.current_iter % self.save_freq == 0 and self.current_iter != 0:
                        self.save_checkpoint(is_best=False)

            if self.current_iter >= self.num_iters:
                break

        self.save_checkpoint(is_best=False)
        logger.info("Training run completed successfully.")

    def save_checkpoint(self, is_best: bool = False) -> None:
        model_state = {key: value.contiguous() for key, value in self.model.state_dict().items()}

        train_state = {
            "current_iter": self.current_iter,
            "best_psnr": self.best_psnr,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "wandb_id": wandb.run.id if wandb.run else None,
        }

        if is_best:
            save_dir = self.checkpoints_dir_path / "best"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Checkpoint] New best model saved (PSNR: {self.best_psnr:.2f} dB.)")
        else:
            save_dir = self.checkpoints_dir_path / f"iter_{self.current_iter}"
            save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Checkpoint] Checkpoint saved to '{save_dir}'.")

        save_file(model_state, save_dir / "model.safetensors")
        torch.save(train_state, save_dir / "state.pth")

    def load_checkpoint(self, checkpoint_dir_path: Path) -> None:
        model_path = checkpoint_dir_path / "model.safetensors"

        if model_path.exists():
            checkpoint_state_dict = load_file(model_path, device=self.device)

            try:
                self.model.load_state_dict(checkpoint_state_dict)
            except RuntimeError:
                logger.error("[Checkpoint] Architecture mismatch during weights loading! Raising error.")
                raise

            logger.info(f"[Checkpoint] Model weights loaded successfully from {checkpoint_dir_path.name}.")
        else:
            logger.warning(f"[Checkpoint] Model weights file not found at '{model_path}'.")

        state_path = checkpoint_dir_path / "state.pth"

        if state_path.exists():
            state = torch.load(checkpoint_dir_path / "state.pth", map_location=self.device)

            self.current_iter = state["current_iter"]
            self.best_psnr = state["best_psnr"]

            self.optimizer.load_state_dict(state["optimizer_state"])

            if self.scheduler and state["scheduler_state"]:
                self.scheduler.load_state_dict(state["scheduler_state"])

            if self.scaler and state["scaler_state"]:
                self.scaler.load_state_dict(state["scaler_state"])

            logger.info("[Checkpoint] Training state loaded successfully.")
        else:
            logger.warning(f"[Checkpoint] Model state file not found at '{state_path}'.")
