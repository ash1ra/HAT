from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from trainer import Trainer


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


@pytest.fixture
def trainer_setup(tmp_path: Path) -> tuple[Trainer, nn.Module, optim.Optimizer, Path]:
    model = DummyModel()

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2)

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    root_dir = tmp_path / "test"

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaling_factor=2,
        num_iters=10,
        log_freq=1,
        val_freq=5,
        save_freq=5,
        root_dir_path=root_dir,
        gradient_clipping_norm=0.5,
        device="cpu",
        dtype=torch.bfloat16,
        use_wandb=False,
    )

    return trainer, model, optimizer, root_dir


def test_trainer_initialization(trainer_setup: tuple) -> None:
    trainer, _, _, root_dir = trainer_setup

    assert trainer.checkpoints_dir_path.exists()
    assert trainer.logs_dir_path.exists()
    assert trainer.imgs_dir_path.exists()

    assert trainer.current_iter == 0
    assert trainer.best_psnr == float("-inf")


def test_trainer_update_avg_time(trainer_setup: tuple) -> None:
    trainer, _, _, _ = trainer_setup
    alpha = 0.1

    trainer.timer.last_iter_duration = 10.0
    trainer._update_avg_time()

    assert trainer.avg_iter_time == 10.0
    assert trainer.timer.last_iter_duration == 0.0

    trainer.timer.last_iter_duration = 5.0
    expected_avg_iter_time = (1 - alpha) * trainer.avg_iter_time + alpha * trainer.timer.last_iter_duration
    trainer._update_avg_time()

    assert trainer.avg_iter_time == expected_avg_iter_time


def test_trainer_train_step(trainer_setup: tuple) -> None:
    trainer, model, optimizer, _ = trainer_setup

    batch = next(iter(trainer.train_dataloader))

    initial_weights = model.conv.weight.clone()

    loss = trainer._train_step(batch, is_accumulating=False)

    assert isinstance(loss, Tensor)
    assert not torch.isnan(loss)

    assert not torch.equal(model.conv.weight, initial_weights)


@patch.object(Trainer, "_train_step")
@patch.object(Trainer, "validate")
@patch.object(Trainer, "save_checkpoint")
def test_trainer_train(mock_save_checkpoint, mock_validate, mock_train_step, trainer_setup: tuple) -> None:
    trainer, _, _, _ = trainer_setup

    trainer.num_iters = 3
    trainer.val_freq = 2
    trainer.save_freq = 2
    trainer.log_freq = 1

    mock_train_step.return_value = torch.tensor(0.5)

    trainer.train()

    assert trainer.current_iter == 3
    assert mock_train_step.call_count == 3

    mock_validate.assert_called_once()

    assert mock_save_checkpoint.call_count == 2

    mock_save_checkpoint.assert_any_call(is_best=False)


@patch("trainer.calculate_psnr")
@patch("trainer.calculate_ssim")
@patch.object(Trainer, "save_checkpoint")
def test_trainer_validate(mock_save_checkpoint, mock_calculate_ssim, mock_calculate_psnr, trainer_setup: tuple) -> None:
    trainer, _, _, _ = trainer_setup

    mock_calculate_psnr.return_value = 35.0
    mock_calculate_ssim.return_value = (0.95, torch.rand(1, 1, 32, 32))

    trainer.validate()

    assert trainer.best_psnr == 35.0
    mock_save_checkpoint.assert_called_once_with(is_best=True)

    mock_save_checkpoint.reset_mock()

    mock_calculate_psnr.return_value = 30.0

    trainer.validate()

    assert trainer.best_psnr == 35.0
    mock_save_checkpoint.assert_not_called()


def test_trainer_gradient_accumulation(trainer_setup: tuple) -> None:
    trainer, model, optimizer, _ = trainer_setup
    trainer.accumulation_steps = 2

    batch = next(iter(trainer.train_dataloader))

    with patch.object(optimizer, "step") as mock_step:
        trainer._train_step(batch, is_accumulating=True)
        mock_step.assert_not_called()

    with patch.object(optimizer, "step") as mock_step:
        trainer._train_step(batch, is_accumulating=False)
        mock_step.assert_called_once()


@patch("trainer.save_file")
@patch("torch.save")
def test_trainer_save_checkpoint(mock_torch_save, mock_save_file, trainer_setup: tuple) -> None:
    trainer, _, _, _ = trainer_setup

    trainer.current_iter = 100
    trainer.best_psnr = 32.1

    trainer.save_checkpoint(is_best=False)

    expected_save_dir = trainer.checkpoints_dir_path / "iter_100"

    mock_save_file.assert_called_once()

    args, kwargs = mock_save_file.call_args
    assert args[1] == expected_save_dir / "model.safetensors"

    mock_torch_save.assert_called_once()

    args, kwargs = mock_torch_save.call_args
    state = args[0]
    assert state["current_iter"] == 100
    assert state["best_psnr"] == 32.1
    assert args[1] == expected_save_dir / "state.pth"


@patch("trainer.load_file")
@patch("torch.load")
def test_trainer_load_checkpoint(mock_torch_load, mock_load_file, trainer_setup: tuple) -> None:
    trainer, model, _, root_dir = trainer_setup

    checkpoint_dir = root_dir / "dummy_checkpoint"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "model.safetensors").touch()
    (checkpoint_dir / "state.pth").touch()

    dummy_state_dict = model.state_dict()
    mock_load_file.return_value = dummy_state_dict

    mock_torch_load.return_value = {
        "current_iter": 50,
        "best_psnr": 30.0,
        "optimizer_state": trainer.optimizer.state_dict(),
        "scheduler_state": None,
        "scaler_state": None,
        "wandb_id": "dummy_id",
    }

    trainer.load_checkpoint(checkpoint_dir)

    assert trainer.current_iter == 50
    assert trainer.best_psnr == 30.0
