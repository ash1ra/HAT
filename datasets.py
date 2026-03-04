import random
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2 as transforms

from prepare_data import imresize
from utils import logger


class DynamicPairDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        scaling_factor: int,
        patch_size: int,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.test_mode = test_mode
        self.dev_mode = dev_mode

        with open(data_path, "r") as f:
            self.img_paths = [Path(line.strip()) for line in f if line.strip()]

        if len(self.img_paths) == 0:
            raise FileNotFoundError("Images not found at specified path.")

        if dev_mode:
            self.img_paths = self.img_paths[: int(len(self.img_paths) * 0.1)]

        self.normalize = transforms.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_path = self.img_paths[index]

        try:
            hr_img_tensor = decode_image(str(img_path), mode=ImageReadMode.RGB)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        if self.test_mode:
            _, img_height, img_width = hr_img_tensor.shape

            img_height_new = img_height - img_height % self.scaling_factor
            img_width_new = img_width - img_width % self.scaling_factor
            hr_img_tensor = hr_img_tensor[:, :img_height_new, :img_width_new]

            hr_patch_np = rearrange(hr_img_tensor, "c h w -> h w c").numpy()
            lr_patch_np = imresize(hr_patch_np, scale=1 / self.scaling_factor)
            lr_patch = rearrange(torch.from_numpy(lr_patch_np), "h w c -> c h w")

            return {
                "hr": self.normalize(hr_img_tensor),
                "lr": self.normalize(lr_patch),
            }

        _, img_height, img_width = hr_img_tensor.shape
        hr_patch_size = self.patch_size * self.scaling_factor

        if img_height < hr_patch_size or img_width < hr_patch_size:
            logger.warning(
                f"[Data] Image '{img_path.name}' ({img_height}x{img_width}) is smaller than patch size ({hr_patch_size}). Skipped."
            )
            return self.__getitem__(random.randint(0, len(self) - 1))

        y = random.randint(0, img_height - hr_patch_size)
        x = random.randint(0, img_width - hr_patch_size)
        hr_patch = hr_img_tensor[:, y : y + hr_patch_size, x : x + hr_patch_size]

        if random.random() < 0.5:
            hr_patch = transforms.functional.hflip(hr_patch)

        if random.random() < 0.5:
            hr_patch = transforms.functional.vflip(hr_patch)

        k = random.randint(0, 3)
        if k > 0:
            hr_patch = torch.rot90(hr_patch, k, [1, 2])

        hr_patch_np = rearrange(hr_patch, "c h w -> h w c").numpy()
        lr_patch_np = imresize(hr_patch_np, scale=1 / self.scaling_factor)
        lr_patch = rearrange(torch.from_numpy(lr_patch_np), "h w c -> c h w")

        return {
            "hr": self.normalize(hr_patch),
            "lr": self.normalize(lr_patch),
        }


class StaticPairDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        scaling_factor: int,
        patch_size: int,
        test_mode: bool = False,
        dev_mode: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.test_mode = test_mode
        self.dev_mode = dev_mode

        self.hr_dir_path = data_path / "HR"
        self.lr_dir_path = data_path / f"LR_x{scaling_factor}"

        if not self.hr_dir_path.exists() or not self.lr_dir_path.exists():
            raise FileNotFoundError(f"[Data] Datasets directories not found in '{data_path}'")

        hr_img_names = {hr_img.name for hr_img in self.hr_dir_path.glob("*.png")}
        lr_img_names = {lr_img.name for lr_img in self.lr_dir_path.glob("*.png")}

        self.img_names = sorted(list(hr_img_names & lr_img_names))

        if not self.test_mode:
            valid_images = []
            for img_name in self.img_names:
                lr_path = self.lr_dir_path / img_name

                with Image.open(lr_path) as img:
                    img_width, img_height = img.size
                    if img_width >= self.patch_size and img_height >= self.patch_size:
                        valid_images.append(img_name)
                    else:
                        logger.warning(f"[Data] Dropped '{img_name}' ({img_width}x{img_height})")

            self.img_names = valid_images

        if len(self.img_names) == 0:
            raise FileNotFoundError(
                f"[Data] No matching files found between '{self.hr_dir_path}' and '{self.lr_dir_path}'"
            )

        if len(hr_img_names) != len(lr_img_names):
            logger.warning(
                f"[Data] Count mismatch! HR: {len(hr_img_names)}, LR: {len(lr_img_names)}. "
                f"Proceeding with {len(self.img_names)} common files."
            )

        if dev_mode:
            self.img_names = self.img_names[: int(len(self.img_names) * 0.1)]

        self.normalize = transforms.ToDtype(torch.float32, scale=True)

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_name = self.img_names[index]

        hr_img_path = str(self.hr_dir_path / img_name)
        lr_img_path = str(self.lr_dir_path / img_name)

        try:
            hr_img_tensor = decode_image(hr_img_path, mode=ImageReadMode.RGB)
            lr_img_tensor = decode_image(lr_img_path, mode=ImageReadMode.RGB)
        except Exception as e:
            if self.test_mode:
                logger.error(f"[Data] Error loading '{img_name}'.")
                raise e
            else:
                logger.warning(f"[Data] Failed to load '{img_name}'. Skipping and resampling.")
                return self.__getitem__(random.randint(0, len(self) - 1))

        hr_img_tensor = self.normalize(hr_img_tensor)
        lr_img_tensor = self.normalize(lr_img_tensor)

        if self.test_mode:
            return {"hr": hr_img_tensor, "lr": lr_img_tensor}

        _, img_height, img_width = lr_img_tensor.shape

        if img_height <= self.patch_size or img_width <= self.patch_size:
            logger.warning(
                f"[Data] Image '{img_name}' ({img_height}x{img_width}) is smaller than patch size ({self.patch_size}). Skipped."
            )
            return self.__getitem__(random.randint(0, len(self) - 1))

        lr_y = random.randint(0, img_height - self.patch_size)
        lr_x = random.randint(0, img_width - self.patch_size)

        hr_y = lr_y * self.scaling_factor
        hr_x = lr_x * self.scaling_factor
        hr_patch_size = self.patch_size * self.scaling_factor

        lr_patch = lr_img_tensor[:, lr_y : lr_y + self.patch_size, lr_x : lr_x + self.patch_size]
        hr_patch = hr_img_tensor[:, hr_y : hr_y + hr_patch_size, hr_x : hr_x + hr_patch_size]

        if random.random() < 0.5:
            lr_patch = transforms.functional.hflip(lr_patch)
            hr_patch = transforms.functional.hflip(hr_patch)

        if random.random() < 0.5:
            lr_patch = transforms.functional.vflip(lr_patch)
            hr_patch = transforms.functional.vflip(hr_patch)

        k = random.randint(0, 3)
        if k > 0:
            lr_patch = torch.rot90(lr_patch, k, [1, 2])
            hr_patch = torch.rot90(hr_patch, k, [1, 2])

        return {"hr": hr_patch, "lr": lr_patch}
