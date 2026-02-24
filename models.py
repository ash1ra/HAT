"""
This file uses copied or rewritten code versions from the following repositories:

HAT: https://github.com/XPixelGroup/HAT/blob/main/hat/archs/hat_arch.py
SwinIR: https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py
"""

import math
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils import checkpoint

from utils import combine_windows_into_img, split_img_into_windows


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, squeeze_factor: int) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels // squeeze_factor,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=num_channels // squeeze_factor,
                out_channels=num_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_channels: int, compress_ratio: int, squeeze_factor: int) -> None:
        super().__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels // compress_ratio,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=num_channels // compress_ratio,
                out_channels=num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ChannelAttention(
                num_channels=num_channels,
                squeeze_factor=squeeze_factor,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.cab(x)


class WMSA(nn.Module):
    def __init__(self, num_channels: int, window_size: int, num_heads: int) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.window_size = window_size
        self.num_heads = num_heads

        self.scale = (num_channels // num_heads) ** -0.5

        self.qkv_layer = nn.Linear(in_features=num_channels, out_features=num_channels * 3)
        self.projection = nn.Linear(in_features=num_channels, out_features=num_channels)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, rpi_sa: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        num_windows, num_pixels_in_window, num_channels = x.shape

        qkv_tensor = rearrange(
            self.qkv_layer(x),
            "nw np (qkv heads d) -> qkv nw heads np d",
            qkv=3,
            heads=self.num_heads,
        )

        queries, keys, values = qkv_tensor[0], qkv_tensor[1], qkv_tensor[2]

        queries *= self.scale
        attention_scores = queries @ keys.transpose(-2, -1)

        relative_position_bias = rearrange(
            self.relative_position_bias_table[rpi_sa],
            "np1 np2 heads -> 1 heads np1 np2",
        )

        attention_scores += relative_position_bias

        if attention_mask is not None:
            nw_img = attention_mask.shape[0]

            attention_scores = rearrange(
                attention_scores,
                "(b nw_img) heads np1 np2 -> b nw_img heads np1 np2",
                nw_img=nw_img,
            )

            attention_scores += rearrange(
                attention_mask,
                "nw_img np1 np2 -> 1 nw_img 1 np1 np2",
                nw_img=nw_img,
            )

            attention_scores = rearrange(
                attention_scores,
                "b nw_img heads np1 np2 -> (b nw_img) heads np1 np2",
            )

        attention_probs = F.softmax(attention_scores, dim=-1)

        x = rearrange(
            attention_probs @ values,
            "nw heads np d -> nw np (heads d)",
        )

        x = self.projection(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class HAB(nn.Module):
    def __init__(
        self,
        num_channels: int,
        compress_ratio: int,
        squeeze_factor: int,
        window_size: int,
        num_heads: int,
        cab_scale: float,
        train_img_size: tuple[int, int],
        shift_size: int,
        mlp_ratio: float | int,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.cab_scale = cab_scale
        self.shift_size = shift_size

        if min(train_img_size) <= window_size:
            self.shift_size = 0
            self.window_size = min(train_img_size)

        if not (0 <= shift_size < window_size):
            raise ValueError(f"Shift size ({shift_size}) must be >= 0 and less than window_size ({window_size})")

        self.layer_norm_1 = nn.LayerNorm(num_channels)
        self.cab = CAB(
            num_channels=num_channels,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
        )
        self.wmsa = WMSA(
            num_channels=num_channels,
            window_size=self.window_size,
            num_heads=num_heads,
        )
        self.layer_norm_2 = nn.LayerNorm(num_channels)
        self.mlp = MLP(
            in_features=num_channels,
            hidden_features=int(num_channels * mlp_ratio),
            out_features=num_channels,
        )

    def forward(
        self,
        x: Tensor,
        x_size: tuple[int, int],
        rpi_sa: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        img_height, img_width = x_size
        batch_size, num_pixels_in_img, num_channels = x.shape

        residual = x

        x = self.layer_norm_1(x)

        x = rearrange(x, "b (h w) c -> b h w c", h=img_height, w=img_width)

        x_cab = rearrange(x, "b h w c -> b c h w").to(memory_format=torch.channels_last)
        x_cab = self.cab(x_cab)
        x_cab = rearrange(x_cab, "b c h w -> b (h w) c")

        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x
            attention_mask = None  # type: ignore

        x_windows = split_img_into_windows(img_tensor=x_shifted, window_size=self.window_size)
        x_windows = rearrange(x_windows, "np ws_h ws_w c -> np (ws_h ws_w) c")

        attention_windows = self.wmsa(x_windows, rpi_sa=rpi_sa, attention_mask=attention_mask)
        attention_windows = rearrange(attention_windows, "np (ws_h ws_w) c -> np ws_h ws_w c", ws_h=self.window_size)

        x_shifted = combine_windows_into_img(
            windows_tensor=attention_windows, img_height=img_height, img_width=img_width
        )

        if self.shift_size > 0:
            x_wmsa = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_wmsa = x_shifted

        x_wmsa = rearrange(x_wmsa, "b h w c -> b (h w) c")

        x = x_wmsa + self.cab_scale * x_cab + residual
        x = self.mlp(self.layer_norm_2(x)) + x

        return x


class OCAB(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_heads: int,
        window_size: int,
        overlap_ratio: int | float,
        mlp_ratio: float | int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.window_size = window_size
        self.overlapped_window_size = int(window_size * overlap_ratio) + window_size
        self.num_heads = num_heads

        self.scale = (num_channels // num_heads) ** -0.5

        self.qkv_layer = nn.Linear(in_features=num_channels, out_features=num_channels * 3)
        self.projection = nn.Linear(in_features=num_channels, out_features=num_channels)
        self.unfold = nn.Unfold(
            kernel_size=(self.overlapped_window_size, self.overlapped_window_size),
            stride=window_size,
            padding=(self.overlapped_window_size - window_size) // 2,
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size + self.overlapped_window_size - 1) * (window_size + self.overlapped_window_size - 1),
                num_heads,
            )
        )

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

        self.layer_norm_1 = nn.LayerNorm(num_channels)

        self.layer_norm_2 = nn.LayerNorm(num_channels)
        self.mlp = MLP(
            in_features=num_channels,
            hidden_features=int(num_channels * mlp_ratio),
            out_features=num_channels,
        )

    def forward(self, x: Tensor, x_size: tuple[int, int], rpi_oca: Tensor) -> Tensor:
        img_height, img_width = x_size
        batch_size, num_pixels_in_img, num_channels = x.shape

        residual = x

        x = self.layer_norm_1(x)
        x = rearrange(x, "b (h w) c -> b h w c", h=img_height, w=img_width)

        qkv = rearrange(
            self.qkv_layer(x),
            "b h w (qkv c) -> qkv b c h w",
            qkv=3,
        )

        queries = rearrange(qkv[0], "b c h w -> b h w c")
        keys_values = rearrange(qkv[1:3], "kv b c h w -> b (kv c) h w")

        q_windows = split_img_into_windows(img_tensor=queries, window_size=self.window_size)
        q_windows = rearrange(q_windows, "nw ws_h ws_w c -> nw (ws_h ws_w) c")

        kv_windows = self.unfold(keys_values)
        kv_windows = rearrange(
            kv_windows,
            "b (kv c ows_h ows_w) nw -> kv (b nw) (ows_h ows_w) c",
            kv=2,
            c=num_channels,
            ows_h=self.overlapped_window_size,
            ows_w=self.overlapped_window_size,
        )
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        queries = rearrange(q_windows, "nw np (heads d) -> nw heads np d", heads=self.num_heads)
        keys = rearrange(k_windows, "nw np (heads d) -> nw heads np d", heads=self.num_heads)
        values = rearrange(v_windows, "nw np (heads d) -> nw heads np d", heads=self.num_heads)

        queries *= self.scale
        attention_scores = queries @ keys.transpose(-2, -1)

        relative_position_bias = rearrange(
            self.relative_position_bias_table[rpi_oca.view(-1)],
            "(np_q np_kv) heads -> 1 heads np_q np_kv",
            np_q=self.window_size**2,
            np_kv=self.overlapped_window_size**2,
        )

        attention_scores += relative_position_bias

        attention_probs = self.softmax(attention_scores)

        attention_windows = rearrange(
            attention_probs @ values,
            "nw heads (ws_h ws_w) d -> nw ws_h ws_w (heads d)",
            ws_h=self.window_size,
        )

        x = combine_windows_into_img(windows_tensor=attention_windows, img_height=img_height, img_width=img_width)
        x = rearrange(x, "b h w c -> b (h w) c")

        x = self.projection(x) + residual
        x = x + self.mlp(self.layer_norm_2(x))

        return x


class RHAG(nn.Module):
    def __init__(
        self,
        num_hab_blocks: int,
        num_channels: int,
        compress_ratio: int,
        squeeze_factor: int,
        window_size: int,
        num_heads: int,
        cab_scale: float,
        train_img_size: tuple[int, int],
        mlp_ratio: float | int,
        overlap_ratio: float | int,
    ) -> None:
        super().__init__()

        self.habs = nn.ModuleList(
            [
                HAB(
                    num_channels=num_channels,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    window_size=window_size,
                    num_heads=num_heads,
                    cab_scale=cab_scale,
                    train_img_size=train_img_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(num_hab_blocks)
            ]
        )

        self.ocab = OCAB(
            num_channels=num_channels,
            num_heads=num_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
        )

        self.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self,
        x: Tensor,
        x_size: tuple[int, int],
        rpi_sa: Tensor,
        rpi_oca: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        img_height, img_width = x_size
        batch_size, num_pixels_in_img, num_channels = x.shape

        residual = x

        for hab in self.habs:
            x = hab(x, x_size, rpi_sa, attention_mask)

        x = self.ocab(x, x_size, rpi_oca)

        x = rearrange(x, "b (h w) c -> b c h w", h=img_height, w=img_width).to(memory_format=torch.channels_last)
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x += residual

        return x


class ImageReconstruction(nn.Module):
    def __init__(
        self,
        scaling_factor: int,
        num_channels: int,
        num_reconstruction_channels: int,
        num_output_channels: int,
    ) -> None:
        super().__init__()

        self.image_reconstruction_list = [
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_reconstruction_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
        ]

        if (scaling_factor & (scaling_factor - 1)) == 0:  # scaling_factor = 2^n
            for _ in range(int(math.log(scaling_factor, 2))):
                self.image_reconstruction_list.extend(
                    [
                        nn.Conv2d(
                            in_channels=num_reconstruction_channels,
                            out_channels=4 * num_reconstruction_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.PixelShuffle(upscale_factor=2),
                    ]
                )
        elif scaling_factor == 3:
            self.image_reconstruction_list.extend(
                [
                    nn.Conv2d(
                        in_channels=num_reconstruction_channels,
                        out_channels=9 * num_reconstruction_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.PixelShuffle(upscale_factor=3),
                ]
            )
        else:
            raise ValueError(
                f"Scaling factor of {scaling_factor} is not supported. Supported scaling factors: 2^n and 3."
            )

        self.image_reconstruction_list.append(
            nn.Conv2d(
                in_channels=num_reconstruction_channels,
                out_channels=num_output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.image_reconstruction = nn.Sequential(*self.image_reconstruction_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.image_reconstruction(x)


class HAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_rhag_blocks: int,
        num_hab_blocks: int,
        num_channels: int,
        compress_ratio: int,
        squeeze_factor: int,
        window_size: int,
        num_heads: int,
        cab_scale: float,
        train_img_size: tuple[int, int],
        mlp_ratio: float | int,
        overlap_ratio: float | int,
        scaling_factor: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.scaling_factor = scaling_factor
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm_before_dfe = nn.LayerNorm(normalized_shape=num_channels)

        self.deep_feature_extraction = nn.ModuleList(
            [
                RHAG(
                    num_hab_blocks=num_hab_blocks,
                    num_channels=num_channels,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    window_size=window_size,
                    num_heads=num_heads,
                    cab_scale=cab_scale,
                    train_img_size=train_img_size,
                    mlp_ratio=mlp_ratio,
                    overlap_ratio=overlap_ratio,
                )
                for i in range(num_rhag_blocks)
            ]
        )

        self.norm_after_dfe = nn.LayerNorm(normalized_shape=num_channels)

        self.conv_after_dfe = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.img_reconstruction = ImageReconstruction(
            scaling_factor=scaling_factor,
            num_channels=num_channels,
            num_reconstruction_channels=64,
            num_output_channels=in_channels,
        )

        self.register_buffer("rpi_sa", self._calculate_rpi_sa())
        self.register_buffer("rpi_oca", self._calculate_rpi_oca())

        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, in_channels, 1, 1) + 0.5

        self.register_buffer("imgs_mean", self.mean)

        self.apply(self._init_weights)

    def _calculate_rpi_sa(self) -> Tensor:
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)

        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = rearrange(coords, "c h w -> c (h w)")

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = rearrange(relative_coords, "c np1 np2 -> np1 np2 c")

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1

        relative_coords[:, :, 0] *= 2 * self.window_size - 1

        return relative_coords.sum(-1)

    def _calculate_rpi_oca(self) -> Tensor:
        window_size_original = self.window_size
        window_size_overlapped = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_original_h = torch.arange(window_size_original)
        coords_original_w = torch.arange(window_size_original)

        coords_overlapped_h = torch.arange(window_size_overlapped)
        coords_overlapped_w = torch.arange(window_size_overlapped)

        coords_original = torch.stack(torch.meshgrid([coords_original_h, coords_original_w], indexing="ij"))
        coords_overlapped = torch.stack(torch.meshgrid([coords_overlapped_h, coords_overlapped_w], indexing="ij"))

        coords_original_flatten = rearrange(coords_original, "c h w -> c (h w)")
        coords_overlapped_flatten = rearrange(coords_overlapped, "c h w -> c (h w)")

        relative_coords = coords_overlapped_flatten[:, None, :] - coords_original_flatten[:, :, None]
        relative_coords = rearrange(relative_coords, "c np_orig np_over -> np_orig np_over c")

        relative_coords[:, :, 0] += window_size_original - 1
        relative_coords[:, :, 1] += window_size_original - 1

        relative_coords[:, :, 0] *= window_size_original + window_size_overlapped - 1

        return relative_coords.sum(-1)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def _calculate_attention_mask(self, x_size: tuple[int, int]) -> Tensor:
        img_height, img_width = x_size
        img_mask = torch.zeros((1, img_height, img_width, 1))

        shift_size = self.window_size // 2

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )

        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -shift_size),
            slice(-shift_size, None),
        )

        count = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = count
                count += 1

        mask_windows = split_img_into_windows(img_tensor=img_mask, window_size=self.window_size)
        mask_windows = rearrange(mask_windows, "b ws_h ws_w c -> (b c) (ws_h ws_w)")

        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask.masked_fill_(attention_mask != 0, float(-100.0))
        attention_mask.masked_fill_(attention_mask == 0, float(0.0))

        return attention_mask

    def _add_padding(self, x: Tensor) -> Tensor:
        _, _, img_height, img_width = x.shape

        mod_pad_height = (self.window_size - img_height % self.window_size) % self.window_size
        mod_pad_width = (self.window_size - img_width % self.window_size) % self.window_size

        if mod_pad_height != 0 or mod_pad_width != 0:
            x = F.pad(x, (0, mod_pad_width, 0, mod_pad_height), "reflect")

        return x

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_channels, img_height, img_width = x.shape

        x = self._add_padding(x)
        _, _, padded_img_height, padded_img_width = x.shape

        self.imgs_mean = self.imgs_mean.type_as(x)
        x -= self.imgs_mean

        x = self.shallow_feature_extraction(x)
        x_after_sfe = x

        x = rearrange(x, "b c h w -> b (h w) c")

        x = self.norm_before_dfe(x)

        attention_mask = self._calculate_attention_mask((padded_img_height, padded_img_width))
        attention_mask = attention_mask.type_as(x)

        for layer in self.deep_feature_extraction:
            if self.use_gradient_checkpointing and x.requires_grad:
                x = checkpoint.checkpoint(
                    layer,
                    x,
                    (padded_img_height, padded_img_width),
                    self.rpi_sa,
                    self.rpi_oca,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x = layer(
                    x,
                    (padded_img_height, padded_img_width),
                    self.rpi_sa,
                    self.rpi_oca,
                    attention_mask,
                )

        x = self.norm_after_dfe(x)

        x = rearrange(x, "b (h w) c -> b c h w", h=padded_img_height, w=padded_img_width).to(
            memory_format=torch.channels_last
        )

        x = self.conv_after_dfe(x) + x_after_sfe

        x = self.img_reconstruction(x)

        x = x[:, :, : img_height * self.scaling_factor, : img_width * self.scaling_factor]

        x += self.imgs_mean

        return x
