from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F

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

        qkv_tensor = (
            self.qkv_layer(x)
            .reshape(
                num_windows,
                num_pixels_in_window,
                3,
                self.num_heads,
                num_channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = qkv_tensor[0], qkv_tensor[1], qkv_tensor[2]

        queries *= self.scale
        attention_scores = queries @ keys.transpose(-2, -1)

        relative_position_bias = (
            self.relative_position_bias_table[rpi_sa.flatten()]
            .view(num_pixels_in_window, num_pixels_in_window, -1)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze_(0)
        )

        attention_scores += relative_position_bias

        if attention_mask is not None:
            num_windows_per_img = attention_mask.shape[0]

            attention_scores = attention_scores.view(
                num_windows // num_windows_per_img,
                num_windows_per_img,
                self.num_heads,
                num_pixels_in_window,
                num_pixels_in_window,
            )
            attention_scores += attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_heads, num_pixels_in_window, num_pixels_in_window)

        attention_probs = F.softmax(attention_scores, dim=-1)

        x = (attention_probs @ values).transpose(1, 2).reshape(num_windows, num_pixels_in_window, num_channels)
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
        alpha: float,
        train_img_size: tuple[int, int],
        shift_size: int,
        mlp_ratio: float | int,
    ) -> None:
        super().__init__()

        self.window_size = window_size
        self.alpha = alpha
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
        x = x.view(batch_size, img_height, img_width, num_channels)

        x_cab = x.permute(0, 3, 1, 2)
        x_cab = self.cab(x_cab)
        x_cab = x_cab.permute(0, 2, 3, 1).reshape(batch_size, img_height * img_width, num_channels)

        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x_shifted = x
            attention_mask = None  # type: ignore

        x_windows = split_img_into_windows(img_tensor=x_shifted, window_size=self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, num_channels)

        attention_windows = self.wmsa(x_windows, rpi_sa=rpi_sa, attention_mask=attention_mask)

        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, num_channels)
        x_shifted = combine_windows_into_img(
            windows_tensor=attention_windows, img_height=img_height, img_width=img_width
        )

        if self.shift_size > 0:
            x_wmsa = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_wmsa = x_shifted

        x_wmsa = x_wmsa.view(batch_size, img_height * img_width, num_channels)

        x = x_wmsa + self.alpha * x_cab + residual
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
        batch_size, num_pixels_in_window, num_channels = x.shape

        residual = x

        x = self.layer_norm_1(x)
        x = x.view(batch_size, img_height, img_width, num_channels)

        qkv = self.qkv_layer(x).reshape(batch_size, img_height, img_width, 3, num_channels).permute(3, 0, 4, 1, 2)
        queries = qkv[0].permute(0, 2, 3, 1)
        keys_values = torch.cat((qkv[1], qkv[2]), dim=1)

        q_windows = split_img_into_windows(img_tensor=queries, window_size=self.window_size)
        q_windows = q_windows.view(-1, self.window_size * self.window_size, num_channels)

        kv_windows = self.unfold(keys_values)
        kv_windows = rearrange(
            kv_windows,
            "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch",
            nc=2,
            ch=num_channels,
            owh=self.overlapped_window_size,
            oww=self.overlapped_window_size,
        ).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.num_channels // self.num_heads

        queries = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)
        keys = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        values = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)

        queries *= self.scale
        attention = queries @ keys.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[rpi_oca.view(-1)].view(
            self.window_size * self.window_size, self.overlapped_window_size * self.overlapped_window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        attention += relative_position_bias.unsqueeze(0)

        attention = self.softmax(attention)
        attention_windows = (attention @ values).transpose(1, 2).reshape(b_, nq, self.num_channels)

        attention_windows = attention_windows.view(-1, self.window_size, self.window_size, self.num_channels)

        x = combine_windows_into_img(windows_tensor=attention_windows, img_height=img_height, img_width=img_width)
        x = x.view(batch_size, img_height * img_width, self.num_channels)

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
        alpha: float,
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
                    alpha=alpha,
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

        x = x.transpose(1, 2).view(batch_size, num_channels, img_height, img_width)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x += residual

        return x


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
        alpha: float,
        train_img_size: tuple[int, int],
        mlp_ratio: float | int,
        overlap_ratio: float | int,
        oca_overlap_ratio: float | int,
        scaling_factor: int,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.window_size = window_size
        self.oca_overlap_ratio = oca_overlap_ratio
        self.scaling_factor = scaling_factor

        self.shallow_feature_extraction = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.deep_feature_extraction = nn.ModuleList(
            [
                RHAG(
                    num_hab_blocks=num_hab_blocks,
                    num_channels=num_channels,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    window_size=window_size,
                    num_heads=num_heads,
                    alpha=alpha,
                    train_img_size=train_img_size,
                    mlp_ratio=mlp_ratio,
                    overlap_ratio=overlap_ratio,
                )
                for i in range(num_rhag_blocks)
            ]
        )

        self.conv_after_dfe = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.img_reconstruction = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels * (scaling_factor**2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(upscale_factor=scaling_factor),
            nn.Conv2d(
                in_channels=num_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
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
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1

        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_position_index

    def _calculate_rpi_oca(self) -> Tensor:
        window_size_original = self.window_size
        window_size_overlapped = self.window_size + int(self.oca_overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_original)
        coords_w = torch.arange(window_size_original)

        coords_original = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_original_flatten = torch.flatten(coords_original, 1)

        coords_h = torch.arange(window_size_overlapped)
        coords_w = torch.arange(window_size_overlapped)

        coords_overlapped = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_overlapped_flatten = torch.flatten(coords_overlapped, 1)

        relative_coords = coords_overlapped_flatten[:, None, :] - coords_original_flatten[:, :, None]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += window_size_original - 1
        relative_coords[:, :, 1] += window_size_original - 1

        relative_coords[:, :, 0] *= window_size_original + window_size_overlapped - 1
        relative_position_index = relative_coords.sum(-1)

        return relative_position_index

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
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

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

        x = x.flatten(2).transpose(1, 2)

        attention_mask = self._calculate_attention_mask((padded_img_height, padded_img_width))
        attention_mask = attention_mask.type_as(x)

        for layer in self.deep_feature_extraction:
            x = layer(
                x,
                (padded_img_height, padded_img_width),
                self.rpi_sa,
                self.rpi_oca,
                attention_mask,
            )

        x = x.transpose(1, 2).view(batch_size, self.num_channels, padded_img_height, padded_img_width)

        x = self.conv_after_dfe(x) + x_after_sfe

        x = self.img_reconstruction(x)

        x = x[:, :, : img_height * self.scaling_factor, : img_width * self.scaling_factor]

        x += self.imgs_mean

        return x
