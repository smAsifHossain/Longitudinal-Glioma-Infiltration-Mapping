from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if data_format not in {"channels_last", "channels_first"}:
            raise NotImplementedError(data_format)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        dims = tuple(range(2, x.ndim))
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        shape = [1, -1] + [1] * (x.ndim - 2)
        return self.weight.view(*shape) * x + self.bias.view(*shape)


class MedNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: int | None = None,
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__()
        self.do_res = do_res
        self.dim = dim
        conv = nn.Conv3d if dim == "3d" else nn.Conv2d
        self.conv1 = conv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        elif norm_type == "layer":
            self.norm = LayerNorm(in_channels, data_format="channels_first")
        else:
            raise ValueError(f"Unknown norm_type={norm_type}")
        self.conv2 = conv(in_channels, exp_r * in_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv3 = conv(exp_r * in_channels, out_channels, kernel_size=1)
        self.grn = grn
        if grn:
            spatial = (1, 1, 1) if dim == "3d" else (1, 1)
            self.grn_beta = nn.Parameter(torch.zeros((1, exp_r * in_channels, *spatial)))
            self.grn_gamma = nn.Parameter(torch.zeros((1, exp_r * in_channels, *spatial)))

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            dims = tuple(range(2, x1.ndim))
            gx = torch.norm(x1, p=2, dim=dims, keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        return x + x1 if self.do_res else x1


class MedNeXtDownBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False,
                 norm_type="group", dim="3d", grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, False, norm_type, dim=dim, grn=grn)
        conv = nn.Conv3d if dim == "3d" else nn.Conv2d
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv1 = conv(
            in_channels, in_channels, kernel_size=kernel_size, stride=2,
            padding=kernel_size // 2, groups=in_channels
        )

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        x1 = super().forward(x)
        if self.resample_do_res:
            x1 = x1 + self.res_conv(x)
        return x1


class MedNeXtUpBlock(MedNeXtBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False,
                 norm_type="group", dim="3d", grn=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, False, norm_type, dim=dim, grn=grn)
        conv = nn.ConvTranspose3d if dim == "3d" else nn.ConvTranspose2d
        self.resample_do_res = do_res
        self.dim = dim
        if do_res:
            self.res_conv = conv(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv1 = conv(
            in_channels, in_channels, kernel_size=kernel_size, stride=2,
            padding=kernel_size // 2, groups=in_channels
        )

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        x1 = super().forward(x)
        pad = (1, 0, 1, 0, 1, 0) if self.dim == "3d" else (1, 0, 1, 0)
        x1 = F.pad(x1, pad)
        if self.resample_do_res:
            x1 = x1 + F.pad(self.res_conv(x), pad)
        return x1


class OutBlock(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, dim: str = "3d"):
        super().__init__()
        conv = nn.ConvTranspose3d if dim == "3d" else nn.ConvTranspose2d
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, dummy_tensor=None) -> torch.Tensor:
        return self.conv_out(x)


class MedNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r=4,
        kernel_size: int = 7,
        enc_kernel_size: int | None = None,
        dec_kernel_size: int | None = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: str | None = None,
        block_counts=(2, 2, 2, 2, 2, 2, 2, 2, 2),
        norm_type: str = "group",
        dim: str = "3d",
        grn: bool = False,
    ):
        super().__init__()
        if checkpoint_style not in {None, "outside_block"}:
            raise ValueError(checkpoint_style)
        if checkpoint_style is not None:
            raise NotImplementedError("Smoke test uses checkpoint_style=None")
        self.do_ds = deep_supervision
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size
        conv = nn.Conv3d if dim == "3d" else nn.Conv2d
        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if isinstance(exp_r, int):
            exp_r = [exp_r] * len(block_counts)

        def seq(count, cin, cout, ratio, k):
            return nn.Sequential(*[
                MedNeXtBlock(cin, cout, ratio, k, do_res, norm_type, dim=dim, grn=grn)
                for _ in range(count)
            ])

        self.enc_block_0 = seq(block_counts[0], n_channels, n_channels, exp_r[0], enc_kernel_size)
        self.down_0 = MedNeXtDownBlock(n_channels, 2*n_channels, exp_r[1], enc_kernel_size, do_res_up_down, norm_type, dim)
        self.enc_block_1 = seq(block_counts[1], 2*n_channels, 2*n_channels, exp_r[1], enc_kernel_size)
        self.down_1 = MedNeXtDownBlock(2*n_channels, 4*n_channels, exp_r[2], enc_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.enc_block_2 = seq(block_counts[2], 4*n_channels, 4*n_channels, exp_r[2], enc_kernel_size)
        self.down_2 = MedNeXtDownBlock(4*n_channels, 8*n_channels, exp_r[3], enc_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.enc_block_3 = seq(block_counts[3], 8*n_channels, 8*n_channels, exp_r[3], enc_kernel_size)
        self.down_3 = MedNeXtDownBlock(8*n_channels, 16*n_channels, exp_r[4], enc_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.bottleneck = seq(block_counts[4], 16*n_channels, 16*n_channels, exp_r[4], dec_kernel_size)
        self.up_3 = MedNeXtUpBlock(16*n_channels, 8*n_channels, exp_r[5], dec_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.dec_block_3 = seq(block_counts[5], 8*n_channels, 8*n_channels, exp_r[5], dec_kernel_size)
        self.up_2 = MedNeXtUpBlock(8*n_channels, 4*n_channels, exp_r[6], dec_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.dec_block_2 = seq(block_counts[6], 4*n_channels, 4*n_channels, exp_r[6], dec_kernel_size)
        self.up_1 = MedNeXtUpBlock(4*n_channels, 2*n_channels, exp_r[7], dec_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.dec_block_1 = seq(block_counts[7], 2*n_channels, 2*n_channels, exp_r[7], dec_kernel_size)
        self.up_0 = MedNeXtUpBlock(2*n_channels, n_channels, exp_r[8], dec_kernel_size, do_res_up_down, norm_type, dim, grn)
        self.dec_block_0 = seq(block_counts[8], n_channels, n_channels, exp_r[8], dec_kernel_size)
        self.out_0 = OutBlock(n_channels, n_classes, dim)
        self.dummy_tensor = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        if deep_supervision:
            self.out_1 = OutBlock(2*n_channels, n_classes, dim)
            self.out_2 = OutBlock(4*n_channels, n_classes, dim)
            self.out_3 = OutBlock(8*n_channels, n_classes, dim)
            self.out_4 = OutBlock(16*n_channels, n_classes, dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        r0 = self.enc_block_0(x)
        x = self.down_0(r0)
        r1 = self.enc_block_1(x)
        x = self.down_1(r1)
        r2 = self.enc_block_2(x)
        x = self.down_2(r2)
        r3 = self.enc_block_3(x)
        x = self.down_3(r3)
        x = self.bottleneck(x)
        ds4 = self.out_4(x) if self.do_ds else None
        x = self.dec_block_3(r3 + self.up_3(x))
        ds3 = self.out_3(x) if self.do_ds else None
        x = self.dec_block_2(r2 + self.up_2(x))
        ds2 = self.out_2(x) if self.do_ds else None
        x = self.dec_block_1(r1 + self.up_1(x))
        ds1 = self.out_1(x) if self.do_ds else None
        x = self.dec_block_0(r0 + self.up_0(x))
        out = self.out_0(x)
        return [out, ds1, ds2, ds3, ds4] if self.do_ds else out


def create_mednext_v1(num_input_channels: int, num_classes: int, model_id: str = "S",
                      kernel_size: int = 3, deep_supervision: bool = False) -> MedNeXt:
    configs = {
        "S": dict(exp_r=2, block_counts=[2]*9, checkpoint_style=None),
        "B": dict(exp_r=[2,3,4,4,4,4,4,3,2], block_counts=[2]*9, checkpoint_style=None),
        "M": dict(exp_r=[2,3,4,4,4,4,4,3,2], block_counts=[3,4,4,4,4,4,4,4,3], checkpoint_style=None),
        "L": dict(exp_r=[3,4,8,8,8,8,8,4,3], block_counts=[3,4,8,8,8,8,8,4,3], checkpoint_style=None),
    }
    cfg = configs[model_id]
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
        do_res=True,
        do_res_up_down=True,
        **cfg,
    )
