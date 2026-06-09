'''
This code is basically on PFT-SR(https://github.com/CVL-UESTC/PFT-SR) & DMNet(https://github.com/PRIS-CV/DMNet)

'''

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper
from basicsr.utils.registry import ARCH_REGISTRY
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import pywt
from einops import rearrange
from torch.autograd import Function
from .idynamicdwconv_util import *
import smm_cuda


# ====================== SMM CUDA Ops ======================

class SMM_QmK(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None


class SMM_AmV(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None


# ====================== DFE  ======================

class DFE(nn.Module):
    """Dual Feature Extraction: bottleneck conv branch × linear gate branch"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        mid_dim = in_features // 5

        self.conv = nn.Sequential(
            nn.Conv2d(in_features, mid_dim, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_dim, out_features, 1, 1, 0),
        )
        self.linear = nn.Conv2d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, H * W).permute(0, 2, 1).contiguous()
        return x

    def flops(self, N):
        C = self.in_features
        mid = C // 5
        flops = 0
        flops += N * C * mid
        flops += N * mid * mid * 9
        flops += N * mid * C
        flops += N * C * C
        return flops


# ====================== DWT / IDWT ======================

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).float())
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).float())

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        filters = torch.cat([
            w_ll.unsqueeze(0).unsqueeze(1),
            w_lh.unsqueeze(0).unsqueeze(1),
            w_hl.unsqueeze(0).unsqueeze(1),
            w_hh.unsqueeze(0).unsqueeze(1),
        ], dim=0)
        self.register_buffer('filters', filters.float())

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


# ====================== WMA Full Subband ======================

class WMA_Full(nn.Module):
    """
    Wavelet Modulation Attention — full 4-subband version.
      Input (B, C, H, W)
        → reduce: C → C//4
        → DWT: (B, C//4, H, W) → (B, C, H/2, W/2)  [LL|LH|HL|HH 拼在 channel]
        → channel attention on all C channels (4 subbands interact)
        → IDynamic filter
        → IDWT: (B, C, H/2, W/2) → (B, C//4, H, W)
        → project_out: C//4 → C
      Output (B, C, H, W)
    """
    def __init__(self, dim, activation='relu'):
        super().__init__()
        self.dim = dim
        c_sub = dim // 4

        # 1. 降維
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, c_sub, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # 2. DWT
        self.dwt = DWT_2D(wave='haar')

        # 3. Channel attention on ALL 4 subbands (dim channels total)
        #    自動計算 head 數
        self.num_heads = 1
        for i in range(8, 0, -1):
            if dim % i == 0:
                self.num_heads = i
                break

        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=False,
        )

        # 4. IDynamic filter
        self.use_idynamic = dim >= 16
        if self.use_idynamic:
            idyn_heads = 1
            for i in range(self.num_heads, 0, -1):
                if dim % i == 0:
                    idyn_heads = i
                    break
            self.filter = IDynamic(channels=dim, kernel_size=7, group_channels=idyn_heads)
        else:
            self.filter = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # 5. IDWT + project back
        self.idwt = IDWT_2D(wave='haar')
        self.project_out = nn.Conv2d(c_sub, dim, kernel_size=1, bias=False)

        self.act = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, dim, H, W)
        Returns:
            (B, dim, H, W)
        """
        # reduce → DWT
        x_reduced = self.reduce(x)          # (B, C//4, H, W)
        x_dwt = self.dwt(x_reduced)         # (B, C, H/2, W/2) = [LL|LH|HL|HH]

        # Channel attention on all 4 subbands
        qkv = self.qkv_dwconv(self.qkv(x_dwt))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature   # (B, heads, C_h, C_h)
        attn = self.act(attn)

        out = attn @ v
        out = rearrange(
            out, 'b head c (h w) -> b (head c) h w',
            head=self.num_heads, h=x_dwt.shape[-2], w=x_dwt.shape[-1],
        )

        # IDynamic filter
        if self.use_idynamic:
            out = self.filter(x_main=out, x=x_dwt)
        else:
            out = self.filter(out) + out

        # IDWT → project back
        out = self.idwt(out)                 # (B, C//4, H, W)
        out = self.project_out(out)          # (B, C, H, W)
        return out


# ====================== Basic building blocks ======================

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super().__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_features, hidden_features, kernel_size=kernel_size,
                stride=1, padding=(kernel_size - 1) // 2, groups=hidden_features,
            ),
            nn.GELU(),
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# ====================== WindowAttention ======================

class WindowAttention(nn.Module):
    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        if dim > 100:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale

        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias
            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        if pfa_values[shift] is not None:
            if not self.training:
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn

        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(
                b_, self.num_heads, n, c // self.num_heads) + v_lepe
            ).transpose(1, 2).reshape(b_, n, c)

        if not self.training:
            del q, k, v, relative_position_bias
            torch.cuda.empty_cache()

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        flops = 0
        if self.layer_id < 2:
            flops += self.num_heads * n * (self.dim // self.num_heads) * n
            flops += self.num_heads * n * n * (self.dim // self.num_heads)
        else:
            flops += self.num_heads * n * (self.dim // self.num_heads) * self.num_topk[self.layer_id - 2]
            flops += self.num_heads * n * self.num_topk[self.layer_id] * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops


# ====================== SpatialFrequencyLayer ======================

class SFMLayer(nn.Module):
    """
    v2 三段式設計:
      Stage 1: LayerNorm → DFE → wqkv → PFA Window Attention → + residual_1
      Stage 2 (optional): LayerNorm → WMA_Full → + residual_2
      Stage 3: LayerNorm → ConvFFN → + residual_3

    use_wma=True 時才啟用 Stage 2（只在每個 SFMB 的最後一層）
    """
    def __init__(self, dim, block_id, layer_id, input_resolution, num_heads, num_topk,
                 window_size, shift_size, convffn_kernel_size, mlp_ratio,
                 qkv_bias=True, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_wma=False):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.use_wma = use_wma

        # ─── Stage 1: DFE + PFA Attention ───
        self.norm1 = norm_layer(dim)
        self.dfe = DFE(dim, dim)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            dim, layer_id=layer_id, window_size=to_2tuple(window_size),
            num_heads=num_heads, num_topk=num_topk, qkv_bias=qkv_bias,
        )

        # ─── Stage 2: WMA (only when use_wma=True) ───
        if self.use_wma:
            self.norm_wma = norm_layer(dim)
            self.wma = WMA_Full(dim)

        # ─── Stage 3: FFN ───
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_features=dim, hidden_features=mlp_hidden_dim,
            kernel_size=convffn_kernel_size, act_layer=act_layer,
        )

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape

        # ═══ Stage 1: DFE → PFA Attention ═══
        shortcut = x
        x_norm = self.norm1(x)

        # DFE 空間細節提取
        x_dfe = self.dfe(x_norm, x_size)

        # QKV projection
        x_qkv = self.wqkv(x_dfe)

        # LePE
        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        # Window partition + PFA attention
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(b, h, w, 4 * c),
                shifts=(-self.shift_size, -self.shift_size), dims=(1, 2),
            )
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, 4 * c)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, 4 * c)

        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices,
            rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift,
        )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x = shortcut + attn_x.view(b, n, c)       # ← residual 1

        # ═══ Stage 2: WMA (only at SFMB last layer) ═══
        if self.use_wma:
            shortcut2 = x
            x_wma_in = self.norm_wma(x)
            # seq → 4D for WMA
            x_4d = x_wma_in.transpose(1, 2).contiguous().view(b, c, h, w)
            x_wma_out = self.wma(x_4d)                                        # (B, C, H, W)
            x_wma_seq = x_wma_out.view(b, c, n).transpose(1, 2).contiguous()  # (B, L, C)
            x = shortcut2 + x_wma_seq              # ← residual 2

        # ═══ Stage 3: FFN ═══
        x = x + self.convffn(self.norm2(x), x_size)  # ← residual 3

        return x, [pfa_values, pfa_indices]

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # DFE
        flops += self.dfe.flops(h * w)
        # wqkv
        flops += self.dim * 3 * self.dim * h * w
        # Window Attention
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        # WMA (if present)
        if self.use_wma:
            N = h * w
            M = N // 4
            C = self.dim
            c_sub = C // 4
            # reduce
            flops += N * C * c_sub
            # qkv on full dim at half resolution
            flops += M * C * C * 3
            # qkv_dwconv
            flops += M * C * 3 * 9
            # channel attention Q@K^T + attn@V
            c_per_head = C // self.wma.num_heads if hasattr(self, 'wma') else C // 4
            flops += self.wma.num_heads * c_per_head * M * c_per_head * 2
            # IDynamic
            red = 8
            idyn_mid = C // red
            idyn_groups = C // 2
            flops += M * (C * idyn_mid + idyn_mid * 49 + idyn_mid * 49 * idyn_groups + C * 49)
            # project_out
            flops += N * c_sub * C
        # ConvFFN
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio
        # LePE
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)

        return flops


# ====================== BasicBlock ======================

class BasicBlock(nn.Module):
    """
    只有每個 SFMB 的最後一層啟用 WMA
    """
    def __init__(self, dim, input_resolution, idx, layer_id, depth,
                 num_heads, num_topk, window_size, convffn_kernel_size,
                 mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            # ★ 只有最後一層啟用 WMA
            is_last_layer = (i == depth - 1)
            self.layers.append(
                SFMLayer(
                    dim=dim,
                    block_id=idx,
                    layer_id=layer_id + i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    use_wma=is_last_layer,     # ← 關鍵改動
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, x_size, params):
        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


# ====================== Remaining classes ======================

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w
        assert h % 2 == 0 and w % 2 == 0
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(b, -1, 4 * c)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class SFMB(nn.Module):
    def __init__(self, dim, idx, layer_id, input_resolution, depth,
                 num_heads, num_topk, window_size, convffn_kernel_size, mlp_ratio,
                 qkv_bias=True, norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim, input_resolution=input_resolution, idx=idx, layer_id=layer_id,
            depth=depth, num_heads=num_heads, num_topk=num_topk, window_size=window_size,
            convffn_kernel_size=convffn_kernel_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

    def forward(self, x, pfa_list, x_size, params):
        x_block, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_block, x_size))) + x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        h, w = self.img_size if input_resolution is None else input_resolution
        return h * w * self.embed_dim if self.norm is not None else 0


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])

    def flops(self, input_resolution=None):
        return 0


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super().__init__(*m)

    def flops(self, input_resolution):
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            return self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            return self.num_feat * 9 * self.num_feat * 9 * x * y


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = [nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1), nn.PixelShuffle(scale)]
        super().__init__(*m)

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        conv = self[0]
        return h * w * conv.in_channels * conv.out_channels * (conv.kernel_size[0] ** 2)


# ====================== SFM Main Model ======================

@ARCH_REGISTRY.register()
class SFMformer(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=90,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 num_topk=[256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64,
                           32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16],
                 window_size=8, convffn_kernel_size=5, mlp_ratio=2.,
                 qkv_bias=True, norm_layer=nn.LayerNorm, ape=False,
                 patch_norm=True, use_checkpoint=False, upscale=2,
                 img_range=1., upsampler='', resi_connection='1conv', **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.layer_id = 0
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SFMB(
                dim=embed_dim, idx=i_layer, layer_id=self.layer_id,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads, num_topk=num_topk,
                window_size=window_size, convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer,
                downsample=None, use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size, resi_connection=resi_connection,
            )
            self.layers.append(layer)
            self.layer_id = self.layer_id + depths[i_layer]

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1),
            )

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        return relative_coords.sum(-1)

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops(resolution)
        return flops


if __name__ == '__main__':
    upscale = 3
    model = SFMformer(
        upscale=upscale,
        img_size=64,
        embed_dim=52,
        depths=[2, 4, 6, 6, 6],
        num_heads=4,
        num_topk=[1024, 1024,
                  256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64,
                  32, 32, 32, 32, 32, 32],
        window_size=32,
        convffn_kernel_size=7,
        img_range=1.,
        mlp_ratio=1,
        upsampler='pixelshuffledirect',
        resi_connection='1conv',
    )

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))

    # Print which layers have WMA
    for i, sfmb in enumerate(model.layers):
        for j, layer in enumerate(sfmb.residual_group.layers):
            wma_tag = " ★ WMA" if layer.use_wma else ""
            print(f"  SFMB[{i}].layer[{j}] (global_id={layer.layer_id}){wma_tag}")