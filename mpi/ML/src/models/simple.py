"""Простые baseline-модели для MPI: U-Net CNN, MoDL, Diffusion.

Не привязаны к конкретной статье — служат «контролем» для оценки того,
сколько даёт перенос конкретной идеи (физическая модель PMCNet,
dual-branch FDS-MPI, DEQ-итерации и т.д.) по сравнению с обычной
прямой сетью.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Базовая U-Net CNN: измерение → концентрация (image → image)
# ---------------------------------------------------------------------------


class MPIReconstructionCNN(nn.Module):
    """Простая U-Net CNN для прямого отображения измерений в изображение.

    Принимает на вход 4-канальное «изображение» из вектора измерений,
    переразложенного в 2D, и возвращает реконструированную карту
    концентрации 51×51 (через билинейное приведение к нужному размеру
    на выходе). Используется как сравнительная база.
    """

    def __init__(self, input_channels: int = 4, output_channels: int = 1,
                 base_filters: int = 32,
                 output_size: Tuple[int, int] = (51, 51)):
        super().__init__()
        self.output_size = tuple(output_size)

        self.enc1 = self._double_conv(input_channels, base_filters)
        self.enc2 = self._double_conv(base_filters, base_filters * 2)
        self.enc3 = self._double_conv(base_filters * 2, base_filters * 4)
        self.pool = nn.MaxPool2d(2)

        self.bridge = self._double_conv(base_filters * 4, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4,
                                      kernel_size=2, stride=2)
        self.dec3 = self._double_conv(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = self._double_conv(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters,
                                      kernel_size=2, stride=2)
        self.dec1 = self._double_conv(base_filters * 2, base_filters)

        self.head = nn.Sequential(
            nn.Conv2d(base_filters, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bridge(self.pool(e3))

        d3 = self.dec3(torch.cat([
            F.interpolate(self.up3(b), size=e3.shape[-2:],
                          mode='bilinear', align_corners=False), e3
        ], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(self.up2(d3), size=e2.shape[-2:],
                          mode='bilinear', align_corners=False), e2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(self.up1(d2), size=e1.shape[-2:],
                          mode='bilinear', align_corners=False), e1
        ], dim=1))

        out = self.head(d1)
        if out.shape[-2:] != self.output_size:
            out = F.interpolate(out, size=self.output_size,
                                mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# MoDL: model-based unrolled deep learning
# ---------------------------------------------------------------------------


class MoDLNetwork(nn.Module):
    """Упрощённый MoDL (Aggarwal et al., 2019):

        x_{k+1} = (Aᵀ·A + λ·I)⁻¹ (Aᵀ·y + λ·D_θ(x_k))

    где D_θ — обучаемый CNN-денойзер. K итераций разворачиваются в одном
    forward, веса D_θ разделяются между итерациями. Используется как
    base для сравнения «классическое разворачивание + обучаемый
    регуляризатор».
    """

    def __init__(self, system_matrix, image_shape,
                 n_iterations: int = 3, lambda_param: float = 0.01,
                 base_filters: int = 32):
        super().__init__()
        if np.iscomplexobj(system_matrix):
            S_ext = np.vstack([system_matrix.real, system_matrix.imag])
        else:
            S_ext = system_matrix
        self.M, self.N = S_ext.shape
        self.image_shape = tuple(image_shape)
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param

        self.register_buffer('A',
                             torch.tensor(S_ext, dtype=torch.float32))
        # Предвычисляем (AᵀA + λ I)⁻¹ один раз
        AtA = self.A.T @ self.A
        Inv = torch.inverse(AtA + lambda_param * torch.eye(self.N))
        self.register_buffer('LS_inverse', Inv)
        self.register_buffer('A_T', self.A.T.contiguous())

        # Обучаемый денойзер: shared между итерациями
        self.denoiser = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, 1, kernel_size=3, padding=1),
        )

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """measurements: (B, M) (вещественные, расширенная форма)."""
        B = measurements.shape[0]
        Aty = measurements @ self.A          # (B, N)
        x = Aty @ self.LS_inverse.T          # начальное приближение

        for _ in range(self.n_iterations):
            x_img = x.view(B, 1, *self.image_shape)
            d = self.denoiser(x_img).view(B, -1)
            rhs = Aty + self.lambda_param * d
            x = rhs @ self.LS_inverse.T

        return torch.sigmoid(x.view(B, 1, *self.image_shape))


# ---------------------------------------------------------------------------
# Diffusion baseline (упрощённый DDPM)
# ---------------------------------------------------------------------------


class _DiffusionUNet(nn.Module):
    """Маленький U-Net для denoise-шага DDPM с time embedding.

    Args:
        in_channels:  размерность входа = 1 (noisy image) + n_condition_channels
                      (условие, обычно Tikhonov-реконструкция).
        base, time_dim: ёмкость + размерность time-embedding.
    """

    def __init__(self, in_channels: int, base: int = 64,
                 time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.enc1 = self._block(in_channels, base)
        self.enc2 = self._block(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.bridge = self._block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2,
                                      kernel_size=2, stride=2)
        self.dec2 = self._block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base,
                                      kernel_size=2, stride=2)
        self.dec1 = self._block(base * 2, base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

        # Time projection
        self.t_proj1 = nn.Linear(time_dim, base)
        self.t_proj2 = nn.Linear(time_dim, base * 2)
        self.t_projB = nn.Linear(time_dim, base * 4)

    @staticmethod
    def _block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) *
            torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        e1 = self.enc1(x) + self.t_proj1(t_emb)[:, :, None, None]
        e2 = self.enc2(self.pool(e1)) + self.t_proj2(t_emb)[:, :, None, None]
        b = self.bridge(self.pool(e2)) + self.t_projB(t_emb)[:, :, None, None]

        d2 = self.dec2(torch.cat([
            F.interpolate(self.up2(b), size=e2.shape[-2:],
                          mode='bilinear', align_corners=False), e2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(self.up1(d2), size=e1.shape[-2:],
                          mode='bilinear', align_corners=False), e1
        ], dim=1))
        return self.head(d1)


class DiffusionModel(nn.Module):
    """Conditional DDPM для MPI: денойзинг GT-концентрации с условием.

    Условие (`condition`) — это, как правило, грубая реконструкция из
    Tikhonov или Kaczmarz, передаётся U-Net'у конкатенацией с зашумлённым
    x_t. Это превращает безусловный baseline в задачно-ориентированную
    модель «улучши грубую реконструкцию до GT»; loss = MSE между
    предсказанным шумом и реальным.

    Если `condition=None`, модель работает в безусловном режиме.

    Args:
        cond_channels: число каналов condition (1 для Tikhonov-recon).
    """

    def __init__(self, n_steps: int = 100,
                 image_size: int = 51, base_filters: int = 64,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 cond_channels: int = 1):
        super().__init__()
        self.n_steps = n_steps
        self.image_size = image_size
        self.cond_channels = cond_channels

        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))

        # Вход U-Net: noisy (1 канал) + condition (cond_channels)
        self.unet = _DiffusionUNet(in_channels=1 + cond_channels,
                                   base=base_filters)

    def _concat_cond(self, x_t: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        if condition is None:
            zeros = torch.zeros(x_t.shape[0], self.cond_channels,
                                *x_t.shape[-2:], device=x_t.device)
            return torch.cat([x_t, zeros], dim=1)
        if condition.shape[-2:] != x_t.shape[-2:]:
            condition = F.interpolate(condition, size=x_t.shape[-2:],
                                       mode='bilinear', align_corners=False)
        return torch.cat([x_t, condition], dim=1)

    def q_sample(self, x_start: torch.Tensor,
                 t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqa = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqom = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqa * x_start + sqom * noise

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor,
                 condition: torch.Tensor = None) -> torch.Tensor:
        """Один обратный шаг диффузии (с conditioning)."""
        inp = self._concat_cond(x_t, condition)
        eps_pred = self.unet(inp, t)
        alpha = self.alphas[t][:, None, None, None]
        alpha_bar = self.alphas_cumprod[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]
        coef = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
        mean = (1.0 / torch.sqrt(alpha)) * (x_t - coef * eps_pred)
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(beta) * noise
        return mean

    def forward(self, x_start: torch.Tensor,
                t: torch.Tensor = None,
                condition: torch.Tensor = None) -> torch.Tensor:
        """Тренировочный forward: возвращает MSE между шумом и предсказанием.

        Args:
            x_start:   (B, 1, H, W) ground-truth концентрация.
            t:         (B,) индексы шагов диффузии; если None — случайные.
            condition: (B, cond_channels, H, W) грубая реконструкция
                       (Tikhonov или Kaczmarz). None = безусловный режим.
        """
        if t is None:
            t = torch.randint(0, self.n_steps, (x_start.shape[0],),
                              device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        inp = self._concat_cond(x_noisy, condition)
        eps_pred = self.unet(inp, t)
        return F.mse_loss(eps_pred, noise)

    def sample(self, condition: torch.Tensor,
               n_steps: int = None) -> torch.Tensor:
        """Условное сэмплирование: condition → x_0 (B, 1, H, W)."""
        n_steps = n_steps or self.n_steps
        device = condition.device
        B = condition.shape[0]
        H, W = condition.shape[-2:]
        x = torch.randn(B, 1, H, W, device=device)
        for step in reversed(range(min(n_steps, self.n_steps))):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition)
        return x


__all__ = ['MPIReconstructionCNN', 'MoDLNetwork', 'DiffusionModel']
