"""FDS-MPI — Fusion Dual-Sampling CNN для super-resolution MPI-изображений.

Источник: Shang Y., Liu J., Zhang L., Wu X., Zhang P., Yin L., Hui H.,
Tian J. «Deep learning for improving the spatial resolution of magnetic
particle imaging», Phys. Med. Biol. 67 (2022) 125012,
https://doi.org/10.1088/1361-6560/ac6e24

## Постановка задачи

FDS-MPI — это **пост-процессинг** уже реконструированных MPI-изображений
(не алгоритм реконструкции). Сеть обучается отображать изображение,
полученное на сканере с **низким градиентным полем** (low-resolution,
LR), в изображение, эквивалентное сканированию с **высоким градиентом**
(high-resolution, HR):

    f* = argmin_f ‖ f(I_LR) − I_HR ‖²₂                    (paper Eq. 2)

Физическая мотивация: FWHM пространственного разрешения MPI ~ 1/G
(paper Eq. 1), поэтому удвоение G даёт удвоение разрешения. Но
высокое G требует мощного оборудования и снижает SNR. FDS-MPI делает
то же программно, без аппаратных затрат.

## Архитектура (paper Sec. 2.2, Fig. 1)

«Fusion Dual-Sampling» — это **две параллельные ветки**, фичи которых
объединяются в конце. Концепт: ветки извлекают **комплементарные**
признаки (одна — с pooling, другая — без), и их fusion компенсирует
ограничения каждой по отдельности.

  Input I_LR (B, 1, H, W)
    │
    ├─── Branch A (FEN_A → FDN_A) — С pooling layers
    │       • удаляет MPI-артефакты;
    │       • сокращает compute через max-pool;
    │       • residual connection вход→выход (skip).
    │
    ├─── Branch B (FEN_B → FDN_B) — БЕЗ pooling layers
    │       • сохраняет структурные детали;
    │       • residual connection вход→выход (skip).
    │
    └─── Fusion: concat(out_A, out_B) → 1×1 conv → I_HR (B, 1, H, W)

### Branch A — с pooling (artefact removal)

  • FEN_A: 4 conv-блока (5×5, 64 filters, ReLU) с 2 max-pool слоями
           между ними (downsample ×4 в простр. размере);
  • FDN_A: симметричные 4 deconv-блока (5×5, 64 filters, ReLU)
           с upsampling обратно;
  • Skip connection: I_LR добавляется к выходу FDN_A (residual learning).

### Branch B — без pooling (detail preservation)

  • FEN_B: 4 conv-блока (5×5, 64 filters, ReLU) без pooling;
  • FDN_B: 4 deconv-блока (5×5, 64 filters, ReLU);
  • Skip connection: I_LR добавляется к выходу FDN_B.

### Fusion (paper Sec. 2.2, последний абзац)

  • Конкатенация выходов веток по каналу: (B, 2, H, W);
  • Два conv-слоя 5×5: (B, 64, H, W) → (B, 1, H, W);
  • Финальная активация — линейная (выход интерпретируется как
    нормированная HR-MPI-картинка).

## Гиперпараметры (paper Sec. 3.2)

  • Размер ядер: 5×5 ВО ВСЕХ слоях;
  • Число фильтров: 64 (последний слой — 1);
  • Stride: 1, padding: 2 (для same-size при kernel 5);
  • Инициализация: Gaussian (paper не специфицирует σ — стандартно
    Kaiming-normal для ReLU-сетей);
  • Loss: MSE по формуле paper Eq. 3
    L(F) = (1/M·N) · Σ_ij (F(I_LR)_ij − I_HR_ij)²;
  • Optimizer: ADAM (typical default для PyTorch);
  • Learning rate: 1e-5 базовый, decay factor 0.5;
  • Batch size: 512;
  • Epochs: 10 000;
  • Тренируется на 8 000 patches 64×64, randomly cropped из MNIST-like
    100×100 симулированных MPI-изображений (paper Sec. 3.1.1).

## Сравнение с альтернативами (paper Sec. 3.3)

Paper сравнивает FDS-MPI с одно-веточными аналогами:
  • UCNN, CNN — только conv-слои (без residual learning);
  • REDUCNN, REDCNN — encoder-decoder + RL.
На 100×100 simulated MPI:
  • FDS-MPI: PSNR=28.11, SSIM=0.94 (paper Table 3) — лучший;
  • REDCNN: PSNR=26.42, SSIM=0.92;
  • CNN: PSNR=25.42, SSIM=0.88.
Дуальная архитектура даёт +1.7 dB PSNR и +0.02 SSIM поверх
лучшего single-network варианта.

## Ограничения метода (paper Sec. 4)

  • Это post-processing, не reconstruction — нужна уже готовая LR-картинка;
  • Только 2D (тренировался на 2D-срезах MNIST), 3D-вариант = будущая
    работа;
  • Обучается на симуляциях — на реальных данных нужен fine-tuning
    или transfer learning (paper Sec. 4).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helper-блоки
# =============================================================================


def _conv_block(c_in: int, c_out: int, kernel_size: int = 5) -> nn.Sequential:
    """Conv 5×5 + ReLU. Padding подбирается для same-size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                   stride=1, padding=kernel_size // 2),
        nn.ReLU(inplace=True),
    )


def _deconv_block(c_in: int, c_out: int, kernel_size: int = 5) -> nn.Sequential:
    """ConvTranspose 5×5 + ReLU. Padding подбирается для same-size при stride=1.

    Paper использует stride=1 (Sec. 3.2: "The strides of convolution and
    deconvolution were 1 with 1 padding"). При stride=1 ConvTranspose
    математически эквивалентен Conv2d с тем же ядром и padding, но
    paper явно различает «convolutional» и «deconvolutional» слои
    архитектурно, как принято в encoder-decoder литературе
    (Noh 2015, Ronneberger 2015).
    """
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=kernel_size,
                            stride=1, padding=kernel_size // 2),
        nn.ReLU(inplace=True),
    )


# =============================================================================
# Branch A — feature encoder/decoder с pooling (artefact removal)
# =============================================================================


class _BranchA(nn.Module):
    """Ветка А: 4 conv + 2 pool → 4 deconv + 2 upsample.

    Paper Sec. 2.2: «branch A contains an auto-encoder architecture
    with a residual connection. Two PLs are added to extract
    representative information and simultaneously remove useless
    information; the MPI image has a large number of artefacts, and
    this system can remove unnecessary artefacts from the MPI image».

    Pool-слои размещены между парами conv-блоков (paper Fig. 1).
    Структура (одна возможная схема, согласованная с paper):

      conv → conv → pool → conv → conv → pool → bottleneck
        ↓                                         ↑
      ╰─ residual skip → конкатенация / сумма ── ╯
        ↓                                         ↑
      deconv → upsample → deconv → deconv → upsample → deconv

    Skip-connection (RL): I_LR + output (paper Sec. 2.2.3).
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64,
                 kernel_size: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv_block(in_channels, base_filters, kernel_size),
            _conv_block(base_filters, base_filters, kernel_size),
            nn.MaxPool2d(2),
            _conv_block(base_filters, base_filters, kernel_size),
            _conv_block(base_filters, base_filters, kernel_size),
            nn.MaxPool2d(2),
        )
        self.bottleneck = _conv_block(base_filters, base_filters, kernel_size)
        # ConvTranspose со stride=2 — upsample; чередуется с stride=1 deconv.
        self.up1 = nn.ConvTranspose2d(
            base_filters, base_filters, kernel_size=2, stride=2,
        )
        self.dec1 = _deconv_block(base_filters, base_filters, kernel_size)
        self.up2 = nn.ConvTranspose2d(
            base_filters, base_filters, kernel_size=2, stride=2,
        )
        self.dec2 = _deconv_block(base_filters, base_filters, kernel_size)
        # Финальный 1-канальный выход; ReLU не применяем — это вход в fusion.
        self.out_conv = nn.Conv2d(base_filters, in_channels,
                                    kernel_size=kernel_size,
                                    stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) → (B, in_channels, H, W)."""
        H, W = x.shape[-2:]
        e = self.encoder(x)
        b = self.bottleneck(e)
        d = self.up1(b)
        d = self.dec1(d)
        d = self.up2(d)
        d = self.dec2(d)
        out = self.out_conv(d)
        # Приведение к исходному размеру (если pooling даёт нестандартный shape)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                 mode='bilinear', align_corners=False)
        # Residual learning: I_LR + processed (paper Sec. 2.2.3)
        return out + x


# =============================================================================
# Branch B — feature encoder/decoder без pooling (detail preservation)
# =============================================================================


class _BranchB(nn.Module):
    """Ветка В: 4 conv → 4 deconv, БЕЗ pool (paper Sec. 2.2.1).

    «In contrast to branch A, branch B has no PL. Consequently, useful
    information from the MPI image is not lost in branch B and can
    reconstruct high-resolution MPI images».

    Без pooling-слоёв обработка идёт на исходном разрешении — это
    сохраняет тонкие пространственные детали, которые в ветке А
    подавляются вместе с артефактами.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64,
                 kernel_size: int = 5, n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4):
        super().__init__()
        enc_layers = [_conv_block(in_channels, base_filters, kernel_size)]
        for _ in range(n_encoder_layers - 1):
            enc_layers.append(
                _conv_block(base_filters, base_filters, kernel_size)
            )
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for _ in range(n_decoder_layers - 1):
            dec_layers.append(
                _deconv_block(base_filters, base_filters, kernel_size)
            )
        # Финальный 1-канальный выход без ReLU (вход в fusion)
        dec_layers.append(
            nn.ConvTranspose2d(base_filters, in_channels,
                                 kernel_size=kernel_size,
                                 stride=1, padding=kernel_size // 2)
        )
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) → (B, in_channels, H, W)."""
        feats = self.encoder(x)
        out = self.decoder(feats)
        # Residual learning: I_LR + processed (paper Sec. 2.2.3)
        return out + x


# =============================================================================
# Fusion DS-MPI — головная модель
# =============================================================================


class FDSMPI(nn.Module):
    """Fusion Dual-Sampling CNN для super-resolution MPI (Shang 2022).

    ## Forward

      x (B, 1, H, W)  — low-resolution MPI image
        ├─ Branch A (+ pooling) →  out_A  (B, 1, H, W)
        └─ Branch B (no pooling) → out_B  (B, 1, H, W)
                                    │
              concat([out_A, out_B], dim=1) → (B, 2, H, W)
                                    │
                          fusion conv layers → (B, 1, H, W)

    ## Применение

    Эта модель — НЕ алгоритм реконструкции, а **пост-процессор**:
    она получает уже реконструированную (Tikhonov / X-space / Kaczmarz)
    LR-картинку и улучшает её до HR-эквивалента. В пайплайн её можно
    поставить ПОСЛЕ любого классического реконструктора.

    Шаг типового использования:

        u_meas → Tikhonov.reconstruct() → I_LR (51×51, низкое разрешение)
                                              ↓
                                       FDSMPI.forward()
                                              ↓
                                I_HR (51×51, повышенное разрешение)

    Args:
        in_channels: число входных каналов изображения (default 1 для MPI).
        base_filters: число свёрточных каналов (paper Sec. 3.2: 64).
        kernel_size: размер ядра (paper: 5).

    Paper Sec. 3.2 hyperparameters использованы как defaults; их
    можно переопределить для экспериментов.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 64,
                 kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        self.branch_a = _BranchA(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
        )
        self.branch_b = _BranchB(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_size=kernel_size,
        )
        # Fusion: concat(out_A, out_B) → conv → conv → 1-канал.
        # Paper Sec. 2.2 (Fig. 1): «concatenate and convolutional layers
        # were fused into dual-channel networks to improve performance».
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * in_channels, base_filters,
                       kernel_size=kernel_size,
                       stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, in_channels,
                       kernel_size=kernel_size,
                       stride=1, padding=kernel_size // 2),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Gaussian-инициализация (paper Sec. 3.2).

        «The convolution and deconvolution layers were initialized using
        a Gaussian kernel». Для ReLU-сетей это эквивалентно
        Kaiming-normal с fan_in mode (де-факто стандарт).
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                          nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: low-resolution MPI image (B, in_channels, H, W).

        Returns: high-resolution-equivalent image (B, in_channels, H, W).
        """
        out_a = self.branch_a(x)
        out_b = self.branch_b(x)
        # Fusion: concatenation по каналу + conv layers
        cat = torch.cat([out_a, out_b], dim=1)        # (B, 2, H, W)
        return self.fusion(cat)


# =============================================================================
# MSE-loss как в paper (Eq. 3) — служебный wrapper для обучения
# =============================================================================


def fds_mpi_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE-loss FDS-MPI (paper Eq. 3).

      L(F) = (1 / (M·N)) · Σ_ij (F(I_LR)_ij − I_HR_ij)²

    Это просто mean squared error, обёрнут отдельно для symmetry с
    нотацией paper'а.

    Args:
        pred:   (B, 1, H, W) — F(I_LR), выход FDSMPI.
        target: (B, 1, H, W) — I_HR, ground truth с высоким градиентом.
    """
    return torch.nn.functional.mse_loss(pred, target, reduction='mean')


__all__ = ['FDSMPI', 'fds_mpi_loss']
