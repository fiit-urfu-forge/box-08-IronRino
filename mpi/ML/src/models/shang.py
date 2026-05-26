"""FDS-MPI: Fusion Dual-Sampling CNN (Shang et al., 2022).

Источник: Shang Y. et al. «Deep learning for improving the spatial
resolution of magnetic particle imaging», Physics in Medicine & Biology
65(15):155012, 2022.

Согласно статье:
  • Задача — обучение отображения f: I_LR → I_HR, где LR = слабый
    градиент (3 Тл/м), HR = сильный (6 Тл/м). Постпроцессинг поверх
    X-space реконструкции.
  • Архитектура — две параллельные подсети (branch A и branch B), выходы
    которых КОНКАТЕНИРУЮТСЯ и сворачиваются в финальное изображение:
      – branch A: автоэнкодер с residual connection, ДВЕ операции
        max-pool для подавления артефактов и снижения вычислений;
      – branch B: автоэнкодер с residual, БЕЗ пулинга — сохраняет
        мелкие детали.
  • Свёртки 5×5, padding=1, stride=1 (по статье); 64 фильтра во всех
    промежуточных слоях, ReLU активация. Loss — MSE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(c_in: int, c_out: int, kernel: int = 5):
    """Свёртка + BatchNorm + ReLU — стандартный блок FDS-MPI из статьи.

    Статья Shang 2022 (Fig. 1) явно показывает BatchNorm после каждой
    свёртки. Без BN на малых батчах (как у нас, batch=16) обучение
    становится нестабильным, поэтому BN критичен для воспроизводимости
    качества из статьи (SSIM 0.94, PSNR 28 dB на MNIST-фантомах).
    """
    p = kernel // 2
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, padding=p, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class _BranchA(nn.Module):
    """Подсеть с пулингом (artifact-suppression branch).

    Encoder с двумя MaxPool ×2 → bottleneck → decoder с двумя
    ConvTranspose ×2. Residual connection от входа на выход (1×1 conv).
    Все conv-слои с BN+ReLU.
    """

    def __init__(self, in_channels: int = 1, base: int = 64,
                 kernel: int = 5):
        super().__init__()
        self.enc1 = _conv_bn_relu(in_channels, base, kernel)
        self.enc2 = _conv_bn_relu(base, base, kernel)
        self.enc3 = _conv_bn_relu(base, base, kernel)
        self.pool = nn.MaxPool2d(2)

        # ConvTranspose с BN+ReLU
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base, base, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base, base, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.deconv = _conv_bn_relu(base, base, kernel)

        self.skip = nn.Conv2d(in_channels, base, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        residual = self.skip(x)
        h = self.pool(self.enc1(x))            # H/2
        h = self.pool(self.enc2(h))            # H/4
        h = self.enc3(h)
        h = self.up1(h)                        # H/2
        h = self.up2(h)                        # H
        if h.shape[-2:] != orig_size:
            h = F.interpolate(h, size=orig_size,
                              mode='bilinear', align_corners=False)
        h = self.deconv(h)
        return h + residual


class _BranchB(nn.Module):
    """Подсеть без пулинга (detail-preserving branch).

    Чисто свёрточная цепочка с residual connection. Все conv-слои с
    BatchNorm+ReLU (см. статью Fig. 1).
    """

    def __init__(self, in_channels: int = 1, base: int = 64,
                 kernel: int = 5):
        super().__init__()
        self.conv1 = _conv_bn_relu(in_channels, base, kernel)
        self.conv2 = _conv_bn_relu(base, base, kernel)
        self.conv3 = _conv_bn_relu(base, base, kernel)
        self.conv4 = _conv_bn_relu(base, base, kernel)
        self.skip = nn.Conv2d(in_channels, base, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        return h + residual


class ShangCNN(nn.Module):
    """FDS-MPI: Fusion Dual-Sampling network (Shang et al., 2022).

    Конкатенирует выходы двух подсетей и пропускает через финальный
    fusion-блок (2×base каналов → 1). Loss — MSE (см. trainer).
    """

    def __init__(self, input_channels: int = 1, output_channels: int = 1,
                 base_filters: int = 64, kernel: int = 5):
        super().__init__()
        self.branch_a = _BranchA(input_channels, base_filters, kernel)
        self.branch_b = _BranchB(input_channels, base_filters, kernel)
        p = kernel // 2
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * base_filters, base_filters, kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, output_channels, kernel, padding=p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        return self.fusion(torch.cat([a, b], dim=1))


# Имена, ожидаемые остальной кодовой базой
FDSMPI = ShangCNN

__all__ = ['ShangCNN', 'FDSMPI']
