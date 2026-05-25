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


class _BranchA(nn.Module):
    """Подсеть с пулингом (artifact-suppression branch).

    Encoder с двумя MaxPool ×2 → bottleneck → decoder с двумя
    ConvTranspose ×2. Residual connection от входа на выход (1×1 conv).
    """

    def __init__(self, in_channels: int = 1, base: int = 64,
                 kernel: int = 5):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv2d(in_channels, base, kernel, padding=p)
        self.conv2 = nn.Conv2d(base, base, kernel, padding=p)
        self.conv3 = nn.Conv2d(base, base, kernel, padding=p)
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.deconv = nn.Conv2d(base, base, kernel, padding=p)

        # Residual: 1×1 для приведения каналов входа к base
        self.skip = nn.Conv2d(in_channels, base, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        residual = self.skip(x)

        h = self.act(self.conv1(x))
        h = self.pool(h)                       # H/2
        h = self.act(self.conv2(h))
        h = self.pool(h)                       # H/4
        h = self.act(self.conv3(h))
        h = self.act(self.up1(h))              # H/2
        h = self.act(self.up2(h))              # H

        if h.shape[-2:] != orig_size:
            h = F.interpolate(h, size=orig_size,
                              mode='bilinear', align_corners=False)
        h = self.act(self.deconv(h))
        return h + residual


class _BranchB(nn.Module):
    """Подсеть без пулинга (detail-preserving branch).

    Чисто свёрточная цепочка с residual connection. Никакого
    пространственного сжатия — мелкие структуры сохраняются.
    """

    def __init__(self, in_channels: int = 1, base: int = 64,
                 kernel: int = 5):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv2d(in_channels, base, kernel, padding=p)
        self.conv2 = nn.Conv2d(base, base, kernel, padding=p)
        self.conv3 = nn.Conv2d(base, base, kernel, padding=p)
        self.conv4 = nn.Conv2d(base, base, kernel, padding=p)
        self.skip = nn.Conv2d(in_channels, base, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.act(self.conv4(h))
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
