"""Mixture of Experts (MoE) для MPI: комбинирование разнотипных моделей.

Идея: уже обученные модели реконструкции (классические, CNN, физико-
ограниченные) имеют разные сильные стороны — Tikhonov хорошо подавляет
точечные артефакты, PMCNet точнее на краях, FDS-MPI лучше с мелкими
деталями. MoE учит комбинировать их выходы так, чтобы итоговая
реконструкция была лучше любого индивидуального эксперта.

Три режима комбинирования (`mode` в `MoEReconstructor`):

  • 'mean'    — простое усреднение, без параметров. Baseline.
  • 'scalar'  — один обучаемый вес α_k на эксперта (softmax по k),
                итог = Σ_k α_k · expert_k. Глобальный блендинг.
  • 'spatial' — per-pixel gating: маленькая CNN смотрит на стек выходов
                всех экспертов и выдаёт K весов в каждом пикселе
                (softmax по k). Самый выразительный режим — позволяет
                одному эксперту «доминировать» в одной области, другому
                — в другой.

Эксперты передаются как `{name: callable}`, где callable принимает
измерение и возвращает 2D-numpy-массив той же формы, что image_shape.
Веса экспертов замораживаются — обучается только gating-механизм MoE.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class _SpatialGatingNet(nn.Module):
    """Маленькая CNN, выдающая per-pixel вероятности по экспертам.

    Вход:  (B, K, H, W) — стек выходов K экспертов.
    Выход: (B, K, H, W) — softmax-нормализованные веса (по оси K).
    """

    def __init__(self, n_experts: int, hidden: int = 32):
        super().__init__()
        self.n_experts = n_experts
        self.net = nn.Sequential(
            nn.Conv2d(n_experts, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_experts, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=1)


class MoEReconstructor(nn.Module):
    """Mixture of Experts для MPI-реконструкции.

    Эксперты — внешние callable'ы (например, `comparator.tikhonov_reconstruction`,
    `comparator.cnn_reconstruction` и т.д.). MoE не дублирует и не
    переобучает их, а учит лишь gating-механизм поверх их выходов.
    """

    def __init__(self, experts: Dict[str, Callable],
                 image_shape: Tuple[int, int],
                 mode: str = 'spatial',
                 device: str = 'cpu'):
        super().__init__()
        if not experts:
            raise ValueError("Передайте хотя бы одного эксперта")
        if mode not in ('mean', 'scalar', 'spatial'):
            raise ValueError(f"Неизвестный mode={mode!r}; ожидается "
                             "'mean' | 'scalar' | 'spatial'")

        self.experts = experts
        self.expert_names = list(experts.keys())
        self.n_experts = len(experts)
        self.image_shape = tuple(image_shape)
        self.mode = mode
        self.device = torch.device(device)

        # Параметры комбинирования
        if mode == 'scalar':
            # Логиты α_k → softmax → веса
            self.scalar_logits = nn.Parameter(torch.zeros(self.n_experts))
            self.gating = None
        elif mode == 'spatial':
            self.gating = _SpatialGatingNet(self.n_experts)
            self.scalar_logits = None
        else:  # 'mean'
            self.gating = None
            self.scalar_logits = None

        self.to(self.device)

    # ---- запуск экспертов ----------------------------------------------------

    def run_experts(self, measurement) -> torch.Tensor:
        """Запустить всех экспертов на одном измерении.

        Returns: (K, H, W) — выходы экспертов, нормированные на [0, 1]
        каждый (как делает comparator).
        """
        recons = []
        for name in self.expert_names:
            img = self.experts[name](measurement)
            if img is None:
                # Если эксперт упал — заменяем нулями (пусть gating
                # сам отнормирует)
                img = np.zeros(self.image_shape, dtype=np.float32)
            recons.append(torch.tensor(img, dtype=torch.float32))
        return torch.stack(recons, dim=0)  # (K, H, W)

    # ---- комбинирование ------------------------------------------------------

    def _combine(self, expert_stack: torch.Tensor) -> torch.Tensor:
        """expert_stack: (B, K, H, W) → (B, 1, H, W)."""
        if self.mode == 'mean':
            return expert_stack.mean(dim=1, keepdim=True)
        if self.mode == 'scalar':
            w = F.softmax(self.scalar_logits, dim=0).view(1, -1, 1, 1)
            return (expert_stack * w).sum(dim=1, keepdim=True)
        # spatial
        gates = self.gating(expert_stack)                       # (B, K, H, W)
        return (expert_stack * gates).sum(dim=1, keepdim=True)

    def forward(self, expert_stack: torch.Tensor) -> torch.Tensor:
        """Прямой проход на ПРЕДВЫЧИСЛЕННЫХ выходах экспертов.

        Используется в обучении gating: эксперты прогоняются один раз
        заранее, а затем gating тренируется на их выходах без
        повторного вызова экспертов.

        Args:
            expert_stack: (B, K, H, W) — выходы экспертов.
        Returns:
            (B, 1, H, W) — скомбинированная реконструкция.
        """
        return self._combine(expert_stack)

    def reconstruct(self, measurement) -> np.ndarray:
        """Полная реконструкция: эксперты + комбинирование.

        Args:
            measurement: измерение в формате, понятном экспертам
                         (обычно (2, M_per_coil) complex).
        Returns:
            (H, W) numpy-массив, нормированный на [0, 1].
        """
        stack = self.run_experts(measurement).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            out = self._combine(stack)[0, 0]
        recon = out.cpu().numpy()
        if recon.max() > 0:
            recon = recon / recon.max()
        return recon

    # ---- обучение gating -----------------------------------------------------

    def precompute_expert_recons(self,
                                  measurements: List,
                                  show_progress: bool = True
                                  ) -> torch.Tensor:
        """Прогон всех экспертов на батче измерений → тензор для обучения.

        Args:
            measurements: список из N измерений (формат под экспертов).
        Returns:
            (N, K, H, W) float32 тензор.
        """
        N = len(measurements)
        H, W = self.image_shape
        out = torch.zeros(N, self.n_experts, H, W, dtype=torch.float32)
        it = range(N)
        if show_progress:
            it = tqdm(it, desc='MoE: precomputing expert recons')
        for i in it:
            out[i] = self.run_experts(measurements[i])
        return out

    def train_gating(self,
                     expert_recons: torch.Tensor,
                     targets: torch.Tensor,
                     epochs: int = 30,
                     lr: float = 1e-3,
                     batch_size: int = 16,
                     verbose: bool = True) -> List[float]:
        """Обучить gating на предвычисленных выходах экспертов.

        Args:
            expert_recons: (N, K, H, W) — предсказания экспертов.
            targets:       (N, 1, H, W) или (N, H, W) — ground truth.
            epochs:        число эпох обучения gating.
            lr:            learning rate (Adam).
        Returns:
            История loss по эпохам.
        """
        if self.mode == 'mean':
            if verbose:
                print("  MoE mode='mean' — обучения нет (фиксированное усреднение)")
            return []

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        expert_recons = expert_recons.to(self.device)
        targets = targets.to(self.device)

        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(expert_recons, targets)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        crit = nn.MSELoss()

        history = []
        for ep in range(epochs):
            self.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = self._combine(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / len(loader)
            history.append(avg)
            if verbose and (ep % max(1, epochs // 10) == 0
                            or ep == epochs - 1):
                weights = self.current_weights()
                msg = f'  MoE ep {ep+1:3d}/{epochs}: loss={avg:.4e}'
                if weights is not None:
                    pretty = ', '.join(
                        f'{n}={w:.2f}'
                        for n, w in zip(self.expert_names, weights)
                    )
                    msg += f'  weights[{pretty}]'
                print(msg)
        return history

    def current_weights(self) -> Optional[np.ndarray]:
        """Текущие глобальные веса (для 'scalar' — softmax от логитов;
        для 'spatial' — None, веса per-pixel)."""
        if self.mode == 'scalar':
            return F.softmax(self.scalar_logits, dim=0).detach().cpu().numpy()
        return None


__all__ = ['MoEReconstructor']
