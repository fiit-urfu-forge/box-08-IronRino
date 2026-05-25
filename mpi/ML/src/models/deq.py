"""DEQ-MPI: Deep Equilibrium Model с обучаемой согласованностью данных.

Источник: Güngör A. et al. «DEQ-MPI: A Deep Equilibrium Reconstruction
With Learned Consistency for Magnetic Particle Imaging», IEEE TMI 43(5),
2024.

Согласно статье:
  • DEQ-MPI разворачивает ADMM в неявную форму: x* = h_θ(x*; y, A) —
    неподвижная точка обучаемого отображения.
  • Архитектура содержит ДВА обучаемых блока:
      – RDN block: residual dense network для регуляризации
        изображения. В статье: 4 residual-модуля, 128 каналов,
        12 conv-слоёв в каждом модуле, ReLU на выходе для
        неотрицательности. У нас уменьшено для скорости (channels=32,
        n_modules=2, n_convs=4 — параметризовано).
      – LC block: learned consistency. Принимает остаток A·x − y и
        измерения y; обрабатывает 1D-свёртками вдоль частотного
        измерения; результат проектируется на ε-окрестность y для
        предотвращения расходимости.
  • Forward — фиксированное число итераций (в статье ~25 с Anderson
    acceleration); здесь — обычные итерации до сходимости или
    `n_iterations`.

В этой реализации ADMM упрощён: вместо полноценного 3-update ADMM
используется чередование RDN-регуляризации и LC-проекции, которое
эквивалентно при ρ → ∞ и проще в обучении.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualDenseModule(nn.Module):
    """Residual Dense Block (RDB): n_convs параллельно растущих свёрток
    с dense-конкатенацией и global residual.

    Каждый conv видит конкатенацию всех предыдущих признаков и добавляет
    новый канал. В конце 1×1 conv сворачивает обратно к channels, и
    результат складывается с входом.
    """

    def __init__(self, channels: int, n_convs: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels * (i + 1), channels, kernel_size=3, padding=1)
            for i in range(n_convs)
        ])
        # Финальная свёртка: (n_convs+1)·channels → channels
        self.fusion = nn.Conv2d(channels * (n_convs + 1), channels,
                                kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for conv in self.convs:
            new_feat = F.relu(conv(torch.cat(feats, dim=1)), inplace=True)
            feats.append(new_feat)
        out = self.fusion(torch.cat(feats, dim=1))
        return out + x  # global residual


class RDNBlock(nn.Module):
    """Residual Dense Network для регуляризации изображения в DEQ-MPI.

    Параметры по умолчанию урезаны относительно статьи (4 модуля, 128
    каналов, 12 conv-слоёв), чтобы модель обучалась за разумное время на
    одном GPU; общая структура (RDB + ReLU выход) сохранена.
    """

    def __init__(self, channels: int = 32, n_modules: int = 2,
                 n_convs_per_module: int = 4):
        super().__init__()
        self.input_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1)
        self.modules_list = nn.ModuleList([
            _ResidualDenseModule(channels, n_convs_per_module)
            for _ in range(n_modules)
        ])
        self.output_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(x)
        for module in self.modules_list:
            h = module(h)
        # ReLU на выходе — неотрицательность концентрации, как в статье
        return F.relu(self.output_conv(h), inplace=True)


class LCBlock(nn.Module):
    """Learned Consistency block.

    Заменяет жёсткую L2-проекцию на ε-шар вокруг y обучаемой свёрточной
    моделью. Вход: stack[A·x − y, y] (2 канала вдоль частотного
    измерения); 1D-свёртки обрабатывают вдоль частот; выход проектируется
    на ε-окрестность y, чтобы гарантировать ограниченную невязку.
    """

    def __init__(self, n_freq: int, hidden: int = 32, kernel: int = 5,
                 epsilon: float = 1e-2):
        super().__init__()
        self.epsilon = epsilon
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=kernel, padding=p),
        )

    def forward(self, residual: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """residual: (B, M), y: (B, M). Возвращает z: (B, M)."""
        combined = torch.stack([residual, y], dim=1)   # (B, 2, M)
        out = self.net(combined).squeeze(1)            # (B, M)
        # Проекция на ε-окрестность y
        diff = out - y
        norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ratio = torch.clamp(self.epsilon / norm, max=1.0)
        return y + diff * ratio


class DEQMPI(nn.Module):
    """Deep Equilibrium MPI: ADMM-подобные итерации с обучаемыми RDN и LC.

    Forward: повторяем (RDN-регуляризация + LC-консистентность) до
    n_iterations раз. Эквивалентно нахождению неподвижной точки h_θ;
    в инференсе можно остановиться раньше при сходимости (см. условие
    в forward).
    """

    def __init__(self, system_matrix, image_shape,
                 n_iterations: int = 5, lambda_param: float = 0.1,
                 rdn_channels: int = 32, n_rdn_modules: int = 2,
                 rdn_convs_per_module: int = 4,
                 lc_epsilon: float = 1e-2, tol: float = 1e-6):
        super().__init__()
        # Расширяем комплексную SM в вещественную (M, N) → (2·M_complex, N)
        if np.iscomplexobj(system_matrix):
            S_ext = np.vstack([system_matrix.real, system_matrix.imag])
        else:
            S_ext = system_matrix
        self.M, self.N = S_ext.shape
        self.image_shape = tuple(image_shape)
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.tol = tol

        self.register_buffer('A',
                             torch.tensor(S_ext, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T.contiguous())

        # Обучаемые блоки
        self.rdn = RDNBlock(channels=rdn_channels, n_modules=n_rdn_modules,
                            n_convs_per_module=rdn_convs_per_module)
        self.lc = LCBlock(n_freq=self.M, epsilon=lc_epsilon)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → A·x: (B, M)."""
        return x.view(x.shape[0], -1) @ self.A_T

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, M) → Aᵀy: (B, 1, H, W)."""
        return (y @ self.A).view(y.shape[0], 1, *self.image_shape)

    def forward(self, measurements: torch.Tensor,
                return_intermediate: bool = False):
        """measurements: (B, M) вещественные (расширенная форма)."""
        B = measurements.shape[0]
        # Начальное приближение: A^T · y, отнормированное
        x = self.adjoint_op(measurements) * self.lambda_param
        x = F.relu(x)
        history = [x] if return_intermediate else None

        for _ in range(self.n_iterations):
            x_prev = x
            # 1) Регуляризация: пропустить x через RDN-денойзер
            x = self.rdn(x)
            # 2) Согласованность: вычислить A·x − y и пропустить через LC
            residual = self.forward_op(x) - measurements
            z = self.lc(residual, measurements)
            # Обновление: x ← x − λ · Aᵀ · z (gradient-descent-подобный шаг)
            x = x - self.lambda_param * self.adjoint_op(z)
            x = F.relu(x)
            if history is not None:
                history.append(x)
            # Проверка сходимости (опционально)
            if torch.norm(x - x_prev) < self.tol:
                break

        if return_intermediate:
            return x, history
        return x


__all__ = ['DEQMPI', 'RDNBlock', 'LCBlock']
