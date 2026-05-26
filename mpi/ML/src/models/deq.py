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

from typing import Optional

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

    Дефолты архитектуры взяты строго из статьи Güngör 2024 (Sec. IV.A):
      RDN: n_res=4, F_R=12, n_conv=12
      LC:  n_LC=1, F_LC=8
      ε = √M для DC, lambda_param ≈ 1.0
      25 итераций фиксированной точки

    Forward: повторяем (RDN-регуляризация + LC-консистентность) до
    `n_iterations` раз с **Anderson acceleration** для ускорения
    сходимости (статья Sec. IV.A).
    """

    def __init__(self, system_matrix, image_shape,
                 n_iterations: int = 25, lambda_param: float = 1.0,
                 rdn_channels: int = 12, n_rdn_modules: int = 4,
                 rdn_convs_per_module: int = 12,
                 lc_hidden: int = 8, lc_epsilon: Optional[float] = None,
                 tol: float = 1e-4,
                 use_anderson: bool = True, anderson_m: int = 5):
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
        self.use_anderson = use_anderson
        self.anderson_m = anderson_m

        # ε = √M по умолчанию (статья Sec IV.A)
        if lc_epsilon is None:
            lc_epsilon = float(np.sqrt(self.M))

        self.register_buffer('A',
                             torch.tensor(S_ext, dtype=torch.float32))
        self.register_buffer('A_T', self.A.T.contiguous())

        # Обучаемые блоки
        self.rdn = RDNBlock(channels=rdn_channels, n_modules=n_rdn_modules,
                            n_convs_per_module=rdn_convs_per_module)
        self.lc = LCBlock(n_freq=self.M, hidden=lc_hidden,
                          epsilon=lc_epsilon)

    def forward_op(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → A·x: (B, M)."""
        return x.view(x.shape[0], -1) @ self.A_T

    def adjoint_op(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, M) → Aᵀy: (B, 1, H, W)."""
        return (y @ self.A).view(y.shape[0], 1, *self.image_shape)

    def _step(self, x: torch.Tensor,
              measurements: torch.Tensor) -> torch.Tensor:
        """Один шаг h_θ(x; y, A): RDN-регуляризация + LC-консистентность."""
        x = self.rdn(x)
        residual = self.forward_op(x) - measurements
        z = self.lc(residual, measurements)
        x = x - self.lambda_param * self.adjoint_op(z)
        return F.relu(x)

    def _anderson_solve(self, x0: torch.Tensor,
                        measurements: torch.Tensor) -> torch.Tensor:
        """Anderson acceleration для поиска неподвижной точки h_θ.

        Хранит последние `m` итераций и решает маленькую задачу LS,
        чтобы найти линейную комбинацию, минимизирующую невязку
        F(x) = x − h_θ(x). Сходится за ~5× меньше шагов, чем простая
        итерация (статья Sec IV.A, ссылка [62]).
        """
        m = self.anderson_m
        B = x0.shape[0]
        N = x0[0].numel()
        # Буферы X (итерации) и F (невязки)
        X = torch.zeros(B, m, N, device=x0.device, dtype=x0.dtype)
        Fbuf = torch.zeros_like(X)

        x = x0
        h = self._step(x, measurements)
        X[:, 0] = x.view(B, -1)
        Fbuf[:, 0] = (h - x).view(B, -1)
        x = h

        for k in range(1, self.n_iterations):
            h = self._step(x, measurements)
            idx = k % m
            X[:, idx] = x.view(B, -1)
            Fbuf[:, idx] = (h - x).view(B, -1)

            n_used = min(k + 1, m)
            # Решаем α: F[:, :n] α = 0, sum(α) = 1 (через нормальное
            # уравнение с регуляризацией)
            Fk = Fbuf[:, :n_used]                          # (B, n, N)
            G = Fk @ Fk.transpose(1, 2)                    # (B, n, n)
            G = G + 1e-4 * torch.eye(n_used, device=x0.device).unsqueeze(0)
            # Лагранжева система через bordered matrix
            ones = torch.ones(B, n_used, 1, device=x0.device, dtype=x0.dtype)
            top = torch.cat([G, ones], dim=2)              # (B, n, n+1)
            bot = torch.cat([ones.transpose(1, 2),
                             torch.zeros(B, 1, 1, device=x0.device,
                                         dtype=x0.dtype)], dim=2)
            A_sys = torch.cat([top, bot], dim=1)           # (B, n+1, n+1)
            b_sys = torch.zeros(B, n_used + 1, 1,
                                device=x0.device, dtype=x0.dtype)
            b_sys[:, -1, 0] = 1.0
            try:
                sol = torch.linalg.solve(A_sys, b_sys)
                alpha = sol[:, :n_used, 0]                 # (B, n)
            except RuntimeError:
                # Fallback: простая итерация
                x = h
                continue

            x_new = (alpha.unsqueeze(-1) *
                     (X[:, :n_used] + Fbuf[:, :n_used])).sum(dim=1)
            x_new = x_new.view(*x.shape)
            x_new = F.relu(x_new)

            if torch.norm(x_new - x) < self.tol:
                x = x_new
                break
            x = x_new

        return x

    def forward(self, measurements: torch.Tensor,
                return_intermediate: bool = False):
        """measurements: (B, M) вещественные (расширенная форма)."""
        # Начальное приближение: A^T·y (см. статья Sec IV.B "x is
        # initialized with the least-squares solution xLS = A†·y"; здесь
        # упрощённо берём adjoint, что эффективно при whitening).
        x = self.adjoint_op(measurements) * self.lambda_param
        x = F.relu(x)

        if return_intermediate:
            # Простая итерация для возможности сохранения промежуточных
            history = [x]
            for _ in range(self.n_iterations):
                x_prev = x
                x = self._step(x, measurements)
                history.append(x)
                if torch.norm(x - x_prev) < self.tol:
                    break
            return x, history

        if self.use_anderson:
            x = self._anderson_solve(x, measurements)
        else:
            for _ in range(self.n_iterations):
                x_prev = x
                x = self._step(x, measurements)
                if torch.norm(x - x_prev) < self.tol:
                    break
        return x

    # ----- инициализация (КРИТИЧНО, см. статья Sec V.A) -----

    def pretrain_rdn(self, x_train: torch.Tensor, sigma1: float = 0.1,
                     epochs: int = 30, lr: float = 1e-3,
                     batch_size: int = 16) -> list:
        """Pre-train RDN как denoiser: x_n = x_clean + N(0, σ₁²I) → x_clean.

        Статья (Sec V.A): без этой инициализации PSNR DEQ-MPI падает с
        37.6 dB до 29.9 dB. Запускайте до основного обучения.

        Args:
            x_train: (N, 1, H, W) — чистые training-изображения.
            sigma1: std шума, 0.1 в статье.
        Returns:
            История loss по эпохам.
        """
        from torch.utils.data import DataLoader, TensorDataset
        dev = next(self.parameters()).device
        ds = TensorDataset(x_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.rdn.parameters(), lr=lr)
        crit = nn.L1Loss()
        history = []
        for ep in range(epochs):
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(dev)
                noise = sigma1 * torch.randn_like(xb)
                opt.zero_grad()
                pred = self.rdn(xb + noise)
                loss = crit(pred, xb)
                loss.backward()
                opt.step()
                total += loss.item()
            history.append(total / max(1, len(loader)))
        return history

    def pretrain_lc(self, y_clean: torch.Tensor, sigma2: float = 0.05,
                    sigma3: float = 0.02, epochs: int = 30, lr: float = 1e-3,
                    batch_size: int = 16) -> list:
        """Pre-train LC, чтобы он повторял каноническую L2-проекцию χ(v_n, y_n).

        Статья (Sec V.A): без этой инициализации PSNR падает с 37.6 dB
        до 20.2 dB (катастрофическое расхождение). y_n = y_clean + N(σ₂),
        v_n = y_clean + N(σ₃); цель — LC(v_n, y_n) ≈ χ(v_n, y_n).

        Args:
            y_clean: (N, M) — чистые training-измерения (без шума).
        """
        from torch.utils.data import DataLoader, TensorDataset
        dev = next(self.parameters()).device
        ds = TensorDataset(y_clean)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.lc.parameters(), lr=lr)
        crit = nn.L1Loss()
        history = []
        for ep in range(epochs):
            total = 0.0
            for (yb,) in loader:
                yb = yb.to(dev)
                yn = yb + sigma2 * torch.randn_like(yb)
                vn = yb + sigma3 * torch.randn_like(yb)
                # Целевая L2-проекция (тот же код что и внутри LCBlock,
                # но без свёртки — это идентичная функция)
                with torch.no_grad():
                    diff = vn - yn
                    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    ratio = torch.clamp(self.lc.epsilon / norm, max=1.0)
                    target = yn + diff * ratio
                opt.zero_grad()
                # LC принимает (residual, y), где residual = v − y
                pred = self.lc(vn - yn, yn)
                loss = crit(pred, target)
                loss.backward()
                opt.step()
                total += loss.item()
            history.append(total / max(1, len(loader)))
        return history


__all__ = ['DEQMPI', 'RDNBlock', 'LCBlock']
