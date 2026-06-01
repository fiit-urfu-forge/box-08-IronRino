"""DEQ-MPI — Deep Equilibrium Reconstruction с Learned Consistency.

Источник: Güngör A., Askin B., Soydan D. A., Top C. B., Saritas E. U.,
Çukur T. «DEQ-MPI: A Deep Equilibrium Reconstruction with Learned
Consistency for Magnetic Particle Imaging», IEEE Trans. Med. Imaging
(2024), arXiv:2212.13233v2 [eess.IV] 6 Sep 2023,
https://doi.org/10.1109/TMI.2023.3327646

## Прорывная идея — Deep Equilibrium Model для MPI

Классические unrolling-методы (MoDL и др.) разворачивают K шагов
итеративной реконструкции в одну глубокую сеть и обучают её
оптимизировать качество выхода **через N_it фиксированных итераций**.
Минусы:
  • N_it мало (5-10), потому что backward через все шаги
    требует O(N_it) памяти и compute;
  • обученный с N_it=5 unrolled-метод **деградирует** при тесте на
    бóльшем числе итераций (paper Fig. 4);
  • используют hand-crafted ℓ2-ball DC-меру (Ψ_χ), которая
    игнорирует корреляции в MPI-данных.

DEQ-MPI: вместо разворачивания тренирует **неявное** отображение

    x* = h_θ(x*; y, A)         (paper Eq. 13)

как **fixed point**. Backward через implicit differentiation работает
независимо от числа итераций forward'а → O(1) память.

## ADMM-формулировка с learned consistency

DEQ-MPI решает задачу

    argmin_{x,z}  f(z)  s.t.  x = z⁽¹⁾,  A·x = z⁽⁰⁾       (paper Eq. 4)

где f(z) = χ(z⁽⁰⁾) + R(z⁽¹⁾):
  • χ — indicator ℓ2-ball вокруг измерения y, эта DC-часть **заменена
    на learned-консистент-блок** Ψ_LC;
  • R — image prior, заменён на Residual Dense Network Ψ_RDN.

Fixed-point итерация (paper Eq. 15-17):

  z⁽⁰⁾_{k+1} = Ψ_LC(A·x_k − d⁽⁰⁾_k, y)       (learned consistency)
  z⁽¹⁾_{k+1} = Ψ_RDN(x_k − d⁽¹⁾_k)           (learned regularization)
  x_{k+1}   = M·(Aᵀ·(z⁽⁰⁾_{k+1} + d⁽⁰⁾_k) + z⁽¹⁾_{k+1} + d⁽¹⁾_k)
  d⁽⁰⁾_{k+1} = d⁽⁰⁾_k + z⁽⁰⁾_{k+1} − A·x_{k+1}
  d⁽¹⁾_{k+1} = d⁽¹⁾_k + z⁽¹⁾_{k+1} − x_{k+1}

где M = (I + Aᵀ·A)⁻¹ — предвычисленная матрица, A — system matrix,
d⁽⁰⁾, d⁽¹⁾ — Lagrange-множители.

## RDN-блок (paper Sec. III.B.1, Eq. 18-23)

Residual Dense Network с **n_res** residual-модулями. Каждый модуль
содержит **n_conv** свёрток с dense-соединениями (выход слоя l
получает конкатенацию ВСЕХ предыдущих слоёв модуля + входа модуля).

Default архитектура (paper Sec. IV.A):
  • n_res = 4 residual modules
  • F_R = 12 channels в основном feature map
  • n_conv = 12 свёрток на модуль
  • ReLU на финальном выходе → c ≥ 0 by construction (MPI nonneg)

## LC-блок (paper Sec. III.B.1, Eq. 24-25)

Learned Consistency — замена ℓ2-ball проекции Ψ_χ:

  Ψ_LC(v, y) = y + ℓ2-ball-clip(Z(v, y) − y, ε)        (paper Eq. 25)

где Z — конволюционная сеть с n_LC = 1 hidden layer, F_LC = 8 channels.
Для FFL-сканеров (single coil): **1D-свёртки по frequency** dimension.
Для FFP-сканеров (multi-coil): 2D по (freq, channel).

Ψ_LC учит, как сглаживать «выбросы» в данных, учитывая корреляции
между гармониками — то, что hand-crafted ℓ2-ball игнорирует.

## Implicit differentiation (paper Sec. III.B.2, Eq. 28-33)

Backward через convergent solution x*:

  (∂x*/∂θ)ᵀ·b = (∂h_θ(x*)/∂θ)ᵀ · s*
  s* решается через fixed-point:
    s_{i+1} = (∂h_θ(x*)/∂x*)ᵀ · s_i + b              (paper Eq. 32)

Память O(1) вне зависимости от forward N_it.

## Pre-training (КРИТИЧНО! paper Sec. III.B.2, Eq. 26-27)

DEQ-MPI без pre-training даёт pSNR на **7-17 dB хуже** (paper Sec.
V.A): «pSNR is 37.6 dB for DEQ-MPI, 29.9 dB when RDN is randomly
initialized, 20.2 dB when LC is randomly initialized».

**Pre-train RDN как denoiser** (σ1 = 0.1):
  argmin_θ_RDN  ‖Ψ_RDN(x_r + n_1) − x_r‖₁              (paper Eq. 26)

**Pre-train LC** mimicking ℓ2-ball DC (σ2 = 0.05, σ3 = 0.02):
  argmin_θ_LC  ‖Ψ_LC(y_r + n_3, y_r + n_2) − Ψ_χ(y_r + n_3, y_r + n_2)‖₁
                                                       (paper Eq. 27)

Initialization x_0 = A†·y (pseudo-inverse), d⁽⁰⁾_0 = d⁽¹⁾_0 = 0.

## Forward solver (paper Sec. III.B.2)

Fixed-point iteration с **Anderson acceleration** (Anderson 1965).
Stop criterion: ‖x_{k+1} − x_k‖₂ < 10⁻⁴.
Max iterations: 25 (paper нашёл — достаточно для convergence).

## Hyperparameters (paper Sec. IV.A)

  n_res = 4, F_R = 12, n_conv = 12, n_LC = 1, F_LC = 8
  lr = 10⁻³, 200 epochs, ADAM (β1=0.9, β2=0.999)
  ε = √M для DC, σ1=0.1, σ2=0.05, σ3=0.02 для pre-training
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# RDN-блок (Residual Dense Network, paper Eq. 18-23)
# =============================================================================


class _ResidualDenseModule(nn.Module):
    """Один residual-dense модуль из RDN (paper Eq. 19-21).

    Внутри n_conv свёрток с **dense-connections**: выход слоя l получает
    конкатенацию входа модуля и выходов всех предыдущих слоёв 1..l-1.

    Все свёртки: 3×3 (стандартно для RDN, paper не оговаривает),
    ReLU после каждой кроме финальной fusion.
    """

    def __init__(self, in_channels: int, growth_rate: int, n_conv: int,
                 kernel_size: int = 3):
        super().__init__()
        self.n_conv = n_conv
        self.growth_rate = growth_rate

        # Dense convolutions: каждая получает все предыдущие feature maps
        self.convs = nn.ModuleList()
        for l in range(n_conv):
            in_ch = in_channels + l * growth_rate
            self.convs.append(nn.Conv2d(
                in_ch, growth_rate, kernel_size=kernel_size,
                stride=1, padding=kernel_size // 2,
            ))

        # Local feature fusion: 1×1 conv ← concat(all dense outputs)
        fuse_in = in_channels + n_conv * growth_rate
        self.fuse = nn.Conv2d(fuse_in, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) → (B, in_channels, H, W)."""
        features = [x]
        for conv in self.convs:
            inp = torch.cat(features, dim=1)
            out = F.relu(conv(inp), inplace=False)
            features.append(out)
        fused = self.fuse(torch.cat(features, dim=1))
        # Local residual learning (paper Eq. 21): module-input + fused
        return x + fused


class _RDNBlock(nn.Module):
    """RDN-блок целиком (paper Eq. 18-23).

    Архитектура:
      Z_0(v) — shallow feature extraction (2 conv)
        ↓
      Stack of n_res residual-dense modules (caсcade)
        ↓
      Z_fuse — 1×1 conv для слияния выходов всех модулей
        ↓
      Z_out — финальная свёртка + ReLU (non-negativity, paper Eq. 23)
        ↓
      output = ReLU(Z_out + v)   ← global residual learning
    """

    def __init__(self, n_res: int = 4, F_R: int = 12, n_conv: int = 12,
                 kernel_size: int = 3):
        super().__init__()
        self.n_res = n_res
        # Shallow feature extractor Z_0 (paper Eq. 18): 2 conv layers
        # 1 канал MPI → F_R каналов
        self.shallow = nn.Sequential(
            nn.Conv2d(1, F_R, kernel_size=kernel_size,
                       stride=1, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_R, F_R, kernel_size=kernel_size,
                       stride=1, padding=kernel_size // 2),
        )
        # n_res каскадных residual-dense модулей (paper Eq. 19-21)
        self.modules_list = nn.ModuleList([
            _ResidualDenseModule(F_R, growth_rate=F_R, n_conv=n_conv,
                                   kernel_size=kernel_size)
            for _ in range(n_res)
        ])
        # Global feature fusion Z_fuse (paper Eq. 22): 1×1 conv
        self.fuse = nn.Conv2d(F_R * n_res, F_R, kernel_size=1)
        # Z_out — финальная свёртка обратно в 1 канал
        self.out = nn.Conv2d(F_R, 1, kernel_size=kernel_size,
                              stride=1, padding=kernel_size // 2)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """v: (B, 1, H, W) — input image after subtracting d⁽¹⁾_k.

        Returns: z⁽¹⁾_{k+1} (B, 1, H, W) с non-negativity.
        """
        u0 = self.shallow(v)
        outputs = []
        u = u0
        for module in self.modules_list:
            u = module(u)
            outputs.append(u)
        # Global feature fusion (paper Eq. 22)
        u_fuse = self.fuse(torch.cat(outputs, dim=1))
        # Output + global residual + ReLU (paper Eq. 23)
        return F.relu(self.out(u_fuse) + v, inplace=False)


# =============================================================================
# LC-блок (Learned Consistency, paper Eq. 24-25)
# =============================================================================


class _LCBlock(nn.Module):
    """Learned Consistency block (paper Eq. 24-25).

    Заменяет hand-crafted ℓ2-ball-проекцию Ψ_χ на обучаемый
    конволюционный модуль Z(v, y), затем ℓ2-clip итогового residual'а:

      Ψ_LC(v, y) = y + ℓ2-clip(Z(v, y) − y, ε)

    Архитектура Z (paper Sec. III.B.1):
      Вход v ∈ C^M, y ∈ C^M. Стэк по реальной/мнимой части — 4 канала
      ((Re v, Im v, Re y, Im y)). 1D conv-свёртки по frequency для
      FFL-сканера; для FFP — 2D-свёртки по (freq, channel).
      n_LC = 1 hidden layer, F_LC = 8 channels.

    Args:
        M: число частотных бинов в y.
        n_hidden: число скрытых свёрточных слоёв (paper default 1).
        F_hidden: число каналов в скрытых слоях (paper default 8).
        kernel_size: размер 1D-ядра по freq dimension (default 3).
    """

    def __init__(self, M: int, n_hidden: int = 1, F_hidden: int = 8,
                 kernel_size: int = 3):
        super().__init__()
        self.M = M
        # 4-канальный вход: (Re v, Im v, Re y, Im y) по freq dim
        layers: List[nn.Module] = []
        in_ch = 4
        for _ in range(n_hidden):
            layers += [
                nn.Conv1d(in_ch, F_hidden, kernel_size=kernel_size,
                            stride=1, padding=kernel_size // 2),
                nn.ReLU(inplace=True),
            ]
            in_ch = F_hidden
        # Финальный слой → 2 канала (Re, Im) output
        layers.append(
            nn.Conv1d(in_ch, 2, kernel_size=kernel_size,
                        stride=1, padding=kernel_size // 2)
        )
        self.z_net = nn.Sequential(*layers)

    def forward(self, v_real: torch.Tensor, v_imag: torch.Tensor,
                y_real: torch.Tensor, y_imag: torch.Tensor,
                epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Learned consistency (paper Eq. 25).

        Args:
            v_real, v_imag, y_real, y_imag: (B, M) tensors.
            epsilon: ε-bound для ℓ2-clip (paper: ε = √M).

        Returns:
            (z_real, z_imag) — оба (B, M).
        """
        # Стэк (Re v, Im v, Re y, Im y) по каналу
        inp = torch.stack([v_real, v_imag, y_real, y_imag], dim=1)   # (B, 4, M)
        z = self.z_net(inp)                                            # (B, 2, M)
        z_re_raw, z_im_raw = z[:, 0], z[:, 1]                          # (B, M)

        # Residual против measurement y
        delta_re = z_re_raw - y_real
        delta_im = z_im_raw - y_imag
        # ℓ2-clip: если ‖δ‖_2 > ε, то δ ← ε · δ/‖δ‖_2 (paper Eq. 25)
        norm = torch.sqrt(delta_re.pow(2).sum(dim=-1)
                            + delta_im.pow(2).sum(dim=-1)).clamp_min(1e-12)
        scale = torch.where(norm > epsilon, epsilon / norm,
                              torch.ones_like(norm))                   # (B,)
        delta_re = delta_re * scale.unsqueeze(-1)
        delta_im = delta_im * scale.unsqueeze(-1)

        return y_real + delta_re, y_imag + delta_im


def l2_ball_projection(v_real: torch.Tensor, v_imag: torch.Tensor,
                        y_real: torch.Tensor, y_imag: torch.Tensor,
                        epsilon: float
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classic ℓ2-ball DC mapping Ψ_χ(v, y) (paper Eq. 10).

      Ψ_χ(v, y) = y + (v − y)                  if ‖v − y‖ ≤ ε
                = y + ε·(v − y)/‖v − y‖        otherwise

    Используется для:
      • baseline-сравнения с LC (LC-ablated variant из paper);
      • **pre-training** LC через target Ψ_χ (paper Eq. 27).
    """
    delta_re = v_real - y_real
    delta_im = v_imag - y_imag
    norm = torch.sqrt(delta_re.pow(2).sum(dim=-1)
                        + delta_im.pow(2).sum(dim=-1)).clamp_min(1e-12)
    scale = torch.where(norm > epsilon, epsilon / norm,
                          torch.ones_like(norm))
    delta_re = delta_re * scale.unsqueeze(-1)
    delta_im = delta_im * scale.unsqueeze(-1)
    return y_real + delta_re, y_imag + delta_im


# =============================================================================
# Anderson acceleration solver (paper Sec. III.B.2)
# =============================================================================


def _anderson_solve(fn, x0: torch.Tensor, m: int = 5,
                     max_iter: int = 25, tol: float = 1e-4,
                     beta: float = 1.0) -> Tuple[torch.Tensor, int]:
    """Anderson acceleration для решения x* = fn(x*).

    DEQ-литература (Bai et al. 2019, Anderson 1965): сохраняем m
    последних residual'ов и решаем bordered least-squares для
    нахождения коэффициентов линейной комбинации.

    Args:
        fn:       callable, принимает x и возвращает fn(x) той же формы.
        x0:       (B, *) начальное приближение.
        m:        размер истории.
        max_iter: лимит итераций.
        tol:      порог ‖x_{k+1} - x_k‖ для остановки.
        beta:     mixing coefficient (1.0 = чистый Anderson).

    Returns:
        (x*, n_iters) — convergent solution и реально пройденное число итераций.
    """
    B = x0.shape[0]
    flat_dim = x0.numel() // B
    X = torch.zeros(B, m, flat_dim, dtype=x0.dtype, device=x0.device)
    F_buf = torch.zeros(B, m, flat_dim, dtype=x0.dtype, device=x0.device)
    X[:, 0] = x0.reshape(B, -1)
    F_buf[:, 0] = fn(x0).reshape(B, -1)
    X[:, 1] = F_buf[:, 0]
    F_buf[:, 1] = fn(F_buf[:, 0].reshape_as(x0)).reshape(B, -1)

    H = torch.zeros(B, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1.0
    y_target = torch.zeros(B, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y_target[:, 0] = 1.0

    res_norm_prev = float('inf')
    for k in range(2, max_iter):
        n = min(k, m)
        G = F_buf[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + 1e-4 * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        try:
            alpha = torch.linalg.solve(H[:, :n + 1, :n + 1],
                                          y_target[:, :n + 1])[:, 1:n + 1, 0]
        except RuntimeError:
            # Singular system — fallback на простое Picard-обновление
            alpha = torch.zeros(B, n, dtype=x0.dtype, device=x0.device)
            alpha[:, -1] = 1.0

        new_x = beta * (alpha[:, :, None] * F_buf[:, :n]).sum(dim=1) \
                + (1.0 - beta) * (alpha[:, :, None] * X[:, :n]).sum(dim=1)

        X[:, k % m] = new_x
        F_buf[:, k % m] = fn(new_x.reshape_as(x0)).reshape(B, -1)

        res = (F_buf[:, k % m] - X[:, k % m]).norm(dim=-1).max().item()
        x_diff = (new_x - X[:, (k - 1) % m]).norm(dim=-1).max().item()
        if x_diff < tol:
            break
        res_norm_prev = res

    return F_buf[:, k % m].reshape_as(x0), k


# =============================================================================
# DEQ-MPI — head-level модель
# =============================================================================


class DEQMPI(nn.Module):
    """Deep Equilibrium Reconstruction для MPI (Güngör 2024).

    ## Использование

    Forward принимает измерение y (комплексное, разложенное в Re/Im)
    и возвращает реконструкцию x. Внутри: fixed-point поиск с Anderson,
    implicit-backward через 1-step approximation (см. backward).

    ## Initialization (КРИТИЧЕСКИ ВАЖНО)

    Перед обучением **обязательно** запустить:
      1. `pretrain_rdn(images, sigma1=0.1, epochs=...)` — учит RDN как denoiser
      2. `pretrain_lc(y_clean, sigma2=0.05, sigma3=0.02, epochs=...)`
         — учит LC mimick'ать ℓ2-ball Ψ_χ

    Без них SSIM падает на 7-17 dB (paper Sec. V.A).

    ## Hyperparameters (paper Sec. IV.A)

    Args:
        system_matrix: A ∈ C^(M × N) — измеренная системная матрица.
        image_shape:   (H, W) — размер реконструкции.
        n_iterations:  макс. число fixed-point итераций (paper: 25).
        epsilon:       ε для DC ℓ2-clip (paper: √M).
        n_res:         residual modules в RDN (paper: 4).
        F_R:           channels в RDN (paper: 12).
        n_conv:        dense-conv слоёв на модуль (paper: 12).
        n_lc_hidden:   hidden слоёв в LC (paper: 1).
        F_lc:          channels в LC (paper: 8).
        anderson_m:    memory size Anderson acceleration.
        tol:           порог сходимости (paper: 1e-4).
    """

    def __init__(self,
                 system_matrix: np.ndarray,
                 image_shape: Tuple[int, int],
                 n_iterations: int = 25,
                 epsilon: Optional[float] = None,
                 n_res: int = 4, F_R: int = 12, n_conv: int = 12,
                 n_lc_hidden: int = 1, F_lc: int = 8,
                 anderson_m: int = 5, tol: float = 1e-4):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.n_iterations = n_iterations
        self.anderson_m = anderson_m
        self.tol = tol

        # Разложить комплексную SM в (Re, Im) для real-arithmetic
        A = np.asarray(system_matrix)
        if np.iscomplexobj(A):
            self.M_complex, self.N = A.shape
            A_re = A.real.astype(np.float32)
            A_im = A.imag.astype(np.float32)
        else:
            raise ValueError(
                "DEQ-MPI ожидает комплексную системную матрицу "
                "(в paper measurement тоже комплексное)."
            )

        self.register_buffer('A_re', torch.tensor(A_re))
        self.register_buffer('A_im', torch.tensor(A_im))
        # Precompute M = (I + AᵀA)⁻¹ для least-squares шага (paper Eq. 17)
        AtA = (self.A_re.T @ self.A_re + self.A_im.T @ self.A_im)
        I = torch.eye(self.N)
        self.register_buffer('LS_inv', torch.linalg.inv(I + AtA))

        # ε для DC: default √M_complex (paper Sec. IV.A)
        self.epsilon = float(epsilon) if epsilon is not None \
            else float(np.sqrt(self.M_complex))

        # RDN-блок (regularization)
        self.rdn = _RDNBlock(n_res=n_res, F_R=F_R, n_conv=n_conv)
        # LC-блок (learned consistency)
        self.lc = _LCBlock(M=self.M_complex, n_hidden=n_lc_hidden,
                            F_hidden=F_lc)

    # ---- Forward operators ----------------------------------------------------

    def _A_forward(self, x: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A · x (комплексное) → (Re, Im) result.

        x: (B, N) — flattened image.
        Returns: (Ax_real, Ax_imag) — оба (B, M).
        """
        Ax_re = x @ self.A_re.T
        Ax_im = x @ self.A_im.T
        return Ax_re, Ax_im

    def _A_adjoint(self, y_re: torch.Tensor, y_im: torch.Tensor
                   ) -> torch.Tensor:
        """Aᵀ · y где Aᵀ = (A_re − j·A_im)ᵀ; берём Re части суммы.

        (Aᵀ y)_n = Σ_m conj(A_{mn}) · y_m
                  = Σ_m (A_re_{mn} y_re_m + A_im_{mn} y_im_m) + j·(...)
        Берём только Re часть — image x в paper вещественный.
        """
        return y_re @ self.A_re + y_im @ self.A_im

    # ---- Fixed-point step h_θ ------------------------------------------------

    def _hθ_step(self, state_flat: torch.Tensor,
                 y_re: torch.Tensor, y_im: torch.Tensor) -> torch.Tensor:
        """Один h_θ шаг (paper Eq. 15-17).

        state_flat: (B, N + 2M) — конкатенированное (x, d⁽⁰⁾_re, d⁽⁰⁾_im,
        d⁽¹⁾) … где d⁽⁰⁾ — комплексное (R-Im), d⁽¹⁾ — вещественное.

        Layout:
          state_flat[:, :N]                 — x
          state_flat[:, N:N+M]              — d⁽⁰⁾_re
          state_flat[:, N+M:N+2M]           — d⁽⁰⁾_im
          state_flat[:, N+2M:N+2M+N]        — d⁽¹⁾
        """
        B = state_flat.shape[0]
        N, M = self.N, self.M_complex
        x = state_flat[:, :N]                            # (B, N)
        d0_re = state_flat[:, N:N + M]                   # (B, M)
        d0_im = state_flat[:, N + M:N + 2 * M]           # (B, M)
        d1 = state_flat[:, N + 2 * M:N + 2 * M + N]      # (B, N)

        # 1) LC step: z⁽⁰⁾_{k+1} = Ψ_LC(Ax_k − d⁽⁰⁾_k, y)
        Ax_re, Ax_im = self._A_forward(x)
        v0_re = Ax_re - d0_re
        v0_im = Ax_im - d0_im
        z0_re, z0_im = self.lc(v0_re, v0_im, y_re, y_im, self.epsilon)

        # 2) RDN step: z⁽¹⁾_{k+1} = Ψ_RDN(x_k − d⁽¹⁾_k)
        v1 = (x - d1).view(B, 1, *self.image_shape)
        z1 = self.rdn(v1).view(B, -1)                    # (B, N)

        # 3) Least-squares step (paper Eq. 17):
        # x_{k+1} = M · (Aᵀ(z⁽⁰⁾_{k+1} + d⁽⁰⁾_k) + z⁽¹⁾_{k+1} + d⁽¹⁾_k)
        At_arg_re = z0_re + d0_re
        At_arg_im = z0_im + d0_im
        At_z0 = self._A_adjoint(At_arg_re, At_arg_im)    # (B, N)
        rhs = At_z0 + z1 + d1
        x_new = rhs @ self.LS_inv.T

        # 4) Lagrange-multiplier updates
        Ax_new_re, Ax_new_im = self._A_forward(x_new)
        d0_re_new = d0_re + z0_re - Ax_new_re
        d0_im_new = d0_im + z0_im - Ax_new_im
        d1_new = d1 + z1 - x_new

        return torch.cat([x_new, d0_re_new, d0_im_new, d1_new], dim=-1)

    # ---- Forward (training + inference) --------------------------------------

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        """Forward с implicit-backward gradient (1-step approximation).

        Стратегия (paper Sec. III.B.2 + Bai 2019):
          • Forward search под torch.no_grad() — даёт сходящийся x*;
          • Один дополнительный шаг h_θ под autograd для backward.
        Это сохраняет O(1) память и даёт состоятельный градиент через
        implicit-differentiation Eq. 28-33.

        Args:
            measurement: (B, 2, M) или (B, 2*M) — Re/Im измерения y.
                         Тип float32, не complex.

        Returns:
            x_recon: (B, 1, H, W) — реконструкция в исходном image_shape.
        """
        B = measurement.shape[0]
        # Reshape (B, 2, M) → (y_re, y_im)
        if measurement.dim() == 3:
            y_re = measurement[:, 0]
            y_im = measurement[:, 1]
        elif measurement.dim() == 2:
            y_re = measurement[:, :self.M_complex]
            y_im = measurement[:, self.M_complex:]
        else:
            raise ValueError(
                f"measurement.shape={measurement.shape} unexpected"
            )

        # Initialization (paper Sec. III.B.2): x_0 = A†·y, d_0 = 0
        with torch.no_grad():
            # Pseudo-inverse через precomputed M·Aᵀ
            x0 = self._A_adjoint(y_re, y_im) @ self.LS_inv.T  # (B, N)
        d0_re = torch.zeros(B, self.M_complex, device=x0.device)
        d0_im = torch.zeros(B, self.M_complex, device=x0.device)
        d1 = torch.zeros(B, self.N, device=x0.device)
        state0 = torch.cat([x0, d0_re, d0_im, d1], dim=-1)

        # Forward search под no_grad
        with torch.no_grad():
            def step(s):
                return self._hθ_step(s, y_re, y_im)
            x_star_state, _n_iter = _anderson_solve(
                step, state0, m=self.anderson_m,
                max_iter=self.n_iterations, tol=self.tol,
            )

        # Один шаг под autograd для implicit backward (Bai 2019)
        x_star_state = self._hθ_step(x_star_state.detach(), y_re, y_im)

        x_star = x_star_state[:, :self.N]
        return x_star.view(B, 1, *self.image_shape)

    # ---- Pre-training (paper Sec. III.B.2, Eq. 26-27) ------------------------

    def pretrain_rdn(self, images: torch.Tensor,
                     sigma1: float = 0.1,
                     epochs: int = 30, lr: float = 1e-3,
                     batch_size: int = 16,
                     device: Optional[str] = None) -> List[float]:
        """Pre-train RDN как denoiser (paper Eq. 26).

          argmin_θ_RDN  ‖Ψ_RDN(x_r + n_1) − x_r‖₁,   n_1 ~ N(0, σ1²)

        ОБЯЗАТЕЛЕН перед основным обучением: без этого pSNR падает с
        37.6 dB до 29.9 dB (paper Sec. V.A).

        Args:
            images: (N_train, 1, H, W) — обучающие GT-картинки x_r.
            sigma1: σ для входного шума (paper: 0.1).
            epochs, lr, batch_size: стандартные training hyperparams.

        Returns:
            История loss по эпохам.
        """
        from torch.utils.data import DataLoader, TensorDataset
        dev = device or next(self.parameters()).device
        self.rdn.to(dev)
        ds = TensorDataset(images)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.rdn.parameters(), lr=lr)
        crit = nn.L1Loss()
        history = []
        for ep in range(epochs):
            total = 0.0
            for (x_r,) in loader:
                x_r = x_r.to(dev)
                noise = sigma1 * torch.randn_like(x_r)
                x_noisy = x_r + noise
                opt.zero_grad()
                pred = self.rdn(x_noisy)
                loss = crit(pred, x_r)
                loss.backward()
                opt.step()
                total += loss.item()
            history.append(total / len(loader))
        return history

    def pretrain_lc(self, y_clean: torch.Tensor,
                    sigma2: float = 0.05, sigma3: float = 0.02,
                    epochs: int = 30, lr: float = 1e-3,
                    batch_size: int = 16,
                    device: Optional[str] = None) -> List[float]:
        """Pre-train LC mimick'ать ℓ2-ball DC (paper Eq. 27).

          argmin_θ_LC  ‖Ψ_LC(v_n, y_n) − Ψ_χ(v_n, y_n)‖₁
              y_n = y_r + n_2,  v_n = y_r + n_3
              n_2 ~ N(0, σ2²),  n_3 ~ N(0, σ3²)

        То есть учим LC ВОСПРОИЗВОДИТЬ поведение классической
        ℓ2-ball-projection. После этого тонкая настройка через
        end-to-end loss перенастраивает её под data distribution.

        Args:
            y_clean: (N_train, M) комплексные ИЛИ (N_train, 2, M) Re/Im —
                     noise-free measurements y_r = A·x_r.
            sigma2, sigma3: σ шумов для y_n и v_n (paper: 0.05, 0.02).
        """
        from torch.utils.data import DataLoader, TensorDataset
        dev = device or next(self.parameters()).device
        self.lc.to(dev)

        if y_clean.dim() == 2 and torch.is_complex(y_clean):
            y_re_all = y_clean.real
            y_im_all = y_clean.imag
        elif y_clean.dim() == 3 and y_clean.shape[1] == 2:
            y_re_all = y_clean[:, 0]
            y_im_all = y_clean[:, 1]
        else:
            raise ValueError(
                f"y_clean.shape={y_clean.shape} — ожидается (N, M) complex "
                "или (N, 2, M) real."
            )

        ds = TensorDataset(y_re_all, y_im_all)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.lc.parameters(), lr=lr)
        crit = nn.L1Loss()
        history = []
        for ep in range(epochs):
            total = 0.0
            for y_re, y_im in loader:
                y_re = y_re.to(dev); y_im = y_im.to(dev)
                n2_re = sigma2 * torch.randn_like(y_re)
                n2_im = sigma2 * torch.randn_like(y_im)
                n3_re = sigma3 * torch.randn_like(y_re)
                n3_im = sigma3 * torch.randn_like(y_im)
                yn_re = y_re + n2_re
                yn_im = y_im + n2_im
                vn_re = y_re + n3_re
                vn_im = y_im + n3_im

                opt.zero_grad()
                lc_re, lc_im = self.lc(vn_re, vn_im, yn_re, yn_im,
                                        self.epsilon)
                # Target: classic ℓ2-ball mapping
                with torch.no_grad():
                    tg_re, tg_im = l2_ball_projection(
                        vn_re, vn_im, yn_re, yn_im, self.epsilon,
                    )
                loss = crit(lc_re, tg_re) + crit(lc_im, tg_im)
                loss.backward()
                opt.step()
                total += loss.item()
            history.append(total / len(loader))
        return history


__all__ = ['DEQMPI', 'l2_ball_projection']
