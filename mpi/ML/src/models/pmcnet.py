"""Physical Model-Constrained Network (PMCNet) для реконструкции MPI.

Основано на: Huang et al., "PMCNet: A Physical Model-Constrained Network for
Magnetic Particle Imaging Reconstruction", IEEE Trans. Magn. 2026
(DOI 10.1109/TMAG.2026.3674068).

Реализованы две модели:

  PMCNet                  -- базовая (Eq. 1, 2, 6, 7, 8 статьи): U-Net φ_θ(z)
                             генерирует концентрацию c, прямой оператор
                             u = S·c (системная матрица заменяет интеграл
                             Eq. 1). Loss = ||S·c − u_meas||₁, без обучающих
                             данных.

  PMCNetWithRefinedPhysics -- расширенная (Eq. 4, 5): добавлена релаксация
                              Дебая с обучаемой τ через softplus-параметр,
                              мульти-цветовой режим (раздельные c_k и τ_k для
                              разных типов МНЧ), TV-регуляризация.
                              Релаксация применяется к u = S·c как
                              частотный фильтр H_τ(f) = 1 / (1 + j·2π·f·τ),
                              эквивалентный временной свёртке с
                              r(t) = (1/τ)·e^(−t/τ)·u(t).

Оба варианта интегрируются в существующий пайплайн через комплексную
системную матрицу, уже загружаемую `MPIReconstructionComparator`.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================================================================
# Конфигурация
# =============================================================================


@dataclass
class PMCNetConfig:
    """Гиперпараметры PMCNet и его уточнённой версии."""

    # Размер реконструируемого изображения
    image_size: Tuple[int, int] = (51, 51)

    # Параметры U-Net
    base_channels: int = 32

    # Оптимизация
    n_iterations: int = 2000
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = field(default_factory=lambda:
                        'cuda' if torch.cuda.is_available() else 'cpu')

    # ---- параметры уточнённой физики (PMCNetWithRefinedPhysics) ----
    # Число одновременно реконструируемых типов МНЧ (multi-color MPI)
    n_colors: int = 1
    # Включить релаксацию Дебая (Eq. 4–5)
    use_debye: bool = False
    # Начальное значение постоянной времени релаксации, секунды
    init_tau_seconds: float = 2.0e-6

    # ---- параметры аналитического прямого оператора ----
    # Все дефолты соответствуют BeihangUniversityData (2D Narrowband MPI
    # System, Beihang University, 2023-09-07) — см. «Описание параметров
    # файла H5.docx». Где у реальной системы анизотропия (G_x ≠ G_y,
    # A_x ≠ A_y), в config приведены значения по оси X; для оси Y указаны
    # отдельные поля `*_y` ниже. Для симметричной симуляции (как в статье
    # PMCNet) задайте равные _x/_y значения вручную.

    # Драйв-частоты (H5: acquisition.drivefield.driveFrequency)
    drive_frequency_x: float = 24510.0   # Гц
    drive_frequency_y: float = 26042.0   # Гц

    # Селективный градиент поля (H5: acquisition.gradient)
    # 0.56 T/m в A/m/m = 0.56 / μ₀ ≈ 4.456·10⁵
    gradient_strength: float = 4.456e5     # G_x в А/м/м (= 0.56 Т/м)
    gradient_strength_y: float = 8.913e5   # G_y в А/м/м (= 1.12 Т/м)

    # Амплитуда драйв-поля (H5: acquisition.drivefield.strength)
    # 4 мТ в A/m = 4·10⁻³ / μ₀ ≈ 3183
    drive_field_amplitude: float = 3183.0     # A_x в А/м (= 4 мТ)
    drive_field_amplitude_y: float = 8913.0   # A_y в А/м (= 11.2 мТ)

    # FOV сканирования (H5: calibration.fieldOfView)
    image_extent_m: float = 0.038        # L_x = L_y, метры (38 мм)

    # Чувствительность приёмной катушки (R_coil для s(r) = 1/(1+(r/R)²)).
    # Прямо в H5 нет; типично ≈ радиус FOV.
    coil_radius_m: float = 0.020

    # Свойства частиц (H5: tracer.name = Perimag; Fe₃O₄ ≈ 0.6 Т насыщения,
    # магнитный диаметр ~20 нм)
    saturation_magnetization_T: float = 0.6
    particle_diameter_nm: float = 20.0
    temperature_K: float = 300.0

    # Число точек по t в траектории FFP. По умолчанию одно «биение»
    # Лиссажу = 1/|f_x − f_y| ≈ 6.527·10⁻⁴ с, что совпадает с H5
    # `acquisition.drivefield.cycle = 6.528·10⁻⁴ с`.
    n_time_samples: int = 1024
    scan_duration_s: Optional[float] = None

    # Регуляризаторы
    lambda_tv: float = 0.0  # вес TV-штрафа на c

    # Схема дискретизации ∂M/∂t для AnalyticalForwardModel:
    #   True  → центральная разность через Conv1d с ядром [-1, 0, +1]/(2Δt)
    #           (точность O(Δt²); используется в PMCNetFinal)
    #   False → forward-difference [-1, +1]/Δt (точность O(Δt); paper-faithful
    #           схема из PMCNetPaper, наследуется PMCNetPhysicsEnhanced)
    use_central_fd: bool = True


# =============================================================================
# Численно-устойчивая функция Ланжевена
# =============================================================================


def langevin_safe(xi: torch.Tensor) -> torch.Tensor:
    """Численно-устойчивая функция Ланжевена L(ξ) = coth(ξ) − 1/ξ.

    Идея совпадает со спецификацией пользователя:
        if |ξ| < ε_lo:  L = ξ/3 − ξ³/45 + …        # ряд Тейлора
        elif |ξ| > 20:  L = sign(ξ)·(1 − 1/|ξ|)    # асимптотика
        else:           L = coth(ξ) − 1/ξ          # точное значение

    Технические уточнения для устойчивости в float32:

      • Tейлоровская ветвь использует пять членов
            L(ξ) ≈ ξ/3 − ξ³/45 + 2ξ⁵/945 − ξ⁷/4725 + 2ξ⁹/93555,
        а не один. Это нужно потому, что «точная» формула coth(ξ) − 1/ξ
        при малых ξ страдает катастрофическим сокращением: например, при
        ξ = 10⁻³ оба слагаемых имеют порядок 10³, а их разность — 3·10⁻⁴,
        что в float32 теряет ~6 значащих цифр. Чтобы Taylor оставался
        точнее float32-floor (~10⁻⁷), границу зоны расширяем до |ξ| < 0.5:
        пятичленный ряд даёт там ошибку ≲ 10⁻⁸, что лучше точной формы.

      • Точная ветвь записана через устойчивое
            coth(ξ) = sign(ξ)·(1 + e^{−2|ξ|}) / (1 − e^{−2|ξ|}),
        чтобы избежать переполнения при |ξ| ≈ 20.

      • Всё реализовано через `torch.where`, чтобы autograd видел один
        дифференцируемый граф (булевы маски с in-place присваиванием в
        исходной реализации PMCNet ломали backprop и давали NaN).

      • Защитные `clamp_min` стоят только в неактивных ветвях
        `torch.where`, не влияя на корректные градиенты в активных.

    Пользовательская спецификация (`|ξ| < 1e-6 → ξ/3`) — это вырожденный
    предельный случай этой формулы при ξ → 0. В арифметике с бесконечной
    точностью её хватило бы, но float32 на стыке 1e-6 терял бы данные
    из-за сокращения; пятичленный ряд + порог 0.5 решают эту проблему
    без потери смысла спецификации (ξ/3 — это первый и доминирующий член).
    """
    abs_xi = xi.abs()
    abs_xi_safe = abs_xi.clamp_min(1e-12)

    # Tейлоровская ветвь: 5 членов, точна до ~10⁻⁸ для |ξ| < 0.5
    xi2 = xi.pow(2)
    small = (xi / 3.0) * (
        1.0 - xi2 / 15.0 + (2.0 / 315.0) * xi2.pow(2)
        - xi2.pow(3) / 1575.0 + (2.0 / 31185.0) * xi2.pow(4)
    )

    # |ξ| > 20: асимптотика, симметризованная по знаку
    # (L — нечётная функция)
    large = torch.sign(xi) * (1.0 - 1.0 / abs_xi_safe)

    # Промежуточная ветвь: устойчивая coth − 1/ξ
    exp_neg = torch.exp(-2.0 * abs_xi_safe)
    coth = torch.sign(xi) * (1.0 + exp_neg) / (1.0 - exp_neg).clamp_min(1e-12)
    xi_safe = torch.where(abs_xi > 1e-12, xi, torch.ones_like(xi))
    mid = coth - 1.0 / xi_safe

    return torch.where(
        abs_xi < 0.5, small,
        torch.where(abs_xi > 20.0, large, mid),
    )


# =============================================================================
# Релаксация Дебая (Eq. 4–5)
# =============================================================================


class DebyeRelaxationFilter(nn.Module):
    """Первое-порядковая модель релаксации Дебая.

    Дифференциальная форма (Eq. 4 статьи):
        τ · dM_D(r,t)/dt = −M_D(r,t) + M(r,t)

    Решение в виде свёртки (Eq. 5):
        M_D(r,t) = M(r,t) ⊛ r(t),   r(t) = (1/τ) · exp(−t/τ) · u_step(t)

    Эквивалентное частотное представление:
        H(f; τ) = 1 / (1 + j·2π·f·τ)

    Особенности реализации:
      • τ — обучаемый параметр (`nn.Parameter`), параметризуется через
        softplus, чтобы гарантировать положительность; градиент течёт
        естественно через autograd.
      • Для multi-color MPI хранится массив τ_k длиной n_colors.
      • Поддерживаются обе формы применения: временная (FFT-свёртка с
        нулевым допаддингом для линейной свёртки) и частотная (умножение
        на H(f) — используется в PMCNetWithRefinedPhysics, где сигнал
        задан в частотной области через системную матрицу).
    """

    def __init__(self, n_colors: int = 1, init_tau: float = 2.0e-6):
        super().__init__()
        self.n_colors = n_colors
        init_raw = self._tau_to_raw(init_tau)
        self.raw_tau = nn.Parameter(
            torch.full((n_colors,), init_raw, dtype=torch.float32)
        )

    @staticmethod
    def _tau_to_raw(tau: float) -> float:
        """Инверсия softplus: τ → raw, т.ч. softplus(raw) = τ.

        Для крупных τ (≥ 80) softplus(x) ≈ x — используем как есть, чтобы
        избежать переполнения в expm1.
        """
        if tau >= 80.0:
            return tau
        return math.log(math.expm1(tau))

    @property
    def tau_seconds(self) -> torch.Tensor:
        return F.softplus(self.raw_tau).clamp_min(1e-12)

    # ---- частотный отклик H(f; τ) = 1 / (1 + j·2π·f·τ) ----
    def freq_response(self, freqs_hz: torch.Tensor,
                      color_idx: int = 0) -> torch.Tensor:
        tau = self.tau_seconds[color_idx]
        omega_tau = 2.0 * math.pi * freqs_hz * tau
        # 1 / (1 + j·ωτ) = (1 − j·ωτ) / (1 + (ωτ)²)
        denom = 1.0 + omega_tau.pow(2)
        real = 1.0 / denom
        imag = -omega_tau / denom
        return torch.complex(real, imag)

    # ---- временное ядро r(t) = (1/τ)·e^{−t/τ}·u(t) ----
    def time_kernel(self, t: torch.Tensor, color_idx: int = 0) -> torch.Tensor:
        tau = self.tau_seconds[color_idx]
        t_rel = (t - t[0]).clamp_min(0.0)
        exponent = (t_rel / tau).clamp(max=50.0)  # защита от underflow
        return (1.0 / tau) * torch.exp(-exponent)

    def apply_time(self, M: torch.Tensor, t: torch.Tensor,
                   color_idx: int = 0) -> torch.Tensor:
        """Линейная свёртка M(...,T) ⊛ r(t) через FFT с нулевым допаддингом."""
        T = M.shape[-1]
        dt = (t[1] - t[0]).abs()
        kernel = self.time_kernel(t, color_idx) * dt  # дискретный интеграл

        N = 2 * T  # размер, гарантирующий линейность свёртки
        M_pad = F.pad(M, (0, N - T))
        k_pad = F.pad(kernel, (0, N - T))
        result = torch.fft.irfft(
            torch.fft.rfft(M_pad) * torch.fft.rfft(k_pad), n=N,
        )
        return result[..., :T]


# =============================================================================
# Прямой оператор: системная матрица (заменяет интегральную Eq. 1)
# =============================================================================


class SystemMatrixForward(nn.Module):
    """Дискретный прямой оператор u = S · c.

    В статье (Eq. 1) сигнал задаётся пространственно-временным интегралом
    u(t) = −μ₀ ∫ s(r)·∂_t M(r,t)·c(r) dr. Для пайплайна, оперирующего
    готовой измеренной/калиброванной системной матрицей S (например, из
    BeihangUniversityData или OpenMPI), его дискретный аналог — просто
    S·c в частотной области гармоник.

    Концентрация c вещественная, поэтому u_real = S_real·c, u_imag = S_imag·c
    (нет смешивания через мнимую часть c).
    """

    def __init__(self, system_matrix: np.ndarray):
        super().__init__()
        S = np.asarray(system_matrix)
        if not np.iscomplexobj(S):
            S = S.astype(np.complex64)
        self.register_buffer('S_real', torch.tensor(S.real, dtype=torch.float32))
        self.register_buffer('S_imag', torch.tensor(S.imag, dtype=torch.float32))

    @property
    def M(self) -> int:
        return int(self.S_real.shape[0])

    @property
    def N(self) -> int:
        return int(self.S_real.shape[1])

    def forward(self, c_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """c_flat: (B, N) вещественные. Возвращает (u_real, u_imag), оба (B, M)."""
        u_real = c_flat @ self.S_real.T
        u_imag = c_flat @ self.S_imag.T
        return u_real, u_imag


# =============================================================================
# Аналитический прямой оператор:
#   FFP-траектория Лиссажу + радиальная чувствительность катушки +
#   намагниченность Ланжевена + интеграл u(t) = −μ₀ ∫ s(r)·c(r)·∂M/∂t dr
#
# Источник физики: LaTeX-документ «Моделирование MPI», уравнения 5–13, 16;
# параметры по умолчанию — Таб. 1 этого же документа. Идея фиксированных
# свёрток для конечных разностей заимствована у Maxwell-PCNN
# (Eq. 12–13 цитируемой статьи), что даёт автоматическую дифференцируемость
# и совместимость с GPU-конвейером без ручных циклов по сетке.
# =============================================================================


class RadialCoilSensitivity(nn.Module):
    """Радиально-зависимая чувствительность приёмной катушки.

        s(x, y) = 1 / (1 + (r / R_coil)²),   r = √(x² + y²)

    (LaTeX Eq. 16.) Заменяет константный профиль из исходной реализации
    PMCNet — это «hard constraint» на уровне модели: на краях FOV
    чувствительность плавно спадает, поэтому реконструированные
    интенсивности у границ не могут переоцениваться.
    """

    def __init__(self, image_size: Tuple[int, int], image_extent_m: float,
                 coil_radius_m: float):
        super().__init__()
        Nx, Ny = image_size
        x = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Nx)
        y = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(X.pow(2) + Y.pow(2))
        s = 1.0 / (1.0 + (r / coil_radius_m).pow(2))
        self.register_buffer('sensitivity', s)
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

    def forward(self) -> torch.Tensor:
        return self.sensitivity


class UniformCoilSensitivity(nn.Module):
    """p(r) ≡ 1 — paper-faithful профиль чувствительности катушки.

    Статья Huang 2026 (Sec. II.B, Eq. 1) определяет u(t) через интеграл
    с произвольным `p(r)`, но в симуляциях статьи (Sec. III.A) явный вид
    `p(r)` не задаётся — фактически используется константа = 1 (идеальный
    приёмник). Это нужно отдельно от `RadialCoilSensitivity`
    (s(r) = 1/(1+(r/R)²)) — последняя наше улучшение сверх статьи.

    Используется в `BasicAnalyticalForwardModel` (paper-faithful PMCNet).
    """

    def __init__(self, image_size: Tuple[int, int], image_extent_m: float):
        super().__init__()
        Nx, Ny = image_size
        x = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Nx)
        y = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        self.register_buffer('sensitivity', torch.ones_like(X))
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)

    def forward(self) -> torch.Tensor:
        return self.sensitivity


class LissajousFFPTrajectory(nn.Module):
    """Траектория FFP в виде фигуры Лиссажу.

    Точка FFP — это место, где полное поле H_total = 0, т.е.
        G_x·r_FFP_x + A_x·sin(2π·f_x·t) = 0
        ⇒ r_FFP_x(t) = − (A_x / G_x) · sin(2π·f_x·t)
    и аналогично по y. Амплитуда хода FFP равна `A / G`, а НЕ полу-FOV:
    для BeihangUniversityData с A_x = 4 мТ, G_x = 0.56 Т/м FFP по x
    раскачивается на ±7.14 мм, хотя FOV = ±19 мм (38% покрытия).

    Условие f_x ≠ f_y (в идеале — иррациональное отношение) даёт
    апериодичность траектории и плотное покрытие активной области FFP.

    Args:
        amp_x_m, amp_y_m: амплитуда FFP по каждой оси (метры). Если задан
            только `image_extent_m` (legacy-вариант), берётся `extent/2`.
    """

    def __init__(self, n_time_samples: int,
                 freq_x_hz: float, freq_y_hz: float,
                 amp_x_m: Optional[float] = None,
                 amp_y_m: Optional[float] = None,
                 image_extent_m: Optional[float] = None,
                 duration_s: Optional[float] = None):
        super().__init__()
        if amp_x_m is None or amp_y_m is None:
            if image_extent_m is None:
                raise ValueError(
                    "Передайте либо (amp_x_m, amp_y_m), либо image_extent_m"
                )
            amp_x_m = amp_x_m if amp_x_m is not None else image_extent_m / 2.0
            amp_y_m = amp_y_m if amp_y_m is not None else image_extent_m / 2.0
        if duration_s is None:
            duration_s = 1.0 / max(abs(freq_x_hz - freq_y_hz), 1.0)
        t = torch.linspace(0.0, duration_s, n_time_samples)
        r_x = amp_x_m * torch.sin(2.0 * math.pi * freq_x_hz * t)
        r_y = amp_y_m * torch.sin(2.0 * math.pi * freq_y_hz * t)
        self.register_buffer('t', t)
        self.register_buffer('r_x', r_x)
        self.register_buffer('r_y', r_y)
        self.duration_s = float(duration_s)
        self.n_time_samples = int(n_time_samples)
        self.amp_x_m = float(amp_x_m)
        self.amp_y_m = float(amp_y_m)

    @property
    def dt(self) -> torch.Tensor:
        return self.t[1] - self.t[0]

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.r_x, self.r_y


class TimeDerivativeFD(nn.Module):
    """Производная по времени через свёртку с фиксированным ядром.

    Идея взята из Maxwell-PCNN (Eq. 12–13): партиальные производные
    реализуются как обычная свёртка с не-обучаемым ядром, что:
      • делает оператор полностью совместимым с autograd «бесплатно»;
      • избегает индексирования и циклов по сетке;
      • легко переносится на GPU и батчевые тензоры.

    Используется центральная разностная схема (точность O(Δt²)) с
    реплицирующим паддингом на границах:
        ∂V/∂t [k] ≈ (V[k+1] − V[k−1]) / (2·Δt)
    что менее шумно, чем forward-difference из LaTeX Eq. 13.
    """

    def __init__(self, dt: float):
        super().__init__()
        kernel = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32) / float(dt)
        self.register_buffer('kernel', kernel.view(1, 1, 3))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """V: (..., T) → ∂V/∂t той же формы."""
        orig_shape = V.shape
        T = orig_shape[-1]
        V_flat = V.reshape(-1, 1, T)
        V_pad = F.pad(V_flat, (1, 1), mode='replicate')
        dV = F.conv1d(V_pad, self.kernel)
        return dV.view(orig_shape)


class TimeDerivativeForwardFD(nn.Module):
    """∂/∂t через forward-difference: ∂V/∂t[k] ≈ (V[k+1] − V[k]) / Δt.

    Реализует буквально theory.md Eq. derivative — простейшую схему,
    к которой по умолчанию приходит paper-PMCNet (статья не специфицирует
    схему дискретизации). Точность O(Δt), на единицу хуже центральной
    разности из `TimeDerivativeFD` (O(Δt²)) — но соответствует
    paper-faithful базовой версии.

    Реплицирующее padding в конце сохраняет длину T.
    """

    def __init__(self, dt: float):
        super().__init__()
        kernel = torch.tensor([-1.0, 1.0], dtype=torch.float32) / float(dt)
        self.register_buffer('kernel', kernel.view(1, 1, 2))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        """V: (..., T) → ∂V/∂t той же формы (forward FD, O(Δt))."""
        orig_shape = V.shape
        T = orig_shape[-1]
        V_flat = V.reshape(-1, 1, T)
        # Реплицируем последний элемент справа, чтобы выход имел длину T
        V_pad = F.pad(V_flat, (0, 1), mode='replicate')
        dV = F.conv1d(V_pad, self.kernel)
        return dV.view(orig_shape)


class LangevinMagnetization(nn.Module):
    """Адиабатическая намагниченность M(H) = m_sat · L(β·|H|) · ê_H.

    β = μ₀·m / (k_B·T)  (LaTeX Eq. 3)

    Возвращает векторные компоненты (M_x, M_y), что нужно для
    направления-зависимой чувствительности (∂M_x/∂t воспринимается
    x-катушкой и т. д.).
    """

    def __init__(self, saturation_magnetization_T: float = 0.6,
                 particle_diameter_nm: float = 30.0,
                 temperature_K: float = 300.0):
        super().__init__()
        mu0 = 4.0 * math.pi * 1e-7
        kB = 1.380649e-23
        d_m = particle_diameter_nm * 1e-9
        # Магнитный момент: m = M_sat · V_core, V_core = π·d³/6, M_sat = B_sat/μ₀
        m_moment = saturation_magnetization_T * (math.pi * d_m**3 / 6.0) / mu0
        beta = mu0 * m_moment / (kB * temperature_K)
        self.register_buffer('mu0', torch.tensor(mu0, dtype=torch.float32))
        self.register_buffer('m_moment', torch.tensor(m_moment, dtype=torch.float32))
        self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))

    def forward(self, H_x: torch.Tensor, H_y: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_mag = torch.sqrt(H_x.pow(2) + H_y.pow(2)).clamp_min(1e-30)
        xi = self.beta * H_mag
        L = langevin_safe(xi)
        scale = self.m_moment * L / H_mag
        return scale * H_x, scale * H_y


class AnalyticalForwardModel(nn.Module):
    """Полностью вычислительный прямой оператор MPI (LaTeX Eq. 1–13):

        u(t) = −μ₀ · ∫_Ω s(r) · c(r) · ∂M(r, t)/∂t dr,
        M(r, t) — Ланжевен от полного поля H(r, t) = G·(r − r_FFP(t)),
        r_FFP(t) — траектория Лиссажу,
        s(r) — радиальная чувствительность катушки.

    Векторизован полностью: внутри нет циклов Python по сетке (в исходной
    реализации `PhysicalModel.forward` был тройной цикл 128×128×1024 —
    миллионы итераций; здесь всё считается одной операцией на тензоре
    формы (B, N_x, N_y, T)).

    Возвращает два скалярных канала (u_x, u_y) — отдельно для x- и
    y-катушки, как в реальных MPI-сканерах. Без релаксации Дебая — она
    подключается отдельно через `DebyeRelaxationFilter.apply_time`.
    """

    def __init__(self, config: 'PMCNetConfig'):
        super().__init__()
        self.config = config
        self.coil = RadialCoilSensitivity(
            image_size=config.image_size,
            image_extent_m=config.image_extent_m,
            coil_radius_m=config.coil_radius_m,
        )
        # FFP-амплитуды строго из физики: A / G. Для BeihangUniversityData
        # A_x = 4 мТ, G_x = 0.56 Т/м → амплитуда ≈ 7.14 мм при FOV ±19 мм
        # (FFP покрывает не весь FOV — нужно учесть это в траектории).
        amp_x_m = config.drive_field_amplitude / config.gradient_strength
        amp_y_m = (config.drive_field_amplitude_y /
                   config.gradient_strength_y)
        self.ffp = LissajousFFPTrajectory(
            n_time_samples=config.n_time_samples,
            freq_x_hz=config.drive_frequency_x,
            freq_y_hz=config.drive_frequency_y,
            amp_x_m=amp_x_m,
            amp_y_m=amp_y_m,
            duration_s=config.scan_duration_s,
        )
        self.langevin = LangevinMagnetization(
            saturation_magnetization_T=config.saturation_magnetization_T,
            particle_diameter_nm=config.particle_diameter_nm,
            temperature_K=config.temperature_K,
        )
        # Схема производной выбирается через config.use_central_fd:
        #   True  → центральная разность (Maxwell-PCNN Eq. 12–13, точнее)
        #   False → forward-difference (paper-faithful, как в PMCNetPaper)
        dt = float(self.ffp.dt.item())
        if config.use_central_fd:
            self.ddt = TimeDerivativeFD(dt=dt)
        else:
            self.ddt = TimeDerivativeForwardFD(dt=dt)
        self.register_buffer('gradient_x',
                             torch.tensor(config.gradient_strength,
                                          dtype=torch.float32))
        self.register_buffer('gradient_y',
                             torch.tensor(config.gradient_strength_y,
                                          dtype=torch.float32))

        # Пиксельные веса для квадратурной интеграции
        Nx, Ny = config.image_size
        dx = config.image_extent_m / (Nx - 1)
        dy = config.image_extent_m / (Ny - 1)
        self.register_buffer('dA', torch.tensor(dx * dy, dtype=torch.float32))

    def _compute_magnetization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """M_x(x,y,t), M_y(x,y,t) — каждое (Nx, Ny, T).

        Анизотропный градиент: H_x = G_x·(x − r_FFP_x), H_y = G_y·(y − r_FFP_y).
        Это соответствует записи реального сканера BeihangUniversityData,
        где G_x ≠ G_y (0.56 vs 1.12 Т/м).
        """
        X = self.coil.X.unsqueeze(-1)             # (Nx, Ny, 1)
        Y = self.coil.Y.unsqueeze(-1)
        r_x = self.ffp.r_x.view(1, 1, -1)         # (1, 1, T)
        r_y = self.ffp.r_y.view(1, 1, -1)
        Hx = self.gradient_x * (X - r_x)
        Hy = self.gradient_y * (Y - r_y)
        return self.langevin(Hx, Hy)

    def forward(self, concentration: torch.Tensor) -> torch.Tensor:
        """concentration: (B, 1, Nx, Ny) → u: (B, 2, T) [x-coil, y-coil]."""
        if concentration.dim() == 3:
            concentration = concentration.unsqueeze(1)
        B = concentration.shape[0]

        Mx, My = self._compute_magnetization()  # (Nx, Ny, T)

        dMx_dt = self.ddt(Mx)
        dMy_dt = self.ddt(My)

        # Веса для интеграла: s(r) · c(r)
        s = self.coil.sensitivity                          # (Nx, Ny)
        c = concentration[:, 0]                            # (B, Nx, Ny)
        weight = (s.unsqueeze(0) * c).unsqueeze(-1)        # (B, Nx, Ny, 1)

        # Интегральная сумма (трапеции аппроксимированы прямоугольниками,
        # ошибка O((dx·dy)²) на гладких c — достаточно для двух-капельных
        # фантомов; для острых распределений можно перейти на trapezoid).
        scale = -self.langevin.mu0 * self.dA
        u_x = scale * (weight * dMx_dt.unsqueeze(0)).sum(dim=(1, 2))  # (B, T)
        u_y = scale * (weight * dMy_dt.unsqueeze(0)).sum(dim=(1, 2))

        return torch.stack([u_x, u_y], dim=1)              # (B, 2, T)


class BasicAnalyticalForwardModel(nn.Module):
    """Paper-faithful PMCNet прямой оператор (Huang 2026, Sec. II.B, Eq. 1–3).

    Реализует ровно те уравнения, что описаны в статье, без улучшений:

        u(t) = −μ₀ ∫ c(r) · p(r) · ∂M(r,t)/∂t dr            [paper Eq. 1]
        M(r,t) = c · m · (coth(αH) − 1/(αH)) · ê_H           [paper Eq. 2]
        α = μ₀·m / (k_B·T)                                   [paper Eq. 3]

    Отличия от `AnalyticalForwardModel` (наша улучшенная версия):
      • p(r) ≡ 1 (`UniformCoilSensitivity`) — статья не специфицирует
        профиль; радиальная p(r) = 1/(1+(r/R)²) — наше улучшение.
      • ∂/∂t через forward-difference (`TimeDerivativeForwardFD`) — это
        theory.md Eq. derivative буквально, как в paper. Центральная
        разность через Conv1d — наше улучшение (см. `TimeDerivativeFD`,
        в духе Maxwell-PCNN Eq. 12–13).
      • Без релаксации Дебая (paper Sec. II.B — adiabatic model);
        Debye-расширение есть в `PMCNetFinal` (paper Sec. II.C).
      • Без multi-color, без TV-штрафа.

    `langevin_safe` используется даже в базовой версии: без него float32
    даёт NaN при ξ → 0 (катастрофическое сокращение в `coth(ξ) − 1/ξ`).
    Это численная необходимость для PyTorch-реализации, не отход от
    статьи (в paper рассматривается математическая форма, без обсуждения
    численной устойчивости float32).
    """

    def __init__(self, config: 'PMCNetConfig'):
        super().__init__()
        self.config = config
        self.coil = UniformCoilSensitivity(
            image_size=config.image_size,
            image_extent_m=config.image_extent_m,
        )
        amp_x_m = config.drive_field_amplitude / config.gradient_strength
        amp_y_m = (config.drive_field_amplitude_y /
                   config.gradient_strength_y)
        self.ffp = LissajousFFPTrajectory(
            n_time_samples=config.n_time_samples,
            freq_x_hz=config.drive_frequency_x,
            freq_y_hz=config.drive_frequency_y,
            amp_x_m=amp_x_m,
            amp_y_m=amp_y_m,
            duration_s=config.scan_duration_s,
        )
        self.langevin = LangevinMagnetization(
            saturation_magnetization_T=config.saturation_magnetization_T,
            particle_diameter_nm=config.particle_diameter_nm,
            temperature_K=config.temperature_K,
        )
        # Forward FD — paper-faithful схема дискретизации производной
        self.ddt = TimeDerivativeForwardFD(dt=float(self.ffp.dt.item()))
        self.register_buffer('gradient_x',
                              torch.tensor(config.gradient_strength,
                                           dtype=torch.float32))
        self.register_buffer('gradient_y',
                              torch.tensor(config.gradient_strength_y,
                                           dtype=torch.float32))
        Nx, Ny = config.image_size
        dx = config.image_extent_m / (Nx - 1)
        dy = config.image_extent_m / (Ny - 1)
        self.register_buffer('dA',
                              torch.tensor(dx * dy, dtype=torch.float32))

    def _compute_magnetization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.coil.X.unsqueeze(-1)
        Y = self.coil.Y.unsqueeze(-1)
        r_x = self.ffp.r_x.view(1, 1, -1)
        r_y = self.ffp.r_y.view(1, 1, -1)
        Hx = self.gradient_x * (X - r_x)
        Hy = self.gradient_y * (Y - r_y)
        return self.langevin(Hx, Hy)

    def forward(self, concentration: torch.Tensor) -> torch.Tensor:
        """concentration: (B, 1, Nx, Ny) → u: (B, 2, T) [x-coil, y-coil]."""
        if concentration.dim() == 3:
            concentration = concentration.unsqueeze(1)
        B = concentration.shape[0]

        Mx, My = self._compute_magnetization()
        dMx_dt = self.ddt(Mx)
        dMy_dt = self.ddt(My)

        s = self.coil.sensitivity                          # (Nx, Ny), ≡ 1
        c = concentration[:, 0]                            # (B, Nx, Ny)
        weight = (s.unsqueeze(0) * c).unsqueeze(-1)        # (B, Nx, Ny, 1)

        scale = -self.langevin.mu0 * self.dA
        u_x = scale * (weight * dMx_dt.unsqueeze(0)).sum(dim=(1, 2))
        u_y = scale * (weight * dMy_dt.unsqueeze(0)).sum(dim=(1, 2))

        return torch.stack([u_x, u_y], dim=1)              # (B, 2, T)


# =============================================================================
# Hard-constraint декомпозиция (Maxwell-PCNN-стиль, Scheinker 2023)
# =============================================================================


class HardConstrainedSpectralForward(nn.Module):
    """Прямой оператор «жёсткой физики», встроенный в архитектуру сети.

    Этот блок заменяет линейный `SystemMatrixForward (S·c)` полной цепочкой
    фиксированных дифференцируемых физических слоёв. Все законы взяты из
    theory.md (раздел «Основное уравнение MPI»):

        H_sel(r)   = G·r                                  [Eq. selection_field]
        r_FFP(t)   = (A/G)·sin(2π·f·t)                    [Eq. ffp_trajectory]
        H(r,t)     = H_sel(r − r_FFP(t)) + H_exc(t)       [Eq. total_field]
        m(r,t)     = m_sat · L(μ₀·m_sat·|H|/(k_B·T)) · ê  [Eq. langevin]
        L(ξ)       = coth(ξ) − 1/ξ                        [Eq. langevin_func]
        ∂m/∂t      через Conv1d ядром [−1, 0, +1]/(2Δt)   [Eq. derivative]
        a(r,t)     = −μ₀ · s(r) · ∂m/∂t                   [Eq. system_function]
        s(r)       = 1 / (1 + (|r|/R_coil)²)              [Eq. coil_sensitivity]
        u(t)       = ∫_Ω a(r,t) · c(r) dr                 [Eq. main_mpi]
        U(f)       = FFT_t{u(t)}; первые M/2 гармоник на катушку.

    Аналогия с Maxwell-PCNN (Scheinker 2023, Eq. 6):
      • PCNN: B = ∇×A — фиксированная свёртка (W_curl) на выходе сети
        гарантирует ∇·B = 0 _by construction_ (Eq. 14).
      • Здесь: вся цепочка операторов на выходе U-Net гарантирует
        согласованность u(t) с физикой MPI _by construction_, без
        штрафов в loss и без предвычисленной линейной аппроксимации S.

    Отличие от `SystemMatrixForward + build_analytical_system_matrix`:
      • S·c — это _предвычисленная_ линейная матрица; физика «застывает»
        в момент сборки SM (R_coil, m_sat, f_x фиксированы при build).
      • HardConstrainedSpectralForward — _архитектурный_ блок: на каждом
        форварде физика пересчитывается через дифференцируемые слои,
        поэтому любой физический параметр потенциально доступен для
        joint blind calibration (через `nn.Parameter` с softplus).

    Математически линеен по `c` (как и S·c), но реализован как функция
    `c → u_freq`, а не как умножение на матрицу. Выход нормирован тем
    же масштабом, что и `build_analytical_system_matrix` (SM_max → 1),
    чтобы существующий формат measurement / loss работали без изменений.
    """

    def __init__(self, config: 'PMCNetConfig', n_meas_bins: int):
        super().__init__()
        if n_meas_bins % 2 != 0:
            raise ValueError(
                f"n_meas_bins ({n_meas_bins}) должно быть чётным "
                f"(по половине на каждую катушку)"
            )
        self.n_meas_bins = int(n_meas_bins)
        self.n_freq_per_coil = self.n_meas_bins // 2

        # Гарантируем nyquist-условие: T ≥ 2·n_freq_per_coil
        cfg = config
        if cfg.n_time_samples < 2 * self.n_freq_per_coil:
            cfg = PMCNetConfig(**{
                **config.__dict__,
                'n_time_samples': 2 * self.n_freq_per_coil,
            })
        self._Nx, self._Ny = cfg.image_size

        # Физический форвард: вся theory.md внутри (Eq. 1–13, 16)
        self.physics = AnalyticalForwardModel(cfg)

        # Нормировочный множитель: max |U(f)| на единичную «дельта» концентрации.
        # Совпадает с SM_max в `build_analytical_system_matrix`, поэтому
        # выход согласован по масштабу с предвычисленной нормированной SM.
        with torch.no_grad():
            Mx, My = self.physics._compute_magnetization()    # (Nx, Ny, T)
            dMx = self.physics.ddt(Mx)
            dMy = self.physics.ddt(My)
            s = self.physics.coil.sensitivity                  # (Nx, Ny)
            scale = -self.physics.langevin.mu0 * self.physics.dA
            kx = scale * s.unsqueeze(-1) * dMx                 # (Nx, Ny, T)
            ky = scale * s.unsqueeze(-1) * dMy
            T = kx.shape[-1]
            Kx_freq = torch.fft.rfft(kx.reshape(-1, T), dim=-1)
            Ky_freq = torch.fft.rfft(ky.reshape(-1, T), dim=-1)
            sm_max = torch.cat([
                Kx_freq[:, :self.n_freq_per_coil].abs().flatten(),
                Ky_freq[:, :self.n_freq_per_coil].abs().flatten(),
            ]).max().clamp_min(1e-30)
        self.register_buffer('output_scale', sm_max)

    @property
    def M(self) -> int:
        return self.n_meas_bins

    @property
    def N(self) -> int:
        return self._Nx * self._Ny

    def forward(self, c_flat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """c_flat: (B, N) → (u_real, u_imag), оба (B, M).

        Полная цепочка: c → AnalyticalForwardModel (theory.md Eq. 1–13)
        → rfft по времени → первые M/2 гармоник на каждую катушку
        → деление на output_scale.
        """
        B = c_flat.shape[0]
        if c_flat.shape[1] != self._Nx * self._Ny:
            raise ValueError(
                f"c_flat имеет {c_flat.shape[1]} пикселей, "
                f"ожидалось {self._Nx * self._Ny}"
            )
        c_img = c_flat.view(B, 1, self._Nx, self._Ny)

        # Полный физический форвард по theory.md Eq. 1–13
        u_time = self.physics(c_img)                           # (B, 2, T)

        # Преобразование в частотную область — формат measurement
        U_x = torch.fft.rfft(u_time[:, 0], dim=-1)             # (B, T//2+1)
        U_y = torch.fft.rfft(u_time[:, 1], dim=-1)
        U = torch.cat([
            U_x[:, :self.n_freq_per_coil],
            U_y[:, :self.n_freq_per_coil],
        ], dim=-1) / self.output_scale                          # (B, M) complex
        return U.real, U.imag


class BasicHardConstrainedSpectralForward(nn.Module):
    """Paper-faithful версия `HardConstrainedSpectralForward`.

    Полностью повторяет интерфейс «улучшенного» спектрального форварда
    (`c_flat → (u_real, u_imag)` в формате measurement), но внутри
    использует `BasicAnalyticalForwardModel` — paper-faithful физику
    без радиальной p(r), без центральной FD, без Debye.

    FFT и нормировка по `output_scale` идентичны улучшенной версии:
    это нужно, чтобы PMCNetPaper мог использовать ту же loss-функцию
    и тот же measurement-формат, что и остальные PMCNet-варианты.
    """

    def __init__(self, config: 'PMCNetConfig', n_meas_bins: int):
        super().__init__()
        if n_meas_bins % 2 != 0:
            raise ValueError(
                f"n_meas_bins ({n_meas_bins}) должно быть чётным "
                f"(по половине на каждую катушку)"
            )
        self.n_meas_bins = int(n_meas_bins)
        self.n_freq_per_coil = self.n_meas_bins // 2

        cfg = config
        if cfg.n_time_samples < 2 * self.n_freq_per_coil:
            cfg = PMCNetConfig(**{
                **config.__dict__,
                'n_time_samples': 2 * self.n_freq_per_coil,
            })
        self._Nx, self._Ny = cfg.image_size

        # Paper-faithful физика: BasicAnalyticalForwardModel
        self.physics = BasicAnalyticalForwardModel(cfg)

        with torch.no_grad():
            Mx, My = self.physics._compute_magnetization()
            dMx = self.physics.ddt(Mx)
            dMy = self.physics.ddt(My)
            s = self.physics.coil.sensitivity                  # ≡ 1
            scale = -self.physics.langevin.mu0 * self.physics.dA
            kx = scale * s.unsqueeze(-1) * dMx
            ky = scale * s.unsqueeze(-1) * dMy
            T = kx.shape[-1]
            Kx_freq = torch.fft.rfft(kx.reshape(-1, T), dim=-1)
            Ky_freq = torch.fft.rfft(ky.reshape(-1, T), dim=-1)
            sm_max = torch.cat([
                Kx_freq[:, :self.n_freq_per_coil].abs().flatten(),
                Ky_freq[:, :self.n_freq_per_coil].abs().flatten(),
            ]).max().clamp_min(1e-30)
        self.register_buffer('output_scale', sm_max)

    @property
    def M(self) -> int:
        return self.n_meas_bins

    @property
    def N(self) -> int:
        return self._Nx * self._Ny

    def forward(self, c_flat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = c_flat.shape[0]
        if c_flat.shape[1] != self._Nx * self._Ny:
            raise ValueError(
                f"c_flat имеет {c_flat.shape[1]} пикселей, "
                f"ожидалось {self._Nx * self._Ny}"
            )
        c_img = c_flat.view(B, 1, self._Nx, self._Ny)
        u_time = self.physics(c_img)                           # (B, 2, T)
        U_x = torch.fft.rfft(u_time[:, 0], dim=-1)
        U_y = torch.fft.rfft(u_time[:, 1], dim=-1)
        U = torch.cat([
            U_x[:, :self.n_freq_per_coil],
            U_y[:, :self.n_freq_per_coil],
        ], dim=-1) / self.output_scale
        return U.real, U.imag


# =============================================================================
# Backbone: U-Net без shallow skip-связей
# =============================================================================


def _double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class PMCNetUNet(nn.Module):
    """U-Net per Fig. 2 статьи PMCNet.

    4 downsampling-блока (MaxPool ×2) → bottleneck → 4 upsampling-блока
    (ConvTranspose ×2). Shallow skip-связи удалены, как в статье
    («to avoid the impact of shallow noise»). Sigmoid на выходе — c ∈ [0, 1].

    Чтобы поддерживать произвольные (в т.ч. нечётные) размеры изображений
    типа 51×51 (BeihangUniversityData) без рассинхронизации форм при
    pool/transposed-conv, вход интерполируется до ближайшего кратного 16,
    а выход приводится обратно к `image_size`.
    """

    def __init__(self, image_size: Tuple[int, int] = (51, 51),
                 out_channels: int = 1, base: int = 32):
        super().__init__()
        self.image_size = tuple(image_size)
        self.base = base

        self.enc1 = _double_conv(1, base)
        self.enc2 = _double_conv(base, base * 2)
        self.enc3 = _double_conv(base * 2, base * 4)
        self.enc4 = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = _double_conv(base * 8, base * 8)

        self.up4 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec4 = _double_conv(base * 4, base * 4)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec3 = _double_conv(base * 2, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec2 = _double_conv(base, base)

        out_base = max(base // 2, 8)
        self.up1 = nn.ConvTranspose2d(base, out_base, kernel_size=2, stride=2)
        self.dec1 = _double_conv(out_base, out_base)

        self.head = nn.Sequential(
            nn.Conv2d(out_base, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _round_up(n: int, base: int = 16) -> int:
        return ((n + base - 1) // base) * base

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        H, W = self.image_size
        H_in = self._round_up(H, 16)
        W_in = self._round_up(W, 16)
        if z.shape[-2:] != (H_in, W_in):
            z = F.interpolate(z, size=(H_in, W_in),
                              mode='bilinear', align_corners=False)

        x = self.enc1(z); x = self.pool(x)
        x = self.enc2(x); x = self.pool(x)
        x = self.enc3(x); x = self.pool(x)
        x = self.enc4(x); x = self.pool(x)
        x = self.bottleneck(x)
        x = self.up4(x); x = self.dec4(x)
        x = self.up3(x); x = self.dec3(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up1(x); x = self.dec1(x)
        x = self.head(x)

        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W),
                              mode='bilinear', align_corners=False)
        return x


# =============================================================================
# Базовая модель PMCNet (без релаксации)
# =============================================================================


class PMCNet(nn.Module):
    """PMCNet в виде модуля: φ_θ(z) → c, затем S·c → (u_real, u_imag).

    Соответствует уравнениям (1), (6), (7), (8) статьи в их дискретной
    форме, где интегральный прямой оператор заменён матричным.
    """

    def __init__(self, system_matrix: np.ndarray,
                 image_shape: Tuple[int, int],
                 base_channels: int = 32):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.unet = PMCNetUNet(image_size=self.image_shape,
                               out_channels=1, base=base_channels)
        self.forward_op = SystemMatrixForward(system_matrix)

    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.unet(z)                          # (B, 1, H, W)
        c_flat = c.view(c.shape[0], -1)           # (B, N)
        u_real, u_imag = self.forward_op(c_flat)  # каждое (B, M)
        return c, u_real, u_imag


# =============================================================================
# Paper-faithful PMCNet (Huang 2026, Sec. II.B + III.B)
# =============================================================================


class PMCNetWithBasicPhysics(nn.Module):
    """PMCNet строго по статье Huang 2026 (без улучшений).

    Архитектура: U-Net φ_θ(z) → концентрация c, физический форвард по
    Eq. 1–3 статьи (paper Sec. II.B) через `BasicAnalyticalForwardModel`:

      Шаг 1. Lissajous-траектория FFP по заданным f_x, f_y, A_x, A_y, G.
      Шаг 2. H_total = G·(r − r_FFP(t)) [paper Eq. (без номера в III.A)].
      Шаг 3. M(H) = m·L(α·|H|)·ê_H через Langevin (paper Eq. 2, 3).
      Шаг 4. ∂M/∂t через forward-difference (paper Eq. derivative).
      Шаг 5. u(t) = −μ₀·∫ p(r)·c(r)·∂M/∂t dr,  p(r) ≡ 1 (paper Eq. 1).
      Шаг 6. U(f) = FFT_t{u(t)} для совместимости с pipeline-форматом
                    measurement.

    Что НЕ входит (это сделано в улучшенных моделях):
      • Радиальная p(r) = 1/(1+(r/R)²) — `PMCNetPhysicsEnhanced`.
      • Центральная конечная разность через Conv1d — `PMCNetPhysicsEnhanced`.
      • Релаксация Дебая (paper Sec. II.C) — `PMCNetFinal`.
      • Multi-color (paper Sec. III) — `PMCNetFinal`.
      • TV-регуляризация — `PMCNetFinal`.

    Соответствует pseudocode в задании пользователя.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: 'PMCNetConfig'):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.unet = PMCNetUNet(image_size=self.image_shape, out_channels=1,
                               base=config.base_channels)
        self.forward_op = BasicHardConstrainedSpectralForward(
            config, n_meas_bins
        )

    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.unet(z)
        c_flat = c.view(c.shape[0], -1)
        u_real, u_imag = self.forward_op(c_flat)
        return c, u_real, u_imag


# =============================================================================
# PMCNet с hard-constraint форвардом (PCNN-стиль)
# =============================================================================


class PMCNetHardConstrained(nn.Module):
    """PMCNet, где S·c заменено цепочкой фиксированных физических слоёв.

    Single-color аналог `PMCNet`: U-Net φ_θ(z) → c, затем
    `HardConstrainedSpectralForward(c) → (u_real, u_imag)`. Loss остаётся
    L1 в частотной области (формат measurement не меняется), а вся физика
    зашита в архитектуру форварда по уравнениям theory.md (см. док-стринг
    `HardConstrainedSpectralForward`).
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: 'PMCNetConfig'):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.unet = PMCNetUNet(image_size=self.image_shape, out_channels=1,
                               base=config.base_channels)
        self.forward_op = HardConstrainedSpectralForward(config, n_meas_bins)

    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.unet(z)                          # (B, 1, H, W), σ ∈ [0,1]
        c_flat = c.view(c.shape[0], -1)
        u_real, u_imag = self.forward_op(c_flat)
        return c, u_real, u_imag


class PMCNetHardConstrainedRefined(nn.Module):
    """PMCNet с hard-constraint форвардом + Debye + multi-color.

    Multi-color аналог `PMCNetWithRefinedPhysics`: U-Net выдаёт K
    концентраций c_1,…,c_K, для каждой считается hard-constrained сигнал,
    далее каждый канал умножается на частотный отклик релаксации
    H_τ_k(f) = 1 / (1 + j·2π·f·τ_k) (см. док-стринг
    `DebyeRelaxationFilter`), и каналы складываются.

    Декомпозиция по цветам — прямой аналог декомпозиции компонент в
    Maxwell-PCNN (Scheinker 2023, Eq. 10): каждая A_k зависит только
    от соответствующей J_k, поэтому K параллельных скалярных подсетей
    компактнее, чем одна K-канальная.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: 'PMCNetConfig',
                 harmonic_frequencies_hz: Optional[np.ndarray] = None):
        super().__init__()
        self.config = config
        self.image_shape = tuple(image_shape)
        self.n_colors = config.n_colors

        self.unet = PMCNetUNet(image_size=self.image_shape,
                               out_channels=self.n_colors,
                               base=config.base_channels)
        self.forward_op = HardConstrainedSpectralForward(config, n_meas_bins)
        self.debye = DebyeRelaxationFilter(n_colors=self.n_colors,
                                            init_tau=config.init_tau_seconds)

        M = self.forward_op.M
        if harmonic_frequencies_hz is None:
            freqs = (np.arange(1, M + 1, dtype=np.float32)
                     * config.drive_frequency_x)
        else:
            freqs = np.asarray(harmonic_frequencies_hz,
                               dtype=np.float32).flatten()
            if freqs.size != M:
                raise ValueError(
                    f"harmonic_frequencies_hz должен содержать {M} элементов, "
                    f"получено {freqs.size}"
                )
        self.register_buffer('freqs_hz',
                             torch.tensor(freqs, dtype=torch.float32))

    def total_concentration(self, c: torch.Tensor) -> torch.Tensor:
        return c.sum(dim=1, keepdim=True)

    def forward(self, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.unet(z)  # (B, K, H, W)
        B = c.shape[0]
        device = c.device

        u_real_total = torch.zeros(B, self.forward_op.M, device=device)
        u_imag_total = torch.zeros(B, self.forward_op.M, device=device)

        for k in range(self.n_colors):
            c_k = c[:, k, :, :].reshape(B, -1)
            u_r, u_i = self.forward_op(c_k)
            if self.config.use_debye:
                H = self.debye.freq_response(self.freqs_hz, color_idx=k)
                Hr, Hi = H.real, H.imag
                u_r, u_i = u_r * Hr - u_i * Hi, u_r * Hi + u_i * Hr
            u_real_total = u_real_total + u_r
            u_imag_total = u_imag_total + u_i

        return c, u_real_total, u_imag_total


# =============================================================================
# Реконструкторы (data-free per-measurement optimization)
# =============================================================================


class _BaseReconstructor:
    """Общая инфраструктура для PMCNet-реконструкторов."""

    def __init__(self, network: nn.Module, config: PMCNetConfig):
        self.network = network
        self.config = config
        self.device = torch.device(config.device)
        self.network.to(self.device)
        self.loss_history: List[float] = []

    def _measurement_to_tensors(self, measurement
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        u = np.asarray(measurement).flatten()
        if not np.iscomplexobj(u):
            u = u.astype(np.complex64)
        u_real = torch.tensor(u.real, dtype=torch.float32, device=self.device)
        u_imag = torch.tensor(u.imag, dtype=torch.float32, device=self.device)
        return u_real, u_imag

    def _reset_unet_weights(self):
        for m in self.network.unet.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    @staticmethod
    def _tv_loss(c: torch.Tensor) -> torch.Tensor:
        dh = (c[..., 1:, :] - c[..., :-1, :]).abs().mean()
        dw = (c[..., :, 1:] - c[..., :, :-1]).abs().mean()
        return dh + dw


class PMCNetReconstructor(_BaseReconstructor):
    """Алгоритм 1 PMCNet: data-free оптимизация на одно измерение.

    Перед каждой реконструкцией веса U-Net реинициализируются (см. Sec. II.E
    статьи). Вход z фиксирован (детерминирован через config.seed).
    """

    def __init__(self, system_matrix: np.ndarray,
                 image_shape: Tuple[int, int],
                 config: Optional[PMCNetConfig] = None):
        cfg = config or PMCNetConfig(image_size=tuple(image_shape))
        net = PMCNet(system_matrix, image_shape, base_channels=cfg.base_channels)
        super().__init__(net, cfg)
        self.image_shape = tuple(image_shape)

    def reconstruct(self, measurement, n_iterations: Optional[int] = None,
                    verbose: bool = False, reset: bool = True) -> np.ndarray:
        n_iter = n_iterations or self.config.n_iterations
        u_real, u_imag = self._measurement_to_tensors(measurement)
        M = self.network.forward_op.M
        if u_real.numel() != M:
            raise ValueError(
                f"Измерение содержит {u_real.numel()} бинов, ожидалось {M}."
            )

        if reset:
            self._reset_unet_weights()

        torch.manual_seed(self.config.seed)
        z = torch.randn(1, 1, *self.image_shape, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.learning_rate)
        self.loss_history.clear()

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc='PMCNet')

        for it in iterator:
            optimizer.zero_grad()
            c, ur, ui = self.network(z)
            loss = (ur[0] - u_real).abs().mean() + (ui[0] - u_imag).abs().mean()
            if self.config.lambda_tv > 0:
                loss = loss + self.config.lambda_tv * self._tv_loss(c)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (it % max(1, n_iter // 20) == 0):
                iterator.set_postfix({'loss': f'{loss.item():.4e}'})

        self.network.eval()
        with torch.no_grad():
            c, _, _ = self.network(z)
        self.network.train()
        return c[0, 0].detach().cpu().numpy()


class PMCNetPaperReconstructor(_BaseReconstructor):
    """Реконструктор для `PMCNetPaper` — алгоритм 1 статьи в чистом виде.

    Совпадает с pseudocode пользователя:

        Init θ ~ random, z ~ N(0, 1) (зафиксирован seed-ом).
        For iter = 1 .. n_iterations:
            ĉ = φ_θ(z)
            û = P(ĉ)                                # paper Eq. 1–3
            L = ‖û − u_meas‖₁                       # paper Eq. 7, 8
            backward, Adam.step()                   # lr = 1e-3
        return ĉ

    По умолчанию `n_iterations = 20000`, `lr = 1e-3`, Adam (paper Sec. III.B).
    """

    def __init__(self, image_shape: Tuple[int, int], n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None):
        cfg = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{**cfg.__dict__,
                              'image_size': tuple(image_shape)})
        net = PMCNetWithBasicPhysics(image_shape, n_meas_bins, cfg)
        super().__init__(net, cfg)
        self.image_shape = tuple(image_shape)

    def reconstruct(self, measurement, n_iterations: Optional[int] = None,
                    verbose: bool = False, reset: bool = True) -> np.ndarray:
        n_iter = n_iterations or self.config.n_iterations
        u_real, u_imag = self._measurement_to_tensors(measurement)
        M = self.network.forward_op.M
        if u_real.numel() != M:
            raise ValueError(
                f"Измерение содержит {u_real.numel()} бинов, ожидалось {M}."
            )

        if reset:
            self._reset_unet_weights()

        torch.manual_seed(self.config.seed)
        z = torch.randn(1, 1, *self.image_shape, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.learning_rate)
        self.loss_history.clear()

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc='PMCNet-Paper')

        for it in iterator:
            optimizer.zero_grad()
            c, ur, ui = self.network(z)
            loss = (ur[0] - u_real).abs().mean() + (ui[0] - u_imag).abs().mean()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (it % max(1, n_iter // 20) == 0):
                iterator.set_postfix({'loss': f'{loss.item():.4e}'})

        self.network.eval()
        with torch.no_grad():
            c, _, _ = self.network(z)
        self.network.train()
        return c[0, 0].detach().cpu().numpy()


class PMCNetHardConstrainedReconstructor(_BaseReconstructor):
    """Реконструктор для PMCNet с hard-constraint форвардом.

    API идентично `PMCNetReconstructor`: `reconstruct(measurement)` принимает
    комплексный вектор гармоник, оптимизирует L1 в частотной области.
    Под капотом — не S·c, а полный физический форвард
    (см. `HardConstrainedSpectralForward`).
    """

    def __init__(self, image_shape: Tuple[int, int], n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None):
        cfg = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{**cfg.__dict__,
                              'image_size': tuple(image_shape)})
        net = PMCNetHardConstrained(image_shape, n_meas_bins, cfg)
        super().__init__(net, cfg)
        self.image_shape = tuple(image_shape)

    def reconstruct(self, measurement, n_iterations: Optional[int] = None,
                    verbose: bool = False, reset: bool = True) -> np.ndarray:
        n_iter = n_iterations or self.config.n_iterations
        u_real, u_imag = self._measurement_to_tensors(measurement)
        M = self.network.forward_op.M
        if u_real.numel() != M:
            raise ValueError(
                f"Измерение содержит {u_real.numel()} бинов, ожидалось {M}."
            )

        if reset:
            self._reset_unet_weights()

        torch.manual_seed(self.config.seed)
        z = torch.randn(1, 1, *self.image_shape, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.learning_rate)
        self.loss_history.clear()

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc='PMCNet-HardConstrained')

        for it in iterator:
            optimizer.zero_grad()
            c, ur, ui = self.network(z)
            loss = (ur[0] - u_real).abs().mean() + (ui[0] - u_imag).abs().mean()
            if self.config.lambda_tv > 0:
                loss = loss + self.config.lambda_tv * self._tv_loss(c)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (it % max(1, n_iter // 20) == 0):
                iterator.set_postfix({'loss': f'{loss.item():.4e}'})

        self.network.eval()
        with torch.no_grad():
            c, _, _ = self.network(z)
        self.network.train()
        return c[0, 0].detach().cpu().numpy()


class PMCNetHardConstrainedRefinedReconstructor(_BaseReconstructor):
    """Реконструктор для `PMCNetHardConstrainedRefined`.

    Возвращает кортеж (концентрация, оценённые τ_k), как
    `PMCNetRefinedReconstructor`. Hard-constraint форвард + Debye в
    частотной области (математически эквивалентен временной свёртке,
    но дешевле, т.к. FFT уже посчитан в выходе форварда).
    """

    def __init__(self, image_shape: Tuple[int, int], n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 harmonic_frequencies_hz: Optional[np.ndarray] = None):
        cfg = config or PMCNetConfig(image_size=tuple(image_shape),
                                      use_debye=True)
        cfg = PMCNetConfig(**{**cfg.__dict__,
                              'image_size': tuple(image_shape)})
        net = PMCNetHardConstrainedRefined(
            image_shape, n_meas_bins, cfg,
            harmonic_frequencies_hz=harmonic_frequencies_hz,
        )
        super().__init__(net, cfg)
        self.image_shape = tuple(image_shape)

    def reconstruct(self, measurement, n_iterations: Optional[int] = None,
                    verbose: bool = False, reset: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
        n_iter = n_iterations or self.config.n_iterations
        u_real, u_imag = self._measurement_to_tensors(measurement)
        M = self.network.forward_op.M
        if u_real.numel() != M:
            raise ValueError(
                f"Измерение содержит {u_real.numel()} бинов, ожидалось {M}."
            )

        if reset:
            self._reset_unet_weights()
            init_raw = DebyeRelaxationFilter._tau_to_raw(
                self.config.init_tau_seconds
            )
            with torch.no_grad():
                self.network.debye.raw_tau.fill_(init_raw)

        torch.manual_seed(self.config.seed)
        z = torch.randn(1, 1, *self.image_shape, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.learning_rate)
        self.loss_history.clear()

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc='PMCNet-HardConstrained-Refined')

        for it in iterator:
            optimizer.zero_grad()
            c, ur, ui = self.network(z)
            loss = (ur[0] - u_real).abs().mean() + (ui[0] - u_imag).abs().mean()
            if self.config.lambda_tv > 0:
                loss = loss + self.config.lambda_tv * self._tv_loss(c)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if verbose and (it % max(1, n_iter // 20) == 0):
                tau_us = (self.network.debye.tau_seconds * 1e6
                          ).detach().cpu().numpy()
                iterator.set_postfix({
                    'loss': f'{loss.item():.4e}',
                    'tau_us': np.array2string(tau_us, precision=3),
                })

        self.network.eval()
        with torch.no_grad():
            c, _, _ = self.network(z)
            taus = self.network.debye.tau_seconds.detach().cpu().numpy()
        self.network.train()

        c_np = c[0].cpu().numpy()
        if self.config.n_colors == 1:
            c_np = c_np[0]
        return c_np, taus



# =============================================================================
# Сборка синтетической системной матрицы из аналитической физики
# =============================================================================


def build_analytical_system_matrix(image_shape: Tuple[int, int],
                                   n_meas_bins: int,
                                   config: Optional[PMCNetConfig] = None
                                   ) -> np.ndarray:
    """Построить аналитическую системную матрицу того же формата, что и
    измеренная (из калибровки сканера).

    Идея: каждая колонка n матрицы S — это частотный отклик прямого
    оператора на дельта-распределение концентрации в пикселе n. Для
    аналитической модели это вычисляется в закрытой форме:

        u_n(t) = −μ₀ · dA · s(r_n) · ∂M(r_n, t)/∂t

    Поскольку M(r, t) общая для всех пикселей, всё семейство колонок
    получается одним батчевым FFT — без циклов по пикселям, за O(N·T·logT).

    Возвращаемая матрица имеет форму `(n_meas_bins, Nx·Ny)` и совпадает
    по схеме раскладки с измеренной SM из BeihangUniversityData
    (см. `main.py`: `SM = S.reshape(2 · n_freq, n_pixels)`), что
    обеспечивает прямую взаимозаменяемость в пайплайне.

    Args:
        image_shape: (Nx, Ny).
        n_meas_bins: общее число строк (= 2 · число гармоник на катушку).
        config:     PMCNetConfig с физическими параметрами. Параметр
                    `n_time_samples` при необходимости поднимается так,
                    чтобы rfft давал ≥ n_meas_bins/2 частотных бинов.
    """
    base_cfg = config or PMCNetConfig(image_size=tuple(image_shape))
    n_freq_per_coil = n_meas_bins // 2
    T_samples = max(base_cfg.n_time_samples, 2 * n_freq_per_coil)
    cfg = PMCNetConfig(**{
        **base_cfg.__dict__,
        'image_size': tuple(image_shape),
        'n_time_samples': T_samples,
    })

    physics = AnalyticalForwardModel(cfg)
    physics.eval()

    with torch.no_grad():
        Mx, My = physics._compute_magnetization()   # (Nx, Ny, T)
        dMx = physics.ddt(Mx)
        dMy = physics.ddt(My)
        s = physics.coil.sensitivity                # (Nx, Ny)
        scale = -physics.langevin.mu0 * physics.dA

        # u_n(t) для каждого пикселя как столбец: shape (Nx, Ny, T)
        ux = scale * s.unsqueeze(-1) * dMx
        uy = scale * s.unsqueeze(-1) * dMy

        Nx, Ny = image_shape
        N = Nx * Ny
        ux_flat = ux.reshape(N, T_samples)          # (N, T)
        uy_flat = uy.reshape(N, T_samples)

        Ux = torch.fft.rfft(ux_flat, dim=-1)        # (N, T//2 + 1) complex
        Uy = torch.fft.rfft(uy_flat, dim=-1)

        # Берём n_freq_per_coil низших гармоник на каждую катушку
        SM = torch.cat([
            Ux[:, :n_freq_per_coil].T,              # (n_freq, N)
            Uy[:, :n_freq_per_coil].T,
        ], dim=0)                                    # (n_meas_bins, N)

        # Нормировка до единичного максимума: физическая SM в СИ-единицах
        # имеет порядок 10⁻²³ (μ₀·m_moment·dA), что в float32 ниже шума и
        # не сопоставимо по масштабу с реальными измерениями. Абсолютная
        # амплитуда SM произвольна (концентрация на выходе всё равно
        # нормируется на [0, 1]); что физически значимо — это паттерн
        # столбцов, который сохраняется при нормировке.
        SM_max = SM.abs().max().clamp_min(1e-30)
        SM = SM / SM_max

    return SM.cpu().numpy().astype(np.complex64)


# =============================================================================
# Три названных варианта PMCNet — pipeline-ready
# =============================================================================
#
# Все три принимают одинаковый формат измерений (комплексный вектор
# гармоник из частотной области, как в `MPIReconstructionComparator`),
# что позволяет напрямую сравнивать их в одной таблице.
#
#   1. PMCNetStandard         — статья Huang et al. (2026) в чистом виде,
#                               прямой оператор = ИЗМЕРЕННАЯ системная
#                               матрица из калибровки сканера.
#
#   2. PMCNetPhysicsEnhanced  — та же сеть, тот же loss, но прямой
#                               оператор построен из АНАЛИТИЧЕСКОЙ
#                               физики (стабильный Ланжевен, радиальная
#                               чувствительность катушки, траектория
#                               Лиссажу, центральная разность через
#                               фиксированную свёртку). Никаких
#                               NN-улучшений: single color, без Дебая,
#                               без TV.
#
#   3. PMCNetFinal            — аналитическая физика + полный набор
#                               NN-оптимизаций: релаксация Дебая с
#                               обучаемой τ_k, multi-color, TV-штраф,
#                               hard constraints by construction
#                               (sigmoid + радиальная s(r) встроены
#                               архитектурно в духе Maxwell-PCNN).
# =============================================================================


class PMCNetStandard(PMCNetReconstructor):
    """PMCNet-baseline: φ_θ(z) → c → S·c, где S — _измеренная_ SM сканера.

    Это _не_ paper-faithful версия. В статье Huang 2026 (Sec. I, IV)
    PMCNet специально позиционируется как метод _без_ системной матрицы:
        "without the need for a system matrix and network training"
        "without the laborious system matrix calibration"
    SM-метод в статье — это _baseline_ для сравнения, а не сам PMCNet.
    Paper-faithful версия — это `PMCNetPaper` (см. ниже).

    Эта модель в нашем пайплайне сохраняется как _независимый baseline_,
    чтобы параллельно с paper-faithful PMCNet видеть метрики SM-подхода
    на тех же фантомах. Прямой оператор — `S_measured · c` через
    `SystemMatrixForward`, loss = L1, оптимизатор — Adam.
    """


class PMCNetPaper(PMCNetPaperReconstructor):
    """Paper-faithful PMCNet — точно по статье Huang 2026.

    Прямой оператор: `BasicAnalyticalForwardModel` (paper Eq. 1–3):
      • Lissajous-траектория FFP по f_x, f_y, A_x/G_x, A_y/G_y;
      • H_total(r,t) = G·(r − r_FFP(t));
      • Langevin (адиабатический, без релаксации Дебая);
      • p(r) ≡ 1 (uniform — paper не специфицирует профиль катушки);
      • ∂M/∂t через forward-difference (theory.md Eq. derivative буквально);
      • u(t) = −μ₀·∫ p(r)·c(r)·∂M/∂t dr.

    Без улучшений: без радиальной p(r), без центральной FD через Conv1d,
    без Дебая, без multi-color, без TV. Только то, что описано в paper
    Sec. II.B + III.B.

    Алгоритм (paper Sec. III.B):
      Init: θ random, z ~ N(0,1) (фикс. seed)
      Loop n_iterations=20000 (default):
        ĉ = φ_θ(z); û = P(ĉ); L = ‖û − u_meas‖₁; backward; Adam.step()
      Return ĉ

    Это вариант, к которому нужно подавать _аналитически_ сгенерированный
    u_meas (см. `synthesize_measurements_analytical` в pipeline.py) —
    paper в симуляциях так же делает: "In the simulation, u_meas and u
    use the same physical model for calculation" (Sec. III.A).
    """


class PMCNetPhysicsEnhanced(PMCNetHardConstrainedReconstructor):
    """PMCNet-Paper + два физических улучшения, не связанных с обучением.

    База — `PMCNetPaper` (paper-faithful: Eq. 1–3 статьи через
    `BasicAnalyticalForwardModel`). Сверх неё добавлены ровно две
    физические корректировки, не использующие методы машинного обучения:

      ▸ Зависимость чувствительности катушки от расстояния до центра:
        p(r) = 1 / (1 + (|r|/R_coil)²)  (RadialCoilSensitivity).
        Ближе к краю FOV сигнал слабее, как в реальной приёмной катушке
        с конечным радиусом. Paper-версия неявно использует p(r) ≡ 1,
        что приводит к переоценке концентрации на периферии.

      ▸ Адаптация функции Ланжевена для числовой стабильности
        (`langevin_safe`): пятичленный ряд Тейлора для |ξ| < 0.5,
        устойчивая форма coth для |ξ| > 0.5 и асимптотика 1 − 1/|ξ|
        для |ξ| > 20. Прямой расчёт coth(ξ) − 1/ξ в float32 даёт
        катастрофическое сокращение при ξ → 0 и регулярно сваливается
        в NaN при типичных параметрах поля — наша версия закрывает
        этот разрыв.

    Что НЕ изменилось относительно paper-PMCNet (`PMCNetPaper`):
      ▸ Архитектура сети (U-Net без shallow skip), вход z, loss L1, Adam.
      ▸ Forward-difference для ∂M/∂t (центральная разность через Conv1d
        перенесена в `PMCNetFinal` как нейросетевое улучшение).
      ▸ Single color, без Debye, без TV (это всё в `PMCNetFinal`).

    Назначение варианта — изолированно оценить вклад «более реалистичной
    физики» при той же NN-стороне, что и в paper-варианте.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            'n_colors': 1,
            'use_debye': False,
            'lambda_tv': 0.0,
            # paper-faithful схема производной — центральная FD перенесена
            # в PMCNetFinal как нейросетевое улучшение (см. требование
            # пользователя)
            'use_central_fd': False,
        })
        super().__init__(image_shape, n_meas_bins, config=cfg)


class PMCNetFinal(PMCNetHardConstrainedRefinedReconstructor):
    """PMCNetPhysicsEnhanced + три нейросетевых/ML-улучшения.

    База — `PMCNetPhysicsEnhanced` (paper-PMCNet + радиальная p +
    langevin_safe). Сверх неё в финальной версии добавлены три метода,
    расширяющие именно нейросетевую часть алгоритма машинного обучения:

      ▸ TV-регуляризация Σ λ·∑|∇c| на выходе сети — подавляет
        высокочастотные шумовые артефакты на реконструированном
        изображении. Default λ_TV = 1e-3 (компромисс между сглаживанием
        и сохранением границ объектов).

      ▸ Точное вычисление производной намагниченности через специальный
        сверточный слой: ∂M/∂t реализуется как Conv1d с фиксированным
        ядром [−1, 0, +1]/(2Δt) — центральная разность, точность O(Δt²)
        вместо O(Δt) у forward-FD из PhysicsEnhanced (Maxwell-PCNN
        Eq. 12–13, `TimeDerivativeFD`). Фиксированное ядро делает
        оператор полностью совместимым с autograd «бесплатно» и
        переносимым на GPU без ручных циклов.

      ▸ Возможность учитывать инерционность частиц через обучаемый
        параметр времени релаксации τ (релаксация Дебая, paper Eq. 4–5):
        применяется как частотный фильтр H_τ(f) = 1/(1 + j·2π·f·τ) к
        выходу форварда, эквивалентный временной свёртке с
        r(t) = (1/τ)·exp(−t/τ). τ обучается совместно с весами сети
        через softplus-параметризацию (положительность _by construction_),
        как в paper Sec. III.B: «we did not provide the magnitude of the
        relaxation time constant directly, but instead estimated it
        through the gradient descent algorithm».

    Дополнительно по умолчанию включён multi-color режим (paper Sec. III):
    U-Net выводит K концентраций c_1, ..., c_K (по типу МНЧ), сигнал —
    U(f) = Σ_k H_τ_k(f) · ForwardOp(c_k). Отключается через `n_colors=1`.

    Hard constraints _by construction_ (PCNN-философия Scheinker 2023):
      – sigmoid в head U-Net → c ∈ [0, 1] архитектурно;
      – softplus(raw_τ)       → τ_k > 0 архитектурно;
      – радиальная p(r) (из `PhysicsEnhanced`) — фиксированный буфер,
        не штраф в loss;
      – Conv1d-ядро для ∂/∂t — фиксированный буфер, не штраф.

    Иерархия: PMCNetPaper (paper-faithful) ⊂ PhysicsEnhanced (+radial p,
    +langevin_safe) ⊂ Final (+TV, +central FD, +обучаемая τ-Debye).
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 n_colors: int = 2):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            'n_colors': n_colors,
            'use_debye': True,
            'lambda_tv': max(base.lambda_tv, 1e-3),
            # Точное ∂M/∂t через сверточный слой — нейросетевое улучшение
            # из спецификации пользователя
            'use_central_fd': True,
        })
        super().__init__(image_shape, n_meas_bins, config=cfg)


__all__ = [
    'PMCNetConfig',
    'langevin_safe',
    # Paper-faithful физические примитивы (Huang 2026, Sec. II.B)
    'UniformCoilSensitivity',
    'TimeDerivativeForwardFD',
    'BasicAnalyticalForwardModel',
    'BasicHardConstrainedSpectralForward',
    # Улучшенные физические модули (наши добавки + Maxwell-PCNN)
    'RadialCoilSensitivity',
    'LissajousFFPTrajectory',
    'TimeDerivativeFD',
    'LangevinMagnetization',
    'AnalyticalForwardModel',
    'HardConstrainedSpectralForward',
    # Дебай и SM-форвард
    'DebyeRelaxationFilter',
    'SystemMatrixForward',
    'build_analytical_system_matrix',
    # U-Net и низкоуровневые модели
    'PMCNetUNet',
    'PMCNet',
    'PMCNetWithBasicPhysics',
    'PMCNetHardConstrained',
    'PMCNetHardConstrainedRefined',
    # Низкоуровневые реконструкторы
    'PMCNetReconstructor',
    'PMCNetPaperReconstructor',
    'PMCNetHardConstrainedReconstructor',
    'PMCNetHardConstrainedRefinedReconstructor',
    # Четыре названных варианта (pipeline-ready)
    'PMCNetStandard',           # SM из калибровки (baseline)
    'PMCNetPaper',              # paper-faithful (Huang 2026, Eq. 1-3)
    'PMCNetPhysicsEnhanced',    # paper + физические улучшения
    'PMCNetFinal',              # PhysicsEnhanced + Debye + multi-color + TV
]
