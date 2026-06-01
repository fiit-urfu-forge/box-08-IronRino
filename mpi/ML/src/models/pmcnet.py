"""PMCNet — нейросеть для реконструкции изображений MPI с физической моделью.

## Что такое MPI и в чём задача

Магнитно-частичная томография (MPI) — медицинский метод визуализации
распределения суперпарамагнитных наночастиц (SPION). Сканер прикладывает
переменное магнитное поле; намагниченность частиц нелинейна, поэтому
индуцированный в приёмной катушке сигнал u(t) содержит гармоники драйв-
частоты, кодирующие пространственное распределение концентрации частиц c(r).

Прямая задача (математическая модель):

    u(t) = −μ₀ · ∫ s(r) · ∂M(r,t)/∂t · c(r) dr            (интегральная форма)

или после дискретизации и преобразования Фурье:

    u_freq = A · c + noise                                (матричная форма)

где A — комплексная системная матрица. Обратная задача (получить c из u) —
некорректная: малые возмущения u приводят к большим изменениям c.

## Идея PMCNet

Нейросеть учит c напрямую через физический форвард, без обучающей выборки:

    Init θ ~ random, z ~ N(0,1) (фиксированный латент)
    For iteration = 1 to N (N ~ 20000):
        ĉ = φ_θ(z)              # U-Net генерирует концентрацию из латента z
        û = P(ĉ)                # физический форвард: ĉ → сигнал
        L = ||û − u_meas||₁     # L1-невязка с реальным измерением
        L.backward(); θ ← θ − η·∇L  # Adam-шаг
    return ĉ

Преимущества: data-free (одно измерение → одна реконструкция, без
тренировочного датасета); физика встроена в loss → реконструкция
физически согласована.

## Четыре варианта в этом модуле

  PMCNetStandard          — SM-baseline (НЕ paper-faithful). Прямой оператор
                            P(c) = S_measured · c, где S_measured —
                            калибровочная матрица сканера, измеренная
                            заранее на сетке дельта-фантомов. Сохранён
                            для сравнения. Paper-вариант избегает SM.

  PMCNetPaper             — paper-faithful. Физика вычисляется явно из
                            физических параметров (драйв-частоты,
                            градиент, амплитуда поля, диаметр частиц,
                            температура) — без измеренной SM. Соответствует
                            идее статьи: «параметры сканера известны,
                            калибровка не нужна».

  PMCNetPhysicsEnhanced   — paper + два физических улучшения:
                            * радиальная чувствительность катушки
                              p(r) = 1/(1+(r/R)²) вместо uniform p(r)≡1
                              (в реальности приёмная катушка ловит сигнал
                              слабее на краях FOV);
                            * langevin_safe — численно-устойчивая функция
                              Ланжевена с разложением Тейлора при малых
                              аргументах (защита от NaN в float32).

  PMCNetSoftConstrained   — PhysicsEnhanced + soft-constraints из
                            Scheinker 2023 (PCNN):
                            * λ_grad · ‖∇c‖₂² — L2-штраф на пространственный
                              градиент c через central-FD ядро;
                            * частотно-взвешенный L1 на u — компенсирует
                              падение амплитуды гармоник как ~1/k.

  PMCNetDebye             — PhysicsEnhanced + ОДНО ML-улучшение:
                            релаксация Дебая (paper Eq. 4-5) как
                            частотный фильтр H_τ(f) = 1/(1+j·2π·f·τ)
                            с обучаемой τ через softplus(raw_τ).

  PMCNetCentralFD         — PhysicsEnhanced + ОДНО физ. улучшение:
                            центральная разность ∂M/∂t через Conv1d
                            с ядром [-1,0,+1]/(2Δt), точность O(Δt²)
                            (Maxwell-PCNN Eq. 12-13).

## Иерархия классов

  Физические примитивы (paper-faithful):
    UniformCoilSensitivity         — p(r) ≡ 1
    TimeDerivativeForwardFD        — forward FD, O(Δt)
    BasicAnalyticalForwardModel    — paper-faithful u(t) = -μ₀·∫s·∂M/∂t·c dr
    BasicHardConstrainedSpectralForward — FFT/DFT обёртка

  Физические примитивы (улучшенные):
    RadialCoilSensitivity          — p(r) = 1/(1+(r/R)²), R может обучаться
    TimeDerivativeFD               — central FD через Conv1d, O(Δt²)
    AnalyticalForwardModel         — то же что Basic + улучшения
    HardConstrainedSpectralForward — FFT обёртка

  Дополнительные:
    LangevinMagnetization, LissajousFFPTrajectory,
    DebyeRelaxationFilter, SystemMatrixForward

  Высокоуровневые nn.Module:
    PMCNet, PMCNetWithBasicPhysics, PMCNetHardConstrained,
    PMCNetHardConstrainedRefined, PMCNetUNet

  Reconstructor'ы (data-free per-measurement optimization):
    PMCNetReconstructor (base for Standard),
    PMCNetPaperReconstructor (base for Paper),
    PMCNetHardConstrainedReconstructor (base for PhysicsEnhanced,
        Soft, CentralFD — все используют HardConstrainedSpectralForward),
    PMCNetHardConstrainedRefinedReconstructor (base for Debye —
        тот же forward + DebyeRelaxationFilter).

  Pipeline-ready user-facing:
    PMCNetStandard, PMCNetPaper, PMCNetPhysicsEnhanced [= база Phys],
    PMCNetSoftConstrained, PMCNetDebye, PMCNetCentralFD.
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
    """Гиперпараметры PMCNet, организованные по тематическим блокам.

    Каждое улучшение модели представлено отдельным флагом, чтобы можно
    было независимо включать и отключать его для ablation-исследования.
    Группы:

      1. БАЗОВЫЕ настройки сети и оптимизатора;
      2. ФИЗИКА: improved Langevin / Debye / coil sensitivity / FD-схема;
      3. АРХИТЕКТУРА ML: multi-color, learnable R_coil, output activation;
      4. SOFT-CONSTRAINTS (Scheinker 2023): grad penalty, freq weighting,
         TV-loss — auxiliary loss-термы, не меняющие forward-операторе;
      5. ПАРАМЕТРЫ СКАНЕРА: драйв-частоты, градиент, FOV, частицы и т.д.

    Соответствие preset-моделей флагам:

      | Preset                | physics | soft         | ML extras    |
      |-----------------------|---------|--------------|--------------|
      | PMCNet-SM (Standard)  | —       | —            | —            |
      | PMCNet-Paper          | base    | —            | —            |
      | PMCNet-Phys           | stable+ | —            | —            |
      | PMCNet-Soft (NEW)     | stable+ | grad + freq  | —            |
      | PMCNet-Full           | stable+ | TV           | Debye + MC   |
    """

    # =========================================================================
    # 1. БАЗА: сеть, оптимизатор, окружение
    # =========================================================================

    image_size: Tuple[int, int] = (51, 51)
    base_channels: int = 32
    n_iterations: int = 2000
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = field(default_factory=lambda:
                        'cuda' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # 2. ФИЗИЧЕСКИЕ УЛУЧШЕНИЯ (на форвард-оператор Paper, флаги независимы)
    # =========================================================================
    # Все три флага независимо переключаются — позволяет конструировать
    # PMCNet-вариант с любой комбинацией. По умолчанию ВСЕ False —
    # это paper-faithful поведение (`PMCNetPaper`). Каждый из четырёх
    # named-вариантов (RadialCoil, Soft, Debye, CentralFD) включает
    # РОВНО ОДНО из улучшений, чтобы изолировать его вклад.

    # 2a. Радиальная чувствительность катушки p(r) = 1/(1+(r/R_coil)²)
    # вместо uniform p(r) ≡ 1. По умолчанию False (paper-faithful uniform).
    # Используется `PMCNetRadialCoil`.
    use_radial_coil: bool = False

    # 2b. Схема ∂M/∂t для аналитического оператора:
    #   True  → центральная разность через Conv1d [-1, 0, +1]/(2Δt) — O(Δt²)
    #   False → forward-FD [-1, +1]/Δt — O(Δt) (paper-faithful)
    # Используется `PMCNetCentralFD`.
    use_central_fd: bool = False

    # 2c. Релаксация Дебая (paper Eq. 4–5) как частотный фильтр
    # H_τ(f) = 1/(1+j·2π·f·τ), применяемый к выходу forward-оператора.
    # τ обучаемая через softplus(raw_τ). Используется `PMCNetDebye`.
    use_debye: bool = False
    # Начальное τ для Дебая (с). 1 нс делает фильтр практически прозрачным
    # до 1 МГц; раньше было 2 мкс, что убивало высокие гармоники до 10%.
    init_tau_seconds: float = 1.0e-9

    # =========================================================================
    # 3. ML-АРХИТЕКТУРА
    # =========================================================================

    # 3a. Число каналов концентрации (multi-color MPI). >1 нужно когда
    # в образце смешаны частицы разных типов с разным τ.
    n_colors: int = 1

    # 3b. Учить R_coil как nn.Parameter через softplus (blind calibration).
    # По умолчанию False — R фиксирован.
    learn_coil_radius: bool = False

    # =========================================================================
    # 4. SOFT-CONSTRAINTS — auxiliary loss-термы (PCNN-style, Scheinker 2023)
    # =========================================================================
    #
    # Все эти штрафы добавляются ВРЕМЯ ОПТИМИЗАЦИИ как L = L_data + Σ λ_i·R_i,
    # не меняя forward operator. Это «soft physics constraints» по
    # терминологии Scheinker 2023 — гибкая регуляризация, в отличие от
    # «hard constraints by construction» (вшитых в архитектуру).

    # 4a. TV-штраф на концентрацию: λ·(‖∂c/∂x‖₁ + ‖∂c/∂y‖₁).
    # Поощряет кусочно-постоянные реконструкции, подавляет рябь.
    lambda_tv: float = 0.0

    # 4b. NEW (Scheinker 2023, Eq. 12-13). L2-штраф на квадрат пространственного
    # градиента: λ·‖∇c‖₂². Гасит мелкомасштабные осцилляции, оставляя
    # резкие границы (в отличие от TV — но мягче). Реализован через
    # фиксированный Conv2d с central-FD ядром.
    lambda_grad: float = 0.0

    # 4c. NEW. Частотно-взвешенный L1-loss: каждая гармоника k получает
    # вес w_k = (k+1)^p, что компенсирует низкое SNR на высоких гармониках
    # MPI. p — степень убывания амплитуды (типично 0.5-1.0).
    # При False (default) — обычный L1 без весов.
    use_freq_weighting: bool = False
    freq_weighting_power: float = 0.5

    # =========================================================================
    # 5. ПАРАМЕТРЫ СКАНЕРА (BeihangUniversityData дефолты)
    # =========================================================================
    # Параметры из «Описание параметров файла H5.docx». При работе с другим
    # сканером значения переопределяются через `_load_scanner_params_from_h5`
    # в pipeline.

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

    # Чувствительность приёмной катушки (R_coil для s(r) = 1/(1+(r/R)²))
    coil_radius_m: float = 0.020

    # Свойства частиц (H5: tracer.name = Perimag; Fe₃O₄ ≈ 0.6 Т насыщения)
    saturation_magnetization_T: float = 0.6
    particle_diameter_nm: float = 20.0
    temperature_K: float = 300.0

    # Число точек по t в траектории FFP. По умолчанию одно «биение» Лиссажу
    n_time_samples: int = 1024
    scan_duration_s: Optional[float] = None


# =============================================================================
# Численно-устойчивая функция Ланжевена
# =============================================================================


def langevin_safe(xi: torch.Tensor) -> torch.Tensor:
    """Функция Ланжевена L(ξ) = coth(ξ) − 1/ξ — численно-устойчивая реализация.

    ## Физический смысл

    Функция Ланжевена описывает среднюю намагниченность ансамбля магнитных
    диполей в тепловом равновесии под действием поля. Для суперпарамагнитной
    наночастицы:

        M(H) = m_sat · L(ξ),  ξ = μ₀ · m · H / (k_B · T)

    где m — магнитный момент частицы, m_sat — насыщающая намагниченность,
    T — температура. ξ — отношение магнитной энергии к тепловой:
      • ξ → 0  → L → ξ/3       (линейный режим: M ≈ const·H)
      • ξ → ∞  → L → 1         (насыщение: все диполи вдоль H)
      • ξ ≈ 1  → переходная S-форма с максимальной нелинейностью

    Именно S-образная нелинейность создаёт спектр гармоник в индуцированном
    сигнале — основу пространственной локализации в MPI. Без неё реконструкция
    была бы невозможна (линейный отклик дал бы только частоту f, без гармоник).

    ## Численные ловушки

    Прямая формула `coth(ξ) − 1/ξ` в float32 катастрофически сокращается при
    ξ → 0: для ξ = 10⁻³ оба слагаемых имеют порядок 10³, их разность 3·10⁻⁴,
    после вычитания теряется около 6 значащих цифр → NaN/Inf в типичных
    параметрах MPI. Кроме того, `coth(ξ)` при |ξ| ≈ 20 переполняется в float32.

    ## Реализация: три ветви

    1) **Малые ξ (|ξ| < 0.5)** — Тейлоровское разложение из пяти членов:
       L(ξ) ≈ ξ/3 − ξ³/45 + 2ξ⁵/945 − ξ⁷/4725 + 2ξ⁹/93555
       Точность ≲ 10⁻⁸ во всём интервале — лучше float32-floor (~10⁻⁷).

    2) **Средние ξ (0.5 ≤ |ξ| ≤ 20)** — точная форма через устойчивый coth:
       coth(ξ) = sign(ξ)·(1 + e^{−2|ξ|}) / (1 − e^{−2|ξ|})
       Эта запись избегает переполнения при больших |ξ|.

    3) **Большие ξ (|ξ| > 20)** — асимптотика sign(ξ)·(1 − 1/|ξ|).
       L(ξ) → 1 насыщается экспоненциально быстро, эта аппроксимация
       точна до 10⁻⁹ при ξ > 20.

    Все три ветви склеены через `torch.where`, что даёт ОДИН
    дифференцируемый граф для autograd (булевы маски с in-place
    присваиванием ломают backprop в PyTorch и дают NaN). Защитные
    `clamp_min(1e-12)` стоят в неактивных ветвях и не влияют на
    корректные градиенты в активных.

    Args:
        xi: тензор любой формы (float32 или float64).

    Returns:
        L(xi) той же формы.
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
    """Релаксация Дебая: «инерция» намагниченности частиц с временем τ.

    ## Физика

    Идеальная модель MPI предполагает, что намагниченность частицы M(t)
    мгновенно следует за приложенным полем H(t) через функцию Ланжевена
    (адиабатический режим). В реальности частицы имеют **инерцию**: M(t)
    отстаёт от H(t) с характерным временем релаксации τ. Это особенно
    важно для крупных частиц (>30 нм) и быстрых драйв-полей (>20 кГц).

    Модель Дебая описывает эту инерцию ODE 1-го порядка:

        τ · dM_D(t)/dt = −M_D(t) + M(t)

    где M(t) — «идеальная» (адиабатическая) намагниченность, M_D(t) —
    реальная (с инерцией). Решение — экспоненциальная свёртка:

        M_D(t) = M(t) ⊛ r(t),   r(t) = (1/τ) · exp(−t/τ) · u_step(t)

    В частотной области это эквивалентно умножению на передаточную функцию:

        H(f; τ) = 1 / (1 + j·2π·f·τ)

    Эта функция работает как **lowpass-фильтр**: гасит высокие гармоники
    (>1/(2πτ)). Например, при τ = 2 мкс гармоники выше ~80 кГц подавляются.

    ## Зачем учить τ

    В реальной MPI τ зависит от:
      • размера частиц (квадратично);
      • вязкости окружающей среды (биологические ткани!);
      • температуры.

    Для одного образца τ фиксирован, но априори неизвестен. Учим его
    совместно с весами сети через градиентный спуск (paper-style:
    «we did not provide τ directly, but instead estimated it through
    the gradient descent algorithm»).

    ## Multi-color MPI

    Если в образце смешаны частицы РАЗНОГО типа (например, разных
    диаметров), каждый тип имеет свой τ_k и реконструирует свою карту
    концентрации c_k. Модуль хранит вектор τ длиной `n_colors`.
    Итоговый сигнал — сумма u(f) = Σ_k H_τ_k(f) · ForwardOp(c_k).

    ## Параметризация через softplus

    τ должен быть строго положительным. Прямая параметризация через
    `nn.Parameter` ломала бы это при больших градиентах (τ может стать
    отрицательным). Решение: хранить `raw_tau`, а τ вычислять как
    softplus(raw_tau) = log(1 + exp(raw_tau)). softplus гладко переводит
    R → R⁺ → положительность гарантирована _by construction_, градиенты
    текут корректно.

    ## Два режима применения

    • **Временная свёртка** (`apply_time`): сигнал в виде временного ряда
      u(t), сворачивается с ядром r(t) через FFT с нулевым допаддингом
      (линейная, не циклическая свёртка). Используется когда форвард
      даёт u(t) явно.
    • **Частотная фильтрация** (`freq_response`): сигнал уже в частотной
      области U(f), умножается на H(f). Эффективнее, используется в
      PMCNetDebye где FFT всё равно делается для применения H_τ(f).

    Args:
        n_colors: число типов МНЧ (multi-color MPI). Default 1.
        init_tau: начальное значение τ в секундах. Default 2 мкс (типично
                  для коммерческих SPION при 25 кГц драйв-поле). Для
                  выключения Debye по умолчанию используйте 1e-9 с
                  (фильтр практически прозрачен).
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
# Аналитический прямой оператор MPI
# =============================================================================
# Реализует полную физическую цепочку:
#     u(t) = −μ₀ · ∫ s(r) · c(r) · ∂M(r,t)/∂t dr
# где:
#     M(r,t) — намагниченность по Ланжевену от полного поля H(r,t),
#     H(r,t) = G·(r − r_FFP(t)) — селективное + возбуждающее поле,
#     r_FFP(t) — Lissajous-траектория FFP,
#     s(r) — пространственная чувствительность приёмной катушки.
#
# Архитектурный приём — реализация конечных разностей и интегралов через
# фиксированные свёртки (Conv1d с не-обучаемым ядром) — заимствован из
# подхода physics-constrained neural networks (Scheinker & Pokharel 2023):
# любая дифференциальная операция представима как свёртка → autograd
# работает «бесплатно», без ручных циклов по сетке, и легко переносится
# на GPU.
# =============================================================================


class RadialCoilSensitivity(nn.Module):
    """Радиально-зависимая чувствительность приёмной катушки s(x,y).

    ## Физика

    Приёмная катушка имеет конечный радиус R_coil. Магнитный поток через
    катушку от точечного диполя на расстоянии r ослабевает по закону
    обратных квадратов. Аппроксимируем спад:

        s(x, y) = 1 / (1 + (r / R_coil)²),   r = √(x² + y²)

    где (x, y) — положение в FOV, отсчитанное от центра катушки. В центре
    (r=0) чувствительность s = 1.0. На краях:
      • r = R_coil    → s = 0.5  (половина максимума)
      • r = 2·R_coil  → s = 0.2
      • r → ∞         → s → 0

    ## Зачем учитывать

    Без радиальной чувствительности (т.е. при s(r) ≡ 1) сеть считает,
    что слабый сигнал на краях FOV = малая концентрация. На самом деле
    он может означать «концентрация есть, но катушка её плохо ловит».
    Это приводит к **переоценке концентрации у границ** при реконструкции.

    ## Обучаемый R_coil (опционально)

    Реальный радиус приёмной катушки сканера может отличаться от
    номинального. Если `learn_radius=True`, R_coil становится
    `nn.Parameter`, параметризованный через softplus(raw_R) для
    гарантированной положительности. Сеть подстраивает R_coil под
    реальные данные через градиентный спуск (joint blind calibration).
    В этом режиме `sensitivity` пересчитывается на каждом forward(),
    чтобы градиент по R_coil корректно тёк через autograd.

    Args:
        image_size: (Nx, Ny) — размер сетки в пикселях.
        image_extent_m: физический размер FOV в метрах (квадратное FOV).
        coil_radius_m: начальное значение R_coil в метрах. Типично
                       ~0.5 × image_extent_m (катушка чуть больше FOV).
        learn_radius: если True, R_coil обучается. Default False.
    """

    def __init__(self, image_size: Tuple[int, int], image_extent_m: float,
                 coil_radius_m: float, learn_radius: bool = False):
        super().__init__()
        Nx, Ny = image_size
        x = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Nx)
        y = torch.linspace(-image_extent_m / 2.0, image_extent_m / 2.0, Ny)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        r = torch.sqrt(X.pow(2) + Y.pow(2))
        self.register_buffer('X', X)
        self.register_buffer('Y', Y)
        self.register_buffer('r', r)
        self.learn_radius = bool(learn_radius)
        if self.learn_radius:
            # softplus(raw_R) = R, чтобы R > 0 by construction
            raw = math.log(math.expm1(coil_radius_m)) \
                if coil_radius_m < 80.0 else coil_radius_m
            self.raw_R = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer('_R',
                                  torch.tensor(coil_radius_m, dtype=torch.float32))

    @property
    def coil_radius(self) -> torch.Tensor:
        """Текущее значение R_coil (softplus(raw_R) или фиксированный буфер)."""
        if self.learn_radius:
            return F.softplus(self.raw_R).clamp_min(1e-6)
        return self._R

    @property
    def sensitivity(self) -> torch.Tensor:
        """s(r) = 1 / (1 + (r/R)²) — пересчитывается каждый вызов.

        При `learn_radius=True` градиент по R автоматически течёт через
        autograd. При фиксированном R — стоимость пересчёта пренебрежимо
        мала (одна broadcast-операция на тензоре Nx×Ny).
        """
        return 1.0 / (1.0 + (self.r / self.coil_radius).pow(2))

    def forward(self) -> torch.Tensor:
        return self.sensitivity


class UniformCoilSensitivity(nn.Module):
    """p(r) ≡ 1 — идеализированный приёмник (paper-faithful PMCNet).

    Самый простой возможный профиль: чувствительность катушки одинакова
    во всём поле зрения. Соответствует идеализации, при которой катушка
    бесконечно велика относительно FOV или однородно покрывает его.

    На практике реальная катушка ловит сигнал слабее на краях (см.
    `RadialCoilSensitivity`), но в paper-варианте PMCNet используется
    именно эта идеализация — статья опирается на нелинейность Ланжевена
    как основной механизм пространственной локализации и не моделирует
    конкретный профиль катушки.

    Используется в `BasicAnalyticalForwardModel` для воспроизведения
    paper-faithful физики.
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
    """Lissajous-траектория точки нулевого поля (FFP) в 2D-сканировании MPI.

    ## Физика

    Сканер MPI создаёт «карандаш» сканирования через комбинацию двух полей:
      • **Селективное поле** H_sel(r) = G·r — статический градиент,
        формирующий точку, где H_sel = 0 (Field-Free Point, FFP);
      • **Возбуждающее поле** H_exc(t) = A·sin(2π·f·t) — однородное
        переменное поле, смещающее FFP в пространстве.

    Полное поле в точке r и момент t:
        H_total(r, t) = G·r + H_exc(t)

    FFP — это та точка r_FFP(t), где H_total = 0:
        G_x·r_FFP_x + A_x·sin(2π·f_x·t) = 0
        ⇒ r_FFP_x(t) = −(A_x / G_x) · sin(2π·f_x·t)
    (аналогично по y). То есть FFP колеблется по гармоническому закону
    с амплитудой A/G.

    ## Важный нюанс: A/G ≠ половина FOV

    Многие путают амплитуду движения FFP с размером FOV. Это разные
    величины:
      • A/G  — насколько далеко FFP отклоняется от центра (определяется
                драйв-полем и градиентом);
      • FOV  — размер реконструируемой области (выбирается пользователем).

    Пример: для типичных параметров BeihangUniversityData (A_x = 10 мТ,
    G_x = 1.12 Т/м) FFP по x раскачивается на ±8.93 мм, тогда как FOV
    обычно ±19 мм. FFP покрывает только 47% FOV.

    Это означает, что **периферия FOV не «видится» сканером напрямую** —
    концентрация там восстанавливается через корреляции с центральной
    зоной, что снижает разрешение на краях.

    ## Lissajous figure: зачем разные f_x и f_y

    Если f_x = f_y, FFP движется по прямой. Чтобы покрыть 2D-область,
    нужны РАЗНЫЕ частоты. Это даёт Lissajous figure:
      • f_x ≈ f_y, иррациональное отношение → плотное 2D-покрытие;
      • f_x >> f_y → 1D-line scan, y-ось почти неподвижна;
      • f_x = k·f_y (целое k) → периодическая траектория, узкое покрытие.

    Период замкнутой Lissajous-фигуры = 1/|f_x − f_y|. Это естественная
    длительность одного «полного цикла» сканирования.

    Args:
        n_time_samples: число дискретных временных отсчётов в одном цикле
                        сканирования.
        freq_x_hz, freq_y_hz: драйв-частоты по осям x и y, Hz.
        amp_x_m, amp_y_m: амплитуда движения FFP по каждой оси (метры).
                          Обычно вычисляется как A/G из физики сканера.
                          Если не передано — берётся `image_extent_m/2`
                          (legacy-вариант, физически некорректный).
        image_extent_m: альтернатива для legacy-режима. Игнорируется,
                         если amp_x_m/amp_y_m заданы.
        duration_s: длительность сканирования в секундах. По умолчанию
                     1/|f_x − f_y| — естественный период Lissajous figure.
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
    """Центральная конечная разность ∂/∂t через фиксированную Conv1d.

    ## Формула

    ∂V/∂t[k] ≈ (V[k+1] − V[k−1]) / (2·Δt)

    Точность аппроксимации O(Δt²) — на порядок лучше forward FD (O(Δt)).
    Для сигнала, содержащего гармоники до частоты f_max, относительная
    ошибка ≈ (π·f_max·Δt)²/6. При Δt = 1 μs и f_max = 100 kHz это
    меньше 0.5% — приемлемо.

    ## Реализация через свёртку

    Любая линейная конечно-разностная схема — это свёртка V с фиксированным
    ядром. Для центральной разности ядро = [−1, 0, +1]/(2Δt). Conv1d с
    `weight_grad=False` (не обучаемое) делает оператор:
      • полностью совместимым с autograd: градиент течёт через свёртку
        «бесплатно», без ручной реализации chain rule;
      • батчируемым: применяется к (B, T) и (B, 1, T) без модификации;
      • GPU-совместимым без специальных индексаций.

    Это архитектурный приём из physics-constrained neural networks: любая
    дифференциальная операция представима как свёртка с фиксированным
    ядром, и тогда вся физика становится цепочкой свёрток в `nn.Module`.

    На границах применяется реплицирующий паддинг (V[−1] = V[0],
    V[T] = V[T−1]), чтобы выход имел ту же длину T. Альтернатива —
    zero-padding — внесла бы артефактные пики на границах.
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
    """Forward-difference ∂/∂t — простейшая односторонняя схема.

    ## Формула

    ∂V/∂t[k] ≈ (V[k+1] − V[k]) / Δt

    Ядро свёртки: [−1, +1]/Δt. Точность O(Δt) — на порядок хуже
    центральной разности в `TimeDerivativeFD`, но проще:
      • Использует только один соседний отсчёт (V[k+1]) — естественно
        реализуется в режиме «онлайн» при поступлении данных.
      • Симметрия по знаку отсутствует — может вносить small
        систематический сдвиг по частоте при анализе спектра.

    ## Зачем оставлена

    Соответствует базовой схеме paper-faithful PMCNet. Статья не
    специфицирует схему численного дифференцирования; forward FD —
    самый простой выбор из стандартных. Это позволяет точно
    воспроизвести paper-результаты и оценить, насколько улучшения в
    схеме (central FD в `TimeDerivativeFD`) добавляют качества.

    На границе справа применяется реплицирующий паддинг (V[T] = V[T−1]),
    чтобы выход имел длину T.
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
    """Адиабатическая намагниченность ансамбля SPION под действием поля H.

    ## Формула

    M(H) = m_sat · L(ξ) · ê_H,    ξ = β · |H|,   β = μ₀·m / (k_B·T)

    где:
      • m — магнитный момент одной частицы [A·m²];
      • m_sat — насыщающая намагниченность [A/m];
      • L(ξ) — функция Ланжевена (см. `langevin_safe`);
      • ê_H — единичный вектор в направлении поля.

    Магнитный момент частицы связан с её объёмом и насыщающей
    намагниченностью материала B_sat (для Fe₃O₄ ≈ 0.6 Т):

        m = M_sat · V_core,   V_core = π·d³/6,   M_sat = B_sat/μ₀

    Параметр β = μ₀·m/(k_B·T) — это «магнитная обратная температура»:
    отношение магнитной энергии единичного поля к тепловой. Для типичной
    SPION 30 нм при 300 K β ≈ 7·10⁻³ м/А, поэтому при H = 10⁴ А/м
    ξ = β·H ≈ 70 — глубокое насыщение. При H = 100 А/м ξ ≈ 0.7 —
    переходная область, самая нелинейная.

    ## Векторный выход

    Возвращает (M_x, M_y) — две компоненты намагниченности по осям x и y.
    Скалярная амплитуда |M| = m_sat · L(|β·H|), а направление совпадает с
    направлением H:
        M_x = |M| · H_x / |H|
        M_y = |M| · H_y / |H|

    Это нужно потому, что приёмные катушки x и y «видят» только
    соответствующие компоненты: x-катушка реагирует на ∂M_x/∂t,
    y-катушка — на ∂M_y/∂t.

    Защита от деления на ноль: |H| зажимается снизу на 1e-30 перед
    вычислением xi (если в какой-то точке H = 0 точно — Langevin даёт
    0 → производная 0 → нулевой вклад, что физически корректно).

    Args:
        saturation_magnetization_T: B_sat материала в Тесла. Для Fe₃O₄
                                     ≈ 0.6 (наиболее распространённый
                                     материал MPI-трасеров).
        particle_diameter_nm: диаметр магнитного ядра частицы, нм.
                               Типично 20–30 нм для коммерческих SPION.
        temperature_K: температура образца, K. Для in vivo MPI ≈ 310 K.
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
            learn_radius=config.learn_coil_radius,
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
    """Параметризованный PMCNet прямой оператор (Huang 2026, Sec. II.B, Eq. 1–3).

    Реализует уравнения статьи:

        u(t) = −μ₀ ∫ c(r) · p(r) · ∂M(r,t)/∂t dr            [paper Eq. 1]
        M(r,t) = c · m · (coth(αH) − 1/(αH)) · ê_H           [paper Eq. 2]
        α = μ₀·m / (k_B·T)                                   [paper Eq. 3]

    **Параметризован двумя независимыми config-флагами:**

    ▸ `use_radial_coil` (default False = paper-faithful):
        False → `UniformCoilSensitivity`, p(r) ≡ 1 (статья не специфицирует
                профиль приёмной катушки);
        True  → `RadialCoilSensitivity`, p(r) = 1/(1+(r/R)²) — радиальная
                чувствительность, моделирующая ослабление сигнала на
                краях FOV (улучшение для `PMCNetRadialCoil`).

    ▸ `use_central_fd` (default False = paper-faithful):
        False → `TimeDerivativeForwardFD`, ∂/∂t = (M[t+1] − M[t])/Δt,
                точность O(Δt) — буквально theory.md Eq. derivative;
        True  → `TimeDerivativeFD` через Conv1d с ядром [−1, 0, +1]/(2Δt),
                точность O(Δt²) — улучшение для `PMCNetCentralFD`
                (Maxwell-PCNN Eq. 12-13).

    Релаксация Дебая (use_debye) применяется НЕ здесь, а на уровне
    `BasicHardConstrainedSpectralForward` (post-multiplication на H_τ(f)
    в частотной области). Multi-color и TV не реализованы.

    `langevin_safe` используется всегда: без него float32 даёт NaN при
    ξ → 0 (катастрофическое сокращение в `coth(ξ) − 1/ξ`). Это численная
    необходимость для PyTorch, не отход от статьи.
    """

    def __init__(self, config: 'PMCNetConfig'):
        super().__init__()
        self.config = config
        # Профиль чувствительности катушки: paper-faithful uniform или
        # радиальная (включается через config.use_radial_coil).
        # Это позволяет одной и той же базовой архитектуре служить как
        # для PMCNet-Paper (uniform), так и для PMCNet-RadialCoil (radial).
        if config.use_radial_coil:
            self.coil = RadialCoilSensitivity(
                image_size=config.image_size,
                image_extent_m=config.image_extent_m,
                coil_radius_m=config.coil_radius_m,
                learn_radius=config.learn_coil_radius,
            )
        else:
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
        # Схема ∂M/∂t: paper-faithful forward-FD (O(Δt)) или central FD
        # через Conv1d с ядром [−1, 0, +1]/(2Δt) (O(Δt²)). Управляется
        # через config.use_central_fd.
        if config.use_central_fd:
            self.ddt = TimeDerivativeFD(dt=float(self.ffp.dt.item()))
        else:
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

    Два режима спектрального преобразования:
      • FFT-режим (frequencies_hz=None) — берёт первые M/2 бин-частот
        rfft(u(t)) на каждую катушку. Простой и быстрый, но физические
        частоты бинов определяются `n_time_samples` × `dt` и могут не
        совпадать с реальной частотной сеткой измерения со сканера.
      • DFT-режим (frequencies_hz=array) — вычисляет U(f) = ∫ u(t)·
        exp(−2πi f t) dt напрямую на _явно заданных_ частотах из
        measurement/frequencySelection в H5. Гарантирует, что L1-loss
        сравнивает гармоники, соответствующие одним и тем же физическим
        частотам. Дороже по операциям, но точнее.

    Калибровка `output_scale`:
      • Без вызова `calibrate_output_scale` — используется sm_max
        (макс |ядра|), как в исходной реализации.
      • С вызовом `calibrate_output_scale(u_meas_re, u_meas_im)` после
        конструкции — output_scale подгоняется так, чтобы средняя
        амплитуда forward (для типичного c ≈ 0.5) совпадала со средней
        амплитудой измерения. Устраняет арбитражное масштабирование.
    """

    def __init__(self, config: 'PMCNetConfig', n_meas_bins: int,
                 frequencies_hz: Optional[np.ndarray] = None):
        super().__init__()
        if n_meas_bins % 2 != 0:
            raise ValueError(
                f"n_meas_bins ({n_meas_bins}) должно быть чётным "
                f"(по половине на каждую катушку)"
            )
        self.n_meas_bins = int(n_meas_bins)
        self.n_freq_per_coil = self.n_meas_bins // 2
        self.use_explicit_dft = frequencies_hz is not None

        cfg = config
        # Для FFT-режима: обеспечиваем nyquist-условие.
        # Для DFT-режима: n_time_samples определяет точность интегрирования,
        # но не привязан к n_freq_per_coil напрямую.
        if not self.use_explicit_dft:
            if cfg.n_time_samples < 2 * self.n_freq_per_coil:
                cfg = PMCNetConfig(**{
                    **config.__dict__,
                    'n_time_samples': 2 * self.n_freq_per_coil,
                })
        self._Nx, self._Ny = cfg.image_size

        # Физика (параметризована флагами cfg.use_radial_coil/use_central_fd)
        self.physics = BasicAnalyticalForwardModel(cfg)

        if self.use_explicit_dft:
            self._setup_explicit_dft(frequencies_hz)
        else:
            self._setup_fft_scale()

        # Опциональная Debye-релаксация (применяется в частотной области
        # к выходу _после_ FFT/DFT). Single-color: один τ для всего сигнала.
        # Активируется через cfg.use_debye → `PMCNetDebye`.
        self.use_debye = bool(cfg.use_debye)
        if self.use_debye:
            self.debye = DebyeRelaxationFilter(
                n_colors=1, init_tau=cfg.init_tau_seconds,
            )
            # Частоты для применения H_τ(f): для FFT-режима — это f_x · k,
            # для DFT-режима — явно заданные frequencies_hz (одинаковые
            # для обеих катушек, поэтому дублируем).
            if self.use_explicit_dft:
                debye_freqs = np.tile(np.asarray(frequencies_hz).flatten(),
                                       2).astype(np.float32)
            else:
                k = np.arange(1, self.n_freq_per_coil + 1, dtype=np.float32)
                debye_freqs = np.tile(k * cfg.drive_frequency_x, 2)
            self.register_buffer(
                'debye_freqs_hz',
                torch.tensor(debye_freqs, dtype=torch.float32),
            )
        else:
            self.debye = None

    def _setup_fft_scale(self) -> None:
        """Нормировка output_scale через max |ядро| (старый путь, FFT)."""
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

    def _setup_explicit_dft(self,
                             frequencies_hz: np.ndarray) -> None:
        """Базис явного DFT на заданных частотах.

        Сохраняем `dft_cos` и `dft_sin` как буферы:
            U_real(f_k) = Σ_t u(t) · cos(2π f_k t) · dt
            U_imag(f_k) = −Σ_t u(t) · sin(2π f_k t) · dt
        Это эквивалентно ∫ u(t)·e^{−2πi f_k t} dt в дискретной форме.
        """
        t = self.physics.ffp.t                          # (T,) seconds
        freqs = torch.tensor(np.asarray(frequencies_hz).flatten(),
                              dtype=torch.float32)        # (K,)
        # arg[t, k] = 2π f_k t — (T, K) тензор
        arg = 2.0 * math.pi * t.unsqueeze(-1) * freqs.unsqueeze(0)

        # ── ВАЖНО: DFT-базис БЕЗ умножения на dt ────────────────────────
        # Если умножать на dt = scan_duration/N_samples ~ 1e-7 с, то
        # значения dft_cos/dft_sin ~ 1e-7. Per-pixel ядра физики
        # μ₀·dA·∂M/∂t в SI-единицах ~ 1e-24. После DFT-свёртки
        # получаем ~1e-24 · 1e-7 · N = 1e-28 → под float32 floor (1e-30),
        # output_scale залипает в floor → forward даёт мусор → SSIM 0.09.
        #
        # FFT-режим работает потому, что `torch.fft.rfft` НЕ множит на dt
        # (это чистая DFT-сумма, не интеграл-аппроксимация). Воспроизводим
        # ту же нормировку: dft_cos[t,k] = cos(2π f_k t) без коэффициента.
        # Финальный масштаб всё равно поглощается через output_scale,
        # поэтому физическая интерпретация результата не меняется.
        self.register_buffer('dft_cos', torch.cos(arg))
        self.register_buffer('dft_sin', torch.sin(arg))
        # При explicit DFT n_freq_per_coil = len(frequencies_hz),
        # n_meas_bins = 2 × это значение
        self.n_freq_per_coil = len(freqs)
        self.n_meas_bins = 2 * self.n_freq_per_coil

        # ── Инициализация output_scale через kernel max (в float64) ────────
        # КРИТИЧЕСКИ ВАЖНО: НЕЛЬЗЯ использовать forward(c=ones) для
        # калибровки масштаба. В MPI равномерная концентрация даёт почти
        # нулевой сигнал (FFP проходит через +c и −c зоны симметрично,
        # вклады сокращаются). Это фундаментальное свойство, благодаря
        # которому MPI способен к локализации — и причина, по которой
        # нормировка по uniform-c падала в floor 1e-30.
        #
        # Правильный подход (как в _setup_fft_scale): вычислить per-pixel
        # ядра K(r, t) = −μ₀·dA·s(r)·∂M(r,t)/∂t, взять их DFT на заданных
        # частотах, и нормировать max |K(r, f)|. Это масштаб системной
        # матрицы — он же ненулевой и для uniform-c кейса (где сигнал
        # = Σ_r K(r,f) = малое из-за интерференции).
        #
        # FLOAT64 ОБЯЗАТЕЛЕН: в SI-единицах MPI ядро ≈ μ₀·dA·∂M/∂t ~ 1e-25.
        # После DFT (без · dt) → ~1e-23. Но `K.pow(2)` = ~1e-46, что ниже
        # минимального нормального float32 (1.2e-38) → underflow в 0,
        # и K_mag = 0 → output_scale = floor 1e-30 → forward в loss-плато.
        # Расчёт в float64 обходит underflow; финальный scale хранится
        # в float32 (нормализованные числа уже в пределах представимости).
        with torch.no_grad():
            Mx, My = self.physics._compute_magnetization()   # (Nx, Ny, T)
            dMx = self.physics.ddt(Mx)
            dMy = self.physics.ddt(My)
            s = self.physics.coil.sensitivity                  # (Nx, Ny)
            scale = -self.physics.langevin.mu0 * self.physics.dA
            kx = (scale * s.unsqueeze(-1) * dMx).double()      # ← float64
            ky = (scale * s.unsqueeze(-1) * dMy).double()      # ← float64
            T = kx.shape[-1]
            kx_flat = kx.reshape(-1, T)                        # (N_pixels, T)
            ky_flat = ky.reshape(-1, T)
            dft_cos_64 = self.dft_cos.double()
            dft_sin_64 = self.dft_sin.double()
            # DFT каждого per-pixel ядра на K заданных частотах
            Kx_re = kx_flat @ dft_cos_64                       # (N, K)
            Kx_im = -kx_flat @ dft_sin_64
            Ky_re = ky_flat @ dft_cos_64
            Ky_im = -ky_flat @ dft_sin_64
            sm_max_64 = torch.cat([
                (Kx_re.pow(2) + Kx_im.pow(2)).sqrt().flatten(),
                (Ky_re.pow(2) + Ky_im.pow(2)).sqrt().flatten(),
            ]).max().clamp_min(1e-30)
            # Конвертация обратно в float32 (значения уже нормальные)
            sm_max = sm_max_64.float()
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

        if self.use_explicit_dft:
            # Explicit DFT при явно заданных частотах из H5
            u_x_t = u_time[:, 0]                               # (B, T)
            u_y_t = u_time[:, 1]
            U_x_re = u_x_t @ self.dft_cos                      # (B, K)
            U_x_im = -u_x_t @ self.dft_sin
            U_y_re = u_y_t @ self.dft_cos
            U_y_im = -u_y_t @ self.dft_sin
            U_real = torch.cat([U_x_re, U_y_re], dim=-1) / self.output_scale
            U_imag = torch.cat([U_x_im, U_y_im], dim=-1) / self.output_scale
        else:
            # FFT-путь (paper-faithful, без явных частот)
            U_x = torch.fft.rfft(u_time[:, 0], dim=-1)
            U_y = torch.fft.rfft(u_time[:, 1], dim=-1)
            U = torch.cat([
                U_x[:, :self.n_freq_per_coil],
                U_y[:, :self.n_freq_per_coil],
            ], dim=-1) / self.output_scale
            U_real, U_imag = U.real, U.imag

        # Опциональная Debye-релаксация: умножение на H_τ(f) в частотной
        # области. Эквивалентно временной свёртке с r(t) = (1/τ)·exp(−t/τ).
        if self.use_debye:
            H = self.debye.freq_response(self.debye_freqs_hz, color_idx=0)
            Hr, Hi = H.real, H.imag
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            U_real_new = U_real * Hr - U_imag * Hi
            U_imag_new = U_real * Hi + U_imag * Hr
            U_real, U_imag = U_real_new, U_imag_new

        return U_real, U_imag

    def calibrate_output_scale(self, u_real: torch.Tensor,
                                u_imag: torch.Tensor,
                                typical_c: float = 0.5) -> float:
        """Подогнать output_scale так, чтобы forward(тестовый_c) совпадал
        по средней амплитуде с measurement.

        Без этой калибровки `output_scale` имеет нормировку «по максимуму
        системного ядра» (см. `_setup_explicit_dft`/`_setup_fft_scale`),
        что не привязано к амплитуде конкретного measurement. После
        калибровки выход forward для типичного фантома даёт ту же
        среднюю |амплитуду|, что и measurement → L1-loss сравнивает
        выходы одного порядка величины.

        ## Что используется как «тестовый фантом»

        КРИТИЧЕСКИ ВАЖНО: НЕЛЬЗЯ использовать uniform-c для калибровки.
        В MPI равномерная концентрация даёт почти нулевой сигнал (FFP
        проходит через +c и −c зоны симметрично, контрибуции сокращаются).
        До исправления тут стояло `c = 0.5 * ones`, что давало
        pred_mag ≈ 0 → калибровка проваливалась → forward на огромный
        делитель → loss-плато → SSIM 0.09 у Paper.

        Решение: тестовый фантом = **гауссова капля в центре** —
        концентрация неравномерная, отдалённо похожа на реальные фантомы,
        даёт меняющийся вдоль FFP-трека сигнал (как реальный measurement).

        Args:
            u_real, u_imag: (M,) или (B, M) — measurement в частотной
                            области (как в reconstruct).
            typical_c: пик гауссовой капли (default 0.5 — типичное
                       значение после sigmoid).
        Returns:
            Финальное значение output_scale.
        """
        with torch.no_grad():
            # Гауссова капля в центре FOV — даёт ненулевой MPI-сигнал
            # (в отличие от uniform-c). σ ≈ четверть от меньшей стороны.
            device = u_real.device
            xs = torch.arange(self._Nx, dtype=torch.float32, device=device)
            ys = torch.arange(self._Ny, dtype=torch.float32, device=device)
            X, Y = torch.meshgrid(xs, ys, indexing='ij')
            cx, cy = (self._Nx - 1) / 2.0, (self._Ny - 1) / 2.0
            sigma = max(min(self._Nx, self._Ny) / 4.0, 1.0)
            blob = torch.exp(-((X - cx).pow(2) + (Y - cy).pow(2))
                              / (2.0 * sigma * sigma))
            blob = blob / blob.max() * typical_c                 # peak = typical_c
            c_test = blob.view(1, -1)                            # (1, N)

            ur_pred, ui_pred = self.forward(c_test)
            pred_mag = (ur_pred.pow(2) + ui_pred.pow(2)).sqrt().mean()
            meas_mag = (u_real.pow(2) + u_imag.pow(2)).sqrt().mean()
            if pred_mag > 1e-30 and meas_mag > 1e-30:
                # output = raw / output_scale. Хотим: output_mag ≈ meas_mag
                #         new_scale = output_scale_old * pred_mag / meas_mag
                ratio = (pred_mag / meas_mag).clamp(1e-6, 1e6)
                self.output_scale.fill_(self.output_scale.item() * ratio.item())
        return float(self.output_scale.item())


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
      • Центральная конечная разность через Conv1d — `PMCNetCentralFD`.
      • Релаксация Дебая (paper Sec. II.C) — `PMCNetDebye`.
      • Multi-color (paper Sec. III) — убрано из публичного API.
      • TV-регуляризация — убрано из публичного API (есть в config).

    Соответствует pseudocode в задании пользователя.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: 'PMCNetConfig',
                 frequencies_hz: Optional[np.ndarray] = None):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.unet = PMCNetUNet(image_size=self.image_shape, out_channels=1,
                               base=config.base_channels)
        self.forward_op = BasicHardConstrainedSpectralForward(
            config, n_meas_bins, frequencies_hz=frequencies_hz,
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
    """Общая инфраструктура для PMCNet-реконструкторов.

    ## Soft-constraint loss-термы

    Базовый класс предоставляет реализации auxiliary-loss'ов, которые
    подключаются в реконструкторах через флаги PMCNetConfig:

      * `_tv_loss`              — L1 на ∇c (paper-style TV);
      * `_spatial_grad_l2_loss` — L2 на ∇c (Scheinker 2023, Eq. 12-13);
      * `_data_loss_freq_weighted` — частотно-взвешенный L1 на u_real/imag.

    Все three по архитектуре — soft constraints (auxiliary penalty при
    оптимизации, не зашиты в forward operator). Hard-constraint
    эквивалент (zero ∇·B by construction в PCNN-стиле) применён в
    forward-операторах `HardConstrainedSpectralForward`.
    """

    def __init__(self, network: nn.Module, config: PMCNetConfig):
        self.network = network
        self.config = config
        self.device = torch.device(config.device)
        self.network.to(self.device)
        self.loss_history: List[float] = []

        # Предвычисляем частотные веса для freq-weighted L1, если включён.
        # Веса w_k = (k+1)^p, где k — индекс гармоники (1..M). Для случая
        # двух катушек массив имеет ту же длину, что forward.M.
        self._freq_weights: Optional[torch.Tensor] = None
        if config.use_freq_weighting and hasattr(network, 'forward_op'):
            try:
                M = network.forward_op.M
                k = torch.arange(1, M + 1, dtype=torch.float32)
                w = k.pow(config.freq_weighting_power)
                w = w / w.mean()                      # нормировка
                self._freq_weights = w.to(self.device)
            except AttributeError:
                pass  # forward_op без .M → выключим freq-weighting

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

    # ---- Soft-constraint losses ----------------------------------------------

    @staticmethod
    def _tv_loss(c: torch.Tensor) -> torch.Tensor:
        """L1-TV: ‖∂c/∂x‖₁ + ‖∂c/∂y‖₁. Поощряет кусочно-постоянное c."""
        dh = (c[..., 1:, :] - c[..., :-1, :]).abs().mean()
        dw = (c[..., :, 1:] - c[..., :, :-1]).abs().mean()
        return dh + dw

    @staticmethod
    def _spatial_grad_l2_loss(c: torch.Tensor) -> torch.Tensor:
        """Squared-norm пространственного градиента: ‖∇c‖₂².

        Вычисляется через **central-FD ядро** (Scheinker 2023, Eq. 12-13):
        ∂c/∂x ≈ (c[i+1] − c[i-1]) / 2,   аналогично для y. Свёртки
        реализованы дешёвыми срезами без alloc'ов, чтобы шаг был ≪ 1мс.

        В отличие от L1-TV, этот штраф **гладкий** — мягко гасит
        мелкомасштабные осцилляции, но не «жмёт» сильные градиенты в
        полку. Полезен когда хотим непрерывных границ.
        """
        dh = (c[..., 2:, :] - c[..., :-2, :]) * 0.5
        dw = (c[..., :, 2:] - c[..., :, :-2]) * 0.5
        return (dh.pow(2).mean() + dw.pow(2).mean())

    def _data_loss(self, ur_pred: torch.Tensor, ui_pred: torch.Tensor,
                   ur_meas: torch.Tensor, ui_meas: torch.Tensor
                   ) -> torch.Tensor:
        """L1-loss с опциональным частотно-взвешенным режимом.

        При `use_freq_weighting=False` возвращает классический
            mean|ur_pred − ur_meas| + mean|ui_pred − ui_meas|

        При `use_freq_weighting=True` каждый член |Δ_k| умножается на
        вес w_k = (k+1)^p / ⟨(k+1)^p⟩. Это компенсирует факт, что в
        MPI амплитуды гармоник падают как ~1/k^α (α≈1-2), и без
        взвешивания низкие гармоники доминируют в loss'е, а высокие
        (резолюция!) игнорируются.
        """
        diff_r = (ur_pred - ur_meas).abs()
        diff_i = (ui_pred - ui_meas).abs()
        if self._freq_weights is not None:
            w = self._freq_weights
            return (w * diff_r).mean() + (w * diff_i).mean()
        return diff_r.mean() + diff_i.mean()

    def _aux_loss(self, c: torch.Tensor) -> torch.Tensor:
        """Все включённые soft-constraints на c. Возвращает 0, если нет."""
        loss = torch.zeros((), device=self.device)
        if self.config.lambda_tv > 0:
            loss = loss + self.config.lambda_tv * self._tv_loss(c)
        if self.config.lambda_grad > 0:
            loss = loss + self.config.lambda_grad * self._spatial_grad_l2_loss(c)
        return loss


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
            loss = self._data_loss(ur[0], ui[0], u_real, u_imag)
            loss = loss + self._aux_loss(c)
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
                 config: Optional[PMCNetConfig] = None,
                 frequencies_hz: Optional[np.ndarray] = None):
        cfg = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{**cfg.__dict__,
                              'image_size': tuple(image_shape)})
        net = PMCNetWithBasicPhysics(image_shape, n_meas_bins, cfg,
                                       frequencies_hz=frequencies_hz)
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

        # ОТКЛЮЧЕНО: calibrate_output_scale с Gaussian-blob target ломает
        # physical-path. На physical-измерении SM_analytical и Paper.forward
        # используют ИДЕНТИЧНУЮ нормировку (max|кернел|), поэтому
        # output_scale_init = SM_norm_factor → forward(GT) = measurement
        # точно (Loss(c=GT) ≈ 5e-6). Калибровка с blob сдвигает scale
        # на 8%, и loss(GT) становится 0.06 вместо 0 → optimizer находит
        # неправильное c. На sm-path калибровка тоже не помогает: модель
        # фундаментально mismatch'нута с измеренной SM, ±8% масштаба
        # ничего не лечит.
        # self.network.forward_op.calibrate_output_scale(u_real, u_imag)

        optimizer = torch.optim.Adam(self.network.parameters(),
                                     lr=self.config.learning_rate)
        self.loss_history.clear()

        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc='PMCNet-Paper')

        for it in iterator:
            optimizer.zero_grad()
            c, ur, ui = self.network(z)
            loss = self._data_loss(ur[0], ui[0], u_real, u_imag)
            loss = loss + self._aux_loss(c)
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
            loss = self._data_loss(ur[0], ui[0], u_real, u_imag)
            loss = loss + self._aux_loss(c)
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
            loss = self._data_loss(ur[0], ui[0], u_real, u_imag)
            loss = loss + self._aux_loss(c)
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
                                   config: Optional[PMCNetConfig] = None,
                                   frequencies_hz: Optional[np.ndarray] = None
                                   ) -> np.ndarray:
    """Построить аналитическую системную матрицу того же формата, что и
    измеренная (из калибровки сканера).

    Идея: каждая колонка n матрицы S — это частотный отклик прямого
    оператора на дельта-распределение концентрации в пикселе n. Для
    аналитической модели это вычисляется в закрытой форме:

        u_n(t) = −μ₀ · dA · s(r_n) · ∂M(r_n, t)/∂t

    Поскольку M(r, t) общая для всех пикселей, всё семейство колонок
    получается одним батчевым FFT / DFT — без циклов по пикселям,
    за O(N·T·logT) или O(N·T·K) соответственно.

    Возвращаемая матрица имеет форму `(n_meas_bins, Nx·Ny)` и совпадает
    по схеме раскладки с измеренной SM из BeihangUniversityData.

    ## Два режима компиляции

    **FFT-режим** (frequencies_hz=None): берёт первые `n_meas_bins/2`
    бинов rfft по каждой катушке. Частоты равны k * (1 / (T·dt)),
    что НЕ совпадает с реальными гармониками сканера. Подходит для
    случаев, когда Paper тоже работает в FFT-режиме.

    **DFT-режим** (frequencies_hz задан): вычисляет U(f_k) =
    ∫ u(t)·exp(−j·2π·f_k·t) dt напрямую на ЗАДАННЫХ частотах. Это
    КРИТИЧНО для согласования с PMCNet-Paper, который тоже работает
    в DFT-режиме на этих же частотах. Без этого — измерение в одной
    частотной сетке, forward в другой → Paper не может сойтись.

    Args:
        image_shape: (Nx, Ny).
        n_meas_bins: общее число строк (= 2 · число гармоник на катушку).
        config:     PMCNetConfig с физическими параметрами.
        frequencies_hz: (опц.) явные частоты для DFT-режима. Если задан,
                        n_meas_bins должно быть = 2 · len(frequencies_hz).
    """
    base_cfg = config or PMCNetConfig(image_size=tuple(image_shape))
    n_freq_per_coil = n_meas_bins // 2

    use_dft = frequencies_hz is not None
    if use_dft:
        if 2 * len(frequencies_hz) != n_meas_bins:
            raise ValueError(
                f"n_meas_bins={n_meas_bins} должно быть 2*len(frequencies_hz)"
                f"={2*len(frequencies_hz)}"
            )

    # Для FFT: повышаем T_samples под Nyquist. Для DFT: оставляем как есть.
    if use_dft:
        T_samples = base_cfg.n_time_samples
    else:
        T_samples = max(base_cfg.n_time_samples, 2 * n_freq_per_coil)

    cfg = PMCNetConfig(**{
        **base_cfg.__dict__,
        'image_size': tuple(image_shape),
        'n_time_samples': T_samples,
        # КРИТИЧНО: paper-faithful настройки (uniform p, forward FD)
        # ВНЕ ЗАВИСИМОСТИ от того, что было в base_cfg.
        'use_radial_coil': False,
        'use_central_fd': False,
        'use_debye': False,
    })

    # BasicAnalyticalForwardModel (paper-faithful), НЕ AnalyticalForwardModel.
    physics = BasicAnalyticalForwardModel(cfg)
    physics.eval()

    with torch.no_grad():
        Mx, My = physics._compute_magnetization()   # (Nx, Ny, T)
        dMx = physics.ddt(Mx)
        dMy = physics.ddt(My)
        s = physics.coil.sensitivity                # (Nx, Ny)
        scale = -physics.langevin.mu0 * physics.dA

        # u_n(t) — float64 для устойчивости pow(2) при нормировке
        ux = (scale * s.unsqueeze(-1) * dMx).double()
        uy = (scale * s.unsqueeze(-1) * dMy).double()

        Nx, Ny = image_shape
        N = Nx * Ny
        ux_flat = ux.reshape(N, T_samples)          # (N, T)
        uy_flat = uy.reshape(N, T_samples)

        if use_dft:
            # Explicit DFT на заданных частотах — точно как в
            # BasicHardConstrainedSpectralForward._setup_explicit_dft.
            # U(f_k) = Σ_t u(t) · exp(−j·2π·f_k·t)
            t = physics.ffp.t.double()                              # (T,)
            freqs = torch.tensor(np.asarray(frequencies_hz).flatten(),
                                  dtype=torch.float64)               # (K,)
            arg = 2.0 * math.pi * t.unsqueeze(-1) * freqs.unsqueeze(0)
            dft_cos = torch.cos(arg)                                 # (T, K)
            dft_sin = torch.sin(arg)
            # SM_x[k, n] = Σ_t u_x_n(t) · exp(−j 2π f_k t)
            Ux_re = ux_flat @ dft_cos                                # (N, K)
            Ux_im = -ux_flat @ dft_sin
            Uy_re = uy_flat @ dft_cos
            Uy_im = -uy_flat @ dft_sin
            Ux = (Ux_re + 1j * Ux_im).to(torch.complex128)
            Uy = (Uy_re + 1j * Uy_im).to(torch.complex128)
            SM = torch.cat([Ux.T, Uy.T], dim=0)                      # (2K, N)
        else:
            Ux = torch.fft.rfft(ux_flat, dim=-1)                     # complex
            Uy = torch.fft.rfft(uy_flat, dim=-1)
            SM = torch.cat([
                Ux[:, :n_freq_per_coil].T,
                Uy[:, :n_freq_per_coil].T,
            ], dim=0)                                                # (M, N)

        # Нормировка до единичного максимума
        SM_max = SM.abs().max().clamp_min(1e-300)
        SM = SM / SM_max

    return SM.cpu().numpy().astype(np.complex64)


# =============================================================================
# Шесть named-вариантов PMCNet для ablation — pipeline-ready
# =============================================================================
#
# Все шесть принимают одинаковый формат измерений (комплексный вектор
# гармоник из частотной области, как в `MPIReconstructionComparator`),
# что позволяет напрямую сравнивать их в одной таблице.
#
# Структура: одна общая «база» (PhysicsEnhanced) + три параллельные
# ветки одиночных улучшений, каждая изолирует один компонент.
#
#   1. PMCNetStandard         — paper Huang 2026 в чистом виде с
#                               ИЗМЕРЕННОЙ системной матрицей сканера
#                               (без аналитической физики).
#
#   2. PMCNetPaper            — paper-faithful реализация Eq. 1-3:
#                               LangevinAdiabatic + uniform p(r) +
#                               forward-FD ∂/∂t (без улучшений).
#
#   3. PMCNetPhysicsEnhanced  — БАЗА для веток улучшений: paper-Phys +
#                               langevin_safe + radial p(r). Forward-FD
#                               по-прежнему, single color, без Debye.
#
#   ─── параллельные ветки от Phys ──────────────────────────────────
#
#   4. PMCNetSoftConstrained  — Phys + Scheinker 2023 (PCNN): soft
#                               L2-штраф на ‖∇c‖₂² + freq-weighted L1
#                               на u. Без изменений forward-оператора.
#
#   5. PMCNetDebye            — Phys + ОДНО улучшение: релаксация Дебая
#                               как частотный фильтр H_τ(f) с обучаемой τ
#                               через softplus.
#
#   6. PMCNetCentralFD        — Phys + ОДНО улучшение: центральная FD
#                               через фиксированный Conv1d, O(Δt²)
#                               вместо O(Δt) у forward-FD (Maxwell-PCNN).
# =============================================================================


class PMCNetStandard(PMCNetReconstructor):
    """PMCNet-baseline: φ_θ(z) → c → S·c с ИЗМЕРЕННОЙ системной матрицей.

    ## Подход

    Прямой оператор реконструкции — линейное матричное умножение S·c, где
    S — комплексная системная матрица, измеренная на сканере заранее
    путём последовательного помещения точечного фантома в каждую точку
    сетки FOV и записи отклика. Размер S: (M, N), где M — число
    спектральных бинов измерения, N — число вокселей FOV.

    Loss = ||S·c − u_meas||₁, оптимизация через Adam, 20000 итераций
    на одно измерение.

    ## Когда использовать

    Этот вариант хорош, когда сканер уже откалиброван:
      • S содержит все реальные несовершенства сканера (нелинейности,
        паразитные гармоники, фазовые сдвиги АЦП) → высокое качество
        реконструкции на «родных» данных;
      • Не нужно знать физические параметры сканера.

    Минусы:
      • Требует **трудоёмкой калибровки**: измерение точечного фантома
        в каждом вокселе FOV занимает часы;
      • Результат привязан к конкретному сканеру: SM нельзя перенести
        на другой прибор.

    ## Сравнение с PMCNetPaper

    `PMCNetPaper` (paper-faithful версия) НЕ использует SM — вместо
    этого вычисляет физику аналитически из параметров сканера. Это
    основная идея метода PMCNet в литературе: «physical model-constrained
    network» снимает требование калибровки.

    Эта Standard-версия сохранена в пакете для прямого сравнения с
    paper-вариантом на одних и тех же данных — чтобы видеть, насколько
    проигрывает аналитическая физика SM-baseline'у при идеально
    откалиброванном сканере.
    """


class PMCNetPaper(PMCNetPaperReconstructor):
    """Paper-faithful PMCNet — реконструкция через явную физику без SM.

    ## Подход

    Прямой оператор `P(c)` строится из физических параметров сканера
    (драйв-частоты, градиент, амплитуда поля, диаметр частиц,
    температура), без какой-либо измеренной системной матрицы.

    Цепочка вычислений в `BasicAnalyticalForwardModel`:

      1. Lissajous-траектория FFP:
         r_FFP(t) = −(A/G) · sin(2π·f·t)  по каждой оси
      2. Полное поле:
         H(r, t) = G · (r − r_FFP(t))
      3. Намагниченность (адиабатическая, без Debye):
         M(r, t) = m_sat · L(α·|H|) · ê_H,  α = μ₀·m/(k_B·T)
      4. Производная намагниченности (forward FD):
         ∂M/∂t ≈ (M[t+1] − M[t]) / Δt
      5. Индуцированный сигнал (с uniform p(r) ≡ 1):
         u(t) = −μ₀ · ∫ p(r) · c(r) · ∂M/∂t · dr

    ## Алгоритм

      Init: θ ~ random, z ~ N(0,1) (фиксированный seed)
      Loop n_iterations (default 20000):
          ĉ = φ_θ(z)                       # U-Net → концентрация
          û = P(ĉ)                         # физический форвард
          L = ||û − u_meas||₁              # L1-невязка
          L.backward(); optimizer.step()    # Adam, lr = 1e-3
      Return ĉ

    Никакой обучающей выборки не требуется — это data-free режим:
    одна реконструкция на каждое измерение.

    ## Важно про измерения

    Для корректного результата u_meas должны быть сгенерированы той же
    физикой, что и P(c). Для синтетики это значит — через
    `BasicAnalyticalForwardModel`. Для реальных измерений сканера
    параметры (`drive_frequency`, `gradient`, etc.) должны быть точно
    подгружены из его metadata (см. `_load_scanner_params_from_h5` в
    pipeline.py) — иначе физика модели и реальности расходятся,
    реконструкция становится бесполезной.

    ## Без улучшений (база для четырёх параллельных веток)

    Этот вариант реализует только базовую физику Eq. 1-3. Четыре
    параллельных preset'а наследуются от него, каждый добавляя ровно
    одно улучшение через config-флаг:

      • `PMCNetRadialCoil`  — `use_radial_coil=True`
      • `PMCNetSoftConstrained` — `lambda_grad=1e-4, use_freq_weighting=True`
      • `PMCNetDebye`       — `use_debye=True`
      • `PMCNetCentralFD`   — `use_central_fd=True`

    Все четыре используют тот же `BasicHardConstrainedSpectralForward`
    с разными значениями флагов в config — структурно они идентичны
    и отличаются только конфигурацией.
    """


# =============================================================================
# Четыре параллельных preset'а на базе PMCNetPaper
# =============================================================================
# Каждый меняет РОВНО ОДИН config-флаг относительно PMCNet-Paper:
#
#   PMCNet-Paper       use_radial_coil=F  use_central_fd=F  use_debye=F  λ_grad=0  freq_w=F
#   PMCNet-RadialCoil  use_radial_coil=T  use_central_fd=F  use_debye=F  λ_grad=0  freq_w=F
#   PMCNet-Soft        use_radial_coil=F  use_central_fd=F  use_debye=F  λ_grad>0  freq_w=T
#   PMCNet-Debye       use_radial_coil=F  use_central_fd=F  use_debye=T  λ_grad=0  freq_w=F
#   PMCNet-CentralFD   use_radial_coil=F  use_central_fd=T  use_debye=F  λ_grad=0  freq_w=F
#
# Все пять наследуются от одного и того же `PMCNetPaperReconstructor`,
# единственное отличие — это конкретные значения config-флагов. Это
# идеальный изолированный ablation: каждый «эксперт» отвечает за ровно
# одну поправку к paper-faithful базе.
# =============================================================================


class PMCNetRadialCoil(PMCNetPaperReconstructor):
    """PMCNet-Paper + ОДНО улучшение: радиальная чувствительность катушки.

    Заменяет uniform p(r) ≡ 1 (paper-faithful) на физически реалистичный
    профиль `p(r) = 1 / (1 + (|r|/R_coil)²)` — конечный радиус приёмной
    катушки делает её слабее на краях FOV. Без этой поправки PMCNet
    переоценивает концентрацию на периферии: модель «думает», что
    дальние точки производят такой же сигнал, что и центральные.

    Никаких других изменений относительно `PMCNetPaper`:
      ▸ Forward-FD для ∂M/∂t (центральная FD — в `PMCNetCentralFD`);
      ▸ Без Debye-релаксации (— в `PMCNetDebye`);
      ▸ Без soft-constraints (— в `PMCNetSoftConstrained`);
      ▸ Loss L1 без частотных весов.

    Прежнее имя класса — `PMCNetPhysicsEnhanced` — содержало в себе
    одновременно радиальную катушку и `langevin_safe`. Поскольку
    `langevin_safe` теперь используется во ВСЕХ форвардах (численная
    необходимость для float32), отличающим компонентом остаётся только
    радиальная катушка. Переименовано в `PMCNetRadialCoil` для ясности.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 frequencies_hz: Optional[np.ndarray] = None):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            # ЕДИНСТВЕННОЕ отличие от Paper:
            'use_radial_coil': True,
            # Всё остальное — paper-faithful
            'use_central_fd': False,
            'use_debye': False,
            'lambda_tv': 0.0,
            'lambda_grad': 0.0,
            'use_freq_weighting': False,
            'n_colors': 1,
        })
        super().__init__(image_shape, n_meas_bins,
                         config=cfg, frequencies_hz=frequencies_hz)


class PMCNetSoftConstrained(PMCNetPaperReconstructor):
    """PMCNet-Paper + ОДНО улучшение: soft constraints (Scheinker 2023).

    Добавляет к paper-faithful базе два auxiliary loss-терма из подхода
    Scheinker & Pokharel 2023 (APL Mach. Learn., «PCNN for electrodynamics»):

      ▸ **L2-штраф на пространственный градиент**: λ · ‖∇c‖₂² через
        фиксированное central-FD ядро. В отличие от L1-TV, гладкий
        штраф мягко гасит мелкомасштабные осцилляции, не выпрямляя
        границы в ступеньки. Аналог Eq. 12-13 из paper Scheinker.

      ▸ **Частотно-взвешенный L1 на данных**: вес `w_k = (k+1)^0.5`
        компенсирует затухание амплитуд гармоник MPI как ~1/k —
        иначе loss доминируется первыми гармониками, а высокие
        (которые несут разрешение) игнорируются сетью.

    Никаких изменений в forward-операторе — это «чисто loss-level»
    модификация над paper-faithful физикой. Forward тот же, что у
    `PMCNetPaper`: uniform p(r), forward-FD, без Debye.

    ## Параметры по умолчанию

      λ_grad = 1e-4, freq_weighting_power = 0.5.

    Подобраны как мягкие: видимый эффект, но без риска разнести
    реконструкцию.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 frequencies_hz: Optional[np.ndarray] = None):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            # ЕДИНСТВЕННОЕ отличие от Paper: soft-constraints
            'lambda_grad': 1e-4,
            'use_freq_weighting': True,
            'freq_weighting_power': 0.5,
            # Всё остальное — paper-faithful
            'use_radial_coil': False,
            'use_central_fd': False,
            'use_debye': False,
            'lambda_tv': 0.0,
            'n_colors': 1,
        })
        super().__init__(image_shape, n_meas_bins,
                         config=cfg, frequencies_hz=frequencies_hz)


class PMCNetDebye(PMCNetPaperReconstructor):
    """PMCNet-Paper + ОДНО улучшение: релаксация Дебая (обучаемая τ).

    Добавляет к paper-faithful базе фильтр H_τ(f) = 1/(1 + j·2π·f·τ),
    применяемый к выходу forward в частотной области. Параметр τ
    обучается совместно с весами сети через `softplus(raw_τ)` —
    гарантирует положительность по построению.

    Физический смысл (paper Eq. 4-5):
      Намагниченность частиц не следует мгновенно за полем — есть
      инерция со временем релаксации τ, моделируемая ODE первого
      порядка: τ · dM_D/dt = −M_D + M. В частотной области это
      эквивалентно умножению на лоупасс H_τ(f).

    paper Sec. III.B: «we did not provide the magnitude of the
    relaxation time constant directly, but instead estimated it
    through the gradient descent algorithm».

    Никаких других изменений относительно PMCNet-Paper:
      ▸ Uniform p(r) ≡ 1 (радиальная — в `PMCNetRadialCoil`);
      ▸ Forward-FD (central FD — в `PMCNetCentralFD`);
      ▸ Loss L1, без soft constraints.
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 frequencies_hz: Optional[np.ndarray] = None):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            # ЕДИНСТВЕННОЕ отличие от Paper: обучаемая Debye
            'use_debye': True,
            # Всё остальное — paper-faithful
            'use_radial_coil': False,
            'use_central_fd': False,
            'lambda_tv': 0.0,
            'lambda_grad': 0.0,
            'use_freq_weighting': False,
            'n_colors': 1,
        })
        super().__init__(image_shape, n_meas_bins,
                         config=cfg, frequencies_hz=frequencies_hz)


class PMCNetCentralFD(PMCNetPaperReconstructor):
    """PMCNet-Paper + ОДНО улучшение: центральная разность через Conv1d.

    Заменяет forward-FD (`(M[t+1] − M[t])/Δt`, точность O(Δt)) на
    центральную разность с симметричным ядром `[−1, 0, +1]/(2Δt)`,
    точность O(Δt²). Реализована как Conv1d с фиксированным
    (не-обучаемым) ядром — совместима с autograd «бесплатно» и
    переносима на GPU без ручных циклов (Scheinker 2023, Eq. 12-13).

    Эффект: forward-FD имеет систематический временной сдвиг Δt/2,
    что в частотной области даёт линейный фазовый сдвиг
    `exp(−j·π·f·Δt)`. Центральная FD симметрична и не вносит сдвига,
    давая ровно вдвое лучшую сходимость к точной производной
    (в пределе Δt → 0 точность ×2).

    Никаких других изменений относительно PMCNet-Paper:
      ▸ Uniform p(r) ≡ 1 (радиальная — в `PMCNetRadialCoil`);
      ▸ Без Debye (— в `PMCNetDebye`);
      ▸ Без soft-constraints (— в `PMCNetSoftConstrained`).
    """

    def __init__(self, image_shape: Tuple[int, int],
                 n_meas_bins: int,
                 config: Optional[PMCNetConfig] = None,
                 frequencies_hz: Optional[np.ndarray] = None):
        base = config or PMCNetConfig(image_size=tuple(image_shape))
        cfg = PMCNetConfig(**{
            **base.__dict__,
            'image_size': tuple(image_shape),
            # ЕДИНСТВЕННОЕ отличие от Paper: central FD через Conv1d
            'use_central_fd': True,
            # Всё остальное — paper-faithful
            'use_radial_coil': False,
            'use_debye': False,
            'lambda_tv': 0.0,
            'lambda_grad': 0.0,
            'use_freq_weighting': False,
            'n_colors': 1,
        })
        super().__init__(image_shape, n_meas_bins,
                         config=cfg, frequencies_hz=frequencies_hz)


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
    # Шесть named-вариантов (pipeline-ready, ablation-структура)
    'PMCNetStandard',           # SM из калибровки (baseline)
    'PMCNetPaper',              # paper-faithful (Huang 2026, Eq. 1-3) — общая БАЗА
    'PMCNetRadialCoil',         # Paper + radial p(r) = 1/(1+(r/R)²)
    'PMCNetSoftConstrained',    # Paper + Scheinker 2023 soft constraints
    'PMCNetDebye',              # Paper + Debye-релаксация (обучаемая τ)
    'PMCNetCentralFD',          # Paper + центральная FD через Conv1d
]
