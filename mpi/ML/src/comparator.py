"""Унифицированное сравнение методов реконструкции MPI.

## Что делает

`MPIReconstructionComparator` принимает на вход:
  • Системную матрицу сканера (из калибровочного H5);
  • Любой набор моделей реконструкции через сеттеры (`set_cnn_model`,
    `set_dip_model`, и т. д.);

и предоставляет:
  • Унифицированный метод `compare_all_methods_on_image(image, measurement)`,
    который прогоняет ВСЕ зарегистрированные методы и собирает метрики
    в единую таблицу;
  • Метод `run_phantom_battery(battery)` для пакетной обработки
    нескольких фантомов;
  • Сводные функции `print_summary_table()`, `save_results_to_file()`,
    `_visualize_all_comparison()` для отчётов.

## Поддерживаемые методы

  Классические:
    • Tikhonov (создаётся автоматически из переданной SM);
    • Kaczmarz/ART (то же).

  Регистрируемые через сеттеры (опциональные):
    • set_cnn_model(trainer)        — CNN baseline;
    • set_modl_model(trainer)       — Model-based Deep Learning;
    • set_diffusion_model(trainer)  — DDPM baseline;
    • set_chae_model(model)         — Chae 2017 single-layer FC;
    • set_dip_model(model)          — Deep Image Prior;
    • set_pmcnet_standard(...)      — PMCNet с измеренной SM;
    • set_pmcnet_paper(...)         — PMCNet paper-faithful;
    • set_pmcnet_radial_coil(...) — + физические улучшения;
    • set_pmcnet_soft(...)          — Phys + soft constraints (Scheinker 2023);
    • set_pmcnet_debye(...)         — Phys + Debye-релаксация (обучаемая τ);
    • set_pmcnet_central_fd(...)    — Phys + центральная FD через Conv1d;
    • set_moe(moe)                  — Mixture of Experts.

  Метод, не зарегистрированный через сеттер, автоматически пропускается
  и помечается «ПРОПУЩЕН» в таблице.

## Унификация форматов

Все методы внутри принимают measurement как complex (2, M/2) и
возвращают reconstruction как (Nx, Ny) float. Compatator скрывает
конверсии между разными внутренними форматами моделей.

## Метрики

  • **SSIM** — structural similarity (визуальное качество структур);
  • **PSNR** — peak signal-to-noise ratio (общее качество в dB);
  • **FWHM** — full width at half maximum точечной функции рассеяния
    (мера разрешения);
  • **time** — секунды на реконструкцию (для оценки practical
    применимости).
"""

import numpy as np
import time
import os
import h5py
from matplotlib import pyplot as plt
from tqdm import tqdm

from .models import (
    TikhonovReconstructor, KatsMarcAlgorithm,
)
from .metrics import MetricsCalculator
from .visualization import Visualization
import torch


class MPIReconstructionComparator:
    """Сравнение методов реконструкции"""

    def __init__(self, system_matrix_path=None):
        self.load_system_matrix(system_matrix_path)
        self.cnn_trainer = None
        self.modl_trainer = None
        self.diffusion_trainer = None

        # Модели по статьям
        self.chae_model = None
        self.chae_multi_model = None  # Chae 2017 multi-layer вариант
        self.dip_model = None
        # PMCNet (Huang et al., 2026) — три варианта в одной системе координат
        self.pmcnet_standard = None                    # 1) измеренная SM (baseline)
        self.pmcnet_paper = None                       # 2) paper-faithful (Huang 2026 Eq. 1-3)
        self.pmcnet_radial_coil = None            # 3) paper + physical улучшения
        self.pmcnet_soft = None                        # 4) Phys + Scheinker 2023 soft constraints
        self.pmcnet_debye = None                       # 5) Phys + только Debye-релаксация
        self.pmcnet_central_fd = None                  # 6) Phys + только центральная FD + TV

        # Mixture of Experts поверх остальных методов
        self.moe = None
        self.katsmarc = self.katsmarc if hasattr(self, 'katsmarc') else None

        self.results = []

    def load_system_matrix(self, path):
        """Загрузка системной матрицы"""
        if path is None:
            possible_paths = [
                './DATA/SystemMatrix.h5',
                '../DATA/SystemMatrix.h5',
                './SystemMatrix.h5',
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    path = p
                    break
            else:
                raise FileNotFoundError("Не найден файл SystemMatrix.h5")

        fSM = h5py.File(path, 'r')
        S_data_r = fSM['/measurement/data/r'][:]
        S_data_i = fSM['/measurement/data/i'][:]
        S = S_data_r + 1j * S_data_i
        isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()
        S = S[:, :, isBG == 0]
        self.SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

        number_Position = fSM['/calibration/size'][:].squeeze()
        self.nx, self.ny = int(number_Position[0]), int(number_Position[1])
        self.image_shape = (self.nx, self.ny)

        fSM.close()

        self.tikhonov_reconstructor = TikhonovReconstructor(self.SM)

        # Инициализация KatsMarc (классический алгоритм)
        # Источник: Kaczmarz, S. (1937)
        self.katsmarc = KatsMarcAlgorithm(self.SM)

        print(f"Размер изображения: {self.image_shape}")
        print(f"Размер системной матрицы: {self.SM.shape}")

    def set_cnn_model(self, cnn_trainer):
        self.cnn_trainer = cnn_trainer

    def set_modl_model(self, modl_trainer):
        self.modl_trainer = modl_trainer

    def set_diffusion_model(self, diffusion_trainer):
        self.diffusion_trainer = diffusion_trainer

    def set_chae_model(self, chae_model):
        """Установка модели Chae (2017) — однослойный вариант."""
        self.chae_model = chae_model

    def set_chae_multi_model(self, chae_multi_model):
        """Установка модели Chae (2017) — двухслойный (multi-layer) вариант.

        Согласно Sec. III.3 статьи: hidden layer даёт двухпорядковое
        улучшение MSE на частицах <40 нм. В нашем сетапе они выступают
        как самостоятельные эксперты в MoE.
        """
        self.chae_multi_model = chae_multi_model

    def set_dip_model(self, dip_model):
        """Установка модели Deep Image Prior (Dittmer et al., 2020)"""
        self.dip_model = dip_model

    def set_pmcnet_standard(self, reconstructor):
        """Вариант 1 — PMCNet-Standard (SM-baseline, _не_ из paper).

        Прямой оператор — ИЗМЕРЕННАЯ системная матрица из калибровки
        сканера (S·c). Сохраняется как независимый baseline параллельно
        с paper-faithful PMCNet, чтобы видеть метрики обоих подходов.
        """
        self.pmcnet_standard = reconstructor

    def set_pmcnet_paper(self, reconstructor):
        """Вариант 2 — PMCNet-Paper (paper-faithful, Huang 2026 Eq. 1-3).

        Прямой оператор — `BasicAnalyticalForwardModel` (paper-faithful
        Langevin adiabatic, p(r) ≡ 1, forward-FD для ∂/∂t, без Debye).
        Соответствует pseudocode Алгоритма 1 статьи: φ_θ(z) → ĉ, P(ĉ)
        через явную физику, loss = L1.
        """
        self.pmcnet_paper = reconstructor

    def set_pmcnet_radial_coil(self, reconstructor):
        """Вариант 3 — PMCNet-RadialCoil (Paper + ОДНО улучшение).

        Заменяет uniform p(r) ≡ 1 на радиальную чувствительность катушки
        p(r) = 1/(1+(r/R)²). Никаких других изменений относительно
        PMCNet-Paper. Параллельная ветка к Soft, Debye, CentralFD.
        """
        self.pmcnet_radial_coil = reconstructor

    def set_pmcnet_soft(self, reconstructor):
        """Вариант 4 — PMCNet-Soft (Phys + soft constraints, Scheinker 2023).

        Добавляет к Phys ТОЛЬКО мягкие auxiliary loss'ы — L2-штраф на ∇c
        (Scheinker Eq. 12-13) и частотно-взвешенный L1 на u. Без обучаемой
        τ и multi-color. Изолирует вклад PINN-style soft constraints.
        """
        self.pmcnet_soft = reconstructor

    def set_pmcnet_debye(self, reconstructor):
        """Вариант 5 — PMCNet-Debye (Phys + только Debye-релаксация).

        Добавляет к Phys ТОЛЬКО релаксацию Дебая (paper Eq. 4-5) с
        обучаемой τ через softplus. Без multi-color, TV, soft-constraints,
        central FD. Изолирует вклад модели инерции намагниченности частиц.
        """
        self.pmcnet_debye = reconstructor

    def set_pmcnet_central_fd(self, reconstructor):
        """Вариант 6 — PMCNet-CentralFD (Phys + только центральная FD).

        Добавляет к Phys ТОЛЬКО более точную дискретизацию ∂M/∂t через
        Conv1d с фиксированным ядром [−1, 0, +1]/(2Δt) — точность O(Δt²)
        вместо O(Δt) у forward-FD в Phys. Изолирует эффект схемы
        дискретизации производной (Maxwell-PCNN Eq. 12-13).
        """
        self.pmcnet_central_fd = reconstructor

    def set_moe(self, moe):
        """Установить Mixture of Experts поверх остальных методов.

        MoE сам внутри прогоняет всех своих экспертов и комбинирует их
        выходы (через mean / scalar / spatial gating).
        """
        self.moe = moe

    def moe_reconstruction(self, measurement):
        """Реконструкция через MoE: эксперты + комбинирование."""
        if self.moe is None:
            raise ValueError("MoE модель не установлена")
        recon = self.moe.reconstruct(measurement)
        return self._postprocess_recon(recon)

    def generate_test_case(self, radius=0.2, distance=0.2, intensity1=0.7, intensity2=0.7):
        """Генерация тестового случая с двумя каплями"""
        image = np.zeros((self.nx, self.ny))

        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        center1_x = -distance / 2
        center2_x = distance / 2

        sigma = radius / 2.5

        r1 = np.sqrt((X - center1_x) ** 2 + Y ** 2)
        image += intensity1 * np.exp(-(r1 ** 2) / (2 * sigma ** 2))

        r2 = np.sqrt((X - center2_x) ** 2 + Y ** 2)
        image += intensity2 * np.exp(-(r2 ** 2) / (2 * sigma ** 2))

        if image.max() > 0:
            image = image / image.max()

        return image

    def generate_measurement(self, image):
        """Генерация измерений из изображения"""
        image_vector = image.reshape(-1, 1)
        measurement = self.SM @ image_vector
        measurement_reshaped = measurement.reshape(2, -1)
        return measurement_reshaped

    def tikhonov_reconstruction(self, measurement, mu=None, kmax=100):
        """Tikhonov-реконструкция. μ берётся из `self.tikhonov_mu`
        (выставляется через `set_tikhonov_mu()` после cross-validation),
        либо из аргумента, либо 1e-3 по умолчанию.
        """
        if mu is None:
            mu = getattr(self, 'tikhonov_mu', 1e-3)
        recon = self.tikhonov_reconstructor.reconstruct(measurement, mu, kmax)
        recon = np.asarray(recon).reshape(self.image_shape)
        if recon.max() > 0:
            recon = recon / recon.max()
        return recon

    def set_tikhonov_mu(self, mu: float):
        """Задать μ для Tikhonov (после `tune_tikhonov_mu` в pipeline)."""
        self.tikhonov_mu = float(mu)

    # ====================================================================
    # МЕТОД 1: Chae (2017) - Однослойная полносвязная нейронная сеть
    # ====================================================================
    @staticmethod
    def _model_device(model):
        """Возвращает device первого параметра nn.Module (или 'cpu')."""
        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device('cpu')

    def _chae_inference(self, model, measurement):
        """Общий код инференса для Chae single/multi.

        Препроцессинг согласован с обучением (см. pipeline._flatten_measurement_batch):
          1. Амплитудный спектр |u| по обеим катушкам (paper Sec. III.1);
          2. Конкатенация в плоский вектор длины 2·M_per_coil;
          3. Per-sample max-нормировка → значения в [0, 1] под сигмоид.

        Re/Im разложение не используется — Chae-веса не сходятся к
        Чебышёв-полиномам на нём.
        """
        # Амплитудный спектр + per-sample max-нормировка (как в pipeline)
        abs_spec = np.abs(measurement).astype(np.float32)
        meas_vector = abs_spec.flatten()
        max_val = float(meas_vector.max())
        if max_val > 0:
            meas_vector = meas_vector / max_val

        dev = self._model_device(model)
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32,
                                   device=dev)

        model.eval()
        with torch.no_grad():
            reconstructed_flat = model(meas_tensor.unsqueeze(0))

        reconstructed = reconstructed_flat.cpu().numpy().reshape(self.image_shape)

        # Финальная нормализация для согласования с метриками SSIM/PSNR
        # (target y_train также был нормирован на max во время обучения).
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()
        return reconstructed

    def chae_reconstruction(self, measurement):
        """Реконструкция Chae (2017) — однослойный вариант."""
        if self.chae_model is None:
            raise ValueError("Chae модель не установлена")
        return self._chae_inference(self.chae_model, measurement)

    def chae_multi_reconstruction(self, measurement):
        """Реконструкция Chae (2017) — двухслойный (multi-layer) вариант.

        Та же логика препроцессинга, что у single — отличаются только
        веса модели. См. Sec. III.3 paper про преимущество multi-layer.
        """
        if self.chae_multi_model is None:
            raise ValueError("Chae-multi модель не установлена")
        return self._chae_inference(self.chae_multi_model, measurement)

    # ====================================================================
    # МЕТОД 2: Dittmer et al. (2020) - Deep Image Prior (DIP)
    # ====================================================================
    def dip_reconstruction(self, measurement, n_iterations=6000,
                            patience: int = 500):
        # n_iterations 3000 → 6000, patience 250 → 500: на предыдущем
        # прогоне DIP давал SSIM 0.05-0.90 (огромный разброс), а early
        # stop часто срабатывал поздно (>2000 итер). Удвоение даёт
        # запас сходимости на сложных фантомах без риска переобучения
        # к шуму — patience всё равно остановит, когда loss-plateau.
        """Реконструкция Deep Image Prior c early stopping и фикс. z.

        Изменения относительно прежней версии (соответствие Dittmer 2020):
          • **Фиксированный z**: убрано `latent_z.requires_grad=True` —
            paper явно фиксирует вход (Sec. II.C), trainable z ломает
            spectral-bias регуляризацию DIP.
          • **Early stopping**: трекаем running-min loss; если за
            `patience` итераций нет улучшения — останавливаемся и
            возвращаем `best_x`. Paper Sec. IV: успех DIP именно
            благодаря выбору момента остановки.
          • **lr=1e-3** (было 0.01) — Adam с lr=0.01 на DIP-генераторе
            типично взрывается за 200 итераций.
          • **n_iterations=6000** дефолт (поднято с 3000) — даёт early
            stopping шанс отработать на сложных фантомах.
        """
        if self.dip_model is None:
            raise ValueError("DIP модель не установлена")

        import torch.optim as optim

        # Создаем копию модели на device оригинала
        import copy
        dip_model = copy.deepcopy(self.dip_model)
        dev = self._model_device(dip_model)
        dip_model.train()

        # Подготовка измерений (на том же device)
        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32,
                                   device=dev).unsqueeze(0)

        # Создаем матрицу A для прямого оператора (один раз, на нужном device)
        cache_key = f'_A_tensor_T_{dev}'
        if not hasattr(self, cache_key):
            if np.iscomplexobj(self.SM):
                A_extended = np.vstack([np.real(self.SM), np.imag(self.SM)])
            else:
                A_extended = self.SM
            A_tensor = torch.tensor(A_extended, dtype=torch.float32, device=dev)
            setattr(self, cache_key, A_tensor.T)
        self.A_tensor_T = getattr(self, cache_key)

        # Оптимизатор — только веса сети
        optimizer = optim.Adam(dip_model.parameters(), lr=1e-3)

        # Латент: ФИКСИРОВАННЫЙ (no gradient) — paper Dittmer 2020 Sec. II.C.
        # Прежнее requires_grad=True позволяло сети «жульничать» через z,
        # обходя architecture-induced регуляризацию.
        latent_z = dip_model.generate_random_latent().to(dev)
        # latent_z.requires_grad остаётся False по умолчанию

        # Early-stopping state
        best_loss = float('inf')
        best_x = None
        no_improve = 0

        print(f"  Оптимизация DIP (до {n_iterations} итераций, "
              f"patience={patience})...")
        for iteration in range(n_iterations):
            optimizer.zero_grad()

            generated_image = dip_model(latent_z)
            generated_flat = generated_image.view(1, -1)
            measurement_pred = generated_flat @ self.A_tensor_T

            # L1-loss согласно paper Sec. II.C ("we use p = 1").
            loss = torch.mean(torch.abs(measurement_pred - meas_tensor))
            loss.backward()
            optimizer.step()

            cur = loss.item()
            if cur < best_loss * 0.999:  # требуем заметного улучшения
                best_loss = cur
                best_x = generated_image.detach().clone()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop @ iter {iteration} "
                          f"(no improve for {patience}), best_loss={best_loss:.6f}")
                    break

            if iteration % 200 == 0:
                print(f"    DIP iter {iteration}, loss: {cur:.6f}, "
                      f"best: {best_loss:.6f}")

        # Возвращаем лучший snapshot, а не финал
        if best_x is None:
            best_x = generated_image.detach()
        reconstructed = best_x[0, 0].cpu().numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        # Изменение размера если необходимо
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        return reconstructed

    # ====================================================================
    # МЕТОД 3: Huang et al. (2026) - PMCNet (Physical Model-Constrained Net)
    # ====================================================================
    def _measurement_to_complex_vector(self, measurement):
        """Стандартный путь: measurement формы (2, n_freq) → комплексный вектор (M,)
        как в `main.py`: Meas = [u_data[0,:], u_data[1,:]]."""
        meas = np.asarray(measurement)
        if meas.ndim == 2 and meas.shape[0] == 2:
            return np.concatenate([meas[0, :], meas[1, :]])
        return meas.flatten()

    def _postprocess_recon(self, reconstructed):
        """Привести реконструкцию к image_shape и нормировать на [0, 1]."""
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(
                reconstructed,
                (self.nx / reconstructed.shape[0],
                 self.ny / reconstructed.shape[1]),
                order=1,
            )
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()
        return reconstructed

    # -- Вариант 1: PMCNet-Standard (измеренная SM) ---------------------------
    def pmcnet_standard_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Standard: u = S_measured · c, L1, _не_ из paper."""
        if self.pmcnet_standard is None:
            raise ValueError("PMCNet-Standard модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_standard.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 2: PMCNet-Paper (paper-faithful, Huang 2026 Eq. 1-3) ---------
    def pmcnet_paper_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Paper: u = BasicAnalyticalForward(c), без улучшений.

        Прямой оператор реализует ровно Eq. 1-3 статьи: Langevin
        adiabatic, p(r) ≡ 1, ∂/∂t через forward-difference. Loss = L1.
        Идеально работает на u_meas, сгенерированных через ту же
        аналитическую физику (paper Sec. III.A: "u_meas and u use the
        same physical model for calculation").
        """
        if self.pmcnet_paper is None:
            raise ValueError("PMCNet-Paper модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_paper.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 3: PMCNet-RadialCoil (Paper + только radial p(r)) ------------
    def pmcnet_radial_coil_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-RadialCoil: Paper + ОДНО улучшение — радиальная p(r).

        Заменяет uniform p(r) ≡ 1 на физически реалистичный профиль
        p(r) = 1/(1+(|r|/R_coil)²). Всё остальное — paper-faithful.
        Параллельная ветка Soft / Debye / CentralFD.
        """
        if self.pmcnet_radial_coil is None:
            raise ValueError("PMCNet-RadialCoil модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_radial_coil.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 4: PMCNet-Soft (Paper + Scheinker soft constraints) ----------
    def pmcnet_soft_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Soft: Paper + ОДНО улучшение — Scheinker 2023 soft loss.

        Forward тот же, что у PMCNet-Paper (uniform p(r), forward-FD).
        В loss добавлены два auxiliary штрафа:
          • λ_grad · ‖∇c‖₂² (Scheinker 2023, Eq. 12-13);
          • frequency-weighted L1 на u_real/u_imag.

        Никаких изменений в forward-операторе.
        """
        if self.pmcnet_soft is None:
            raise ValueError("PMCNet-Soft модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_soft.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 5: PMCNet-Debye (Phys + только Debye-релаксация) -------------
    def pmcnet_debye_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Debye: Phys + обучаемая Debye-релаксация (без multi-color, без TV).

        Оценённое τ доступно через `self.pmcnet_debye.network.debye.tau_seconds`
        для отчёта (в отличие от прежнего PMCNet-Final, где τ возвращалось
        вторым элементом кортежа — теперь reconstruct() даёт чистый image).
        """
        if self.pmcnet_debye is None:
            raise ValueError("PMCNet-Debye модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_debye.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        # Сохраним τ для отчёта (single-color → один скаляр)
        try:
            self.last_pmcnet_taus_seconds = (
                self.pmcnet_debye.network.debye.tau_seconds
                .detach().cpu().numpy()
            )
        except AttributeError:
            self.last_pmcnet_taus_seconds = None
        return self._postprocess_recon(recon)

    # -- Вариант 6: PMCNet-CentralFD (Phys + только central FD) ---------------
    def pmcnet_central_fd_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-CentralFD: Phys + центральная разность через Conv1d.

        Всё то же, что у PhysicsEnhanced, но с точностью O(Δt²) у
        производной ∂M/∂t. Симметричное ядро [−1, 0, +1]/(2Δt) реализовано
        фиксированным Conv1d без обучаемых весов.
        """
        if self.pmcnet_central_fd is None:
            raise ValueError("PMCNet-CentralFD модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_central_fd.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # ====================================================================
    # МЕТОД 6: KatsMarc (Алгоритм Кацмарца, 1937)
    # ====================================================================
    def katsmarc_reconstruction(self, measurement, n_iterations=None,
                                 relaxation=None):
        """Реконструкция алгоритмом Кацмарца (Kaczmarz, 1937).

        Параметры по умолчанию задаются в `KatsMarcAlgorithm.reconstruct`
        (n_iterations=20, relaxation=0.3, damp_schedule=False) — они
        откалиброваны под globally-normalised rows. Здесь None означает
        «использовать дефолты алгоритма».
        """
        if self.katsmarc is None:
            print("  KatsMarc алгоритм не инициализирован")
            return None

        meas_vector = np.concatenate([measurement[0, :], measurement[1, :]])
        kwargs = {}
        if n_iterations is not None:
            kwargs['n_iterations'] = n_iterations
        if relaxation is not None:
            kwargs['relaxation'] = relaxation
        reconstructed = self.katsmarc.reconstruct(meas_vector, **kwargs)
        reconstructed = reconstructed.reshape(self.image_shape)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def cnn_reconstruction(self, measurement):
        if self.cnn_trainer is None:
            raise ValueError("CNN модель не установлена")

        reconstructed = self.cnn_trainer.predict(measurement)

        # Обработка разных форматов вывода
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        if reconstructed.ndim == 4:
            reconstructed = reconstructed[0, 0, :, :]
        elif reconstructed.ndim == 3:
            reconstructed = reconstructed[0, :, :]
        elif reconstructed.ndim == 2:
            pass  # уже правильная форма
        else:
            raise ValueError(f"Неожиданная форма вывода: {reconstructed.shape}")

        # Изменение размера до нужного
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    def modl_reconstruction(self, measurement):
        """Реконструкция методом MoDL"""
        if self.modl_trainer is None:
            raise ValueError("MoDL модель не установлена")

        import torch

        # Измерения имеют форму (2, n_measurements)
        # Для MoDL нужно объединить реальную и мнимую части в вектор
        real_part = measurement.real  # (2, n_measurements)
        imag_part = measurement.imag  # (2, n_measurements)

        # Объединяем: сначала реальные части обеих катушек, затем мнимые
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)
        meas_tensor = meas_tensor.to(self.modl_trainer.device)

        self.modl_trainer.model.eval()
        with torch.no_grad():
            reconstructed = self.modl_trainer.model(meas_tensor)

        # Обработка вывода
        if isinstance(reconstructed, tuple):
            reconstructed = reconstructed[0]

        if reconstructed.ndim == 4:
            reconstructed = reconstructed[0, 0].cpu().numpy()
        elif reconstructed.ndim == 3:
            reconstructed = reconstructed[0].cpu().numpy()
        else:
            reconstructed = reconstructed.cpu().numpy()

        # Нормализация
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        # Изменение размера при необходимости
        if reconstructed.shape != self.image_shape:
            from scipy import ndimage
            reconstructed = ndimage.zoom(reconstructed,
                                         (self.nx / reconstructed.shape[0],
                                          self.ny / reconstructed.shape[1]),
                                         order=1)

        return reconstructed

    def diffusion_reconstruction(self, measurement):
        """Условная реконструкция: condition = Tikhonov-recon измерения.

        Diffusion обучен в режиме «уточни грубую Tikhonov-реконструкцию
        до GT», поэтому inference тоже идёт с этим условием.
        """
        if self.diffusion_trainer is None:
            raise ValueError("Diffusion модель не установлена")

        # 1) Грубая Tikhonov-реконструкция как condition
        meas_vector = np.concatenate([measurement[0, :], measurement[1, :]])
        try:
            cond = self.tikhonov_reconstructor.reconstruct(
                meas_vector, mu=1e-2, kmax=15).reshape(self.image_shape)
        except Exception:
            cond = np.zeros(self.image_shape, dtype=np.float32)
        cond = cond.astype(np.float32)
        if cond.max() > 0:
            cond = cond / cond.max()

        dev = self._model_device(self.diffusion_trainer.model)
        cond_tensor = torch.tensor(cond[None, None],
                                    dtype=torch.float32, device=dev)

        # 2) Условное сэмплирование. Используем ПОЛНЫЕ n_steps модели —
        # x инициализируется как чистый шум (соответствует t = n_steps-1),
        # поэтому обратный цикл должен идти от того же максимального t.
        # Передача n_steps=50 при model.n_steps=100 ломала train/test
        # consistency: модель училась с t ∈ [0, 100), а sample стартовал
        # с t=49 при pure-noise входе → out-of-distribution → шум на выходе.
        self.diffusion_trainer.model.eval()
        with torch.no_grad():
            x = self.diffusion_trainer.model.sample(
                cond_tensor,
                n_steps=self.diffusion_trainer.model.n_steps,
            )
            reconstructed = x[0, 0].cpu().numpy()

        # Концентрация ≥ 0 (физика). DDPM может выдать отрицательные
        # значения в первых итерациях обучения — клиппим, иначе деление
        # на max при отрицательном максимуме переворачивает изображение.
        reconstructed = np.clip(reconstructed, 0.0, None)
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()
        return reconstructed

    def load_openmpi_data(self, data_dir=None):
        """Загрузка OpenMPI датасета для проверки моделей.

        data_dir=None — путь определяется автоматически
        (локальная папка ChineseData/OpenMPIData, без скачивания).
        """
        from .data.openmpi import OpenMPIDataManager

        print("\n" + "=" * 70)
        print("ЗАГРУЗКА OPENMPI ДАТАСЕТА ДЛЯ ВАЛИДАЦИИ")
        print("=" * 70)

        data_manager = OpenMPIDataManager(data_dir=data_dir, validation_split=0.2)
        data = data_manager.load_and_prepare_data()

        self.openmpi_train_dataset = data['train_dataset']
        self.openmpi_test_dataset = data['test_dataset']
        self.openmpi_system_matrix = data['system_matrix']
        self.openmpi_image_shape = data['image_shape']

        print(f"\n  OpenMPI данные загружены:")
        print(f"    Обучающая выборка: {data['n_train']} образцов")
        print(f"    Тестовая выборка: {data['n_test']} образцов")
        print(f"    Фантомы в обучении: {data['train_phantoms']}")
        print(f"    Фантомы в тестировании: {data['test_phantoms']}")

        return data

    def validate_on_openmpi(self, model_name, model_func,
                            use_train_split=False, n_samples=None):
        """
        Валидация модели на OpenMPI датасете

        Args:
            model_name: имя модели для отчета
            model_func: функция реконструкции (принимает measurement)
            use_train_split: использовать обучающую выборку (иначе тестовую)
            n_samples: ограничить количество образцов
        """
        if not hasattr(self, 'openmpi_test_dataset'):
            self.load_openmpi_data()

        dataset = self.openmpi_train_dataset if use_train_split else self.openmpi_test_dataset

        if n_samples is not None:
            indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        else:
            indices = range(len(dataset))

        results = []

        print(f"\nВалидация {model_name} на OpenMPI датасете...")
        print(f"  Выборка: {'обучающая' if use_train_split else 'тестовая'}")
        print(f"  Количество образцов: {len(indices)}")

        for idx in tqdm(indices):
            measurement, ground_truth = dataset[idx]

            # Измерение в формате numpy
            meas_np = measurement.numpy()
            gt_np = ground_truth.numpy()

            # Преобразование измерения в формат (2, n_measurements)
            if meas_np.ndim == 3:
                # (channels, h, w) -> комплексные измерения
                n_channels, h, w = meas_np.shape
                real_part = meas_np[:n_channels // 2] if n_channels >= 2 else meas_np
                imag_part = meas_np[n_channels // 2:] if n_channels >= 2 else np.zeros_like(real_part)
                measurement_complex = real_part + 1j * imag_part
                measurement_flat = measurement_complex.reshape(2, -1)
            else:
                measurement_flat = meas_np.reshape(2, -1)

            # Реконструкция
            start_time = time.time()
            try:
                recon = model_func(measurement_flat)
                elapsed_time = time.time() - start_time

                if recon is not None:
                    # Изменение размера до ground truth
                    if recon.shape != gt_np.shape[-2:]:
                        from scipy import ndimage
                        recon = ndimage.zoom(recon,
                                             (gt_np.shape[-2] / recon.shape[0],
                                              gt_np.shape[-1] / recon.shape[1]),
                                             order=1)

                    metrics = MetricsCalculator.calculate_all_metrics(gt_np[0], recon)
                    metrics['time'] = elapsed_time

                    results.append({
                        'idx': idx,
                        'metrics': metrics,
                        'reconstruction': recon
                    })
            except Exception as e:
                print(f"  Ошибка на образце {idx}: {e}")
                continue

        # Агрегация результатов
        if results:
            avg_metrics = {
                'ssim': np.mean([r['metrics']['ssim'] for r in results]),
                'psnr': np.mean([r['metrics']['psnr'] for r in results]),
                'mse': np.mean([r['metrics']['mse'] for r in results]),
                'fwhm': np.mean([r['metrics']['fwhm'] for r in results]),
                'time': np.mean([r['metrics']['time'] for r in results])
            }

            print(f"\nРезультаты валидации {model_name} на OpenMPI:")
            print(f"  Средний SSIM: {avg_metrics['ssim']:.4f}")
            print(f"  Средний PSNR: {avg_metrics['psnr']:.2f} дБ")
            print(f"  Средний FWHM: {avg_metrics['fwhm']:.2f} пикс.")
            print(f"  Среднее время: {avg_metrics['time']:.4f} с")

            return avg_metrics, results

        return None, []

    def compare_all_on_openmpi(self):
        """Сравнение всех методов на OpenMPI датасете"""
        print("\n" + "=" * 70)
        print("СРАВНЕНИЕ МЕТОДОВ НА OPENMPI ДАТАСЕТЕ")
        print("=" * 70)

        # Загрузка данных
        self.load_openmpi_data()

        # Определение методов
        methods = [
            ('Тихонов', self.tikhonov_reconstruction, "Tikhonov (1963)"),
            ('KatsMarc', lambda m: self.katsmarc_reconstruction(m) if self.katsmarc else None, "Kaczmarz (1937)"),
            ('Chae(2017)', self.chae_reconstruction if hasattr(self, 'chae_model') and self.chae_model else None,
             "Chae - Single Layer NN"),
            ('Chae-Multi(2017)',
             self.chae_multi_reconstruction
             if hasattr(self, 'chae_multi_model') and self.chae_multi_model else None,
             "Chae - Multi Layer NN"),
            ('DIP(2020)',
             lambda m: self.dip_reconstruction(m) if hasattr(self, 'dip_model') and self.dip_model else None,
             "Dittmer et al."),
            ('PMCNet-Std(2026)',
             self.pmcnet_standard_reconstruction if self.pmcnet_standard else None,
             "Huang et al. - Standard"),
            ('PMCNet-RadialCoil(2026)',
             self.pmcnet_radial_coil_reconstruction if self.pmcnet_radial_coil else None,
             "Paper + radial p(r) = 1/(1+(r/R)²)"),
            ('PMCNet-Soft(2026)',
             self.pmcnet_soft_reconstruction if self.pmcnet_soft else None,
             "Phys + Scheinker 2023 soft constraints"),
            ('PMCNet-Debye(2026)',
             self.pmcnet_debye_reconstruction if self.pmcnet_debye else None,
             "Phys + Debye-релаксация (обучаемая τ)"),
            ('PMCNet-CentralFD(2026)',
             self.pmcnet_central_fd_reconstruction if self.pmcnet_central_fd else None,
             "Phys + центральная FD (O(Δt²))"),
            ('CNN', self.cnn_reconstruction if hasattr(self, 'cnn_trainer') and self.cnn_trainer else None, "CNN"),
            ('MoDL', self.modl_reconstruction if hasattr(self, 'modl_trainer') and self.modl_trainer else None, "MoDL"),
            ('MoE', self.moe_reconstruction if self.moe else None,
             "Mixture of Experts (per-pixel gating)"),
        ]

        openmpi_results = {}

        print("\nОценка на тестовой выборке (непересекающиеся фантомы):")
        print("-" * 80)
        print(f"{'Метод':<15} {'Источник':<30} {'SSIM':<8} {'PSNR':<10} {'FWHM':<8} {'Время':<8}")
        print("-" * 80)

        for name, method, source in methods:
            if method is None:
                print(f"{name:<15} {source:<30} ПРОПУЩЕН")
                continue

            try:
                avg_metrics, results = self.validate_on_openmpi(
                    name, method, use_train_split=False, n_samples=20
                )

                if avg_metrics:
                    openmpi_results[name] = {
                        'metrics': avg_metrics,
                        'source': source,
                        'n_samples': len(results)
                    }

                    print(f"{name:<15} {source:<30} {avg_metrics['ssim']:<8.4f} "
                          f"{avg_metrics['psnr']:<10.2f} {avg_metrics['fwhm']:<8.2f} "
                          f"{avg_metrics['time']:<8.4f}")
            except Exception as e:
                print(f"{name:<15} {source:<30} ОШИБКА: {str(e)[:30]}")

        # Сохранение результатов
        self.openmpi_results = openmpi_results

        # Визуализация
        self._visualize_openmpi_results(openmpi_results)

        return openmpi_results

    def _visualize_openmpi_results(self, results):
        """Визуализация результатов на OpenMPI датасете"""
        if not results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        methods = list(results.keys())
        ssims = [results[m]['metrics']['ssim'] for m in methods]
        psnrs = [results[m]['metrics']['psnr'] for m in methods]

        # SSIM bar plot
        bars1 = axes[0].bar(methods, ssims, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('SSIM')
        axes[0].set_title('SSIM на OpenMPI датасете')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 1])

        # Добавление значений на столбцы
        for bar, val in zip(bars1, ssims):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # PSNR bar plot
        bars2 = axes[1].bar(methods, psnrs, color='coral', alpha=0.7)
        axes[1].set_ylabel('PSNR (дБ)')
        axes[1].set_title('PSNR на OpenMPI датасете')
        axes[1].tick_params(axis='x', rotation=45)

        for bar, val in zip(bars2, psnrs):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Сравнение методов реконструкции MPI на OpenMPI датасете', fontsize=12)
        plt.tight_layout()

        save_path = './DATA/results/openmpi_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен: {save_path}")
        plt.close()

    def compare_all_methods(self, radius=0.2, distance=0.2):
        """Сравнение всех доступных методов на двух-капельном фантоме
        (legacy API). Делегирует в `compare_all_methods_on_image`."""
        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)
        label = f'two_droplets_r{radius}_d{distance}'
        result = self.compare_all_methods_on_image(
            original_image, measurement, label=label,
            metadata={'radius': radius, 'distance': distance},
        )
        # Совместимость со старыми ключами
        result['radius'] = radius
        result['distance'] = distance
        return result

    def compare_all_methods_on_image(self, original_image, measurement,
                                     label: str = '',
                                     metadata: dict = None):
        """Универсальное сравнение: image + measurement → метрики по всем методам.

        Args:
            original_image: ground-truth изображение (Nx, Ny) — для метрик.
            measurement:    (2, M_per_coil) complex — измерение, формат
                            как из `generate_measurement` (через измеренную
                            SM) или из аналитической SM такой же формы.
            label:          строка-идентификатор фантома для отчёта/имён файлов.
            metadata:       любой dict дополнительных полей (например,
                            phantom_type, generation_method) — сохраняется
                            в результате.
        """
        metadata = dict(metadata) if metadata else {}
        print(f"\nСравнение методов для {label or '(unnamed)'}")
        print("-" * 90)

        results = {}

        # Список всех методов с их источниками
        methods = [
            ('Тихонов', self.tikhonov_reconstruction, "Tikhonov (1963)"),
            ('KatsMarc', lambda m: self.katsmarc_reconstruction(m) if self.katsmarc else None,
             "Kaczmarz (1937)"),
            ('Chae(2017)', self.chae_reconstruction if self.chae_model else None, "Chae - Single Layer NN"),
            ('Chae-Multi(2017)',
             self.chae_multi_reconstruction if self.chae_multi_model else None,
             "Chae - Multi Layer NN (с hidden слоем)"),
            ('DIP(2020)', lambda m: self.dip_reconstruction(m) if self.dip_model else None,
             "Dittmer et al. - Deep Image Prior"),
            ('PMCNet-Std(2026)',
             self.pmcnet_standard_reconstruction if self.pmcnet_standard else None,
             "Huang et al. - SM-baseline (измеренная SM)"),
            ('PMCNet-Paper(2026)',
             self.pmcnet_paper_reconstruction if self.pmcnet_paper else None,
             "Huang et al. - paper-faithful (явная физика)"),
            ('PMCNet-RadialCoil(2026)',
             self.pmcnet_radial_coil_reconstruction if self.pmcnet_radial_coil else None,
             "Paper + radial p(r) = 1/(1+(r/R)²)"),
            ('PMCNet-Soft(2026)',
             self.pmcnet_soft_reconstruction if self.pmcnet_soft else None,
             "Phys + Scheinker 2023 soft (∇c² + freq-weighted L1)"),
            ('PMCNet-Debye(2026)',
             self.pmcnet_debye_reconstruction if self.pmcnet_debye else None,
             "Phys + Debye-релаксация (обучаемая τ)"),
            ('PMCNet-CentralFD(2026)',
             self.pmcnet_central_fd_reconstruction if self.pmcnet_central_fd else None,
             "Phys + центральная FD через Conv1d (O(Δt²))"),
            ('CNN', self.cnn_reconstruction if self.cnn_trainer else None, "CNN (UNet)"),
            ('MoDL', self.modl_reconstruction if self.modl_trainer else None, "MoDL Network"),
            ('Diffusion', self.diffusion_reconstruction if self.diffusion_trainer else None, "Diffusion Model"),
            ('MoE', self.moe_reconstruction if self.moe else None,
             "Mixture of Experts (комбинирование моделей)"),
        ]

        print(f"\n{'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}")
        print("-" * 90)

        for name, method, source in methods:
            if method is None:
                print(f"{name:<15} {source:<35} ПРОПУЩЕН (модель не загружена)")
                continue

            print(f"Реконструкция методом {name}...")
            start_time = time.time()
            try:
                recon = method(measurement)
                if recon is None:
                    print(f"  {name}: Ошибка - метод вернул None")
                    continue

                elapsed_time = time.time() - start_time

                if recon is None or np.isnan(recon).any() or np.isinf(recon).any():
                    print(f"  {name}: Ошибка - некорректный результат")
                    continue

                metrics = MetricsCalculator.calculate_all_metrics(original_image, recon)
                metrics['time'] = elapsed_time

                results[name] = {
                    'image': recon,
                    'metrics': metrics,
                    'source': source
                }

                print(f"{name:<15} {source:<35} {metrics['ssim']:<8.4f} {metrics['psnr']:<12.2f} "
                      f"{metrics['fwhm']:<8.2f} {metrics['time']:<10.4f}")
            except Exception as e:
                print(f"{name:<15} {source:<35} ОШИБКА: {str(e)[:50]}")
                continue

        # Визуализация — имя файла берётся из label
        safe_label = label.replace(' ', '_').replace('/', '_') or 'unnamed'
        self._visualize_all_comparison(original_image, results, safe_label,
                                        metadata)

        return {
            'original': original_image,
            'results': results,
            'label': label,
            **metadata,
        }

    def _visualize_all_comparison(self, original, results, label, metadata):
        """Визуализация сравнения всех методов под уникальным именем фантома."""
        save_path = f'./DATA/results/all_methods_{label}.png'
        # Для legacy-визуализации передадим radius/distance, если они есть
        radius = metadata.get('radius', 0.0)
        distance = metadata.get('distance', 0.0)
        Visualization.plot_all_methods_comparison(
            original, results, radius, distance,
            save_path=save_path, show_plot=False
        )

    def run_full_comparison(self, radius=0.2, distances=[0.2, 0.15, 0.1, 0.05]):
        """Запуск полного сравнения для разных расстояний"""
        print("=" * 90)
        print("ПОЛНОЕ СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI")
        print("=" * 90)
        print("\nСравниваемые методы и их источники:")
        print("  1. Тихонов              - Tikhonov regularization (1963)")
        print("  2. KatsMarc             - Kaczmarz algorithm (1937) - ART")
        print("  3. Chae(2017)           - Single-layer FC NN (ETRI Journal)")
        print("  4. DIP(2020)            - Deep Image Prior (Dittmer et al.)")
        print("  5. PMCNet-Std(2026)        - PMCNet Standard, измеренная SM (Huang et al.)")
        print("  6. PMCNet-Paper(2026)      - PMCNet paper-faithful (явная физика)")
        print("  7. PMCNet-RadialCoil(2026) - Paper + radial p(r)")
        print("  8. PMCNet-Soft(2026)       - Paper + Scheinker 2023 soft constraints")
        print("  9. PMCNet-Debye(2026)      - Paper + Debye-релаксация (обучаемая τ)")
        print(" 10. PMCNet-CentralFD(2026)  - Paper + центральная FD (O(Δt²))")
        print(" 11. CNN                     - U-Net baseline")
        print(" 12. MoDL                    - Model-based Deep Learning")
        print(" 13. Diffusion               - DDPM baseline")
        print(" 14. MoE                     - Mixture of Experts (комбинирование моделей)")
        print("=" * 90)

        all_results = []

        for distance in distances:
            print(f"\n{'=' * 50}")
            print(f"Эксперимент: радиус={radius}, расстояние={distance}")
            print(f"{'=' * 50}")

            result = self.compare_all_methods(radius, distance)
            all_results.append(result)

        self.results = all_results
        return all_results

    def run_phantom_battery(self, battery):
        """Сравнение всех методов на батарее фантомов.

        Args:
            battery: список словарей со схемой
                {'label': str,
                 'image': np.ndarray (Nx, Ny),
                 'measurement': np.ndarray (2, M_per_coil) complex,
                 'metadata': dict (например, {'phantom_type': 'B',
                                              'generation_method': 'sm'})}

        Возвращает список результатов; также сохраняется в `self.results`,
        чтобы работали `print_summary_table` и `save_results_to_file`.
        """
        print("=" * 90)
        print(f"БАТАРЕЯ ФАНТОМОВ: {len(battery)} экспериментов")
        print("=" * 90)

        all_results = []
        for i, item in enumerate(battery, 1):
            label = item.get('label', f'phantom_{i}')
            image = item['image']
            measurement = item['measurement']
            metadata = item.get('metadata', {})
            print(f"\n{'=' * 50}")
            print(f"[{i}/{len(battery)}] Эксперимент: {label}")
            print(f"{'=' * 50}")
            result = self.compare_all_methods_on_image(
                image, measurement, label=label, metadata=metadata,
            )
            all_results.append(result)

        self.results = all_results
        return all_results

    def print_summary_table(self):
        """Вывод сводной таблицы результатов"""
        if not self.results:
            print("Нет результатов для вывода")
            return

        print("\n" + "=" * 120)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 120)

        header = (f"{'Фантом':<32} {'Метод':<18} {'SSIM':<8} "
                  f"{'PSNR':<10} {'FWHM':<8} {'Время':<8}")
        print(header)
        print("-" * 120)

        for result in self.results:
            label = result.get('label') or (
                f"r={result.get('radius')},d={result.get('distance')}"
                if 'radius' in result else 'unnamed')
            for name, data in result['results'].items():
                m = data['metrics']
                print(f"{label:<32} {name:<18} {m['ssim']:<8.4f} "
                      f"{m['psnr']:<10.2f} {m['fwhm']:<8.2f} {m['time']:<8.4f}")

        # Статистика
        print("\n" + "=" * 120)
        print("СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ")
        print("=" * 120)

        # Собираем метрики по методам
        methods_metrics = {}
        for result in self.results:
            for name, data in result['results'].items():
                if name not in methods_metrics:
                    methods_metrics[name] = {'ssim': [], 'psnr': [], 'fwhm': [], 'time': []}
                methods_metrics[name]['ssim'].append(data['metrics']['ssim'])
                methods_metrics[name]['psnr'].append(data['metrics']['psnr'])
                methods_metrics[name]['fwhm'].append(data['metrics']['fwhm'])
                methods_metrics[name]['time'].append(data['metrics']['time'])

        for name, metrics in methods_metrics.items():
            print(f"\n{name}:")
            print(f"  Средний SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}")
            print(f"  Средний PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} дБ")
            print(f"  Средний FWHM: {np.mean(metrics['fwhm']):.2f} ± {np.std(metrics['fwhm']):.2f} пикс.")
            print(f"  Среднее время: {np.mean(metrics['time']):.4f} ± {np.std(metrics['time']):.4f} с")

    def save_results_to_file(self, filename='./DATA/results/all_methods_summary.txt'):
        """Сохранение результатов в файл"""
        if not self.results:
            print("Нет результатов для сохранения")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ MPI\n")
            f.write("=" * 120 + "\n\n")

            f.write("СПИСОК МЕТОДОВ И ИСТОЧНИКОВ:\n")
            f.write("-" * 60 + "\n")
            f.write("1. Тихонов              - Tikhonov regularization (1963)\n")
            f.write("2. KatsMarc             - Kaczmarz algorithm (1937) - ART\n")
            f.write("3. Chae(2017)           - Single-layer FC NN (ETRI Journal)\n")
            f.write("4. DIP(2020)            - Deep Image Prior (Dittmer et al.)\n")
            f.write("5. PMCNet-Std(2026)        - PMCNet Standard, измеренная SM (Huang et al.)\n")
            f.write("6. PMCNet-Paper(2026)      - PMCNet paper-faithful (явная физика)\n")
            f.write("7. PMCNet-RadialCoil(2026) - Paper + radial p(r)\n")
            f.write("8. PMCNet-Soft(2026)       - Paper + Scheinker 2023 soft constraints\n")
            f.write("9. PMCNet-Debye(2026)      - Paper + Debye-релаксация (обучаемая τ)\n")
            f.write("10. PMCNet-CentralFD(2026) - Paper + центральная FD (O(Δt²))\n")
            f.write("11. CNN                     - U-Net baseline\n")
            f.write("12. MoDL                    - Model-based Deep Learning\n")
            f.write("13. Diffusion               - DDPM baseline\n")
            f.write("14. MoE                     - Mixture of Experts (комбинирование моделей)\n")
            f.write("\n" + "=" * 120 + "\n\n")

            for result in self.results:
                label = result.get('label') or (
                    f"radius={result.get('radius')}, distance={result.get('distance')}"
                    if 'radius' in result else 'unnamed')
                f.write(f"ЭКСПЕРИМЕНТ: {label}\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Метод':<18} {'Источник':<35} {'SSIM':<8} "
                        f"{'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}\n")
                f.write("-" * 90 + "\n")

                for name, data in result['results'].items():
                    m = data['metrics']
                    source = data.get('source', '')
                    f.write(f"{name:<18} {source:<35} {m['ssim']:<8.4f} "
                            f"{m['psnr']:<12.2f} "
                            f"{m['fwhm']:<8.2f} {m['time']:<10.4f}\n")
                f.write("\n")

        print(f"\nРезультаты сохранены в файл: {filename}")

    def _total_variation(self, img):
        """Вычисление тотальной вариации для регуляризации"""
        diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
        tv_loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))
        return tv_loss