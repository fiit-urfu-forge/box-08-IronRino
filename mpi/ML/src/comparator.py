"""Сравнение методов реконструкции MPI:
Tikhonov / Kaczmarz / Chae / DIP / Shang / DEQ-MPI / CNN / MoDL / Diffusion / PMCNet-trio.
"""

import numpy as np
import time
import os
import h5py
from matplotlib import pyplot as plt
from tqdm import tqdm

from .models import (
    TikhonovReconstructor, KatsMarcAlgorithm,
    ChaeSingleLayerNN, ChaeMultiLayerNN,
    DeepImagePrior, ShangCNN, DEQMPI,
    PMCNetReconstructor, PMCNetRefinedReconstructor,
    PMCNetStandard, PMCNetPhysicsEnhanced, PMCNetFinal,
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
        self.dip_model = None
        self.shang_model = None
        self.deq_model = None
        # PMCNet (Huang et al., 2026) — три варианта в одной системе координат
        self.pmcnet_reconstructor = None              # legacy alias for Standard
        self.pmcnet_refined_reconstructor = None       # legacy alias for Final
        self.pmcnet_standard = None                    # 1) измеренная SM
        self.pmcnet_physics_enhanced = None            # 2) аналитическая SM
        self.pmcnet_final = None                       # 3) аналитическая SM + Debye + multi-color + TV

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
        """Установка модели Chae (2017)"""
        self.chae_model = chae_model

    def set_dip_model(self, dip_model):
        """Установка модели Deep Image Prior (Dittmer et al., 2020)"""
        self.dip_model = dip_model

    def set_shang_model(self, shang_model):
        """Установка модели Shang et al. (2020)"""
        self.shang_model = shang_model

    def set_deq_model(self, deq_model):
        """Установка модели DEQ-MPI (Güngör et al., 2024)"""
        self.deq_model = deq_model

    def set_pmcnet_model(self, pmcnet_reconstructor):
        """[legacy] Алиас для `set_pmcnet_standard`."""
        self.pmcnet_reconstructor = pmcnet_reconstructor
        if self.pmcnet_standard is None:
            self.pmcnet_standard = pmcnet_reconstructor

    def set_pmcnet_refined_model(self, pmcnet_refined_reconstructor):
        """[legacy] Алиас для `set_pmcnet_final`."""
        self.pmcnet_refined_reconstructor = pmcnet_refined_reconstructor
        if self.pmcnet_final is None:
            self.pmcnet_final = pmcnet_refined_reconstructor

    def set_pmcnet_standard(self, reconstructor):
        """Установить вариант 1: PMCNet-Standard (Huang et al., 2026).

        Прямой оператор — ИЗМЕРЕННАЯ системная матрица из калибровки
        сканера. Никаких физических/NN-улучшений.
        """
        self.pmcnet_standard = reconstructor

    def set_pmcnet_physics_enhanced(self, reconstructor):
        """Установить вариант 2: PMCNet-Physics-Enhanced.

        Прямой оператор — АНАЛИТИЧЕСКАЯ системная матрица (стабильный
        Langevin, радиальная s(r), Лиссажу, центральная разность через
        фиксированную свёртку). NN не трогаем — single color, без Дебая,
        без TV.
        """
        self.pmcnet_physics_enhanced = reconstructor

    def set_pmcnet_final(self, reconstructor):
        """Установить вариант 3: PMCNet-Final.

        Аналитическая SM (как в Physics-Enhanced) + полный набор NN-
        оптимизаций: релаксация Дебая с обучаемой τ_k, multi-color,
        TV-регуляризация, hard constraints by construction.
        """
        self.pmcnet_final = reconstructor

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

    def tikhonov_reconstruction(self, measurement, mu=0.001, kmax=100):
        recon = self.tikhonov_reconstructor.reconstruct(measurement, mu, kmax)
        recon = np.asarray(recon).reshape(self.image_shape)
        if recon.max() > 0:
            recon = recon / recon.max()
        return recon

    # ====================================================================
    # МЕТОД 1: Chae (2017) - Однослойная полносвязная нейронная сеть
    # ====================================================================
    def chae_reconstruction(self, measurement):
        """Реконструкция методом Chae (2017) с батчевой обработкой"""
        if self.chae_model is None:
            raise ValueError("Chae модель не установлена")

        # Подготовка входных данных
        real_part = measurement.real  # (2, n_measurements)
        imag_part = measurement.imag  # (2, n_measurements)

        # Формирование вектора как при обучении
        meas_vector = np.concatenate([
            real_part[0, :],  # real катушка 1
            imag_part[0, :],  # imag катушка 1
            real_part[1, :],  # real катушка 2
            imag_part[1, :]  # imag катушка 2
        ])

        # Нормализация если есть scaler
        if hasattr(self.chae_model, 'scaler_mean'):
            meas_vector = (meas_vector - self.chae_model.scaler_mean) / self.chae_model.scaler_scale

        # Предсказание
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32)

        self.chae_model.eval()
        with torch.no_grad():
            reconstructed_flat = self.chae_model(meas_tensor.unsqueeze(0))

        # Преобразование в изображение
        reconstructed = reconstructed_flat.numpy().reshape(self.image_shape)

        # Постобработка
        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 2: Dittmer et al. (2020) - Deep Image Prior (DIP)
    # ====================================================================
    def dip_reconstruction(self, measurement, n_iterations=500):
        """Полноценная реконструкция Deep Image Prior"""
        if self.dip_model is None:
            raise ValueError("DIP модель не установлена")

        import torch.optim as optim

        # Подготовка измерений
        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        # Создаем копию модели для этого конкретного измерения
        import copy
        dip_model = copy.deepcopy(self.dip_model)
        dip_model.train()

        # Создаем матрицу A для прямого оператора
        if not hasattr(self, 'A_tensor'):
            # Создаем расширенную матрицу для комплексных измерений
            if np.iscomplexobj(self.SM):
                SM_real = np.real(self.SM)
                SM_imag = np.imag(self.SM)
                A_extended = np.vstack([SM_real, SM_imag])
            else:
                A_extended = self.SM

            self.A_tensor = torch.tensor(A_extended, dtype=torch.float32)
            self.A_tensor_T = self.A_tensor.T

        # Оптимизатор
        optimizer = optim.Adam(dip_model.parameters(), lr=0.01)

        # Генерируем латентный вектор
        latent_z = dip_model.generate_random_latent()
        latent_z.requires_grad = True

        # Оптимизация
        print(f"  Оптимизация DIP (до {n_iterations} итераций)...")
        for iteration in range(n_iterations):
            optimizer.zero_grad()

            # Генерация изображения
            generated_image = dip_model(latent_z)
            generated_flat = generated_image.view(1, -1)

            # Прямой оператор
            measurement_pred = generated_flat @ self.A_tensor_T

            # Loss: несоответствие измерениям + регуляризация тотальной вариации
            data_loss = torch.mean((measurement_pred - meas_tensor) ** 2)

            # TV регуляризация
            tv_loss = self._total_variation(generated_image)

            loss = data_loss + 0.01 * tv_loss
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(f"    DIP iteration {iteration}, loss: {loss.item():.6f}")

        # Финальная реконструкция
        dip_model.eval()
        with torch.no_grad():
            reconstructed = dip_model(latent_z)[0, 0].numpy()

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
    # МЕТОД 3: Shang et al. (2020) - CNN для улучшения разрешения
    # ====================================================================
    def shang_reconstruction(self, measurement):
        """Реконструкция методом Shang et al. (2020)"""
        if self.shang_model is None:
            raise ValueError("Shang модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])

        # Для Shang CNN нужен вход в виде изображения
        # Сначала делаем грубую реконструкцию через псевдообращение
        try:
            A_pinv = np.linalg.pinv(self.SM)
            initial_recon = (A_pinv @ meas_vector).reshape(self.image_shape)
        except:
            initial_recon = np.zeros(self.image_shape)

        input_tensor = torch.tensor(initial_recon, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.shang_model.eval()
        with torch.no_grad():
            reconstructed = self.shang_model(input_tensor)[0, 0].numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 4: Güngör et al. (2024) - DEQ-MPI
    # ====================================================================
    def deq_reconstruction(self, measurement):
        """Реконструкция методом DEQ-MPI (Güngör et al., 2024)"""
        if self.deq_model is None:
            raise ValueError("DEQ-MPI модель не установлена")

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0)

        self.deq_model.eval()
        with torch.no_grad():
            reconstructed = self.deq_model(meas_tensor)[0, 0].numpy()

        if reconstructed.max() > 0:
            reconstructed = reconstructed / reconstructed.max()

        return reconstructed

    # ====================================================================
    # МЕТОД 5.5: Huang et al. (2026) - PMCNet (Physical Model-Constrained Net)
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
        """PMCNet-Standard: u = S_measured · c, L1, без улучшений."""
        if self.pmcnet_standard is None:
            raise ValueError("PMCNet-Standard модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_standard.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 2: PMCNet-Physics-Enhanced (аналитическая SM) ----------------
    def pmcnet_physics_enhanced_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Physics-Enhanced: u = S_analytical · c, без NN-улучшений.

        Системная матрица собрана из стабильного Langevin, радиальной
        чувствительности s(r), траектории Лиссажу и центральной разности
        через фиксированную свёртку. Architecture identical to Standard.
        """
        if self.pmcnet_physics_enhanced is None:
            raise ValueError("PMCNet-Physics-Enhanced модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        recon = self.pmcnet_physics_enhanced.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        return self._postprocess_recon(recon)

    # -- Вариант 3: PMCNet-Final (физика + все NN-улучшения) -----------------
    def pmcnet_final_reconstruction(self, measurement, n_iterations=None):
        """PMCNet-Final: аналитическая SM + Debye + multi-color + TV.

        Оцененные τ_k сохраняются в `self.last_pmcnet_taus_seconds`
        для отчёта; для multi-color карты концентраций суммируются по
        цветовым каналам.
        """
        if self.pmcnet_final is None:
            raise ValueError("PMCNet-Final модель не установлена")
        u_complex = self._measurement_to_complex_vector(measurement)
        c_np, taus = self.pmcnet_final.reconstruct(
            u_complex, n_iterations=n_iterations, verbose=False,
        )
        self.last_pmcnet_taus_seconds = taus
        recon = c_np.sum(axis=0) if c_np.ndim == 3 else c_np
        return self._postprocess_recon(recon)

    # -- legacy-обёртки (сохраняют прежний интерфейс) -------------------------
    def pmcnet_reconstruction(self, measurement, n_iterations=None):
        """[legacy] Алиас для `pmcnet_standard_reconstruction`."""
        return self.pmcnet_standard_reconstruction(measurement, n_iterations)

    def pmcnet_refined_reconstruction(self, measurement, n_iterations=None):
        """[legacy] Алиас для `pmcnet_final_reconstruction`."""
        return self.pmcnet_final_reconstruction(measurement, n_iterations)

    # ====================================================================
    # МЕТОД 6: KatsMarc (Алгоритм Кацмарца, 1937)
    # ====================================================================
    def katsmarc_reconstruction(self, measurement, n_iterations=20, relaxation=1.0):
        """Реконструкция алгоритмом Кацмарца (Kaczmarz, 1937)"""
        if self.katsmarc is None:
            print("  KatsMarc алгоритм не инициализирован")
            return None

        meas_vector = np.concatenate([measurement[0, :], measurement[1, :]])
        reconstructed = self.katsmarc.reconstruct(meas_vector, n_iterations, relaxation)
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
        """Реконструкция методом диффузионной модели"""
        if self.diffusion_trainer is None:
            raise ValueError("Diffusion модель не установлена")

        import torch

        real_part = measurement.real
        imag_part = measurement.imag
        meas_vector = np.concatenate([real_part.flatten(), imag_part.flatten()])
        meas_tensor = torch.tensor(meas_vector, dtype=torch.float32).unsqueeze(0).to(self.diffusion_trainer.device)

        self.diffusion_trainer.model.eval()
        with torch.no_grad():
            # Для диффузионной модели нужен обратный процесс
            batch_size = meas_tensor.shape[0]
            x_t = torch.randn(batch_size, 1, self.nx, self.ny).to(self.diffusion_trainer.device)

            # Упрощенная версия обратного процесса
            for t in reversed(range(min(50, self.diffusion_trainer.model.n_steps))):
                t_tensor = torch.full((batch_size,), t, device=self.diffusion_trainer.device, dtype=torch.long)
                x_t = self.diffusion_trainer.model.p_sample(x_t, t_tensor)

            reconstructed = x_t[0, 0].cpu().numpy()

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
            ('DIP(2020)',
             lambda m: self.dip_reconstruction(m) if hasattr(self, 'dip_model') and self.dip_model else None,
             "Dittmer et al."),
            ('Shang(2022)', self.shang_reconstruction if hasattr(self, 'shang_model') and self.shang_model else None,
             "Shang et al. - FDS-MPI"),
            ('DEQ-MPI(2024)', self.deq_reconstruction if hasattr(self, 'deq_model') and self.deq_model else None,
             "Güngör et al."),
            ('PMCNet-Std(2026)',
             self.pmcnet_standard_reconstruction if self.pmcnet_standard else None,
             "Huang et al. - Standard"),
            ('PMCNet-Phys(2026)',
             self.pmcnet_physics_enhanced_reconstruction if self.pmcnet_physics_enhanced else None,
             "Huang et al. - +улучшенная физика"),
            ('PMCNet-Final(2026)',
             self.pmcnet_final_reconstruction if self.pmcnet_final else None,
             "Huang et al. - +физика+NN-оптимизации"),
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
        """Сравнение всех доступных методов"""
        print(f"\nСравнение методов для radius={radius}, distance={distance}")
        print("-" * 90)

        original_image = self.generate_test_case(radius, distance)
        measurement = self.generate_measurement(original_image)

        results = {}

        # Список всех методов с их источниками
        methods = [
            ('Тихонов', self.tikhonov_reconstruction, "Tikhonov (1963)"),
            ('KatsMarc', lambda m: self.katsmarc_reconstruction(m, n_iterations=20) if self.katsmarc else None,
             "Kaczmarz (1937)"),
            ('Chae(2017)', self.chae_reconstruction if self.chae_model else None, "Chae - Single Layer NN"),
            ('DIP(2020)', lambda m: self.dip_reconstruction(m, n_iterations=300) if self.dip_model else None,
             "Dittmer et al. - Deep Image Prior"),
            ('Shang(2022)', self.shang_reconstruction if self.shang_model else None, "Shang et al. - FDS-MPI dual-branch"),
            ('DEQ-MPI(2024)', self.deq_reconstruction if self.deq_model else None, "Güngör et al. - DEQ-MPI"),
            ('PMCNet-Std(2026)',
             self.pmcnet_standard_reconstruction if self.pmcnet_standard else None,
             "Huang et al. - Standard (измеренная SM)"),
            ('PMCNet-Phys(2026)',
             self.pmcnet_physics_enhanced_reconstruction if self.pmcnet_physics_enhanced else None,
             "Huang et al. - +улучшенная физика (аналитическая SM)"),
            ('PMCNet-Final(2026)',
             self.pmcnet_final_reconstruction if self.pmcnet_final else None,
             "Huang et al. - +Debye+multi-color+TV (физика+NN)"),
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

        # Визуализация
        self._visualize_all_comparison(original_image, results, radius, distance)

        return {
            'original': original_image,
            'results': results,
            'radius': radius,
            'distance': distance
        }

    def _visualize_all_comparison(self, original, results, radius, distance):
        """Визуализация сравнения всех методов"""
        save_path = f'./DATA/results/all_methods_r{radius}_d{distance}.png'

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
        print("  5. Shang(2022)          - FDS-MPI dual-branch CNN (PMB)")
        print("  6. DEQ-MPI(2024)        - Deep Equilibrium Model (Güngör et al., IEEE TMI)")
        print("  7. PMCNet-Std(2026)     - PMCNet Standard, измеренная SM (Huang et al.)")
        print("  8. PMCNet-Phys(2026)    - PMCNet + улучшенная физика (аналитическая SM)")
        print("  9. PMCNet-Final(2026)   - PMCNet + физика + NN-оптимизации")
        print(" 10. CNN                  - U-Net baseline")
        print(" 11. MoDL                 - Model-based Deep Learning")
        print(" 12. Diffusion            - DDPM baseline")
        print(" 13. MoE                  - Mixture of Experts (комбинирование моделей)")
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

    def print_summary_table(self):
        """Вывод сводной таблицы результатов"""
        if not self.results:
            print("Нет результатов для вывода")
            return

        print("\n" + "=" * 120)
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("=" * 120)

        header = f"{'Расст.':<8} {'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR':<10} {'FWHM':<8} {'Время':<8}"
        print(header)
        print("-" * 120)

        for result in self.results:
            distance = result['distance']
            for name, data in result['results'].items():
                m = data['metrics']
                source = data.get('source', '')
                print(f"{distance:<8.3f} {name:<15} {source:<35} {m['ssim']:<8.4f} "
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
            f.write("5. Shang(2022)          - FDS-MPI dual-branch CNN (PMB)\n")
            f.write("6. DEQ-MPI(2024)        - Deep Equilibrium Model (Güngör et al., IEEE TMI)\n")
            f.write("7. PMCNet-Std(2026)     - PMCNet Standard, измеренная SM (Huang et al.)\n")
            f.write("8. PMCNet-Phys(2026)    - PMCNet + улучшенная физика (аналитическая SM)\n")
            f.write("9. PMCNet-Final(2026)   - PMCNet + физика + NN-оптимизации\n")
            f.write("10. CNN                  - U-Net baseline\n")
            f.write("11. MoDL                 - Model-based Deep Learning\n")
            f.write("12. Diffusion            - DDPM baseline\n")
            f.write("13. MoE                  - Mixture of Experts (комбинирование моделей)\n")
            f.write("\n" + "=" * 120 + "\n\n")

            for result in self.results:
                distance = result['distance']
                f.write(f"ЭКСПЕРИМЕНТ: radius={result['radius']}, distance={distance}\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Метод':<15} {'Источник':<35} {'SSIM':<8} {'PSNR (дБ)':<12} {'FWHM':<8} {'Время (с)':<10}\n")
                f.write("-" * 90 + "\n")

                for name, data in result['results'].items():
                    m = data['metrics']
                    source = data.get('source', '')
                    f.write(f"{name:<15} {source:<35} {m['ssim']:<8.4f} {m['psnr']:<12.2f} "
                            f"{m['fwhm']:<8.2f} {m['time']:<10.4f}\n")
                f.write("\n")

        print(f"\nРезультаты сохранены в файл: {filename}")

    def _total_variation(self, img):
        """Вычисление тотальной вариации для регуляризации"""
        diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
        diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
        tv_loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))
        return tv_loss