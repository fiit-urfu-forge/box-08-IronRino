"""Симуляторы синтетических MPI-измерений: два независимых пути.

Согласно постановке Chae (2017), сигнал MPI можно получить либо через
прямые физические уравнения (функция Ланжевена + производная по
времени драйв-поля), либо через готовую системную матрицу, в которой
столбцы соответствуют полиномам Чебышёва второго рода.

В этом модуле:
  • `ChebyshevSystemFunction` — Chebyshev-системная матрица (Chae 2017,
    Rahmer 2009);
  • `PhysicalMPISimulator`   — прямые физические уравнения (адиабатическая
    модель Ланжевена);
  • `SyntheticDatasetGenerator` — высокоуровневый генератор обучающих
    выборок: создаёт пары `(image, measurement)` через любой из двух
    методов, с шумом или без, для всех типов фантомов.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyu
from scipy.ndimage import gaussian_filter

from .phantoms import PhantomType, PhantomGenerator


# ---------------------------------------------------------------------------
# Параметры наночастиц
# ---------------------------------------------------------------------------


@dataclass
class NanoparticleProperties:
    """Физические свойства SPIO-частиц и параметров сканера.

    Дефолты приведены к BeihangUniversityData (2D Narrowband MPI System,
    Beihang University, 2023-09-07), параметры из «Описание параметров
    файла H5.docx»:

      • Tracer  — Perimag (магнитный диаметр ~20 нм, M_s ≈ 0.6 Т).
      • Drive   — 4 мТ на x-катушке при 24.51 кГц.
      • Gradient — 0.56 Т/м (x-ось; y = 1.12 Т/м, не симметрично).

    Для воспроизведения экспериментов Chae 2017 (40 нм частицы, 20 кГц,
    32 мТ, 2 Т/м) задайте значения явно через kwargs.
    """
    diameter_nm: float = 20.0                    # Perimag core
    saturation_magnetization: float = 0.6        # Ms / μ₀, Fe₃O₄
    temperature: float = 300.0                   # K
    drive_field_amplitude: float = 4.0           # мТ (H5: drivefield.strength[x])
    gradient_strength: float = 0.56              # Т/м  (H5: gradient[x])
    drive_field_frequency: float = 24510.0       # Гц   (H5: driveFrequency[x])

    @property
    def magnetic_moment(self) -> float:
        d_m = self.diameter_nm * 1e-9
        return self.saturation_magnetization * np.pi * d_m ** 3 / 6.0

    @property
    def alpha(self) -> float:
        """α = μ₀·m / (k_B·T)."""
        mu0 = 4.0 * np.pi * 1e-7
        kB = 1.380649e-23
        return mu0 * self.magnetic_moment / (kB * self.temperature)

    @property
    def langevin_fwhm_mm(self) -> float:
        """FWHM производной L: ~ 1/(G·d) (Chae 2017, Fig. 2(b))."""
        return 0.5 / (self.gradient_strength * (self.diameter_nm / 40.0))

    @property
    def theoretical_resolution_mm(self) -> float:
        return self.langevin_fwhm_mm

    @property
    def convolution_effect(self) -> float:
        """Эффект свёртки в спектре: маленькие частицы → сильнее эффект."""
        if self.diameter_nm >= 50:
            return 0.2
        elif self.diameter_nm <= 20:
            return 1.0
        return 1.0 - (self.diameter_nm - 20) / 30 * 0.8


# ---------------------------------------------------------------------------
# Системная матрица через полиномы Чебышёва
# ---------------------------------------------------------------------------


class ChebyshevSystemFunction:
    """Аналитическая SM на основе полиномов Чебышёва U_n(x).

    Источник: Chae (2017), Rahmer et al. (2009). Структура SM объясняет,
    почему Chae-сети успешно обучаются — обратная задача в этом базисе
    разрешима явно (псевдо-обращение).
    """

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.n_pixels = nx * ny

    @staticmethod
    def chebyshev_polynomial(n: int, x: np.ndarray) -> np.ndarray:
        return eval_chebyu(n, x)

    def system_function_1d(self, z: np.ndarray,
                           particle_size_nm: float = 40.0) -> np.ndarray:
        """1D-системная функция с эффектом свёртки от размера частиц."""
        if particle_size_nm >= 50:
            sigma_conv = 0.05
        elif particle_size_nm <= 20:
            sigma_conv = 0.3
        else:
            sigma_conv = 0.3 - (particle_size_nm - 20) / 30 * 0.25

        n_polys = min(self.n_harmonics, len(z))
        system_matrix = np.zeros((n_polys, len(z)))
        for n in range(n_polys):
            T_n = self.chebyshev_polynomial(n, z[:len(z)])
            if sigma_conv > 0:
                T_n = gaussian_filter(T_n, sigma=sigma_conv)
            system_matrix[n, :] = T_n / (n + 1)
        return system_matrix

    def generate_system_matrix(self,
                               particle_size_nm: float = 40.0) -> np.ndarray:
        """2D SM формы (2·n_harmonics, n_pixels) — комплексная."""
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, _ = np.meshgrid(x, y, indexing='ij')

        n_meas = self.n_harmonics * 2
        S = np.zeros((n_meas, self.n_pixels), dtype=np.complex128)

        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                z = X[i, j]
                S_1d = self.system_function_1d(np.array([z]), particle_size_nm)
                S[:self.n_harmonics, idx] = S_1d[0, :].real
                S[self.n_harmonics:, idx] = S_1d[0, :].imag
        return S


# ---------------------------------------------------------------------------
# Физический симулятор (Chae 2017)
# ---------------------------------------------------------------------------


class PhysicalMPISimulator:
    """Прямой физический симулятор MPI: H(t) → M(H) → ∂M/∂t → u(f)."""

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.n_pixels = nx * ny
        self.system_func = ChebyshevSystemFunction(nx, ny, n_harmonics)

    @staticmethod
    def langevin(x: np.ndarray) -> np.ndarray:
        """L(ξ) = coth(ξ) − 1/ξ (numpy-версия для генерации данных)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x) > 1e-8,
                            1.0 / np.tanh(x) - 1.0 / x, 0.0)

    @staticmethod
    def langevin_derivative(x: np.ndarray) -> np.ndarray:
        """dL/dξ = 1/ξ² − 1/sinh²(ξ)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.abs(x) > 1e-8,
                            1.0 / x ** 2 - 1.0 / np.sinh(x) ** 2,
                            1.0 / 3.0)

    def magnetization_derivative(self, H: np.ndarray,
                                 props: NanoparticleProperties) -> np.ndarray:
        return props.magnetic_moment * props.alpha * self.langevin_derivative(
            props.alpha * H)

    def signal_spectrum(self, concentration: np.ndarray,
                        props: NanoparticleProperties) -> np.ndarray:
        """Спектр сигнала: первые `n_harmonics` гармоник FFT по ∂M/∂t."""
        n_time = 1024
        T = 1.0 / props.drive_field_frequency
        t = np.linspace(0, T, n_time)
        H = (props.drive_field_amplitude * 1e-3 *
             np.cos(2 * np.pi * props.drive_field_frequency * t))
        dM_dH = self.magnetization_derivative(H, props)
        dHdt = (-2 * np.pi * props.drive_field_frequency *
                props.drive_field_amplitude * 1e-3 *
                np.sin(2 * np.pi * props.drive_field_frequency * t))
        dMdt = dM_dH * dHdt * float(np.mean(concentration))
        return np.fft.fft(dMdt)[:self.n_harmonics]

    def generate_measurement(self, concentration: np.ndarray,
                             props: Optional[NanoparticleProperties] = None,
                             add_noise: bool = True,
                             snr_db: float = 30.0) -> np.ndarray:
        """Путь 1: прямые физические уравнения."""
        if props is None:
            props = NanoparticleProperties()
        spectrum = self.signal_spectrum(concentration, props)
        meas = np.zeros((2, self.n_harmonics), dtype=np.complex128)
        meas[0, :] = spectrum
        meas[1, :] = spectrum * np.exp(1j * np.pi / 2)
        if add_noise:
            std = np.std(np.abs(meas)) / (10 ** (snr_db / 20))
            meas = meas + std * (np.random.randn(*meas.shape)
                                 + 1j * np.random.randn(*meas.shape))
        return meas

    def generate_from_system_matrix(self, concentration: np.ndarray,
                                    particle_size_nm: float = 40.0,
                                    add_noise: bool = True,
                                    snr_db: float = 30.0) -> np.ndarray:
        """Путь 2: через готовую системную матрицу (быстрее, более линейно)."""
        S = self.system_func.generate_system_matrix(particle_size_nm)
        meas = S @ concentration.reshape(-1, 1)
        meas = meas.reshape(2, -1)
        if add_noise:
            std = np.std(np.abs(meas)) / (10 ** (snr_db / 20))
            meas = meas + std * (np.random.randn(*meas.shape)
                                 + 1j * np.random.randn(*meas.shape))
        return meas


# ---------------------------------------------------------------------------
# Высокоуровневый генератор датасета
# ---------------------------------------------------------------------------


class SyntheticDatasetGenerator:
    """Высокоуровневая обёртка: phantom × particle_size × method → dataset."""

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.phantom_gen = PhantomGenerator(nx, ny)
        self.phys_sim = PhysicalMPISimulator(nx, ny, n_harmonics)

    def _generate_image(self, phantom_type: PhantomType) -> np.ndarray:
        try:
            if phantom_type == PhantomType.TWO_DROPLETS:
                radius = np.random.uniform(0.1, 0.3)
                distance = np.random.uniform(radius, min(2 * radius, 0.8))
                return self.phantom_gen.two_droplets(radius, distance)
            if phantom_type == PhantomType.CONCENTRATION:
                return self.phantom_gen.concentration_phantom()
            if phantom_type == PhantomType.RESOLUTION:
                return self.phantom_gen.resolution_phantom()
            if phantom_type == PhantomType.ROTATION:
                return self.phantom_gen.rotation_phantom(
                    np.random.uniform(0, 360))
            if phantom_type == PhantomType.SHAPE:
                return self.phantom_gen.shape_phantom(np.random.choice(
                    ['cone', 'square', 'ring', 'cross', 'spiral']))
            if phantom_type == PhantomType.RANDOM:
                return self.phantom_gen.random_phantom(np.random.randint(3, 8))
            if phantom_type == PhantomType.PATTERN:
                return self.phantom_gen.pattern_phantom(
                    np.random.choice(['checkerboard', 'stripes_h',
                                      'stripes_v', 'radial']),
                    np.random.randint(3, 6))
            if phantom_type == PhantomType.PHANTOM_4:
                return self.phantom_gen.phantom_4()
            return self.phantom_gen.random_phantom(4)
        except Exception as e:
            print(f"    Warning: phantom {phantom_type} fallback: {e}")
            return self.phantom_gen.two_droplets(0.2, 0.2)

    def generate_dataset(self, n_samples: int = 1000,
                         method: str = 'system_matrix',
                         phantom_types: Optional[List[PhantomType]] = None,
                         particle_sizes_nm: Optional[List[float]] = None,
                         add_noise: bool = True,
                         snr_db: float = 30.0,
                         test_split: float = 0.2,
                         random_seed: int = 42) -> Dict:
        """Сгенерировать пары (image, measurement) методом `method`.

        Args:
            method: 'system_matrix' (Chebyshev SM, путь 2) или
                    'physical' (прямые уравнения, путь 1).
        """
        np.random.seed(random_seed)
        if phantom_types is None:
            phantom_types = list(PhantomType)
        if particle_sizes_nm is None:
            particle_sizes_nm = [20, 30, 40, 50, 60]

        images, measurements, metadata = [], [], []
        n_combinations = len(phantom_types) * len(particle_sizes_nm)
        per_combo = max(1, n_samples // n_combinations)

        for phantom_type in phantom_types:
            for size in particle_sizes_nm:
                for _ in range(per_combo):
                    try:
                        image = self._generate_image(phantom_type)
                        props = NanoparticleProperties(diameter_nm=size)
                        if method == 'system_matrix':
                            meas = self.phys_sim.generate_from_system_matrix(
                                image, size, add_noise, snr_db)
                        else:
                            meas = self.phys_sim.generate_measurement(
                                image, props, add_noise, snr_db)
                        images.append(image)
                        measurements.append(meas)
                        metadata.append({
                            'phantom_type': phantom_type.value,
                            'particle_size_nm': size,
                            'method': method,
                            'snr_db': snr_db if add_noise else np.inf,
                        })
                    except Exception as e:
                        print(f"    sample dropped: {e}")
                        continue
        if not images:
            raise RuntimeError("No samples generated")

        images = np.array(images, dtype=np.float32)
        measurements = np.array(measurements, dtype=np.complex64)
        n_train = int(len(images) * (1 - test_split))
        indices = np.random.permutation(len(images))
        tr, te = indices[:n_train], indices[n_train:]

        return {
            'X_train': measurements[tr],
            'X_test': measurements[te],
            'y_train': images[tr],
            'y_test': images[te],
            'metadata_train': [metadata[i] for i in tr],
            'metadata_test': [metadata[i] for i in te],
            'image_shape': (self.nx, self.ny),
            'n_harmonics': self.n_harmonics,
            'method': method,
        }

    @staticmethod
    def save_dataset(dataset: Dict,
                     filename_prefix: str = './DATA/dataset/synthetic'):
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        np.save(f'{filename_prefix}_X_train.npy', dataset['X_train'])
        np.save(f'{filename_prefix}_X_test.npy', dataset['X_test'])
        np.save(f'{filename_prefix}_y_train.npy', dataset['y_train'])
        np.save(f'{filename_prefix}_y_test.npy', dataset['y_test'])
        with open(f'{filename_prefix}_metadata.json', 'w') as f:
            json.dump({
                'metadata_train': dataset['metadata_train'],
                'metadata_test': dataset['metadata_test'],
                'image_shape': dataset['image_shape'],
                'n_harmonics': dataset['n_harmonics'],
                'method': dataset['method'],
            }, f, indent=2)

    def create_training_pipeline_dataset(self, n_samples: int = 5000,
                                         include_all_phantoms: bool = True,
                                         save: bool = True) -> Dict:
        phantom_types = (list(PhantomType) if include_all_phantoms else
                         [PhantomType.TWO_DROPLETS, PhantomType.CONCENTRATION,
                          PhantomType.RESOLUTION, PhantomType.ROTATION,
                          PhantomType.SHAPE])
        dataset = self.generate_dataset(
            n_samples=n_samples, method='system_matrix',
            phantom_types=phantom_types,
            particle_sizes_nm=[20, 30, 40, 50, 60],
            add_noise=True, snr_db=30.0, test_split=0.2,
        )
        if save:
            self.save_dataset(dataset, './DATA/dataset/synthetic_complete')
        return dataset


__all__ = [
    'NanoparticleProperties',
    'ChebyshevSystemFunction',
    'PhysicalMPISimulator',
    'SyntheticDatasetGenerator',
]
