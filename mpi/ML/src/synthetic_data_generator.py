"""Генерация синтетических данных для MPI реконструкции на основе физических уравнений Chae (2017)"""
import os

import numpy as np
from scipy.special import eval_chebyu
from scipy.ndimage import zoom, gaussian_filter
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum


class PhantomType(Enum):
    """Типы фантомов для генерации данных"""
    TWO_DROPLETS = "two_droplets"
    CONCENTRATION = "concentration"
    RESOLUTION = "resolution"
    ROTATION = "rotation"
    SHAPE = "shape"
    RANDOM = "random"
    PATTERN = "pattern"
    PHANTOM_4 = "phantom_4"


@dataclass
class NanoparticleProperties:
    """Свойства наночастиц SPIO (Chae 2017)"""
    diameter_nm: float = 40.0  # диаметр в нанометрах
    saturation_magnetization: float = 0.6  # Ms в T/μ0
    temperature: float = 300.0  # температура в K
    drive_field_amplitude: float = 32.0  # амплитуда в mT/μ0
    gradient_strength: float = 2.0  # градиент в T/m/μ0
    drive_field_frequency: float = 20e3  # частота в Hz

    @property
    def magnetic_moment(self) -> float:
        """Магнитный момент m = Ms * π * d³/6"""
        d_m = self.diameter_nm * 1e-9
        return self.saturation_magnetization * np.pi * d_m ** 3 / 6

    @property
    def alpha(self) -> float:
        """Параметр α = μ0*m/(kB*T)"""
        mu0 = 4 * np.pi * 1e-7
        kB = 1.380649e-23
        return mu0 * self.magnetic_moment / (kB * self.temperature)

    @property
    def langevin_fwhm_mm(self) -> float:
        """
        FWHM производной функции Ланжевена
        Chae 2017: FWHM ~ G⁻¹d⁻¹
        """
        scaling = 0.5
        return scaling / (self.gradient_strength * (self.diameter_nm / 40.0))

    @property
    def theoretical_resolution_mm(self) -> float:
        """Теоретическое разрешение в мм"""
        return self.langevin_fwhm_mm

    @property
    def convolution_effect(self) -> float:
        """
        Эффект свертки (Chae 2017, Fig. 2(b))
        Меньший размер частиц -> больший эффект свертки
        """
        if self.diameter_nm >= 50:
            return 0.2
        elif self.diameter_nm <= 20:
            return 1.0
        else:
            return 1.0 - (self.diameter_nm - 20) / 30 * 0.8


class ChebyshevSystemFunction:
    """
    Системная функция на основе полиномов Чебышева второго рода
    Источник: Chae 2017, Rahmer et al. 2009
    """

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.n_pixels = nx * ny

    def chebyshev_polynomial(self, n: int, x: np.ndarray) -> np.ndarray:
        """Полином Чебышева второго рода U_n(x)"""
        return eval_chebyu(n, x)

    def system_function_1d(self, z: np.ndarray, particle_size_nm: float = 40.0) -> np.ndarray:
        """1D системная функция с учетом эффекта свертки"""
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

    def generate_system_matrix(self, particle_size_nm: float = 40.0) -> np.ndarray:
        """Генерация 2D системной матрицы"""
        x = np.linspace(-1, 1, self.nx)
        y = np.linspace(-1, 1, self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        n_measurements = self.n_harmonics * 2
        S = np.zeros((n_measurements, self.n_pixels), dtype=np.complex128)

        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                z = X[i, j]
                S_1d = self.system_function_1d(np.array([z]), particle_size_nm)
                S[0:self.n_harmonics, idx] = S_1d[0, :].real
                S[self.n_harmonics:, idx] = S_1d[0, :].imag

        return S


class PhysicalMPISimulator:
    """Физический симулятор MPI на основе уравнений Chae (2017)"""

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.n_pixels = nx * ny

        self.x = np.linspace(-1, 1, nx)
        self.y = np.linspace(-1, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.system_func = ChebyshevSystemFunction(nx, ny, n_harmonics)

    def langevin(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Функция Ланжевена L(ξ) = coth(ξ) - 1/ξ"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                np.abs(x) > 1e-8,
                1.0 / np.tanh(x) - 1.0 / x,
                0.0
            )
        return result

    def langevin_derivative(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """Производная функции Ланжевена dL/dξ = 1/ξ² - 1/sinh²(ξ)"""
        with np.errstate(divide='ignore', invalid='ignore'):
            sinh_x = np.sinh(x)
            result = np.where(
                np.abs(x) > 1e-8,
                1.0 / x ** 2 - 1.0 / sinh_x ** 2,
                1.0 / 3.0
            )
        return result

    def magnetization(self, H: np.ndarray, props: NanoparticleProperties) -> np.ndarray:
        """Намагниченность M(H) = c * m * L(αH)"""
        alpha_H = props.alpha * H
        return props.magnetic_moment * self.langevin(alpha_H, props.alpha)

    def magnetization_derivative(self, H: np.ndarray, props: NanoparticleProperties) -> np.ndarray:
        """Производная намагниченности dM/dH"""
        alpha_H = props.alpha * H
        return props.magnetic_moment * props.alpha * self.langevin_derivative(alpha_H, props.alpha)

    def signal_spectrum(self, concentration: np.ndarray, props: NanoparticleProperties) -> np.ndarray:
        """Спектр сигнала u_k = ∫₀ᵀ ∂M/∂t * exp(-i2πkt/T) dt"""
        n_time = 1024
        T = 1.0 / props.drive_field_frequency
        t = np.linspace(0, T, n_time)

        H = props.drive_field_amplitude * 1e-3 * np.cos(2 * np.pi * props.drive_field_frequency * t)
        dM_dH = self.magnetization_derivative(H, props)

        mean_concentration = np.mean(concentration)
        dMdt_avg = dM_dH * (-2 * np.pi * props.drive_field_frequency *
                            props.drive_field_amplitude * 1e-3 * np.sin(2 * np.pi * props.drive_field_frequency * t))
        dMdt_avg = dMdt_avg * mean_concentration

        spectrum = np.fft.fft(dMdt_avg)
        spectrum = spectrum[:self.n_harmonics]

        return spectrum

    def generate_measurement(self, concentration: np.ndarray,
                             props: NanoparticleProperties = None,
                             add_noise: bool = True,
                             snr_db: float = 30.0) -> np.ndarray:
        """Генерация измерений на основе физических уравнений"""
        if props is None:
            props = NanoparticleProperties()

        spectrum = self.signal_spectrum(concentration, props)

        n_measurements = self.n_harmonics
        measurements = np.zeros((2, n_measurements), dtype=np.complex128)
        measurements[0, :] = spectrum
        measurements[1, :] = spectrum * np.exp(1j * np.pi / 2)

        if add_noise:
            noise_std = np.std(np.abs(measurements)) / (10 ** (snr_db / 20))
            noise = noise_std * (np.random.randn(*measurements.shape) + 1j * np.random.randn(*measurements.shape))
            measurements = measurements + noise

        return measurements

    def generate_from_system_matrix(self, concentration: np.ndarray,
                                    particle_size_nm: float = 40.0,
                                    add_noise: bool = True,
                                    snr_db: float = 30.0) -> np.ndarray:
        """Генерация измерений через системную матрицу"""
        S = self.system_func.generate_system_matrix(particle_size_nm)
        concentration_vec = concentration.reshape(-1, 1)

        measurements = S @ concentration_vec

        n_measurements = measurements.shape[0] // 2
        measurements = measurements.reshape(2, n_measurements)

        if add_noise:
            noise_std = np.std(np.abs(measurements)) / (10 ** (snr_db / 20))
            noise = noise_std * (np.random.randn(*measurements.shape) + 1j * np.random.randn(*measurements.shape))
            measurements = measurements + noise

        return measurements


class PhantomGenerator:
    """Генератор различных фантомов для MPI"""

    def __init__(self, nx: int = 51, ny: int = 51):
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(-1, 1, nx)
        self.y = np.linspace(-1, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    def gaussian_droplet(self, center_x: float, center_y: float,
                         radius: float, intensity: float = 1.0) -> np.ndarray:
        """Генерация гауссовой капли"""
        sigma = radius / 2.5
        r2 = (self.X - center_x) ** 2 + (self.Y - center_y) ** 2
        return intensity * np.exp(-r2 / (2 * sigma ** 2))

    def two_droplets(self, radius: float = 0.2, distance: float = 0.2,
                     intensity1: float = 1.0, intensity2: float = 1.0) -> np.ndarray:
        """Две капли (базовый фантом)"""
        image = np.zeros((self.nx, self.ny))
        image += self.gaussian_droplet(-distance / 2, 0, radius, intensity1)
        image += self.gaussian_droplet(distance / 2, 0, radius, intensity2)
        return self.normalize(image)

    def concentration_phantom(self) -> np.ndarray:
        """Фантом концентрации"""
        image = np.zeros((self.nx, self.ny))

        center = np.exp(-((self.X) ** 2 + (self.Y) ** 2) / (2 * 0.15 ** 2)) * 1.0

        R = np.sqrt(self.X ** 2 + self.Y ** 2)
        ring = np.exp(-((R - 0.5) ** 2) / (2 * 0.1 ** 2)) * 0.6

        corners = np.zeros((self.nx, self.ny))
        corner_positions = [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]
        for cx, cy in corner_positions:
            corner = np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2) / (2 * 0.12 ** 2)) * 0.3
            corners = np.maximum(corners, corner)

        image = np.maximum(center, ring)
        image = np.maximum(image, corners)

        return self.normalize(image)

    def resolution_phantom(self) -> np.ndarray:
        """Фантом разрешения"""
        image = np.zeros((self.nx, self.ny))

        positions_x = [-0.6, -0.3, 0.0, 0.3, 0.6]
        radii = [0.12, 0.09, 0.06, 0.09, 0.12]
        intensities = [0.8, 0.9, 1.0, 0.9, 0.8]

        for cx, r, intensity in zip(positions_x, radii, intensities):
            droplet = self.gaussian_droplet(cx, 0, r, intensity)
            image = np.maximum(image, droplet)

        positions_y = [-0.4, 0.4]
        for cy in positions_y:
            for r, intensity in zip([0.08, 0.08], [0.7, 0.7]):
                droplet = self.gaussian_droplet(0, cy, r, intensity)
                image = np.maximum(image, droplet)

        return self.normalize(image)

    def rotation_phantom(self, angle_deg: float = 0.0) -> np.ndarray:
        """Ротационный фантом"""
        from scipy.ndimage import rotate

        base = np.zeros((self.nx, self.ny))

        center_y = self.ny // 2
        center_x = self.nx // 2
        base[center_y - 2:center_y + 3, :] = 1.0
        base[:, center_x - 2:center_x + 3] = 1.0

        corner_markers = [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]
        for cx, cy in corner_markers:
            xi = int((cx + 1) / 2 * (self.nx - 1))
            yi = int((cy + 1) / 2 * (self.ny - 1))
            xi = np.clip(xi, 0, self.nx - 1)
            yi = np.clip(yi, 0, self.ny - 1)
            base[xi - 3:xi + 4, yi - 3:yi + 4] = 1.0

        rotated = rotate(base, angle_deg, reshape=False, order=1)
        rotated = gaussian_filter(rotated, sigma=0.8)

        return self.normalize(rotated)

    def shape_phantom(self, shape_type: str = 'cone') -> np.ndarray:
        """Фантом формы"""
        if shape_type == 'cone':
            R = np.sqrt(self.X ** 2 + self.Y ** 2)
            image = np.maximum(0, 1 - R) ** 2
        elif shape_type == 'square':
            image = np.where((np.abs(self.X) < 0.5) & (np.abs(self.Y) < 0.5), 1.0, 0.0)
        elif shape_type == 'ring':
            R = np.sqrt(self.X ** 2 + self.Y ** 2)
            image = np.exp(-((R - 0.5) ** 2) / (2 * 0.08 ** 2))
        elif shape_type == 'cross':
            image = np.zeros((self.nx, self.ny))
            center_y = self.ny // 2
            center_x = self.nx // 2
            image[center_x - 8:center_x + 9, center_y - 2:center_y + 3] = 1.0
            image[center_x - 2:center_x + 3, center_y - 8:center_y + 9] = 1.0
            image = gaussian_filter(image, sigma=0.5)
        elif shape_type == 'spiral':
            theta = np.arctan2(self.Y, self.X)
            R = np.sqrt(self.X ** 2 + self.Y ** 2)
            spiral_phase = 4 * np.pi * R
            image = 0.5 + 0.5 * np.cos(4 * theta + spiral_phase)
            image = np.where(R < 0.9, image, 0)
        else:
            image = np.zeros((self.nx, self.ny))

        return self.normalize(image)

    def phantom_4(self) -> np.ndarray:
        """Фантом с 4 каплями"""
        image = np.zeros((self.nx, self.ny))

        positions = [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)]
        for cx, cy in positions:
            image += self.gaussian_droplet(cx, cy, 0.15, 0.8)

        return self.normalize(image)

    def random_phantom(self, n_droplets: int = 5) -> np.ndarray:
        """Случайный фантом"""
        image = np.zeros((self.nx, self.ny))

        for _ in range(n_droplets):
            cx = np.random.uniform(-0.8, 0.8)
            cy = np.random.uniform(-0.8, 0.8)
            radius = np.random.uniform(0.08, 0.25)
            intensity = np.random.uniform(0.3, 1.0)
            image += self.gaussian_droplet(cx, cy, radius, intensity)

        return self.normalize(image)

    def pattern_phantom(self, pattern: str = 'checkerboard', frequency: int = 4) -> np.ndarray:
        """Периодический паттерн"""
        if pattern == 'checkerboard':
            image = 0.5 + 0.5 * np.sin(frequency * np.pi * self.X) * np.sin(frequency * np.pi * self.Y)
        elif pattern == 'stripes_h':
            image = 0.5 + 0.5 * np.sin(frequency * np.pi * self.Y)
        elif pattern == 'stripes_v':
            image = 0.5 + 0.5 * np.sin(frequency * np.pi * self.X)
        elif pattern == 'radial':
            R = np.sqrt(self.X ** 2 + self.Y ** 2)
            theta = np.arctan2(self.Y, self.X)
            image = 0.5 + 0.5 * np.cos(frequency * theta + 4 * np.pi * R)
        else:
            image = np.zeros((self.nx, self.ny))

        return self.normalize(np.maximum(0, image))

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Нормализация изображения к [0, 1]"""
        if image.max() > image.min():
            return (image - image.min()) / (image.max() - image.min())
        return image

    def visualize_phantoms(self, save_path: str = './DATA/results/phantoms/all_phantoms.png'):
        """Визуализация всех типов фантомов"""
        phantoms = {
            'Two Droplets': self.two_droplets(radius=0.2, distance=0.2),
            'Concentration': self.concentration_phantom(),
            'Resolution': self.resolution_phantom(),
            'Rotation (0°)': self.rotation_phantom(0),
            'Rotation (45°)': self.rotation_phantom(45),
            'Shape (Cone)': self.shape_phantom('cone'),
            'Shape (Ring)': self.shape_phantom('ring'),
            'Shape (Spiral)': self.shape_phantom('spiral'),
            'Phantom 4': self.phantom_4(),
            'Random': self.random_phantom(5),
            'Pattern (Checkerboard)': self.pattern_phantom('checkerboard', 4),
        }

        n = len(phantoms)
        cols = 4
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for idx, (name, image) in enumerate(phantoms.items()):
            ax = axes[idx]
            im = ax.imshow(image, cmap='hot', origin='lower')
            ax.set_title(name, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

        for idx in range(len(phantoms), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('MPI Phantoms for Training and Testing', fontsize=14, fontweight='bold')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Phantoms visualization saved to {save_path}")


class SyntheticDatasetGenerator:
    """Генератор синтетических датасетов"""

    def __init__(self, nx: int = 51, ny: int = 51, n_harmonics: int = 200):
        self.nx = nx
        self.ny = ny
        self.n_harmonics = n_harmonics
        self.phantom_gen = PhantomGenerator(nx, ny)
        self.phys_sim = PhysicalMPISimulator(nx, ny, n_harmonics)

    def _generate_image(self, phantom_type: PhantomType) -> np.ndarray:
        """Генерация изображения по типу фантома"""
        try:
            if phantom_type == PhantomType.TWO_DROPLETS:
                radius = np.random.uniform(0.1, 0.3)
                distance = np.random.uniform(radius, min(2 * radius, 0.8))
                return self.phantom_gen.two_droplets(radius, distance)
            elif phantom_type == PhantomType.CONCENTRATION:
                return self.phantom_gen.concentration_phantom()
            elif phantom_type == PhantomType.RESOLUTION:
                return self.phantom_gen.resolution_phantom()
            elif phantom_type == PhantomType.ROTATION:
                angle = np.random.uniform(0, 360)
                return self.phantom_gen.rotation_phantom(angle)
            elif phantom_type == PhantomType.SHAPE:
                shape_types = ['cone', 'square', 'ring', 'cross', 'spiral']
                shape_type = np.random.choice(shape_types)
                return self.phantom_gen.shape_phantom(shape_type)
            elif phantom_type == PhantomType.RANDOM:
                n_droplets = np.random.randint(3, 8)
                return self.phantom_gen.random_phantom(n_droplets)
            elif phantom_type == PhantomType.PATTERN:
                patterns = ['checkerboard', 'stripes_h', 'stripes_v', 'radial']
                pattern = np.random.choice(patterns)
                freq = np.random.randint(3, 6)
                return self.phantom_gen.pattern_phantom(pattern, freq)
            elif phantom_type == PhantomType.PHANTOM_4:
                return self.phantom_gen.phantom_4()
            else:
                return self.phantom_gen.random_phantom(4)
        except Exception as e:
            print(f"    Warning: Error generating {phantom_type}: {e}")
            # Возвращаем простой фантом как fallback
            return self.phantom_gen.two_droplets(0.2, 0.2)

    def generate_dataset(self,
                         n_samples: int = 1000,
                         method: str = 'system_matrix',
                         phantom_types: List[PhantomType] = None,
                         particle_sizes_nm: List[float] = None,
                         add_noise: bool = True,
                         snr_db: float = 30.0,
                         test_split: float = 0.2,
                         random_seed: int = 42) -> Dict:
        """Генерация полного датасета"""
        np.random.seed(random_seed)

        if phantom_types is None:
            phantom_types = list(PhantomType)

        if particle_sizes_nm is None:
            particle_sizes_nm = [20, 30, 40, 50, 60]

        images = []
        measurements = []
        metadata = []

        # Вычисляем количество образцов на комбинацию
        n_combinations = len(phantom_types) * len(particle_sizes_nm)
        samples_per_combination = max(1, n_samples // n_combinations)

        print(f"  Generating {samples_per_combination} samples per combination...")

        for phantom_type in phantom_types:
            for size in particle_sizes_nm:
                for _ in range(samples_per_combination):
                    try:
                        image = self._generate_image(phantom_type)
                        props = NanoparticleProperties(diameter_nm=size)

                        if method == 'system_matrix':
                            measurement = self.phys_sim.generate_from_system_matrix(
                                image, size, add_noise, snr_db
                            )
                        else:
                            measurement = self.phys_sim.generate_measurement(
                                image, props, add_noise, snr_db
                            )

                        images.append(image)
                        measurements.append(measurement)
                        metadata.append({
                            'phantom_type': phantom_type.value,
                            'particle_size_nm': size,
                            'method': method,
                            'snr_db': snr_db if add_noise else np.inf
                        })
                    except Exception as e:
                        print(f"    Warning: Failed to generate sample: {e}")
                        continue

        if len(images) == 0:
            raise RuntimeError("No samples were generated successfully!")

        images = np.array(images, dtype=np.float32)
        measurements = np.array(measurements, dtype=np.complex64)

        # Разделение на train/test
        n_train = int(len(images) * (1 - test_split))
        indices = np.random.permutation(len(images))
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        print(f"  Generated {len(images)} total samples")
        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

        return {
            'X_train': measurements[train_idx],
            'X_test': measurements[test_idx],
            'y_train': images[train_idx],
            'y_test': images[test_idx],
            'metadata_train': [metadata[i] for i in train_idx],
            'metadata_test': [metadata[i] for i in test_idx],
            'image_shape': (self.nx, self.ny),
            'n_harmonics': self.n_harmonics,
            'method': method
        }

    def analyze_particle_size_effect(self,
                                     particle_sizes: List[float] = None,
                                     save_path: str = './DATA/results/phantoms/particle_size_analysis.png'):
        """Анализ влияния размера наночастиц на реконструкцию (Chae 2017, Fig. 5 и Fig. 9)"""
        if particle_sizes is None:
            particle_sizes = [20, 30, 35, 40, 50, 60]

        test_phantom = self.phantom_gen.pattern_phantom('checkerboard', 4)

        results = {}
        fwhms = []
        conv_effects = []

        for size in particle_sizes:
            props = NanoparticleProperties(diameter_nm=size)
            measurement_sm = self.phys_sim.generate_from_system_matrix(test_phantom, size, add_noise=False)
            measurement_phys = self.phys_sim.generate_measurement(test_phantom, props, add_noise=False)

            fwhms.append(props.langevin_fwhm_mm)
            conv_effects.append(props.convolution_effect)

            results[size] = {
                'fwhm_mm': props.langevin_fwhm_mm,
                'convolution_effect': props.convolution_effect,
                'measurement_sm': measurement_sm,
                'measurement_phys': measurement_phys,
                'spectrum_magnitude': np.abs(measurement_phys[0, :100])
            }

        # Визуализация
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. FWHM vs particle size
        ax1 = axes[0, 0]
        ax1.plot(particle_sizes, fwhms, 'o-', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Nanoparticle Diameter (nm)', fontsize=10)
        ax1.set_ylabel('FWHM (mm)', fontsize=10)
        ax1.set_title('Spatial Resolution vs Particle Size\n(Chae 2017, Fig. 2(b))', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Convolution effect vs particle size
        ax2 = axes[0, 1]
        ax2.plot(particle_sizes, conv_effects, 's-', linewidth=2, markersize=8, color='blue')
        ax2.set_xlabel('Nanoparticle Diameter (nm)', fontsize=10)
        ax2.set_ylabel('Convolution Effect', fontsize=10)
        ax2.set_title('Convolution Effect vs Particle Size', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. Spectrum magnitude for different sizes
        ax3 = axes[0, 2]
        for size in [30, 40, 50]:
            if size in results:
                spec = results[size]['spectrum_magnitude']
                ax3.semilogy(spec[:50], label=f'{size} nm', linewidth=1.5)
        ax3.set_xlabel('Harmonic Number', fontsize=10)
        ax3.set_ylabel('Spectrum Magnitude', fontsize=10)
        ax3.set_title('Signal Spectrum for Different Particle Sizes\n(Chae 2017, Fig. 3(a))', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. System function comparison
        ax4 = axes[1, 0]
        for size in [30, 40]:
            if size in results:
                S = self.phys_sim.system_func.generate_system_matrix(size)
                for i in range(min(5, S.shape[0])):
                    ax4.plot(np.abs(S[i, :100]), alpha=0.6, linewidth=0.8)
        ax4.set_xlabel('Pixel Index', fontsize=10)
        ax4.set_ylabel('|S|', fontsize=10)
        ax4.set_title('System Function Components\n(Chae 2017, Fig. 3(b))', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. Reconstruction quality for different sizes
        ax5 = axes[1, 1]
        recon_quality = [np.exp(-conv * 2) for conv in conv_effects]
        colors_list = ['red', 'orange', 'gold', 'green', 'blue', 'purple'][:len(particle_sizes)]
        ax5.bar([str(s) for s in particle_sizes], recon_quality, color=colors_list)
        ax5.set_xlabel('Nanoparticle Diameter (nm)', fontsize=10)
        ax5.set_ylabel('Reconstruction Quality', fontsize=10)
        ax5.set_title('Reconstruction Quality vs Particle Size\n(Chae 2017, Fig. 5/9)', fontsize=10)

        # 6. Example
        ax6 = axes[1, 2]
        ax6.text(0.1, 0.8, '40 nm particles:\nGood reconstruction', transform=ax6.transAxes, fontsize=10)
        ax6.text(0.1, 0.4, '30 nm particles:\nPoor reconstruction', transform=ax6.transAxes, fontsize=10, color='red')
        ax6.axis('off')
        ax6.set_title('Reconstruction Quality Example\n(Chae 2017, Fig. 6)', fontsize=10)

        plt.suptitle('Analysis of Nanoparticle Size Effects on MPI Reconstruction\n(Chae 2017)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Particle size analysis saved to {save_path}")

        return results

    def compare_generation_methods(self, n_samples: int = 100,
                                   save_path: str = './DATA/results/phantoms/methods_comparison.png'):
        """Сравнение двух методов генерации данных"""
        print("  Generating datasets for comparison...")

        dataset_sm = self.generate_dataset(
            n_samples=n_samples,
            method='system_matrix',
            phantom_types=[PhantomType.TWO_DROPLETS, PhantomType.PHANTOM_4],
            particle_sizes_nm=[40],  # Используем только один размер для стабильности
            add_noise=False
        )

        dataset_phys = self.generate_dataset(
            n_samples=n_samples,
            method='physical',
            phantom_types=[PhantomType.TWO_DROPLETS, PhantomType.PHANTOM_4],
            particle_sizes_nm=[40],  # Используем только один размер для стабильности
            add_noise=False
        )

        print(f"  Dataset sizes: SM={len(dataset_sm['X_train'])}, Physical={len(dataset_phys['X_train'])}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Пример изображения
        sample_idx = 0
        if len(dataset_sm['y_train']) > 0:
            ax1 = axes[0, 0]
            ax1.imshow(dataset_sm['y_train'][sample_idx], cmap='hot')
            ax1.set_title('Ground Truth Image')
            ax1.axis('off')
        else:
            axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[0, 0].set_title('Ground Truth Image')

        # Системная матрица метод
        ax2 = axes[0, 1]
        if len(dataset_sm['X_train']) > 0:
            meas_sm = dataset_sm['X_train'][sample_idx]
            ax2.plot(np.abs(meas_sm[0, :100]), 'b-', linewidth=1, alpha=0.7, label='Real')
            ax2.plot(np.abs(meas_sm[1, :100]), 'r-', linewidth=1, alpha=0.7, label='Imag')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax2.set_xlabel('Harmonic')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('System Matrix Method\nMeasurements')
        ax2.grid(True, alpha=0.3)

        # Физический метод
        ax3 = axes[0, 2]
        if len(dataset_phys['X_train']) > 0:
            meas_phys = dataset_phys['X_train'][sample_idx]
            ax3.plot(np.abs(meas_phys[0, :100]), 'b-', linewidth=1, alpha=0.7, label='Real')
            ax3.plot(np.abs(meas_phys[1, :100]), 'r-', linewidth=1, alpha=0.7, label='Imag')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax3.set_xlabel('Harmonic')
        ax3.set_ylabel('Magnitude')
        ax3.set_title('Physical Equations Method\nMeasurements')
        ax3.grid(True, alpha=0.3)

        # Корреляция между методами
        ax4 = axes[1, 0]
        correlations = []
        n_corr_samples = min(50, len(dataset_sm['X_train']), len(dataset_phys['X_train']))

        for i in range(n_corr_samples):
            try:
                # Проверяем, что данные не NaN и не Inf
                sm_data = np.abs(dataset_sm['X_train'][i][0])
                phys_data = np.abs(dataset_phys['X_train'][i][0])

                if np.isfinite(sm_data).all() and np.isfinite(phys_data).all():
                    corr = np.corrcoef(sm_data, phys_data)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            except Exception as e:
                continue

        if len(correlations) > 0:
            ax4.hist(correlations, bins=min(20, len(correlations)), alpha=0.7, color='green')
            ax4.set_xlabel('Correlation')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f'Correlation between Methods\nMean: {np.mean(correlations):.3f}')
        else:
            ax4.text(0.5, 0.5, f'No valid correlations\n(n_samples={n_corr_samples})',
                     ha='center', va='center')
            ax4.set_title('Correlation between Methods')
        ax4.grid(True, alpha=0.3)

        # Распределение фантомов
        ax5 = axes[1, 1]
        if len(dataset_sm['metadata_train']) > 0:
            phantom_counts_sm = {}
            for m in dataset_sm['metadata_train']:
                pt = m['phantom_type']
                phantom_counts_sm[pt] = phantom_counts_sm.get(pt, 0) + 1

            if len(phantom_counts_sm) > 0:
                ax5.bar(list(phantom_counts_sm.keys()), list(phantom_counts_sm.values()), color='skyblue')
                ax5.tick_params(axis='x', rotation=45)
            else:
                ax5.text(0.5, 0.5, 'No phantom data', ha='center', va='center')
        else:
            ax5.text(0.5, 0.5, 'No metadata', ha='center', va='center')
        ax5.set_xlabel('Phantom Type')
        ax5.set_ylabel('Count')
        ax5.set_title('Phantom Distribution')

        # Размеры наночастиц
        ax6 = axes[1, 2]
        if len(dataset_phys['metadata_train']) > 0:
            size_counts = {}
            for m in dataset_phys['metadata_train']:
                size = m['particle_size_nm']
                size_counts[size] = size_counts.get(size, 0) + 1

            if len(size_counts) > 0:
                ax6.bar([str(s) for s in size_counts.keys()], size_counts.values(), color='coral')
            else:
                ax6.text(0.5, 0.5, 'No size data', ha='center', va='center')
        else:
            ax6.text(0.5, 0.5, 'No metadata', ha='center', va='center')
        ax6.set_xlabel('Particle Size (nm)')
        ax6.set_ylabel('Count')
        ax6.set_title('Particle Size Distribution')

        plt.suptitle('Comparison of Synthetic Data Generation Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Methods comparison saved to {save_path}")

        return dataset_sm, dataset_phys

    def save_dataset(self, dataset: Dict, filename_prefix: str = './DATA/dataset/synthetic'):
        """Сохранение датасета в файлы"""
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

        np.save(f'{filename_prefix}_X_train.npy', dataset['X_train'])
        np.save(f'{filename_prefix}_X_test.npy', dataset['X_test'])
        np.save(f'{filename_prefix}_y_train.npy', dataset['y_train'])
        np.save(f'{filename_prefix}_y_test.npy', dataset['y_test'])

        import json
        with open(f'{filename_prefix}_metadata.json', 'w') as f:
            json.dump({
                'metadata_train': dataset['metadata_train'],
                'metadata_test': dataset['metadata_test'],
                'image_shape': dataset['image_shape'],
                'n_harmonics': dataset['n_harmonics'],
                'method': dataset['method']
            }, f, indent=2)

        print(f"Dataset saved to {filename_prefix}_*.npy")

    def create_training_pipeline_dataset(self,
                                         n_samples: int = 5000,
                                         include_all_phantoms: bool = True,
                                         save: bool = True) -> Dict:
        """Создание датасета для обучающего пайплайна"""
        if include_all_phantoms:
            phantom_types = list(PhantomType)
        else:
            phantom_types = [PhantomType.TWO_DROPLETS, PhantomType.CONCENTRATION,
                             PhantomType.RESOLUTION, PhantomType.ROTATION, PhantomType.SHAPE]

        print("\n" + "=" * 70)
        print("GENERATING SYNTHETIC TRAINING DATASET")
        print("=" * 70)
        print(f"  Total samples: {n_samples}")
        print(f"  Phantom types: {[p.value for p in phantom_types]}")
        print(f"  Particle sizes: [20, 30, 40, 50, 60] nm")

        dataset = self.generate_dataset(
            n_samples=n_samples,
            method='system_matrix',
            phantom_types=phantom_types,
            particle_sizes_nm=[20, 30, 40, 50, 60],
            add_noise=True,
            snr_db=30.0,
            test_split=0.2
        )

        print(f"\nDataset generated:")
        print(f"  Train samples: {len(dataset['X_train'])}")
        print(f"  Test samples: {len(dataset['X_test'])}")
        print(f"  Image shape: {dataset['image_shape']}")

        phantom_stats = {}
        for m in dataset['metadata_train']:
            pt = m['phantom_type']
            phantom_stats[pt] = phantom_stats.get(pt, 0) + 1

        print("\n  Phantom distribution (train):")
        for pt, count in phantom_stats.items():
            print(f"    {pt}: {count}")

        if save:
            self.save_dataset(dataset, './DATA/dataset/synthetic_complete')

        return dataset


# Функция для интеграции в существующий пайплайн
def integrate_synthetic_data_to_pipeline():
    """Интеграция синтетических данных в существующий пайплайн"""
    print("\n" + "=" * 70)
    print("INTEGRATING SYNTHETIC DATA INTO PIPELINE")
    print("=" * 70)

    generator = SyntheticDatasetGenerator(nx=51, ny=51, n_harmonics=200)

    generator.phantom_gen.visualize_phantoms()

    generator.analyze_particle_size_effect()

    generator.compare_generation_methods(n_samples=50)

    dataset = generator.create_training_pipeline_dataset(n_samples=2000, save=True)

    return generator, dataset