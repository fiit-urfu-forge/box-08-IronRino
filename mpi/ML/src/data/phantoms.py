"""Геометрические 2D-фантомы для синтетических MPI-измерений.

Никакой физики MNP — только пиксельные распределения концентрации,
которые подаются на вход симуляторов в `data/simulators.py`.

Поддерживаемые типы (см. `PhantomType`):
  • TWO_DROPLETS — две гауссовы капли (исторически базовый тест MPI);
  • CONCENTRATION — центральная капля + кольцо + 4 угловых точки;
  • RESOLUTION   — пять точек переменного размера для оценки разрешения;
  • ROTATION     — крест с угловыми маркерами под заданным углом;
  • SHAPE        — конус / квадрат / кольцо / крест / спираль;
  • RANDOM       — случайные капли (для аугментации обучающей выборки);
  • PATTERN      — периодические узоры (шахматка, полосы, радиальный);
  • PHANTOM_4    — 4 угловые капли (классический «четыре точки»).
"""

import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate


class PhantomType(Enum):
    """Типы синтетических 2D-фантомов."""
    TWO_DROPLETS = "two_droplets"
    CONCENTRATION = "concentration"
    RESOLUTION = "resolution"
    ROTATION = "rotation"
    SHAPE = "shape"
    RANDOM = "random"
    PATTERN = "pattern"
    PHANTOM_4 = "phantom_4"


class PhantomGenerator:
    """Генератор пиксельных распределений концентрации.

    Координатная сетка нормализована: x, y ∈ [−1, 1]. Все методы возвращают
    `float64`-массив формы `(nx, ny)`, нормированный на `[0, 1]`.
    """

    def __init__(self, nx: int = 51, ny: int = 51):
        self.nx = nx
        self.ny = ny
        self.x = np.linspace(-1, 1, nx)
        self.y = np.linspace(-1, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

    # ---- одиночные капли -----------------------------------------------------

    def gaussian_droplet(self, center_x: float, center_y: float,
                         radius: float, intensity: float = 1.0) -> np.ndarray:
        sigma = radius / 2.5
        r2 = (self.X - center_x) ** 2 + (self.Y - center_y) ** 2
        return intensity * np.exp(-r2 / (2 * sigma ** 2))

    def two_droplets(self, radius: float = 0.2, distance: float = 0.2,
                     intensity1: float = 1.0, intensity2: float = 1.0) -> np.ndarray:
        image = np.zeros((self.nx, self.ny))
        image += self.gaussian_droplet(-distance / 2, 0, radius, intensity1)
        image += self.gaussian_droplet(distance / 2, 0, radius, intensity2)
        return self.normalize(image)

    # ---- составные фантомы ---------------------------------------------------

    def concentration_phantom(self) -> np.ndarray:
        """Центральная капля + кольцо + 4 угловых точки разной интенсивности."""
        center = np.exp(-(self.X ** 2 + self.Y ** 2) / (2 * 0.15 ** 2)) * 1.0
        R = np.sqrt(self.X ** 2 + self.Y ** 2)
        ring = np.exp(-((R - 0.5) ** 2) / (2 * 0.1 ** 2)) * 0.6

        corners = np.zeros((self.nx, self.ny))
        for cx, cy in [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]:
            corner = np.exp(-((self.X - cx) ** 2 + (self.Y - cy) ** 2)
                            / (2 * 0.12 ** 2)) * 0.3
            corners = np.maximum(corners, corner)

        return self.normalize(np.maximum.reduce([center, ring, corners]))

    def resolution_phantom(self) -> np.ndarray:
        """5 точек разного размера по оси x + 2 точки по оси y."""
        image = np.zeros((self.nx, self.ny))
        for cx, r, intensity in zip(
            [-0.6, -0.3, 0.0, 0.3, 0.6],
            [0.12, 0.09, 0.06, 0.09, 0.12],
            [0.8, 0.9, 1.0, 0.9, 0.8],
        ):
            image = np.maximum(image, self.gaussian_droplet(cx, 0, r, intensity))
        for cy in [-0.4, 0.4]:
            image = np.maximum(image, self.gaussian_droplet(0, cy, 0.08, 0.7))
        return self.normalize(image)

    def rotation_phantom(self, angle_deg: float = 0.0) -> np.ndarray:
        """Крест в центре + угловые маркеры, повернутые на `angle_deg`."""
        base = np.zeros((self.nx, self.ny))
        cy, cx = self.ny // 2, self.nx // 2
        base[cy - 2:cy + 3, :] = 1.0
        base[:, cx - 2:cx + 3] = 1.0

        for fx, fy in [(-0.7, -0.7), (0.7, -0.7), (0.7, 0.7), (-0.7, 0.7)]:
            xi = int(np.clip((fx + 1) / 2 * (self.nx - 1), 0, self.nx - 1))
            yi = int(np.clip((fy + 1) / 2 * (self.ny - 1), 0, self.ny - 1))
            base[xi - 3:xi + 4, yi - 3:yi + 4] = 1.0

        rotated = rotate(base, angle_deg, reshape=False, order=1)
        return self.normalize(gaussian_filter(rotated, sigma=0.8))

    def shape_phantom(self, shape_type: str = 'cone') -> np.ndarray:
        """Различные геометрические формы."""
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
            cy, cx = self.ny // 2, self.nx // 2
            image[cx - 8:cx + 9, cy - 2:cy + 3] = 1.0
            image[cx - 2:cx + 3, cy - 8:cy + 9] = 1.0
            image = gaussian_filter(image, sigma=0.5)
        elif shape_type == 'spiral':
            theta = np.arctan2(self.Y, self.X)
            R = np.sqrt(self.X ** 2 + self.Y ** 2)
            image = 0.5 + 0.5 * np.cos(4 * theta + 4 * np.pi * R)
            image = np.where(R < 0.9, image, 0)
        else:
            image = np.zeros((self.nx, self.ny))
        return self.normalize(image)

    def phantom_4(self) -> np.ndarray:
        """Четыре угловые капли."""
        image = np.zeros((self.nx, self.ny))
        for cx, cy in [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)]:
            image += self.gaussian_droplet(cx, cy, 0.15, 0.8)
        return self.normalize(image)

    def random_phantom(self, n_droplets: int = 5) -> np.ndarray:
        """N случайно расположенных капель — для аугментации обучающей выборки."""
        image = np.zeros((self.nx, self.ny))
        for _ in range(n_droplets):
            image += self.gaussian_droplet(
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(-0.8, 0.8),
                np.random.uniform(0.08, 0.25),
                np.random.uniform(0.3, 1.0),
            )
        return self.normalize(image)

    def pattern_phantom(self, pattern: str = 'checkerboard',
                        frequency: int = 4) -> np.ndarray:
        """Периодические узоры — для тестирования передаточной функции."""
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

    # ---- утилиты -------------------------------------------------------------

    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        if image.max() > image.min():
            return (image - image.min()) / (image.max() - image.min())
        return image

    def visualize_phantoms(self,
                           save_path: str = './DATA/results/phantoms/all_phantoms.png'):
        """Сетка изображений всех типов фантомов — для отчётов."""
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
        cols = 4
        rows = (len(phantoms) + cols - 1) // cols
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
        plt.suptitle('MPI Phantoms for Training and Testing',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Phantoms visualization saved to {save_path}")


__all__ = ['PhantomType', 'PhantomGenerator']
