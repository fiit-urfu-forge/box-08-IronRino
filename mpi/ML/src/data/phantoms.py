"""Геометрические 2D-фантомы для синтетических MPI-измерений.

Никакой физики MNP — только пиксельные распределения концентрации,
которые подаются на вход симуляторов в `data/simulators.py`.

Все типы фантомов покрывают тестовую батарею в `build_phantom_battery`:
  • TWO_DROPLETS — две гауссовы капли (test: two_droplets);
  • PHANTOM_4    — 4 угловые капли (test: phantom_4);
  • ROTATION     — крест с маркерами под углом (test: rotation_45);
  • RANDOM       — случайные капли (test: random);
  • MULTI_POINT  — произвольное число точек (2–10) с min-distance —
                   расширяет two_droplets и phantom_4;
  • LINES        — 1–3 непрерывных линейных сегмента (сосудистые фантомы);
  • CIRCLES      — 1–3 непрерывных кольца/диска (расширяет shape_ring).

Также есть служебные методы:
  • `shape_phantom('ring')`  — детерминированное кольцо для тестов;
  • `letter_phantom('B')`    — приближение буквы B для real-data validation
                               (как approximate ground truth для
                               MeasurementData_B.h5).
"""

import os
from enum import Enum
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate


class PhantomType(Enum):
    """Типы синтетических 2D-фантомов (совпадают с типами в тест-батарее)."""
    TWO_DROPLETS = "two_droplets"
    ROTATION = "rotation"
    RANDOM = "random"
    PHANTOM_4 = "phantom_4"
    MULTI_POINT = "multi_point"
    LINES = "lines"
    CIRCLES = "circles"


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

    def shape_phantom(self, shape_type: str = 'ring') -> np.ndarray:
        """Детерминированное центрированное кольцо (R=0.5, толщина 0.08).

        Используется тест-батареей как `shape_ring` фантом. Параметр
        `shape_type` оставлен для совместимости интерфейса; поддерживается
        только 'ring'. Для произвольных колец с варьируемыми параметрами
        см. `circles_phantom`.
        """
        if shape_type != 'ring':
            raise ValueError(
                f"shape_phantom поддерживает только 'ring' (получено: "
                f"{shape_type!r}). Для других форм см. circles_phantom, "
                f"multi_point_phantom, lines_phantom."
            )
        R = np.sqrt(self.X ** 2 + self.Y ** 2)
        image = np.exp(-((R - 0.5) ** 2) / (2 * 0.08 ** 2))
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

    # ---- новые типы для расширения обучающей выборки ------------------------

    def multi_point_phantom(self, n_min: int = 2, n_max: int = 10,
                            min_distance: float = 0.25,
                            radius_range: tuple = (0.06, 0.18),
                            concentration_range: tuple = (0.3, 1.0),
                            max_attempts_per_point: int = 50) -> np.ndarray:
        """Произвольное число точек (N ∈ [n_min, n_max]) с минимальным
        расстоянием между ними и разной концентрацией.

        Расширенная версия RANDOM: точки гарантированно не сливаются благодаря
        rejection-sampling с `min_distance`. Покрывает phantom_4-подобные
        конфигурации (2, 3, 4, 5, ... точек), где модель должна научиться
        локализовать дискретные источники с переменным количеством.

        Args:
            n_min, n_max: диапазон случайного выбора числа точек.
            min_distance: минимальное расстояние между центрами в [−1, 1]
                          координатах (0.25 = 25% радиуса FOV).
            radius_range: (min, max) размер каждой капли (sigma в gaussian).
            concentration_range: (min, max) амплитуда каждой капли.
            max_attempts_per_point: сколько попыток до отказа от текущей
                                     точки (предотвращает бесконечный цикл).
        """
        n = np.random.randint(n_min, n_max + 1)
        image = np.zeros((self.nx, self.ny))
        centers = []
        for _ in range(n):
            for _attempt in range(max_attempts_per_point):
                cx = np.random.uniform(-0.85, 0.85)
                cy = np.random.uniform(-0.85, 0.85)
                ok = all(
                    (cx - px) ** 2 + (cy - py) ** 2 >= min_distance ** 2
                    for px, py in centers
                )
                if ok:
                    centers.append((cx, cy))
                    image += self.gaussian_droplet(
                        cx, cy,
                        np.random.uniform(*radius_range),
                        np.random.uniform(*concentration_range),
                    )
                    break
            # Если не нашли место за max_attempts — пропускаем эту точку
        return self.normalize(image)

    def lines_phantom(self, n_lines_min: int = 1, n_lines_max: int = 3,
                      thickness_range: tuple = (0.025, 0.06),
                      intensity_range: tuple = (0.5, 1.0)) -> np.ndarray:
        """1–3 непрерывных линейных сегмента — приближение сосудистых фантомов.

        Каждая линия задаётся двумя случайными концами в FOV. Толщина —
        гауссово размытие вдоль перпендикуляра. Несколько линий могут
        пересекаться (имитация бифуркаций сосудов).

        Используется как:
          (а) Тест разрешения в нелинейных режимах (длинные линии нагружают
              продольный отклик частиц);
          (б) Структурно-непрерывный фантом для регуляризационных моделей
              (TV-регуляризация в Final ожидает кусочно-гладкие изображения).
        """
        n_lines = np.random.randint(n_lines_min, n_lines_max + 1)
        image = np.zeros((self.nx, self.ny))
        for _ in range(n_lines):
            # Случайные концы отрезка
            x0, y0 = np.random.uniform(-0.8, 0.8, 2)
            x1, y1 = np.random.uniform(-0.8, 0.8, 2)
            thickness = np.random.uniform(*thickness_range)
            intensity = np.random.uniform(*intensity_range)
            # Расстояние от каждого пикселя до отрезка [(x0,y0)→(x1,y1)]
            dx, dy = x1 - x0, y1 - y0
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-8:
                continue  # вырожденный отрезок (точка)
            # Проекция (X-x0, Y-y0) на (dx, dy), параметризованная t ∈ [0, 1]
            t = ((self.X - x0) * dx + (self.Y - y0) * dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            # Ближайшая точка на отрезке
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            # Расстояние до отрезка
            dist_sq = (self.X - proj_x) ** 2 + (self.Y - proj_y) ** 2
            # Гауссов профиль вдоль перпендикуляра
            image += intensity * np.exp(-dist_sq / (2.0 * thickness ** 2))
        return self.normalize(image)

    def circles_phantom(self, n_circles_min: int = 1, n_circles_max: int = 3,
                        radius_range: tuple = (0.18, 0.5),
                        thickness_range: tuple = (0.03, 0.08),
                        intensity_range: tuple = (0.5, 1.0),
                        allow_filled: bool = True) -> np.ndarray:
        """1–3 непрерывных кольца/диска со случайными параметрами.

        Покрывает shape_ring-подобные конфигурации, но с варьируемой
        геометрией. Если `allow_filled=True`, ~30% колец заполняются
        (диски), остальные остаются полыми (кольца). Несколько колец могут
        быть концентрическими.

        Args:
            n_circles_min, n_circles_max: диапазон числа колец.
            radius_range: (min, max) радиус каждого кольца.
            thickness_range: (min, max) толщина границы (sigma).
            intensity_range: (min, max) амплитуда.
            allow_filled: разрешить заполненные диски (иначе только кольца).
        """
        n = np.random.randint(n_circles_min, n_circles_max + 1)
        image = np.zeros((self.nx, self.ny))
        for _ in range(n):
            # Центр и радиус кольца. Центр сдвигается так, чтобы кольцо
            # помещалось в FOV (с запасом)
            R = np.random.uniform(*radius_range)
            max_off = max(0.0, 0.9 - R)
            cx = np.random.uniform(-max_off, max_off)
            cy = np.random.uniform(-max_off, max_off)
            thickness = np.random.uniform(*thickness_range)
            intensity = np.random.uniform(*intensity_range)
            r_pixel = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
            if allow_filled and np.random.random() < 0.3:
                # Заполненный диск: 1 внутри R, плавный край
                disk_dist = np.maximum(r_pixel - R, 0.0)
                image += intensity * np.exp(-disk_dist ** 2 / (2.0 * thickness ** 2))
            else:
                # Кольцо: гауссова "толщина" вокруг r=R
                image += intensity * np.exp(
                    -(r_pixel - R) ** 2 / (2.0 * thickness ** 2)
                )
        return self.normalize(image)

    def letter_phantom(self, letter: str = 'B') -> np.ndarray:
        """Стилизованная буква 'B' — приближение к реальному фантому из
        BeihangUniversityData (MeasurementData_B.h5).

        Используется тест-батареей как approximate ground truth при
        валидации на real-data (точная форма фантома B известна по
        дизайну; точное изображение со сканера недоступно). Параметр
        `letter` оставлен для совместимости интерфейса; поддерживается
        только 'B'.

        Контуры рисуются по нормированной сетке [−1, 1] — пропорции
        сохраняются при разных `nx, ny`.
        """
        if letter.upper() != 'B':
            raise ValueError(
                f"letter_phantom поддерживает только 'B' (получено: "
                f"{letter!r})."
            )
        img = np.zeros((self.nx, self.ny), dtype=np.float32)

        def rect(x0, x1, y0, y1, v=1.0):
            mask = ((self.X >= x0) & (self.X <= x1) &
                    (self.Y >= y0) & (self.Y <= y1))
            img[mask] = np.maximum(img[mask], v)

        def disk(cx, cy, r, v=1.0):
            d2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2
            mask = d2 <= r * r
            img[mask] = np.maximum(img[mask], v)

        # Вертикальная палка (left bar) + две полуокружности справа
        rect(-0.6, -0.4, -0.8, 0.8)
        disk(-0.1, 0.4, 0.4)
        disk(-0.1, -0.4, 0.4)
        disk(0.0, 0.4, 0.22, 0.0)        # вырезаем «дырку» = 0
        disk(0.0, -0.4, 0.22, 0.0)
        return self.normalize(img)

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
            'Rotation (45°)': self.rotation_phantom(45),
            'Phantom 4': self.phantom_4(),
            'Random': self.random_phantom(5),
            'Multi-Point': self.multi_point_phantom(n_min=4, n_max=8),
            'Lines': self.lines_phantom(n_lines_min=2, n_lines_max=3),
            'Circles': self.circles_phantom(n_circles_min=2, n_circles_max=3),
            'Shape (Ring)': self.shape_phantom('ring'),
            'Letter B': self.letter_phantom('B'),
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
