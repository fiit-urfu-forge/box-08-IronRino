"""Метрики оценки качества реконструкции MPI.

## Используемые метрики

  • **SSIM** (Structural Similarity Index, Wang et al. 2004) — мера
    структурного сходства между восстановленным и эталонным
    изображениями. Диапазон [−1, 1], где 1 = идеальное совпадение.
    Учитывает не только пиксельные значения, но и локальную
    статистику (среднее, дисперсию, ковариацию). Лучше отражает
    «визуальное» качество, чем MSE/PSNR.

  • **PSNR** (Peak Signal-to-Noise Ratio) — отношение пиковой
    интенсивности к среднеквадратичной ошибке, в децибелах:
        PSNR = 10·log₁₀(max(I)² / MSE).
    Высокие значения = низкая ошибка. Типичные значения для
    реконструкции MPI: 15–25 dB.

  • **FWHM** (Full Width at Half Maximum) — полуширина точечной
    функции рассеяния на половине высоты. Мера пространственного
    разрешения: меньше FWHM = лучше разрешение мелких деталей.
    Измеряется в пикселях.

  • **time** — время выполнения реконструкции в секундах. Важно
    для practical применимости: data-free методы (DIP, PMCNet)
    тратят минуты на измерение, классические (Tikhonov) — секунды,
    обученные сети (CNN, MoDL) — миллисекунды.

## Унификация

Все метрики автоматически нормализуют входы перед сравнением
(масштабируют к [0, 1]), чтобы метрики не зависели от абсолютной
амплитуды реконструкции. Это критично, потому что разные методы
могут давать выходы в разных шкалах (sigmoid даёт [0, 1], ReLU
оставляет неограниченным).
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import ndimage


class MetricsCalculator:
    """Калькулятор метрик для оценки качества реконструкции"""

    @staticmethod
    def normalize(image):
        """Нормализация изображения к диапазону [0, 1]"""
        if image.max() - image.min() > 1e-10:
            return (image - image.min()) / (image.max() - image.min())
        return image

    @staticmethod
    def calculate_ssim(original, reconstructed, win_size=7):
        """Вычисление SSIM (Structural Similarity Index)"""
        orig_norm = MetricsCalculator.normalize(original)
        recon_norm = MetricsCalculator.normalize(reconstructed)

        # Автоматический выбор win_size
        min_dim = min(orig_norm.shape)
        if win_size > min_dim:
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1

        return ssim(orig_norm, recon_norm, data_range=1.0,
                    win_size=win_size, channel_axis=None)

    @staticmethod
    def calculate_psnr(original, reconstructed):
        """Вычисление PSNR (Peak Signal-to-Noise Ratio)"""
        orig_norm = MetricsCalculator.normalize(original)
        recon_norm = MetricsCalculator.normalize(reconstructed)
        return psnr(orig_norm, recon_norm, data_range=1.0)

    @staticmethod
    def calculate_mse(original, reconstructed):
        """Вычисление MSE (Mean Squared Error)"""
        orig_norm = MetricsCalculator.normalize(original)
        recon_norm = MetricsCalculator.normalize(reconstructed)
        return np.mean((orig_norm - recon_norm) ** 2)

    @staticmethod
    def calculate_fwhm(image):
        """Вычисление FWHM (Full Width at Half Maximum) для оценки разрешения"""
        max_val = np.max(image)
        if max_val == 0:
            return 0

        half_max = max_val / 2
        binary = image > half_max

        if not np.any(binary):
            return 0

        labeled, num_features = ndimage.label(binary)

        if num_features == 0:
            return 0

        fwhm_values = []
        for i in range(1, num_features + 1):
            component = labeled == i

            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)

            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]

                fwhm_h = cmax - cmin
                fwhm_v = rmax - rmin
                fwhm_values.extend([fwhm_h, fwhm_v])

        return np.mean(fwhm_values) if fwhm_values else 0

    @staticmethod
    def calculate_all_metrics(original, reconstructed):
        """Вычисление всех метрик"""
        return {
            'ssim': MetricsCalculator.calculate_ssim(original, reconstructed),
            'psnr': MetricsCalculator.calculate_psnr(original, reconstructed),
            'mse': MetricsCalculator.calculate_mse(original, reconstructed),
            'fwhm': MetricsCalculator.calculate_fwhm(reconstructed)
        }

    @staticmethod
    def print_metrics(metrics, method_name):
        """Вывод метрик в консоль"""
        print(f"\n{method_name}:")
        print(f"  SSIM:  {metrics['ssim']:.4f}")
        print(f"  PSNR:  {metrics['psnr']:.2f} дБ")
        print(f"  MSE:   {metrics['mse']:.6f}")
        print(f"  FWHM:  {metrics['fwhm']:.2f} пикс.")


class MetricsTracker:
    """Трекер метрик для сравнения методов"""

    def __init__(self):
        self.history = []

    def add_result(self, method_name, metrics, time_seconds=None):
        """Добавление результата"""
        result = {
            'method': method_name,
            **metrics
        }
        if time_seconds is not None:
            result['time'] = time_seconds
        self.history.append(result)

    def get_summary(self):
        """Получение сводной статистики"""
        if not self.history:
            return {}

        methods = set(r['method'] for r in self.history)
        summary = {}

        for method in methods:
            method_results = [r for r in self.history if r['method'] == method]
            summary[method] = {
                'ssim_mean': np.mean([r['ssim'] for r in method_results]),
                'ssim_std': np.std([r['ssim'] for r in method_results]),
                'psnr_mean': np.mean([r['psnr'] for r in method_results]),
                'psnr_std': np.std([r['psnr'] for r in method_results]),
                'mse_mean': np.mean([r['mse'] for r in method_results]),
                'mse_std': np.std([r['mse'] for r in method_results])
            }
            if 'time' in method_results[0]:
                summary[method]['time_mean'] = np.mean([r['time'] for r in method_results])

        return summary

    def print_summary(self):
        """Вывод сводной статистики"""
        summary = self.get_summary()
        if not summary:
            print("Нет данных для статистики")
            return

        print("\n" + "=" * 60)
        print("СВОДНАЯ СТАТИСТИКА МЕТРИК")
        print("=" * 60)

        for method, stats in summary.items():
            print(f"\n{method}:")
            print(f"  SSIM:  {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}")
            print(f"  PSNR:  {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} дБ")
            print(f"  MSE:   {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
            if 'time_mean' in stats:
                print(f"  Time:  {stats['time_mean']:.4f} с")

    def get_comparison_table(self):
        """Получение таблицы сравнения"""
        if len(self.history) < 2:
            return None

        methods = list(set(r['method'] for r in self.history))
        if len(methods) != 2:
            return None

        method1, method2 = methods
        results1 = [r for r in self.history if r['method'] == method1]
        results2 = [r for r in self.history if r['method'] == method2]

        n = min(len(results1), len(results2))
        if n == 0:
            return None

        improvements = {
            'ssim': np.mean([r2['ssim'] - r1['ssim']
                             for r1, r2 in zip(results1[:n], results2[:n])]),
            'psnr': np.mean([r2['psnr'] - r1['psnr']
                             for r1, r2 in zip(results1[:n], results2[:n])]),
            'mse': np.mean([r1['mse'] - r2['mse']
                            for r1, r2 in zip(results1[:n], results2[:n])])
        }

        return {
            'method1': method1,
            'method2': method2,
            'improvements': improvements
        }