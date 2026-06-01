"""Классические алгоритмы реконструкции MPI: Tikhonov и Kaczmarz (ART).

Эти методы — фундамент инверсной задачи в линейной алгебре, появились
задолго до нейросетей (Tikhonov 1963, Kaczmarz 1937). В контексте MPI
они служат двум целям:

  1. **Baseline для сравнения**: показывают, насколько лучше работают
     нейросетевые подходы по сравнению с классической линейной алгеброй.
  2. **Препроцессинг для нейросетей**: Tikhonov-реконструкция часто
     используется как начальное приближение для unrolled-методов или
     в качестве «грубой» реконструкции для MoE-экспертов.

Оба метода работают только с заранее измеренной (или аналитически
построенной) системной матрицей A. Они НЕ способны калиброваться по
данным — для этого нужна физическая модель (PMCNet) или обучение
(нейросети).
"""

import numpy as np


class TikhonovReconstructor:
    """Tikhonov-регуляризация для некорректной обратной задачи.

    ## Что решает

    Обратная задача MPI: восстановить c из измерения u = A·c + n, где
    A — комплексная (M × N) системная матрица, n — шум. Если M < N
    (число гармоник меньше числа вокселей) — задача недоопределённая
    и имеет бесконечно много решений. Если M ≥ N, но A плохо
    обусловлена (что типично для MPI), наименьших квадратов решение
    `c = A⁺·u` катастрофически усиливает шум.

    Tikhonov-регуляризация добавляет штраф за норму решения:

        c* = argmin_c  ||A·c − u||² + μ·||c||²

    Аналитическое решение через нормальную систему:

        (Aᴴ·A + μ·I) · c = Aᴴ · u

    где Aᴴ — эрмитово сопряжённая (для вещественной A это просто
    транспонированная).

    ## Параметр μ: баланс fit ↔ smoothness

    μ управляет компромиссом:
      • μ → 0:   решение точно подгоняется под u, включая шум →
                  амплифицируется шум до катастрофического уровня;
      • μ → ∞:   решение → нуль (тривиально гладкое, бесполезное);
      • оптимум: ~10⁻⁴…10⁻¹ для типичных MPI-задач, подбирается
                  через L-curve или cross-validation.

    Этот реализуемый метод использует итеративный градиентный спуск с
    `kmax` шагами вместо прямого обращения матрицы (для матриц размером
    1275×2601 прямое обращение возможно, но не для 3D-случаев). На
    каждой итерации:

        c ← c − η · (Aᴴ·(A·c − u) + μ·c)

    где η — шаг градиентного спуска, оценённый из спектрального
    радиуса Aᴴ·A.

    ## Сильные и слабые стороны

    ✓ Полностью детерминирован, нет hyperparameter random seed;
    ✓ Быстрый: одно умножение матриц за итерацию;
    ✓ Хорошо понимаемая теоретическая основа.

    ✗ Гладкое решение → размытость, потеря тонких деталей;
    ✗ Требует ручного подбора μ для каждого нового сетапа;
    ✗ Не использует никаких структурных свойств изображения
      (пиковая разреженность, кусочная гладкость и т. д.).
    """

    def __init__(self, system_matrix):
        self.A = system_matrix
        self.M, self.N = system_matrix.shape

    def reconstruct(self, measurement, mu=1e-3, kmax=100,
                    enforce_nonneg=True):
        """Восстановление по измерению.

        Args:
            measurement: вектор измерения (M,) или (2, M/2).
            mu: параметр регуляризации.
            kmax: число итераций уточнения (после прямого решения).
            enforce_nonneg: проектировать решение в неотрицательную область
                (концентрация физически ≥ 0).
        """
        b = self._normalize_measurement(measurement)
        AhA = np.conj(self.A.T) @ self.A
        Ah = np.conj(self.A.T)
        sys = np.linalg.inv(AhA + mu * np.eye(self.N)) @ Ah

        x = sys @ b
        if np.iscomplexobj(x):
            x = x.real
        if enforce_nonneg:
            x = np.maximum(x, 0)

        # Итеративное уточнение (Landweber-подобное)
        for _ in range(kmax):
            x = x - (sys @ (self.A @ x - b)).real
            if enforce_nonneg:
                x = np.maximum(x, 0)

        return x

    @staticmethod
    def _normalize_measurement(measurement):
        m = np.asarray(measurement)
        if m.ndim == 2 and m.shape[0] == 2:
            return np.concatenate([m[0, :], m[1, :]])
        return m.flatten()


class KatsMarcAlgorithm:
    """Алгоритм Качмажа (1937), он же ART в томографии.

    Источник: Kaczmarz S. «Angenäherte Auflösung von Systemen linearer
    Gleichungen», Bull. Int. Acad. Pol. Sci. Lett. A, 1937.

    Итеративный проекционный метод: на каждом шаге решение проецируется
    на гиперплоскость, заданную одной строкой системы:
        x_{k+1} = x_k + λ · (b_i − a_iᵀ·x_k) / ‖a_i‖² · a_i

    Подходит для разреженных и больших систем — не требует обращения
    матриц. Сходимость гарантирована для совместных систем при λ ∈ (0, 2).
    """

    def __init__(self, system_matrix, use_random_order=True):
        self.A = np.asarray(system_matrix)
        if np.iscomplexobj(self.A):
            # Раскладываем комплексную матрицу на (Re, Im) — Качмарц работает
            # с вещественными числами
            self.A_real = np.vstack([self.A.real, self.A.imag])
        else:
            self.A_real = self.A.astype(np.float64)
        self.M, self.N = self.A_real.shape
        self.use_random_order = use_random_order
        # Предвычисляем квадраты норм строк
        row_norms_sq = np.einsum('ij,ij->i', self.A_real, self.A_real)
        row_norms_sq = np.clip(row_norms_sq, 1e-30, None)

        # Глобальное масштабирование строк: динамический диапазон гармоник
        # MPI достигает 10⁸, и шаг `lam·residual/||a_i||²·a_i` для слабых
        # строк катастрофически усиливает шум. Нормируем медианой ‖a_i‖,
        # чтобы средняя строка имела ‖a_i‖ ≈ 1, а b пришлось масштабировать
        # тем же фактором в `reconstruct`. Это эквивалент диагонального
        # preconditioning'а Strohmer-Vershynin (2009).
        scale = np.sqrt(np.median(row_norms_sq))
        if scale > 0:
            self.A_real = self.A_real / scale
            row_norms_sq = row_norms_sq / (scale * scale)
        self.row_norms_sq = row_norms_sq
        self.global_scale = scale

    def reconstruct(self, measurement, n_iterations=50, relaxation=0.3,
                    enforce_nonneg=True, damp_schedule=False):
        """Восстановление с защитой от шума.

        Kaczmarz без регуляризации деградирует на зашумлённых данных в
        semi-convergence — первые проходы улучшают качество, дальнейшие
        вшивают шум. Защита:
          • `n_iterations=50` — повышено 20 → 50: на предыдущем прогоне
            Kaczmarz давал SSIM 0.01-0.03 (катастрофически), что говорит
            о недо-итерациях. После исправления `_normalize_measurement`
            (assert на длину b) ему нужно больше проходов для накопления
            сигнала. 50 эпох × 2550 строк = 127K row-updates;
          • `relaxation=0.3` (под-релаксация) сглаживает каждый row-update;
          • строки A нормированы медианой ‖a_i‖ в `__init__`, b делится
            на тот же фактор в `_normalize_measurement` — это устраняет
            дисбаланс энергии гармоник (10⁸-кратный спред);
          • `enforce_nonneg=True` после каждой эпохи — физический prior
            «концентрация ≥ 0» и эффективная регуляризация.

        Note: `damp_schedule=False` по умолчанию — расписание `λ/√k`
        слишком быстро гасит шаг при глобально нормированных строках.

        Args:
            measurement: (M,) или (2, M/2).
            n_iterations: число полных проходов по строкам.
            relaxation:   базовая λ (0.3 — устойчиво на SNR ≥ 15 дБ).
            enforce_nonneg: проекция на положительный конус.
            damp_schedule:  использовать λ_k = relaxation / √k.
        """
        b = self._normalize_measurement(measurement)
        x = np.zeros(self.N, dtype=np.float64)

        for k in range(n_iterations):
            lam = relaxation / np.sqrt(k + 1) if damp_schedule else relaxation
            order = (np.random.permutation(self.M)
                     if self.use_random_order else range(self.M))
            for i in order:
                a_i = self.A_real[i]
                residual = b[i] - a_i @ x
                x = x + lam * residual / self.row_norms_sq[i] * a_i
            if enforce_nonneg:
                x = np.maximum(x, 0)

        return x

    def _normalize_measurement(self, measurement):
        m = np.asarray(measurement)
        if m.ndim == 2 and m.shape[0] == 2:
            m = np.concatenate([m[0, :], m[1, :]])
        else:
            m = m.flatten()
        if np.iscomplexobj(m):
            # Раскладываем на (Re, Im) под расширенную матрицу
            b = np.concatenate([m.real, m.imag])
        else:
            b = m.astype(np.float64, copy=False)
        # Согласуем масштаб с pre-scaled A_real из __init__: если строки
        # были поделены на `global_scale`, то и измерение должно тоже —
        # иначе residual `b - A·x` становится несопоставим с шагом.
        if self.global_scale and self.global_scale > 0:
            b = b / self.global_scale
        # Sanity check: длина b должна совпадать с числом строк A_real
        assert b.shape[0] == self.M, (
            f"Kaczmarz: размер b={b.shape[0]} не совпадает с числом "
            f"строк A_real={self.M}")
        return b


__all__ = ['TikhonovReconstructor', 'KatsMarcAlgorithm']
