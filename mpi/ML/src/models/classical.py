"""Классические алгоритмы реконструкции MPI: Tikhonov и Kaczmarz (ART)."""

import numpy as np


class TikhonovReconstructor:
    """Tikhonov-регуляризация для недоопределённой системы (M < N).

    Решает задачу:
        x* = argmin_x  ||A·x − b||² + μ·||x||²
    что эквивалентно нормальной системе:
        (Aᴴ·A + μ·I)·x = Aᴴ·b

    Полезно как «нижняя планка» для сравнения с нейросетевыми методами:
    линейный, не требует обучения, понятный физический смысл.

    Параметр μ управляет балансом «соответствие данным ↔ гладкость
    решения»; рекомендуемые значения 10⁻⁴…10⁻¹ для MPI-задач.
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
            self.A_real = self.A
        self.M, self.N = self.A_real.shape
        self.use_random_order = use_random_order
        # Предвычисляем квадраты норм строк
        self.row_norms_sq = np.einsum('ij,ij->i', self.A_real, self.A_real)
        self.row_norms_sq = np.clip(self.row_norms_sq, 1e-30, None)

    def reconstruct(self, measurement, n_iterations=5, relaxation=0.5,
                    enforce_nonneg=True, damp_schedule=True):
        """Восстановление с защитой от шума (early stopping + затухание).

        Kaczmarz без регуляризации расходится на зашумлённых данных —
        каждый проход «выпиливает» шум обратно в решение. Защита:
          • короткий по умолчанию `n_iterations=5` — ранняя остановка;
          • `relaxation < 1` (под-релаксация) гасит вклад каждой строки;
          • `damp_schedule=True` — λ_k = relaxation / √k уменьшает шаг с
            итерациями (Polyak-стиль);
          • `enforce_nonneg=True` — после каждого прохода проектируем
            x ≥ 0, что физически соответствует концентрации МНЧ и
            подавляет осциллирующий шум.

        Args:
            measurement: (M,) или (2, M/2).
            n_iterations: число полных проходов по строкам.
            relaxation:   базовая λ (0.5 — устойчиво на SNR ≥ 20 дБ).
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
            return np.concatenate([m.real, m.imag])
        return m


__all__ = ['TikhonovReconstructor', 'KatsMarcAlgorithm']
