import numpy as np
import scipy as sp
import numpy.matlib as npm
import numpy.linalg as npl
from Nargout import Nargout

class ReconstructionMethod(object):
    @staticmethod
    def Tikhonov_underdetermined(sysMatrix, measImage, alfa, iters):
        """
        Метод Тихонова для решения НЕДООПРЕДЕЛЕННЫХ систем (M < N).

        Решает уравнение: sysMatrix * x = measImage
        где неизвестных (N) больше, чем уравнений (M).

        Параметры:
        -----------
        sysMatrix : numpy.ndarray (M x N)
            Системная матрица (прямая задача)
        measImage : numpy.ndarray (M,)
            Вектор измерений (правая часть)
        alfa : float
            Параметр регуляризации Тихонова (штрафует большие решения)
        iters : int
            Количество итераций для улучшения решения (после начальной оценки)

        Возвращает:
        -----------
        xk : numpy.ndarray (N,)
            Восстановленное изображение (вектор концентраций частиц)
        error : list
            История значений ошибки (среднеквадратичное отклонение)

        Источник:
        ---------
        Yang W, Peng L. Image reconstruction algorithms for electrical capacitance tomography[J].
        """
        (m, n) = sysMatrix.shape
        vectorMeasImage = measImage

        # Единичная матрица для регуляризации (размер N x N)
        eyeMatrix = np.eye(n, n)

        # Псевдообращение Тихонова: (A^H * A + α*I)^(-1) * A^H
        # A^H - сопряженное транспонирование (эрмитово сопряжение)
        sys = np.dot(
            np.linalg.inv(np.dot(np.conj(sysMatrix.T), sysMatrix) + alfa * eyeMatrix),
            np.conj(sysMatrix.T)
        )

        xk0 = np.dot(sys, vectorMeasImage)  # начальное решение методом Тихонова
        xk0.imag = 0  # отбрасываем мнимую часть (физический смысл только у real)
        xk0[xk0 < 0] = 0  # неотрицательность концентрации (физическое ограничение)
        xk = xk0

        # Итеративное уточнение (градиентный спуск)
        # Формула: x_{k+1} = x_k - S * (A * x_k - b)
        for i in range(iters):
            xk = xk - np.dot(sys, (np.dot(sysMatrix, xk) - vectorMeasImage))
            xk.imag = 0  # обрезаем мнимую часть на каждом шаге
            xk[xk < 0] = 0  # применяем неотрицательное ограничение

        # Среднеквадратичная ошибка: ||A*x - b||^2 / N
        error = []
        error.append(np.sum((sysMatrix.dot(xk) - vectorMeasImage) ** 2) / np.size(xk))

        return Nargout(xk, error)
