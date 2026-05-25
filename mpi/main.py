import h5py
import matplotlib.pyplot as plt
import numpy as np

from ReconstructionMethod import ReconstructionMethod as RM  # Класс с методами реконструкции


def main():
    filenameSM = r".\DATA\SystemMatrix.h5"  # Файл с системной матрицей
    filenameMeas = r'.\DATA\MeasurementData_B.h5'  # Файл с измерительными данными

    # ==================== ЗАГРУЗКА СИСТЕМНОЙ МАТРИЦЫ ====================
    fSM = h5py.File(filenameSM, 'r')

    S_data_r = fSM['/measurement/data/r'][:]  # Действительная часть
    S_data_i = fSM['/measurement/data/i'][:]  # Мнимая часть

    # Формирование комплексной системной матрицы
    S = S_data_r + 1j * S_data_i

    # Получение массива фоновых кадров (background frames)
    isBG = fSM['/measurement/isBackgroundFrame'][:].squeeze()

    # Исключение фоновых кадров, оставляем только полезные измерения (в исходных данных фоновые измерения отсутствуют)
    S = S[:, :, isBG == 0]

    # Преобразование 3D массива в 2D матрицу (reshaping) для использования в алгоритмах реконструкции
    # Объединение первых двух измерений в одну размерность
    SM = S.reshape(S.shape[0] * S.shape[1], S.shape[2])

    # ==================== ЗАГРУЗКА ИЗМЕРИТЕЛЬНЫХ ДАННЫХ ====================
    fMeas = h5py.File(filenameMeas, 'r')

    u_data_r = fMeas['/measurement/data/r'][:]  # Действительная часть
    u_data_i = fMeas['/measurement/data/i'][:]  # Мнимая часть

    u_data = u_data_r + 1j * u_data_i

    # Объединение измерений для двух кадров в один вектор
    # Meas = [кадр1_гармоника1, кадр1_гармоника2, ..., кадр2_гармоника1, ...]
    Meas = np.concatenate([u_data[0, :], u_data[1, :]])

    # ==================== ПОЛУЧЕНИЕ РАЗМЕРОВ СЕТКИ ====================
    # Извлечение количества позиций из калибровочных данных
    number_Position = fSM['/calibration/size'][:].squeeze()

    # ==================== РЕШЕНИЕ ОБРАТНОЙ ЗАДАЧИ ====================
    kmax = 100  # Максимальное количество итераций для итеративных методов
    mu = 0.001  # Параметр регуляризации

    # Применение метода Тихонова для недоопределенной системы (количество неизвестных больше количества уравнений)
    # Количество неизвестных (вокселей) = 19 * 19 = 361 (только x и y координаты)
    # Количество уравнений = 93 (количество гармоник)
    recImage_Tik_x = RM.Tikhonov_underdetermined(SM, Meas, mu, kmax)

    # Преобразование результата: берем действительную часть и изменяем форму для визуализации
    recImage_Tik_x = recImage_Tik_x.real.reshape([int(number_Position[0]), int(number_Position[1])])

    # ==================== ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
    plt.figure()  # Создание нового окна для графика
    plt.subplot(1, 1, 1)  # Создание подграфика (1x1 сетка, текущий номер 1)
    plt.imshow(recImage_Tik_x)  # Отображение восстановленного изображения
    plt.colorbar()  # Добавление цветовой шкалы для оценки значений
    plt.title("Reconstruction with x coil and Tik")  # Заголовок графика
    plt.show()  # Отображение графика на экране


if __name__ == '__main__':
    main()