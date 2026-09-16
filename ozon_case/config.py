from dataclasses import dataclass
@dataclass(frozen=True)
class SystemConfig:
    #параметры конвейера и зоны сканирования
    BELT_WIDTH_MM: float = 600.0 #ширина рабочей зоны ленты
    SCAN_WINDOW_MM: float = 600.0 #длина окна захвата одного груза
    BELT_SPEED_MM_S: float = 1000.0 #скорость движения ленты

    #разрешение дискретизации
    DY_MM: float = 0.5 #шаг сканирования по оси Y (энкодер с делителем)
    X_POINTS: int = 1920 #число отсчётов профилометра по ширине

    #метрические параметры по X на уровне ленты (FOV_belt = 918 мм)
    FOV_BELT_MM: float = 918.0
    DX_MM: float = 918.0 / 1920 #шаг сетки по оси X

    #пороги фильтрации и сегментации
    Z_THRESHOLD_MM: float = 3.0 #порог бинаризации над опорной плоскостью ленты
    MIN_CONTOUR_AREA_MM2: float = 50.0 #отсечение шума и мусора
    Z_PERCENTILE: float = 99.5 #процентиль для оценки высоты

    #систематические поправки
    GRID_OFFSET_MM: float = 0.25 #компенсация дискретизации сетки
    TRIANGULATION_ANGLE_DEG: float = 30.0 #угол оптической триангуляции