import time
import numpy as np
import cv2
from scipy.ndimage import median_filter
from pydantic import BaseModel
from config import SystemConfig

class DimensionResult(BaseModel):
    length_mm: float
    width_mm: float
    height_mm: float
    volume_cm3: float
    orientation_deg: float
    confidence: float
    processing_time_ms: float
    status: str


def process_depth_map(depth_map: np.ndarray, cfg: SystemConfig = SystemConfig()) -> DimensionResult:
    start_time = time.perf_counter()

    #предобработка и фильтрация
    #заполняем пропуски NaN нулями для фильтрации
    valid_mask = ~np.isnan(depth_map)
    clean_depth = np.where(valid_mask, depth_map, 0.0)

    # \медианный фильтр 3x3 для устранения импульсных всплесков
    filtered_depth = cv2.medianBlur(clean_depth.astype(np.float32), 3)

    #сегментация объекта
    #пороговая бинаризация (Z > 3.0 мм)
    binary_mask = (filtered_depth >= cfg.Z_THRESHOLD_MM).astype(np.uint8) * 255

    #морфология: замыкание 5x5 и размыкание 3x3
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel_open)

    #выделение контуров
    contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        elapsed = (time.perf_counter() - start_time) * 1000
        return DimensionResult(
            length_mm=0, width_mm=0, height_mm=0, volume_cm3=0,
            orientation_deg=0, confidence=0.0, processing_time_ms=elapsed, status="NO_OBJECT"
        )

    #выбираем крупнейшую связную компоненту
    main_contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(main_contour)
    area_mm2 = area_px * (cfg.DX_MM * cfg.DY_MM)

    if area_mm2 < cfg.MIN_CONTOUR_AREA_MM2:
        elapsed = (time.perf_counter() - start_time) * 1000
        return DimensionResult(
            length_mm=0, width_mm=0, height_mm=0, volume_cm3=0,
            orientation_deg=0, confidence=0.0, processing_time_ms=elapsed, status="NOISE_DISCARDED"
        )

    #вычисление высоты H
    obj_pixels_z = filtered_depth[morph_mask > 0]
    height_measured = float(np.percentile(obj_pixels_z, cfg.Z_PERCENTILE))

    #расчет OBB через вращающиеся калиперы
    #преобразуем точки контура из пикселей в метрические миллиметры
    contour_points = main_contour.squeeze().astype(np.float32)
    contour_points_mm = np.empty_like(contour_points)
    contour_points_mm[:, 0] = contour_points[:, 0] * cfg.DX_MM #ось X
    contour_points_mm[:, 1] = contour_points[:, 1] * cfg.DY_MM #ось Y

    #выпуклая оболочка в метрических координатах
    hull_mm = cv2.convexHull(contour_points_mm)

    #минимальный ориентированный прямоугольник
    rect = cv2.minAreaRect(hull_mm)
    (cx, cy), (dim1, dim2), angle = rect

    #определение длины и ширины с внесением систематической поправки шага сетки[
    dim_max = max(dim1, dim2) + cfg.GRID_OFFSET_MM
    dim_min = min(dim1, dim2) + cfg.GRID_OFFSET_MM

    #нормализация угла ориентации
    orientation = angle if dim1 >= dim2 else angle + 90.0
    orientation = orientation % 180.0

    #оценка достоверности
    valid_points_ratio = float(np.sum(valid_mask & (morph_mask > 0)) / np.sum(morph_mask > 0))
    confidence = min(0.99, round(valid_points_ratio, 3))

    #объем описанного параллелепипеда
    volume_cm3 = (dim_max * dim_min * height_measured) / 1000.0

    elapsed = (time.perf_counter() - start_time) * 1000

    return DimensionResult(
        length_mm=round(dim_max, 1),
        width_mm=round(dim_min, 1),
        height_mm=round(height_measured, 1),
        volume_cm3=round(volume_cm3, 1),
        orientation_deg=round(orientation, 1),
        confidence=confidence,
        processing_time_ms=round(elapsed, 2),
        status="OK"
    )