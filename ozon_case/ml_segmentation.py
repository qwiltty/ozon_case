
#модуль инстанс-сегментации для разрешения нештатных ситуаций (Edge Cases):
#разделение слипшихся посылок и отсечение складок мягких пакетов

import numpy as np
import cv2
from scipy.ndimage import sobel
from config import SystemConfig


class InstanceSegmentationModule:
    def __init__(self, weights_path: str | None = None):
        self.model = None
        if weights_path:
            try:
                from ultralytics import YOLO
                self.model = YOLO(weights_path)
                print(f"[ML] YOLOv8-Seg модель успешно загружена из {weights_path}")
            except Exception as e:
                print(f"[ML] Предупреждение: Не удалось загрузить YOLO веса ({e}). Включен fallback-режим.")

    def separate_touching_instances(self, depth_map: np.ndarray, binary_mask: np.ndarray,
                                    cfg: SystemConfig = SystemConfig()) -> list[np.ndarray]:

        #если загружена реальная модель YOLOv8-Seg
        if self.model is not None:
            # Формируем 3-канальный тензор [Высота Z, Градиент dX, Градиент dY]
            grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

            norm_z = np.clip((depth_map / 300.0) * 255, 0, 255).astype(np.uint8)
            norm_gx = np.clip(np.abs(grad_x) * 10, 0, 255).astype(np.uint8)
            norm_gy = np.clip(np.abs(grad_y) * 10, 0, 255).astype(np.uint8)

            input_tensor = np.stack([norm_z, norm_gx, norm_gy], axis=-1)
            results = self.model.predict(input_tensor, verbose=False)

            masks = []
            for r in results:
                if r.masks is not None:
                    for m in r.masks.data.cpu().numpy():
                        resized_m = cv2.resize(m, (depth_map.shape[1], depth_map.shape[0]))
                        masks.append((resized_m > 0.5).astype(np.uint8) * 255)
            if masks:
                return masks

        #1.Вычисляем градиент высоты для обнаружения вертикального шва между соприкасающимися коробками
        dz_dx = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        #границы коробок имеют высокий градиент
        border_mask = (grad_mag > 15.0).astype(np.uint8) * 255

        #2.Очищаем маску от линии шва
        eroded_mask = cv2.bitwise_and(binary_mask, cv2.bitwise_not(border_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_mask = cv2.morphologyEx(eroded_mask, cv2.MORPH_OPEN, kernel)

        #3.Поиск раздельных компонент
        num_labels, labels = cv2.connectedComponents(eroded_mask)

        isolated_masks = []
        for label_id in range(1, num_labels):
            comp_mask = (labels == label_id).astype(np.uint8) * 255
            area_px = cv2.countNonZero(comp_mask)
            area_mm2 = area_px * (cfg.DX_MM * cfg.DY_MM)

            #отсекаем мусор
            if area_mm2 >= cfg.MIN_CONTOUR_AREA_MM2:
                #восстанавливаем геометрию до границ объекта
                restored_mask = cv2.dilate(comp_mask, kernel, iterations=2)
                restored_mask = cv2.bitwise_and(restored_mask, binary_mask)
                isolated_masks.append(restored_mask)

        #если разделить не удалось, возвращаем исходную общую маску
        return isolated_masks if isolated_masks else [binary_mask]