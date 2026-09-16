import numpy as np
from config import SystemConfig

def generate_synthetic_scan(
        length_mm: float,
        width_mm: float,
        height_mm: float,
        angle_deg: float,
        cfg: SystemConfig = SystemConfig(),
        noise_sigma_mm: float = 0.5,
        dropout_rate: float = 0.02
) -> tuple[np.ndarray, dict]:

    #генерирует матрицу глубины (1200 x 1920) с движущейся коробкой
    #возвращает матрицу карты высот и Ground Truth параметры

    n_rows = int(cfg.SCAN_WINDOW_MM / cfg.DY_MM) #1200 срезов
    n_cols = cfg.X_POINTS #1920 точек

    #задаем сетку координат конвейера (X поперек, Y вдоль)
    y_coords = np.arange(n_rows) * cfg.DY_MM
    x_coords = np.arange(n_cols) * cfg.DX_MM - (cfg.FOV_BELT_MM - cfg.BELT_WIDTH_MM) / 2

    X, Y = np.meshgrid(x_coords, y_coords)

    #базовая плоскость конвейера с нормальным шумом вибраций
    depth_map = np.random.normal(0, noise_sigma_mm, (n_rows, n_cols))

    #преобразование координат в систему объекта
    center_x = cfg.BELT_WIDTH_MM / 2.0
    center_y = cfg.SCAN_WINDOW_MM / 2.0
    rad = np.radians(-angle_deg)

    Xr = (X - center_x) * np.cos(rad) - (Y - center_y) * np.sin(rad)
    Yr = (X - center_x) * np.sin(rad) + (Y - center_y) * np.cos(rad)

    #маска коробки
    box_mask = (np.abs(Xr) <= length_mm / 2.0) & (np.abs(Yr) <= width_mm / 2.0)
    depth_map[box_mask] += height_mm

    #имитация оптической тени триангуляции (сзади по X)[cite: 3]
    shadow_width_mm = height_mm * (1.0 / np.tan(np.radians(cfg.TRIANGULATION_ANGLE_DEG)))
    #зона за объектом затеняется
    shadow_mask = (Xr > length_mm / 2.0) & (Xr <= length_mm / 2.0 + shadow_width_mm * 0.1) & (
                np.abs(Yr) <= width_mm / 2.0)
    depth_map[shadow_mask] = np.nan

    #моделирование бликов
    dropouts = (np.random.rand(n_rows, n_cols) < dropout_rate) & box_mask
    depth_map[dropouts] = np.nan

    ground_truth = {
        "length_mm": float(max(length_mm, width_mm)),
        "width_mm": float(min(length_mm, width_mm)),
        "height_mm": float(height_mm),
        "angle_deg": float(angle_deg)
    }

    return depth_map, ground_truth


def generate_touching_boxes_scan(
        box1_dims: tuple[float, float, float] = (150.0, 120.0, 80.0),
        box2_dims: tuple[float, float, float] = (140.0, 100.0, 60.0),
        cfg: SystemConfig = SystemConfig(),
        noise_sigma_mm: float = 0.5
) -> tuple[np.ndarray, list[dict]]:

    #Генерирует сложный краевой сценарий: две коробки, лежащие вплотную друг к другу (Double Pick).

    n_rows = int(cfg.SCAN_WINDOW_MM / cfg.DY_MM)
    n_cols = cfg.X_POINTS

    y_coords = np.arange(n_rows) * cfg.DY_MM
    x_coords = np.arange(n_cols) * cfg.DX_MM - (cfg.FOV_BELT_MM - cfg.BELT_WIDTH_MM) / 2.0
    X, Y = np.meshgrid(x_coords, y_coords)

    depth_map = np.random.normal(0, noise_sigma_mm, (n_rows, n_cols))

    center_x = cfg.BELT_WIDTH_MM / 2.0
    center_y = cfg.SCAN_WINDOW_MM / 2.0

    l1, w1, h1 = box1_dims
    l2, w2, h2 = box2_dims

    #коробка 1 смещена чуть левее
    c1_x = center_x - l1 / 2.0
    c1_y = center_y
    mask1 = (np.abs(X - c1_x) <= l1 / 2.0) & (np.abs(Y - c1_y) <= w1 / 2.0)
    depth_map[mask1] += h1

    #коробка 2 пристыкована вплотную к коробке 1 по оси X (зазор 0 мм)
    c2_x = c1_x + (l1 / 2.0) + (l2 / 2.0)
    c2_y = center_y
    mask2 = (np.abs(X - c2_x) <= l2 / 2.0) & (np.abs(Y - c2_y) <= w2 / 2.0)
    depth_map[mask2] += h2

    gt_boxes = [
        {"length_mm": max(l1, w1), "width_mm": min(l1, w1), "height_mm": h1},
        {"length_mm": max(l2, w2), "width_mm": min(l2, w2), "height_mm": h2}
    ]

    return depth_map, gt_boxes