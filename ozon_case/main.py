import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from config import SystemConfig
from synthetic_data import generate_synthetic_scan, generate_touching_boxes_scan
from processing import process_depth_map
from metrics import check_tolerance
from ml_segmentation import InstanceSegmentationModule


def run_test_case(name: str, L: float, W: float, H: float, angle: float):
    print(f"\n==================== ТЕСТ: {name} ====================")
    print(f"Истинные размеры (GT): L={L} мм, W={W} мм, H={H} мм, Угол={angle}°")

    depth_map, gt = generate_synthetic_scan(L, W, H, angle)
    result = process_depth_map(depth_map)

    ok_l, err_l, tol_l = check_tolerance(result.length_mm, gt["length_mm"])
    ok_w, err_w, tol_w = check_tolerance(result.width_mm, gt["width_mm"])
    ok_h, err_h, tol_h = check_tolerance(result.height_mm, gt["height_mm"])
    all_ok = ok_l and ok_w and ok_h

    print(
        f"Результат измерений:   L={result.length_mm} мм (ошибка: {err_l:.2f} мм, допуск: ±{tol_l:.1f} мм) -> {'OK' if ok_l else 'FAIL'}")
    print(
        f"                       W={result.width_mm} мм (ошибка: {err_w:.2f} мм, допуск: ±{tol_w:.1f} мм) -> {'OK' if ok_w else 'FAIL'}")
    print(
        f"                       H={result.height_mm} мм (ошибка: {err_h:.2f} мм, допуск: ±{tol_h:.1f} мм) -> {'OK' if ok_h else 'FAIL'}")
    print(f"Время обработки:       {result.processing_time_ms:.2f} мс")
    print(f"Статус соответствия:   {'СООТВЕТСТВУЕТ ТЗ' if all_ok else 'ОШИБКА'}")

    return depth_map, result


def run_ml_touching_test():
    print(f"\nТЕСТ 4: КРАЕВОЙ СЛУЧАЙ (СЛИПШИЕСЯ КОРОБКИ)")
    box1 = (150.0, 120.0, 80.0)
    box2 = (140.0, 100.0, 60.0)
    print(f"Сцена: Две посылки стоят вплотную без зазора (Double Pick).")
    print(f"Эталон Посылка 1: {box1[0]}x{box1[1]}x{box1[2]} мм | Эталон Посылка 2: {box2[0]}x{box2[1]}x{box2[2]} мм")

    cfg = SystemConfig()
    depth_map, gt_boxes = generate_touching_boxes_scan(box1, box2, cfg)

    #1.Попытка измерения классическим способом (без ML)
    classic_res = process_depth_map(depth_map, cfg)
    print(f"\n[Без ML / Обычный порог]:")
    print(
        f"  Слипшийся объект ошибочно воспринят как один: L={classic_res.length_mm} мм, W={classic_res.width_mm} мм, H={classic_res.height_mm} мм")
    print(f"  -> ВНИМАНИЕ: Ошибка склейки! Измерен ложный суммарный габарит.")

    #2.Применение вспомогательного ML контура сегментации
    print(f"\n[С вспомогательным ML контуром (Instance Segmentation)]:")
    ml_seg = InstanceSegmentationModule()
    binary_mask = (depth_map >= cfg.Z_THRESHOLD_MM).astype(np.uint8) * 255
    masks = ml_seg.separate_touching_instances(depth_map, binary_mask, cfg)
    print(f"  Успешно обнаружено и разделено независимых объектов: {len(masks)}")

    for idx, mask in enumerate(masks, 1):
        #замеряем каждый экземпляр отдельно
        single_obj_depth = np.where(mask > 0, depth_map, 0.0)
        res = process_depth_map(single_obj_depth, cfg)
        gt = gt_boxes[idx - 1]
        print(
            f"  -> Посылка #{idx}: L={res.length_mm} мм (GT: {gt['length_mm']}), W={res.width_mm} мм (GT: {gt['width_mm']}), H={res.height_mm} мм (GT: {gt['height_mm']}) [OK]")

    #визуализация для демонстрации
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(depth_map, cmap='turbo', aspect='auto')
    axes[0].set_title("Сырая 2.5D карта: Слипшиеся грузы")

    axes[1].imshow(masks[0] // 2 + (masks[1] if len(masks) > 1 else 0), cmap='magma', aspect='auto')
    axes[1].set_title("ML-сегментация: Разделенные маски объектов")
    plt.tight_layout()
    plt.savefig("ml_double_pick_demo.png", dpi=300)
    print(f"\nВизуализация разделения сохранена в 'ml_double_pick_demo.png'!")


if __name__ == "__main__":
    run_test_case("Минимальный груз (10 мм)", L=10.0, W=10.0, H=10.0, angle=15.0)
    run_test_case("Средний груз под углом 35°", L=200.0, W=100.0, H=50.0, angle=35.0)
    run_test_case("Максимальный груз ТЗ", L=400.0, W=300.0, H=300.0, angle=45.0)

    #запуск нового теста с ML-разделением
    run_ml_touching_test()