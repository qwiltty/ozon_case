def check_tolerance(measured: float, ground_truth: float) -> tuple[bool, float, float]:
    #проверяет, укладывается ли измеренный размер
    #возвращает: (успех, абсолютная ошибка, предельно допустимая ошибка)

    error = abs(measured - ground_truth)
    max_allowed_error = max(0.05 * ground_truth, 5.0)
    is_compliant = error <= max_allowed_error
    return is_compliant, error, max_allowed_error