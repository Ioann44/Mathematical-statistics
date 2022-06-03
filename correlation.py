from numpy import linalg


def correlation_ratio(x, y, nMatrix: list[list[int]]):
    """
    Корреляционное отношение Y к X

    Аргументы x, y определяют варианты случайных величин X, Y.
    Матрица n определяет частоты точек вида (x, y), где строки соответствуют y, столбцы - x
    (фактически, x здесь не используется)
    """
    nMatrix = nMatrix
    iMax = len(nMatrix)
    jMax = len(nMatrix[0])

    nAll = sum(sum(ni) for ni in nMatrix)
    yAvg = sum(y[i] * sum(nMatrix[i]) for i in range(iMax)) / nAll

    std_deviation_general = (
        sum(sum(nMatrix[i]) * (y[i] - yAvg) ** 2 for i in range(iMax)) / nAll
    ) ** 0.5

    nx = [sum(nMatrix[i][j] for i in range(iMax)) for j in range(jMax)]
    yx = [sum(y[i] * nMatrix[i][j] for i in range(iMax)) / nx[j] for j in range(jMax)]
    std_deviation_groups = (
        sum(nx[j] * (yx[j] - yAvg) ** 2 for j in range(jMax)) / nAll
    ) ** 0.5

    return std_deviation_groups / std_deviation_general


def parabolic_func_by_matrix(x, y, nMatrix: list[list[int]]):
    """
    Находит функцию регрессии вида A*x^2 + B*x + C

    Аргументы x, y определяют варианты случайных величин X, Y.
    Матрица n определяет частоты точек вида (x, y), где строки соответствуют y, столбцы - x
    """
    iMax = len(nMatrix)
    jMax = len(nMatrix[0])
    nAll = sum(sum(ni) for ni in nMatrix)

    nx = [sum(nMatrix[i][j] for i in range(iMax)) for j in range(jMax)]
    yx = [sum(y[i] * nMatrix[i][j] for i in range(iMax)) / nx[j] for j in range(jMax)]

    def f(exp):
        """Функция для более простого заполнения матрицы A"""
        return sum(nx[j] * x[j] ** exp for j in range(jMax))

    def g(exp):
        """Функция для более простого заполнения матрицы B"""
        return sum(nx[j] * yx[j] * x[j] ** exp for j in range(jMax))

    # solve system A*solution=B
    A = [
        [f(4), f(3), f(2)],
        [f(3), f(2), f(1)],
        [f(2), f(1), nAll],
    ]
    B = [g(2), g(1), g(0)]

    solution = linalg.solve(A, B)
    return f"{solution[0]}*x^2 + {solution[1]}*x + {solution[2]}"


def linear_func_by_dots(x, y):
    """
    Возвращает коэффициенты (k, b), определяющие уравнение регрессии вида y = kx + b

    Аргументы x, y определяют полученные точки (x, y), встречающиеся по одному разу
    """
    xs = sum(x)
    ys = sum(y)
    qs = sum(xi ** 2 for xi in x)
    xys = sum(a * b for a, b in zip(x, y))
    n = len(x)

    k = (n * xys - xs * ys) / (n * qs - xs ** 2)
    b = (qs * ys - xs * xys) / (n * qs - xs ** 2)
    return k, b


def linear_func_by_matrix(x, y, nMatrix: list[list[int]]):
    """
    Возвращает коэффициенты (k, b), определяющие уравнение регрессии вида y = kx + b

    Аргументы x, y определяют варианты случайных величин X, Y.
    Матрица n определяет частоты точек вида (x, y), где строки соответствуют y, столбцы - x
    """
    rows = len(nMatrix)
    cols = len(nMatrix[0])

    xs = sum(x[j] * sum(nMatrix[i][j] for i in range(rows)) for j in range(cols))
    qs = sum(x[j] ** 2 * sum(nMatrix[i][j] for i in range(rows)) for j in range(cols))
    ys = sum(y[i] * sum(nMatrix[i][j] for j in range(cols)) for i in range(rows))
    xys = sum(x[j] * y[i] * nMatrix[i][j] for i in range(rows) for j in range(cols))
    n = sum(nMatrix[i][j] for i in range(rows) for j in range(cols))

    k = (n * xys - xs * ys) / (n * qs - xs ** 2)
    b = (qs * ys - xs * xys) / (n * qs - xs ** 2)
    return k, b


def Spearman(x, y):
    '''Коэффициент ранговой корреляции Спирмена'''
    n = len(x)
    return 1 - 6 * sum((x[i] - y[i]) ** 2 for i in range(n)) / (n ** 3 - n)


def Kendall(x, y):
    '''Коэффициент ранговой корреляции Кендалла'''
    X, Y = zip(*sorted(zip(x, y)))
    n = len(X)
    R = 0
    for i in range(n):
        for j in range(i, n):
            if Y[i] < Y[j]:
                R += 1
    return 4 * R / (n * (n - 1)) - 1
