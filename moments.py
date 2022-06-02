def initial_moment(x, n, k):
    """Начальный момент порядка k"""
    return sum(xi ** k * ni for xi, ni in zip(x, n)) / sum(n)


def central_moment(x, n, k):
    """Центральный момент порядка k"""
    a = A(x, n)
    return sum((xi - a) ** k * ni for xi, ni in zip(x, n)) / sum(n)


def D(x, n):
    """
    Variance

    Дисперсия с учётом частот
    """
    a = A(x, n)
    return central_moment(x, n, 2)


def A(x, n):
    """
    Expected value

    Математическое ожидание с учётом частот
    """
    return initial_moment(x, n, 1)


def asymmetry(x, n):
    return central_moment(x, n, 3) / D(x, n) ** 1.5


def excess(x, n):
    return central_moment(x, n, 4) / D(x, n) ** 2 - 3
