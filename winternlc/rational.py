"""
Rational functions for use in the non-linear corrections.
"""


def rational_func(
    x,
    a0: float,
    a1: float,
    a2: float,
    a3: float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
):
    """
    A rational function with 4th order polynomials in the numerator and denominator.

    :param x: Input value
    :param a0: Coefficient for the constant term in the numerator
    :param a1: Coefficient for the linear term in the numerator
    :param a2: Coefficient for the quadratic term in the numerator
    :param a3: Coefficient for the cubic term in the numerator
    :param b0: Coefficient for the constant term in the denominator
    :param b1: Coefficient for the linear term in the denominator
    :param b2: Coefficient for the quadratic term in the denominator
    :param b3: Coefficient for the cubic term in the denominator

    :return: Rational function value
    """
    return (a0 + a1 * x + a2 * x**2 + a3 * x**3) / (
        1 + b0 * x + b1 * x**2 + b2 * x**3 + b3 * x**4
    )
