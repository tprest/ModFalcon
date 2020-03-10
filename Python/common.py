"""
This file contains methods and objects reused through multiple files.
"""

"""q = 7681 is the modulus used in Kyber."""
q_7681 = 15 * 512 + 1

"""q = 12289 is the modulus used in Falcon."""
q_12289 = 12 * 1024 + 1


def split(f):
    """
    Split a polynomial f in two polynomials.

    Input:
    f           A polynomial

    Output:
    f0, f1      Two polynomials

    Format:     Coefficient
    """
    n = len(f)
    f0 = [f[2 * i + 0] for i in range(n // 2)]
    f1 = [f[2 * i + 1] for i in range(n // 2)]
    return [f0, f1]


def merge(f_list):
    """
    Merge two polynomials into a single polynomial f.

    Input:
    [f0, f1]    A list of two polynomials

    Output:
    f           A polynomial

    Format:     Coefficient
    """
    f0, f1 = f_list
    n = 2 * len(f0)
    f = [0] * n
    for i in range(n // 2):
        f[2 * i + 0] = f0[i]
        f[2 * i + 1] = f1[i]
    return f


def sqnorm(v):
    """
    Compute the squared euclidean norm of the vector v.

    Input:
    v           A vector of polynomials

    Output:
    res         The squared euclidean norm of v

    Format:     Coefficient
    """
    res = 0
    for elt in v:
        for coef in elt:
            res += coef ** 2
    return res
