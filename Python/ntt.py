"""This file contains an implementation of the NTT.

The NTT implemented here is for polynomials in Z_q[x]/(phi), with:
- The integer modulus q = 12289 or 7681
- The polynomial modulus phi = x ** n + 1, with n a power of two, n =< 1024

The code is voluntarily very similar to the code of the FFT.
It is probably possible to use templating to merge both implementations.
"""


# Import split and merged
from common import split, merge

# Import NTT constants for q = 12289
from const_12289 import roots_dict_Zq as roots_dict_Zq_12289
from const_12289 import inv_mod_q as inv_mod_q_12289
from const_12289 import i2 as i2_12289
from const_12289 import sqr1 as sqr1_12289

# Import NTT constants for q = 7681
from const_7681 import roots_dict_Zq as roots_dict_Zq_7681
from const_7681 import inv_mod_q as inv_mod_q_7681
from const_7681 import i2 as i2_7681
from const_7681 import sqr1 as sqr1_7681

# Map modulus q to constants
roots_dict_Zq = {7681: roots_dict_Zq_7681, 12289: roots_dict_Zq_12289}
inv_mod_q = {7681: inv_mod_q_7681, 12289: inv_mod_q_12289}
i2 = {7681: i2_7681, 12289: i2_12289}
sqr1 = {7681: sqr1_7681, 12289: sqr1_12289}


def split_ntt(f_ntt, q):
    """Split a polynomial f in two or three polynomials.

    Args:
        f_ntt: a polynomial

    Format: NTT
    """
    n = len(f_ntt)
    w = roots_dict_Zq[q][n]
    f0_ntt = [0] * (n // 2)
    f1_ntt = [0] * (n // 2)
    for i in range(n // 2):
        f0_ntt[i] = (i2[q] * (f_ntt[2 * i] + f_ntt[2 * i + 1])) % q
        f1_ntt[i] = (i2[q] * (f_ntt[2 * i] - f_ntt[2 * i + 1]) * inv_mod_q[q][w[2 * i]]) % q
    return [f0_ntt, f1_ntt]


def merge_ntt(f_list_ntt, q):
    """Merge two or three polynomials into a single polynomial f.

    Args:
        f_list_ntt: a list of polynomials

    Format: NTT
    """
    f0_ntt, f1_ntt = f_list_ntt
    n = 2 * len(f0_ntt)
    w = roots_dict_Zq[q][n]
    f_ntt = [0] * n
    for i in range(n // 2):
        f_ntt[2 * i + 0] = (f0_ntt[i] + w[2 * i] * f1_ntt[i]) % q
        f_ntt[2 * i + 1] = (f0_ntt[i] - w[2 * i] * f1_ntt[i]) % q
    return f_ntt


def ntt(f, q):
    """Compute the NTT of a polynomial.

    Args:
        f: a polynomial

    Format: input as coefficients, output as NTT
    """
    n = len(f)
    if (n > 2):
        f0, f1 = split(f)
        f0_ntt = ntt(f0, q)
        f1_ntt = ntt(f1, q)
        f_ntt = merge_ntt([f0_ntt, f1_ntt], q)
    elif (n == 2):
        f_ntt = [0] * n
        f_ntt[0] = (f[0] + sqr1[q] * f[1]) % q
        f_ntt[1] = (f[0] - sqr1[q] * f[1]) % q
    return f_ntt


def intt(f_ntt, q):
    """Compute the inverse NTT of a polynomial.

    Args:
        f_ntt: a NTT of a polynomial

    Format: input as NTT, output as coefficients
    """
    n = len(f_ntt)
    if (n > 2):
        f0_ntt, f1_ntt = split_ntt(f_ntt, q)
        f0 = intt(f0_ntt, q)
        f1 = intt(f1_ntt, q)
        f = merge([f0, f1])
    elif (n == 2):
        f = [0] * n
        f[0] = (i2[q] * (f_ntt[0] + f_ntt[1])) % q
        f[1] = (i2[q] * inv_mod_q[q][sqr1[q]] * (f_ntt[0] - f_ntt[1])) % q
    return f


def add_zq(f, g, q):
    """Addition of two polynomials (coefficient representation)."""
    assert len(f) == len(g)
    deg = len(f)
    return [(f[i] + g[i]) % q for i in range(deg)]


def neg_zq(f, q):
    """Negation of a polynomials (any representation)."""
    deg = len(f)
    return [(- f[i]) % q for i in range(deg)]


def sub_zq(f, g, q):
    """Substraction of two polynomials (any representation)."""
    return add_zq(f, neg_zq(g, q), q)


def mul_zq(f, g, q):
    """Multiplication of two polynomials (coefficient representation)."""
    return intt(mul_ntt(ntt(f, q), ntt(g, q), q), q)


def div_zq(f, g, q):
    """Division of two polynomials (coefficient representation)."""
    try:
        return intt(div_ntt(ntt(f, q), ntt(g, q), q), q)
    except ZeroDivisionError:
        raise


def add_ntt(f_ntt, g_ntt, q):
    """Addition of two polynomials (NTT representation)."""
    return add_zq(f_ntt, g_ntt, q)


def sub_ntt(f_ntt, g_ntt, q):
    """Substraction of two polynomials (NTT representation)."""
    return sub_zq(f_ntt, g_ntt, q)


def mul_ntt(f_ntt, g_ntt, q):
    """Multiplication of two polynomials (coefficient representation)."""
    assert len(f_ntt) == len(g_ntt)
    deg = len(f_ntt)
    return [(f_ntt[i] * g_ntt[i]) % q for i in range(deg)]


def div_ntt(f_ntt, g_ntt, q):
    """Division of two polynomials (NTT representation)."""
    assert len(f_ntt) == len(g_ntt)
    deg = len(f_ntt)
    if any(elt == 0 for elt in g_ntt):
        raise ZeroDivisionError
    return [(f_ntt[i] * inv_mod_q[q][g_ntt[i]]) % q for i in range(deg)]


"""This value is the ratio between:
    - The degree n
    - The number of complex coefficients of the NTT
While here this ratio is 1, it is possible to develop a short NTT such that it is 2.
"""
ntt_ratio = 1

