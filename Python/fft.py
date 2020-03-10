"""
This file contains an implementation of the FFT.

The FFT implemented here is for polynomials in R[x]/(phi), with:
- The polynomial modulus phi = x ** n + 1, with d a power of two, n =< 1024

The code is voluntarily very similar to the code of the NTT.
It is probably possible to use templating to merge both implementations.
"""

from common import split, merge         # Import split and merge
from fft_constants import roots_dict    # Import constants useful for the FFT


def split_fft(f_fft):
    """
    Split a polynomial f_fft in two polynomials.

    Input:
    f_fft           A polynomial

    Output:
    f0_fft, f1_fft  Two polynomials

    Format:         FFT

    Similar to algorithm splitfft of Falcon's documentation.
    """
    n = len(f_fft)
    w = roots_dict[n]
    f0_fft = [0] * (n // 2)
    f1_fft = [0] * (n // 2)
    for i in range(n // 2):
        f0_fft[i] = 0.5 * (f_fft[2 * i] + f_fft[2 * i + 1])
        f1_fft[i] = 0.5 * (f_fft[2 * i] - f_fft[2 * i + 1]) * w[2 * i].conjugate()
    return [f0_fft, f1_fft]


def merge_fft(f_list_fft):
    """Merge two or three polynomials into a single polynomial f.

    Input:
    [f0, f1]    A list of two polynomials

    Output:
    f           A polynomial

    Format:     FFT

    Corresponds to algorithm mergefft of Falcon's documentation.
    """
    f0_fft, f1_fft = f_list_fft
    n = 2 * len(f0_fft)
    w = roots_dict[n]
    f_fft = [0] * n
    for i in range(n // 2):
        f_fft[2 * i + 0] = f0_fft[i] + w[2 * i] * f1_fft[i]
        f_fft[2 * i + 1] = f0_fft[i] - w[2 * i] * f1_fft[i]
    return f_fft


def fft(f):
    """
    Compute the FFT of a polynomial mod (x ** n + 1).

    Input:
    f           A polynomial

    Output:
    f_fft       The FFT of f

    Format:     Coefficient (Input)
                FFT (Output)
    """
    n = len(f)
    if (n > 2):
        f0, f1 = split(f)
        f0_fft = fft(f0)
        f1_fft = fft(f1)
        f_fft = merge_fft([f0_fft, f1_fft])
    elif (n == 2):
        f_fft = [0] * n
        f_fft[0] = f[0] + 1j * f[1]
        f_fft[1] = f[0] - 1j * f[1]
    return f_fft


def ifft(f_fft):
    """
    Compute the inverse FFT of a polynomial mod (x ** n + 1).

    Input:
    f_fft       The FFT of a polynomial f

    Output:
    f           A polynomial

    Format:     FFT (Input)
                Coefficient (Output)
    """
    n = len(f_fft)
    if (n > 2):
        f0_fft, f1_fft = split_fft(f_fft)
        f0 = ifft(f0_fft)
        f1 = ifft(f1_fft)
        f = merge([f0, f1])
    elif (n == 2):
        f = [0] * n
        f[0] = f_fft[0].real
        f[1] = f_fft[0].imag
    return f


def add(f, g):
    """Addition of two polynomials (coefficient representation)."""
    assert len(f) == len(g)
    deg = len(f)
    return [f[i] + g[i] for i in range(deg)]


def neg(f):
    """Negation of a polynomials (any representation)."""
    deg = len(f)
    return [- f[i] for i in range(deg)]


def sub(f, g):
    """Substraction of two polynomials (any representation)."""
    return add(f, neg(g))


def mul(f, g):
    """Multiplication of two polynomials (coefficient representation)."""
    return ifft(mul_fft(fft(f), fft(g)))


def div(f, g):
    """Division of two polynomials (coefficient representation)."""
    return ifft(div_fft(fft(f), fft(g)))


def adj(f):
    """Ajoint of a polynomial (coefficient representation)."""
    return ifft(adj_fft(fft(f)))


def add_fft(f_fft, g_fft):
    """Addition of two polynomials (FFT representation)."""
    return add(f_fft, g_fft)


def sub_fft(f_fft, g_fft):
    """Substraction of two polynomials (FFT representation)."""
    return sub(f_fft, g_fft)


def mul_fft(f_fft, g_fft):
    """Multiplication of two polynomials (FFT representation)."""
    deg = len(f_fft)
    return [f_fft[i] * g_fft[i] for i in range(deg)]


def div_fft(f_fft, g_fft):
    """Division of two polynomials (FFT representation)."""
    assert len(f_fft) == len(g_fft)
    deg = len(f_fft)
    return [f_fft[i] / g_fft[i] for i in range(deg)]


def adj_fft(f_fft):
    """Ajoint of a polynomial (FFT representation)."""
    deg = len(f_fft)
    return [f_fft[i].conjugate() for i in range(deg)]


"""This value is the ratio between:
    - The degree n
    - The number of complex coefficients of the FFT
While here this ratio is 1, it is possible to develop a short FFT such that it is 2.
"""
fft_ratio = 1
