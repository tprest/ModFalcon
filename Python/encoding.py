"""Compress and decompress signatures"""
import sys
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.


def compress(v, rate=7):
    """
    Compress a list of polynomials into a list of integers.
    The lowest (rate) bits and the signs are encoded naively,
    the highest bits are encoded using a unary code.

    Input:
    v           A list of polynomials
    rate        The cut-off rate

    Output:
    rep         A list of integers

    Format:     Coefficient
    """
    u = ""
    enc = '#0' + str(rate + 2) + 'b'
    for poly in v:
        for coef in poly:
            s = "1" if coef > 0 else "0"
            s += format((abs(coef) % (1 << rate)), enc)[:1:-1]
            s += "0" * (abs(coef) >> rate) + "1"
            u += s
    u += "0" * ((8 - len(u)) % 8)
    return [int(u[8 * i: 8 * i + 8], 2) for i in range(len(u) // 8)]


def decompress(t, degree, rate=7):
    """
    Decompress a list of integers into a list of polynomials.
    It holds that decompress(compress(v, rate), degree, rate) = v.

    Input:
    t           A list of integers
    degree      The degree (-1) of the polynomials
    rate        The cut-off rate

    Output:
    rep         A list of polynomials

    Format:     Coefficient
    ."""
    u = ""
    for elt in t:
        u += bin((1 << 8) ^ elt)[3:]
    v = []
    while u[-1] == "0":
        u = u[:-1]
    while u != "":
        sign = 1 if u[0] == "1" else -1
        low = int(u[rate:0:-1], 2)
        i, high = rate + 1, 0
        while u[i] == "0":
            i += 1
            high += 1
        elt = sign * (low + (high << rate))
        v += [elt]
        u = u[i + 1:]
    return [v[degree * j: degree * (j + 1)] for j in range(len(v) // degree)]
