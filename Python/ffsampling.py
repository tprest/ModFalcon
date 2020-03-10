"""This file contains important algorithms for Module-Falcon.

- the Fast Fourier orthogonalization (in coefficient and FFT representation)
- the Fast Fourier nearest plane (in coefficient and FFT representation)
- the Fast Fourier sampling (only in FFT)
.
"""
from common import split, merge                     # Split, merge
from fft import add, sub, mul, adj                  # Operations in coef.
from fft import add_fft, sub_fft, mul_fft, adj_fft  # Ops in FFT
from fft import split_fft, merge_fft, fft_ratio     # FFT
from linalg import ldl, ldl_fft
from sampler import sampler_z                       # Gaussian sampler in Z
from math import sqrt


def ffldl(G):
    """
    Compute the ffLDL decomposition tree of G.

    Input:
    G           A Gram matrix

    Output:
    T           The ffLDL decomposition tree of G

    Format:     Coefficient

    Similar to algorithm ffLDL of Falcon's documentation,
    except it's in polynomial representation.
    """
    m = len(G) - 1
    d = len(G[0][0])
    # LDL decomposition
    L, D = ldl(G)
    # General case
    if (d > 2):
        rep = [L]
        for i in range(m + 1):
            # Split the output
            d0, d1 = split(D[i][i])
            Gi = [[d0, d1], [adj(d1), d0]]
            # Recursive call on the split parts
            rep += [ffldl(Gi)]
        return rep
    # Bottom case
    elif (d == 2):
        D[0][0][1] = 0
        D[1][1][1] = 0
        return [L, D[0][0], D[1][1]]


def ffldl_fft(G):
    """
    Compute the ffLDL decomposition tree of G.

    Input:
    G           A Gram matrix

    Output:
    T           The ffLDL decomposition tree of G

    Format:     FFT

    Similar to algorithm ffLDL of Falcon's documentation.
    """
    m = len(G) - 1
    d = len(G[0][0]) * fft_ratio
    # LDL decomposition
    L, D = ldl_fft(G)
    # General case
    if (d > 2):
        rep = [L]
        for i in range(m + 1):
            # Split the output
            d0, d1 = split_fft(D[i][i])
            Gi = [[d0, d1], [adj_fft(d1), d0]]
            # Recursive call on the split parts
            rep += [ffldl_fft(Gi)]
        return rep
    # Bottom case
    elif (d == 2):
        # Each element is real
        return [L, D[0][0], D[1][1]]


def ffnp(t, T):
    """
    Compute the FFNP reduction of t, using T as auxilary information.

    Input:
    t           A vector
    T           The LDL decomposition tree of an (implicit) matrix G

    Output:
    z           An integer vector such that (t - z) * B is short

    Format:     Coefficient
    """
    m = len(t)
    n = len(t[0])
    z = [None] * m
    # General case
    if (n > 1):
        L = T[0]
        for i in range(m - 1, -1, -1):
            # t[i] is "corrected", taking into accounts the t[j], z[j] (j > i)
            tib = t[i][:]
            for j in range(m - 1, i, -1):
                tib = add(tib, mul(sub(t[j], z[j]), L[j][i]))
            # Recursive call
            z[i] = merge(ffnp(split(tib), T[i + 1]))
        return z
    # Bottom case: round each coefficient in parallel
    elif (n == 1):
        z[0] = [round(t[0][0])]
        z[1] = [round(t[1][0])]
        return z


def ffnp_fft(t, T):
    """
    Compute the FFNP reduction of t, using T as auxilary information.

    Input:
    t           A vector
    T           The LDL decomposition tree of an (implicit) matrix G

    Output:
    z           An integer vector such that (t - z) * B is short

    Format:     FFT
    """
    m = len(t)
    n = len(t[0]) * fft_ratio
    z = [None] * m
    # General case
    if (n > 1):
        L = T[0]
        for i in range(m - 1, -1, -1):
            # t[i] is "corrected", taking into accounts the t[j], z[j] (j > i)
            tib = t[i][:]
            for j in range(m - 1, i, -1):
                tib = add_fft(tib, mul_fft(sub_fft(t[j], z[j]), L[j][i]))
            # Recursive call
            z[i] = merge_fft(ffnp_fft(split_fft(tib), T[i + 1]))
        return z
    # Bottom case: round each coefficient in parallel
    elif (n == 1):
        z[0] = [round(t[0][0].real)]
        z[1] = [round(t[1][0].real)]
        return z


def ffsampling_fft(t, T):
    """
    Compute the fast Fourier sampling of t, using T as auxilary information.

    Input:
    t           A vector
    T           The LDL decomposition tree of an (implicit) matrix G

    Output:
    z           An integer vector such that (t - z) * B is short

    Format:     FFT

    This algorithim is a randomized version of ffnp_fft,
    such that z * B is distributed as a spherical Gaussian
    centered around t * B.
    """
    m = len(t)
    n = len(t[0]) * fft_ratio
    z = [None] * m
    # General case
    if (n > 1):
        L = T[0]
        for i in range(m - 1, -1, -1):
            # t[i] is "corrected", taking into accounts the t[j], z[j] (j > i)
            tib = t[i][:]
            for j in range(m - 1, i, -1):
                tib = add_fft(tib, mul_fft(sub_fft(t[j], z[j]), L[j][i]))
            # Recursive call
            z[i] = merge_fft(ffsampling_fft(split_fft(tib), T[i + 1]))
        return z
    # Bottom case: round each coefficient in parallel
    elif (n == 1):
        z[0] = [sampler_z(T[0], t[0][0].real)]
        z[1] = [sampler_z(T[0], t[1][0].real)]
        return z


def print_tree(tree, pref=""):
    """
    Display a LDL tree in a readable form.

    Input:
    tree        An LDL decomposition tree

    Output:
    a           A string visually representing the LDL decomposition tree

    Format:     Coefficient or FFT
    """
    leaf = "|_____> "
    top = "|_______"
    son1 = "|       "
    son2 = "        "
    width = len(top)

    a = ""
    lt = len(tree)
    if lt >= 3:
        if (pref == ""):
            a += pref + str(tree[0]) + "\n"
        else:
            a += pref[:-width] + top + str(tree[0]) + "\n"
        for i in range(1, lt - 1):
            a += print_tree(tree[i], pref + son1)
        a += print_tree(tree[lt - 1], pref + son2)
        return a
    else:
        return (pref[:-width] + leaf + str(tree) + "\n")


def normalize_tree(tree, sigma):
    """
    Normalize a LDL decomposition tree, by doing
    leave := sigma / sqrt(leave) for all the leaves.

    Input:
    tree        An LDL decomposition tree
    sigma       A standard deviation
    """
    lt = len(tree)
    if lt >= 3:
        for i in range(1, lt):
            normalize_tree(tree[i], sigma)
    else:
        tree[0] = sigma / sqrt(tree[0].real)
        if tree[0] >= 2:
            tree[0] = 2
            # FIXME
            # raise ValueError
        tree[1] = 0
