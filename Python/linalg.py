"""
Algorithms for simple linear algebra.
"""
from fft import add, sub, mul, div, adj
from fft import add_fft, sub_fft, mul_fft, div_fft, adj_fft
from ntt import add_zq, mul_zq


def gram(B):
    """
    Compute the Gram matrix of B.

    Input:
    B           A matrix

    Output:
    G           The Gram matrix of B: G = B * (B*)

    Format:     Coefficient
    """
    rows = range(len(B))
    ncols = len(B[0])
    deg = len(B[0][0])
    G = [[[0 for coef in range(deg)] for j in rows] for i in rows]
    for i in rows:
        for j in rows:
            for k in range(ncols):
                G[i][j] = add(G[i][j], mul(B[i][k], adj(B[j][k])))
    return G


def gram_fft(B):
    """
    Compute the Gram matrix of B.

    Input:
    B           A matrix

    Output:
    G           The Gram matrix of B: G = B * (B*)

    Format:     FFT
    """
    rows = range(len(B))
    ncols = len(B[0])
    deg = len(B[0][0])
    G = [[[0 for coef in range(deg)] for j in rows] for i in rows]
    for i in rows:
        for j in rows:
            for k in range(ncols):
                G[i][j] = add_fft(G[i][j], mul_fft(B[i][k], adj_fft(B[j][k])))
    return G


def ldl(G):
    """
    Compute the LDL decomposition of G.

    Input:
    G           A self-adjoint matrix (i.e. G is equal to its conjugate transpose)

    Output:
    L, D        The LDL decomposition of G, that is G = L * D * (L*), where:
                - L is lower triangular with a diagonal of 1's
                - D is diagonal

    Format:     Coefficient
    """
    deg = len(G[0][0])
    dim = len(G)
    L = [[[0 for k in range(deg)] for j in range(dim)] for i in range(dim)]
    D = [[[0 for k in range(deg)] for j in range(dim)] for i in range(dim)]
    for i in range(dim):
        L[i][i] = [1] + [0 for j in range(deg - 1)]
        D[i][i] = G[i][i]
        for j in range(i):
            L[i][j] = G[i][j]
            for k in range(j):
                L[i][j] = sub(L[i][j], mul(mul(L[i][k], adj(L[j][k])), D[k][k]))
            L[i][j] = div(L[i][j], D[j][j])
            D[i][i] = sub(D[i][i], mul(mul(L[i][j], adj(L[i][j])), D[j][j]))
    return [L, D]


def ldl_fft(G):
    """
    Compute the LDL decomposition of G.

    Input:
    G           A self-adjoint matrix (i.e. G is equal to its conjugate transpose)

    Output:
    L, D        The LDL decomposition of G, that is G = L * D * (L*), where:
                - L is lower triangular with a diagonal of 1's
                - D is diagonal

    Format:     FFT
    """
    deg = len(G[0][0])
    dim = len(G)
    L = [[[0 for k in range(deg)] for j in range(dim)] for i in range(dim)]
    D = [[[0 for k in range(deg)] for j in range(dim)] for i in range(dim)]
    for i in range(dim):
        L[i][i] = [1 for j in range(deg)]
        D[i][i] = G[i][i]
        for j in range(i):
            L[i][j] = G[i][j]
            for k in range(j):
                L[i][j] = sub_fft(L[i][j], mul_fft(mul_fft(L[i][k], adj_fft(L[j][k])), D[k][k]))
            L[i][j] = div_fft(L[i][j], D[j][j])
            D[i][i] = sub_fft(D[i][i], mul_fft(mul_fft(L[i][j], adj_fft(L[i][j])), D[j][j]))
    return [L, D]


def vecmatmul(t, B, integer=False, modulus=None):
    """
    Compute the product t * B, where t is a vector and B is a square matrix.

    Input:
    t           A row vector
    B           A matrix
    integer     This flag should be set (to True) iff the elements are in Z[x] / (x ** n + 1)
    modulus     This flag should be set (to q) iff the elements are in Z_q[x] / (x ** n + 1)

    Output:
    v           The row vector t * B

    Format:     Coefficient
    """
    nrows = len(B)
    ncols = len(B[0])
    deg = len(B[0][0])
    assert(len(t) == nrows)
    v = [[0 for k in range(deg)] for j in range(ncols)]
    if modulus is not None:
        for j in range(ncols):
            for i in range(nrows):
                v[j] = add_zq(v[j], mul_zq(t[i], B[i][j], modulus), modulus)
        return v
    else:
        for j in range(ncols):
            for i in range(nrows):
                v[j] = add(v[j], mul(t[i], B[i][j]))
        if integer is True:
            v = [[int(round(elt)) for elt in poly] for poly in v]
        return v


def vecmatmul_fft(t, B):
    """
    Compute the product t * B, where t is a vector and B is a square matrix.

    Input:
    t           A row vector
    B           A matrix

    Output:
    v           The row vector t * B

    Format:     FFT
    """
    nrows = len(B)
    ncols = len(B[0])
    deg = len(B[0][0])
    assert(len(t) == nrows)
    v = [[0 for k in range(deg)] for j in range(ncols)]
    for j in range(ncols):
        for i in range(nrows):
            v[j] = add_fft(v[j], mul_fft(t[i], B[i][j]))
    return v


def vecvecmul(u, v, integer=False, modulus=None):
    """
    Compute the product u * (v^t), u and v are row vectors, and v^t denotes the transose of v.

    Input:
    u           A row vector
    v           A row vector
    integer     This flag should be set (to True) iff the elements are in Z[x] / (x ** n + 1)
    modulus     This flag should be set (to q) iff the elements are in Z_q[x] / (x ** n + 1)

    Output:
    rep         The product u * (v^t) = sum(u[i] * v[i] for i in range(len(u)))

    Format:     Coefficient
    """
    m = len(u)
    deg = len(u[0])
    assert(len(u) == len(v))
    rep = [0 for k in range(deg)]
    if modulus is not None:
        for i in range(m):
            rep = add_zq(rep, mul_zq(u[i], v[i], modulus), modulus)
        return rep
    else:
        for i in range(m):
            rep = add(rep, mul(u[i], v[i]))
        if integer is True:
            rep = [int(round(elt)) for elt in rep]
        return rep


# def matmul(A, B):
#     """
#     Compute the product A * B, where A and B are matrices.
# 
#     Input:
#     A           A matrix
#     B           A matrix
# 
#     Output:
#     rep        The A * B
# 
#     Format:     Coefficient
#     """
#     nrow = len(A)
#     m = len(B)
#     ncol = len(B[0])
#     d = len(A[0][0])
#     assert (m == len(A[0]))
#     rep = [[[0] * d for j in range(ncol)] for i in range(nrow)]
#     for i in range(nrow):
#         for j in range(ncol):
#             for k in range(m):
#                 rep[i][j] = add(rep[i][j], karamul(A[i][k], B[k][j]))
#     return rep
