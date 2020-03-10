"""
This file implements the "module-ntru solver" part of the key generation."""
from random import gauss
from math import sqrt
from fft import fft, ifft, add_fft, mul_fft, adj_fft, div_fft  # FFT
from fft import add, mul, div, neg              # regular operations
from ntt import div_zq, add_zq, mul_zq                         # NTT
from linalg import gram_fft, ldl_fft                # Linear algebra


def karatsuba(a, b, n):
    """
    Karatsuba multiplication between polynomials.

    Input:
    a           A polynomial
    b           A polynomial
    n           A power-of-two integer such that deg(a), deg(b) < n

    Output:
    ab          The product a * b

    Format:     Coefficient
    """
    if n == 1:
        return [a[0] * b[0], 0]
    else:
        n2 = n // 2
        a0 = a[:n2]
        a1 = a[n2:]
        b0 = b[:n2]
        b1 = b[n2:]
        ax = [a0[i] + a1[i] for i in range(n2)]
        bx = [b0[i] + b1[i] for i in range(n2)]
        a0b0 = karatsuba(a0, b0, n2)
        a1b1 = karatsuba(a1, b1, n2)
        axbx = karatsuba(ax, bx, n2)
        for i in range(n):
            axbx[i] -= (a0b0[i] + a1b1[i])
        ab = [0] * (2 * n)
        for i in range(n):
            ab[i] += a0b0[i]
            ab[i + n] += a1b1[i]
            ab[i + n2] += axbx[i]
        return ab


def karamul(a, b):
    """
    Karatsuba multiplication, followed by reduction mod (x ** n + 1).

    Input:
    a           A polynomial (n coefficients)
    b           A polynomial (n coefficients)

    Output:
    ab          The product a * b mod (x ** n + 1)

    Format:     Coefficient
    """
    n = len(a)
    assert(n == len(b))
    ab = karatsuba(a, b, n)
    abr = [ab[i] - ab[i + n] for i in range(n)]
    return abr


def galois_conjugate(a):
    """
    Galois conjugate of an element a in Q[x] / (x ** n + 1).
    In this case, the Galois conjugate of a(x) is simply a(-x).

    Input:
    a           A polynomial (n coefficients)

    Output:
    rep         The Galois conjugate of a

    Format:     Coefficient
    """
    n = len(a)
    return [((-1) ** i) * a[i] for i in range(n)]


def project(a):
    """
    Project an element a of Q[x] / (x ** n + 1) onto Q[y] / (y ** (n // 2) + 1).
    Only works if n is a power-of-two. This projection is done using the field norm:

            project(fe(x^2) + x * fo(x^2)) = fe^2(y) - y * fo^2(y).

    Input:
    a           A polynomial of Q[x] / (x ** n + 1)

    Output:
    res         The field norm projection of a onto Q[y] / (y ** (n // 2) + 1)

    Format:     Coefficient
    """
    n2 = len(a) // 2
    ae = [a[2 * i] for i in range(n2)]
    ao = [a[2 * i + 1] for i in range(n2)]
    ae_squared = karamul(ae, ae)
    ao_squared = karamul(ao, ao)
    res = ae_squared[:]
    for i in range(n2 - 1):
        res[i + 1] -= ao_squared[i]
    res[0] += ao_squared[n2 - 1]
    return res


def lift(a):
    """
    Lift an element a of Q[y] / (y ** (n // 2) + 1) up to Q[x] / (x ** n + 1).
    The lift of a(y) is simply a(x ** 2).

    Input:
    a           A polynomial of Q[y] / (y ** (n // 2) + 1)

    Output:
    res         The lift of a in Q[x] / (x ** n + 1)

    Format:     Coefficient
    """
    n = len(a)
    res = [0] * (2 * n)
    for i in range(n):
        res[2 * i] = a[i]
    return res


def bitsize(a):
    """
    Compute the bitsize of an element of Z (not counting the sign).
    """
    val = abs(a)
    res = 0
    while val:
        res += 1
        val >>= 1
    return res


def reduce(f, g, F, G):
    """
    Reduce (F, G) relatively to (f, g).

    This is done via Babai's reduction.
    (F, G) <-- (F, G) - k * (f, g), where k = round((F f* + G g*) / (f f* + g g*)).
    Similar to algorithm Reduce of Falcon's documentation.

    Input:
    f, g, F, G      Four polynomials mod (x ** n + 1)

    Output:
    None            The inputs are reduced as detailed above.

    Format:         Coefficient
    """
    n = len(f)
    size = max(53, bitsize(min(f)), bitsize(max(f)), bitsize(min(g)), bitsize(max(g)))

    f_adjust = [elt >> (size - 53) for elt in f]
    g_adjust = [elt >> (size - 53) for elt in g]
    fa_fft = fft(f_adjust)
    ga_fft = fft(g_adjust)

    while(1):
        # Because we are working in finite precision to reduce very large polynomials,
        # we may need to perform the reduction several times.
        Size = max(53, bitsize(min(F)), bitsize(max(F)), bitsize(min(G)), bitsize(max(G)))
        if Size < size:
            break

        F_adjust = [elt >> (Size - 53) for elt in F]
        G_adjust = [elt >> (Size - 53) for elt in G]
        Fa_fft = fft(F_adjust)
        Ga_fft = fft(G_adjust)

        den_fft = add_fft(mul_fft(fa_fft, adj_fft(fa_fft)), mul_fft(ga_fft, adj_fft(ga_fft)))
        num_fft = add_fft(mul_fft(Fa_fft, adj_fft(fa_fft)), mul_fft(Ga_fft, adj_fft(ga_fft)))
        k_fft = div_fft(num_fft, den_fft)
        k = ifft(k_fft)
        k = [int(round(elt)) for elt in k]
        if all(elt == 0 for elt in k):
            break
        fk = karamul(f, k)
        gk = karamul(g, k)
        for i in range(n):
            F[i] -= fk[i] << (Size - size)
            G[i] -= gk[i] << (Size - size)
    return F, G


def xgcd(b, n):
    """
    Compute the extended GCD of two integers b and n.

    Input:
    b, n        Two integers

    Output:
    d, u, v     Three integers such that d = u * b + v * n, and d is the GCD of b, n.
    """
    x0, x1, y0, y1 = 1, 0, 0, 1
    while n != 0:
        q, b, n = b // n, n, b % n
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return b, x0, y0


def ntru_solve(f, g, q):
    """
    Solve the NTRU equation for f and g.
    Corresponds to algorithm NTRUSolve of Falcon's documentation.

    Input:
    f, g        Two polynomials mod (x ** n + 1)
    q           An integer

    Output:
    F, G        Two polynomials mod (x ** n + 1) such that f * G - g * F = q
    """
    n = len(f)
    # Bottom case
    if n == 1:
        f0 = f[0]
        g0 = g[0]
        d, u, v = xgcd(f0, g0)
        if d != 1:
            raise ValueError
        else:
            return [- q * v], [q * u]
    # General case
    else:
        # We project the equation onto a subring
        fp = project(f)
        gp = project(g)
        # Recursive call
        Fp, Gp = ntru_solve(fp, gp, q)
        # We lift the solution
        F = karamul(lift(Fp), galois_conjugate(g))
        G = karamul(lift(Gp), galois_conjugate(f))
        # Once we have a solution, we reduce its size
        F, G = reduce(f, g, F, G)
        return F, G


def submatrix(M, row, col):
    """
    Compute the matrix obtained from M by removing the row-th row and the col-th column.
    This submatrix is then useful for computing the determinant and adjugate matrix of M.

    Input:
    M           A matrix
    row         The index of the row to remove
    col         The index of the column to remove

    Output:
    rep         A submatrx

    Format:     Any
    """
    nrow = len(M)
    ncol = len(M[0])
    rep = [[M[i][j] for j in range(ncol) if j != col] for i in range(nrow) if i != row]
    return rep


def my_det(M):
    """
    Compute the determinant of M.
    If M has coefficients in a ring R, then the determinant is in R.
    This determinant is computed using Cramer's rule: this is less efficient
    than other methods but allows to do all operations in R.

    Input:
    M               A matrix

    Output:
    determinant     The determinant of M

    Format:         Coefficient
    """
    nrow = len(M)
    ncol = len(M[0])
    d = len(M[0][0])
    assert (nrow == ncol)
    if (nrow == 1):
        return M[0][0]
    else:
        determinant = [0] * d
        for i in range(nrow):
            Mp = submatrix(M, i, 0)
            det_Mp = my_det(Mp)
            if (i & 1):
                det_Mp = [- elt for elt in det_Mp]
            determinant = add(determinant, karamul(M[i][0], det_Mp))
    return determinant


def my_adjugate(M):
    """
    Compute the adjugate matrix adj(M) of M, the transpose of its cofactor matrix.
    If M has its coefficients in a ring R, then so does adj(M).
    It verifies adj(M) * M = det(M).

    Input:
    M           A matrix

    Output:
    rep         The adjugate matrix of M

    Format:     Coefficient
    """
    nrow = len(M)
    ncol = len(M[0])
    if (nrow == 1):
        deg = len(M[0][0])
        one = [1] + [0 for _ in range(1, deg)]
        return [[one]]
    rep = [[None for j in range(ncol)] for i in range(nrow)]
    for i in range(nrow):
        for j in range(ncol):
            Mp = submatrix(M, i, j)
            rep[j][i] = my_det(Mp)
            if (i + j) & 1:
                rep[j][i] = [- elt for elt in rep[j][i]]
    return rep


def module_ntru_gen(d, q, m):
    """
    Take as input system parameters, and output two "module-NTRU" matrices A and B such that:
    - B * A = 0 [mod q]
    - B has small polynomials
    - A is in Hermite normal form
    Also compute the inverse of B (over the field K = Q[x] / (x ** d + 1)).

    Input:
    d           The degree of the underlying ring R = Z[x] / (x ** d + 1)
    q           An integer
    m           An integer

    Output:
    A           A matrix in R ^ ((m + 1) x 1)
    B           A matrix in R ^ ((m + 1) x (m + 1))
    inv_B       A matrix in K ^ ((m + 1) x (m + 1))
    sq_gs_norm  A real number, the square of the Gram-Schmidt norm of B

    Format:     Coefficient
    """
    if m == 1:
        magic_constant = [1.15]
        gs_slack = 1.17
    elif m == 2:
        magic_constant = [1.07, 1.14]
        gs_slack = 1.17
    elif m == 3:
        magic_constant = [1.21, 1.10, 1.06]
        gs_slack = 1.24
    else:
        print("No parameters implemented yet for m = {m}".format(m=m))
        return
    max_gs_norm = gs_slack * (q ** (1 / (m + 1)))
    while True:
        # We generate all rows of B except the last
        B = [[None for j in range(m + 1)] for i in range(m + 1)]
        for i in range(m):
            for j in range(m + 1):
                # Each coefficient B[i][j] is a polynomial
                sigma = magic_constant[i] * (q ** (1 / (m + 1))) # ==> ||bi~|| = gs_slack * q^(1/(m+1))
                sig = sqrt(1 / (d * (m + 1 - i))) * sigma        # sig = stdv. dev. des coef de bi
                B[i][j] = [int(round(gauss(0, sig))) for k in range(d)]
        # We check that the GS norm is not larger than tolerated
        Bp_fft = [[fft(poly) for poly in row] for row in B[:-1]]
        Gp = gram_fft(Bp_fft)
        [Lp_fft, Dp_fft] = ldl_fft(Gp)
        Dp = [[[0] * d for col in range(m)] for row in range(m)]
        for i in range(m):
            Dp[i][i] = ifft(Dp_fft[i][i])
        prod_di = [1] + [0] * (d - 1)
        for i in range(m):
            prod_di = mul(prod_di, Dp[i][i])
        last = div([q ** 2] + [0] * (d - 1), prod_di)
        norms = [Dp[i][i][0] for i in range(m)] + [last[0]]
        # If the GS norm is too large, restart
        if sqrt(max(norms)) > max_gs_norm:
            continue
        # Try to solve the module-NTRU equation
        f = submatrix(B, m, 0)
        f = [[neg(elt) for elt in row] for row in f]
        g = [B[j][0] for j in range(m)]
        fp = my_det(f)
        adjf = my_adjugate(f)
        gp = [0] * d
        for i in range(m):
            gp = add(gp, karamul(adjf[0][i], g[i]))
        try:
            # Compute f^(-1) mod q
            fp_q = [elt % q for elt in fp]
            inv_f = [[elt[:] for elt in row] for row in adjf]
            for i in range(m):
                for j in range(m):
                    inv_f[i][j] = [elt % q for elt in inv_f[i][j]]
                    inv_f[i][j] = div_zq(inv_f[i][j], fp_q, q)
            # Compute h = f^(-1) * g mod q and A = [1 | h]
            h = [None] * m
            for i in range(m):
                elt = [0] * d
                for j in range(m):
                    elt = add_zq(elt, mul_zq(inv_f[i][j], g[j], q), q)
                h[i] = elt
            one = [1] + [0 for _ in range(1, d)]
            A = [one] + h
            Fp, Gp = ntru_solve(fp, gp, q)
            B[m][0] = Gp
            B[m][1] = [- coef for coef in Fp]
            for i in range(2, m + 1):
                B[m][i] = [0 for _ in range(d)]
            # Compute the inverse of B
            det_B = my_det(B)
            inv_B = my_adjugate(B)
            inv_B = [[div(poly, det_B) for poly in row] for row in inv_B]
            return A, B, inv_B, max(norms)
        # If any step failed, restart
        except (ZeroDivisionError, ValueError):
            continue
