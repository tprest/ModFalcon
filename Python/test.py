"""
This file implements tests for various parts of the Falcon.py library.

Test the code with:
> make test
"""
from common import q_12289, q_7681, sqnorm
from fft import add, sub, mul, div, fft, ifft
from ntt import add_zq, mul_zq, div_zq
from sampler import sampler_z
from ffsampling import ffldl, ffldl_fft, ffnp, ffnp_fft
from random import randint, random, gauss
from math import pi, sqrt, floor, ceil, exp
from ntrugen import module_ntru_gen, my_det
from module_falcon import SecretKey, PublicKey
from encoding import compress, decompress
from linalg import vecmatmul, gram
import sys
if sys.version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.


def test_fft(n, iterations=10):
    """Test the FFT."""
    for i in range(iterations):
        f = [randint(-3, 4) for j in range(n)]
        g = [randint(-3, 4) for j in range(n)]
        h = mul(f, g)
        k = div(h, f)
        k = [int(round(elt)) for elt in k]
        if k != g:
            print("(f * g) / f =", k)
            print("g =", g)
            print("mismatch")
            return False
    return True


def test_ntt(n, q, iterations=10):
    """Test the NTT."""
    for i in range(iterations):
        f = [randint(0, q - 1) for j in range(n)]
        g = [randint(0, q - 1) for j in range(n)]
        h = mul_zq(f, g, q)
        try:
            k = div_zq(h, f, q)
            if k != g:
                print("(f * g) / f =", k)
                print("g =", g)
                print("mismatch")
                return False
        except ZeroDivisionError:
            continue
    return True


def gaussian(sigma, mu, x):
    """The Gaussian function."""
    return exp(- ((x - mu) ** 2) / (2. * (sigma ** 2)))


def test_sampler_z(sigma, mu, iterations):
    """Test the integer Gaussian sampler."""
    den = sqrt(2 * pi) * sigma
    start = int(floor(mu - 10 * sigma))
    end = int(ceil(mu + 10 * sigma))
    index = range(start, end)
    ref_table = {z: int(round(iterations * gaussian(sigma, mu, z)) / den) for z in index}
    obs_table = {z: 0 for z in index}
    for i in range(iterations):
        z = sampler_z(sigma, mu)
        obs_table[z] += 1
    delta = sum(abs(ref_table[i] - obs_table[i]) for i in index) / float(iterations)
    # print obs_table
    print(delta)
    # return obs_table


def test_module_ntru_gen(d, m, iterations):
    q = q_12289
    for _ in range(iterations):
        A, B, inv_B, sqr_gsnorm = module_ntru_gen(d, q, m)
        # Check that the determinant of B is q
        if (my_det(B) != [q] + [0] * (d - 1)):
            print("det(B) != q")
            return False
        # Check that B * A = 0 mod q
        C = [None] * (m + 1)
        for i in range(m + 1):
            elt = [0] * d
            for j in range(m + 1):
                elt = add_zq(elt, mul_zq(B[i][j], A[j], q), q)
            C[i] = elt
        if any(elt != [0] * d for elt in C):
            print("Error: A and B are not orthogonal")
            return False
        # If all the checks passed, return True
    return True


def test_ffnp(d, m, iterations):
    """Test ffnp.

    This functions check that:
    1. the two versions (coefficient and FFT embeddings) of ffnp are consistent
    2. ffnp output lattice vectors close to the targets.
    """
    q = q_12289
    A, B, inv_B, sqr_gsnorm = module_ntru_gen(d, q, m)
    G0 = gram(B)
    G0_fft = [[fft(elt) for elt in row] for row in G0]
    T = ffldl(G0)
    T_fft = ffldl_fft(G0_fft)
    th_bound = (m + 1) * d * sqr_gsnorm / 4.

    mn = 0
    for i in range(iterations):
        t = [[random() for coef in range(d)] for poly in range(m + 1)]
        t_fft = [fft(elt) for elt in t]

        z = ffnp(t, T)
        z_fft = ffnp_fft(t_fft, T_fft)

        zb = [ifft(elt) for elt in z_fft]
        zb = [[round(coef) for coef in elt] for elt in zb]
        if z != zb:
            print("ffnp and ffnp_fft are not consistent")
            return False
        diff = [sub(t[i], z[i]) for i in range(m + 1)]
        diffB = vecmatmul(diff, B)
        norm_zmc = int(round(sqnorm(diffB)))
        mn = max(mn, norm_zmc)

    if mn > th_bound:
        print("z = {z}".format(z=z))
        print("t = {t}".format(t=t))
        print("mn = {mn}".format(mn=mn))
        print("th_bound = {th_bound}".format(th_bound=th_bound))
        print("sqr_gsnorm = {sqr_gsnorm}".format(sqr_gsnorm=sqr_gsnorm))
        print("Warning: the algorithm outputs vectors longer than expected")
        return False
    else:
        return True


def test_compress(d, q, iterations):
    """Test compression and decompression."""
    sigma = 1.5 * sqrt(q)
    for i in range(iterations):
        initial = [[int(round(gauss(0, sigma))) for coef in range(d)]]
        for rate in range(6, 9):
            compressed = compress(initial, rate=rate)
            decompressed = decompress(compressed, degree=d, rate=rate)
            if decompressed != initial:
                return False
    return True


def test_module_falcon(d, m, q, iterations=10):
    """Test Falcon."""
    sk = SecretKey(d, m, q)
    pk = PublicKey(sk)
    for i in range(iterations):
        message = "0"
        sig = sk.sign(message)
        s, t = sig
        # print(len(s) + len(t))
        if pk.verify(message, sig) is False:
            return False
    return True


def make_matrix(v):
    n = len(v)
    M = [[v[i] * v[j] for j in range(n)] for i in range(n)]
    return M


def test_covariance(d, m, q, iterations=100):
    """
    Compute the covariance matrix of the signatures distribution.

    For an isotropic Gaussian, the covariance matrix is
    proportional to the identity matrix.
    """
    sk = SecretKey(d, m, q)
    dim = (m + 1) * d
    liste_sig = []
    mean = [0] * dim
    Cov = [[0 for _ in range(dim)] for _ in range(dim)]
    for i in range(iterations):
        message = "0"
        r, t = sk.sign(message)
        s = decompress(t, sk.d, sk.rate)
        s = sum([elt for elt in s], [])
        mean = add(mean, s)
        liste_sig += [s]
    # print("mean = {mean}".format(mean=mean))
    for s in liste_sig:
        s = [iterations * elt for elt in s]
        s = [(s[i] - mean[i]) for i in range(dim)]
        M = make_matrix(s)
        for i in range(dim):
            Cov[i] = add(Cov[i], M[i])
    # We normalize only at the end to work with integers as long as possible
    for i in range(dim):
        for j in range(dim):
            Cov[i][j] /= (iterations ** 3)
            Cov[i][j] = int(round(Cov[i][j]))
    # print(Cov)
    # return Cov


def test(d, q, iterations=1000):
    """A battery of tests."""
    # sys.stdout.write('Test FFT                    : ')
    # print("OK" if test_fft(d, iterations) else "FAIL")
    # sys.stdout.write('Test NTT                    : ')
    # print("OK" if test_ntt(d, q, iterations) else "FAIL")
    # sys.stdout.write('Test ffSampling (covariance): ')
    # test_covariance(d, 1, q, iterations)
    # test_covariance(d, 2, q, iterations)
    # print("OK" if test_fft(d, iterations) else "FAIL")
    # sys.stdout.write('Test module_ntru_gen (m = 1): ')
    # print("OK" if test_module_ntru_gen(d, 1, iterations // 100) else "FAIL")
    # sys.stdout.write('Test module_ntru_gen (m = 2): ')
    # print("OK" if test_module_ntru_gen(d, 2, iterations // 100) else "FAIL")
    # sys.stdout.write('Test module_ntru_gen (m = 3): ')
    # print("OK" if test_module_ntru_gen(d, 3, iterations // 100) else "FAIL")
    # sys.stdout.write('Test ffnp            (m = 1): ')
    # print("OK" if test_ffnp(d, 1, iterations) else "FAIL")
    # sys.stdout.write('Test ffnp            (m = 2): ')
    # print("OK" if test_ffnp(d, 2, iterations) else "FAIL")
    # # print("OK" if test_ffnp(d, 3, iterations) else "FAIL")
    # sys.stdout.write('Test compression            : ')
    # print("OK" if test_compress(d, q, iterations) else "FAIL")
    sys.stdout.write('Test module-Falcon   (m = 1): ')
    print("OK" if test_module_falcon(d, 1, q, iterations // 10) else "FAIL")
    sys.stdout.write('Test module-Falcon   (m = 2): ')
    print("OK" if test_module_falcon(d, 2, q, iterations // 10) else "FAIL")
    sys.stdout.write('Test module-Falcon   (m = 3): ')
    print("OK" if test_module_falcon(d, 3, q, iterations // 10) else "FAIL")


# Run all the tests
if (__name__ == "__main__"):
    for q in [q_12289]:
        print("")
        for i in range(10, 11):
            print("")
            d = (1 << i)
            print("Test battery for d = {d}, q = {q}".format(d=d, q=q))
            test(d, q)
