"""
This script computes parameters and security estimates for ModFalcon.

References:
- [BDGL16]: ia.cr/2015/1128
- [DLP14]: ia.cr/2014/794
- [Laa16]: https://pure.tue.nl/ws/files/14673128/20160216_Laarhoven.pdf
- [Lyu12]: ia.cr/2011/537
- [MR07]: https://cims.nyu.edu/~regev/papers/average.pdf
- [MW16]: ia.cr/2015/1123
- [NIST]: https://csrc.nist.gov/CSRC/media/Projects/Post-Quantum-Cryptography
          /documents/call-for-proposals-final-dec-2016.pdf
- [Pre17]: ia.cr/2017/480
"""
# from Crypto.Util.number import isPrime
from math import sqrt, exp, log, pi, floor
# For debugging purposes
from sys import version_info
if version_info >= (3, 4):
    from importlib import reload  # Python 3.4+ only.


def smooth(eps, m, normalized=True):
    """
    Compute the smoothing parameter eta_epsilon(Z^m).
    - if normalized is True, take the definition from [Pre17,Falcon]
    - if normalized is False, take the definition from [MR07]
    """
    rep = sqrt(log(2 * m * (1 + 1 / eps)) / pi)
    if normalized is True:
        return rep / sqrt(2 * pi)
    else:
        return rep


def delta_func(beta):
    """
    Compute delta_beta as per Heuristic 1.
    """
    rep = ((pi * beta) ** (1 / beta)) * beta / (2 * pi * exp(1))
    rep = rep ** (1 / (2 * (beta - 1)))
    return rep


def dimensionsforfree(B):
    """
    d in [Duc18].
    """
    return round(B * log(4 / 3) / log(B / (2 * pi * exp(1))))


class ModFalcon:

    def __init__(self, d, n, target_bitsec):
        """
        Initialize a ModFalcon object

        Input:
        - a ring degree d
        - an integer n
        - a target bit-security target_bitsec

        Output:
        - a ModFalcon object with:
          - the ring degree d
          - the rank n + 1
          - the integer modulus q
          - the Gram-Schmidt norm gs_norm
          - the signature standard deviation sigma
          - the tailcut rate and rejection rate
          - Regarding forgery:
            - the required BKZ blocksize
            - the classical bit-security
            - the quantum bit-security
        """

        # d is the degree of the ring Z[x]/(x^d + 1)
        self.d = d
        # n + 1 is the (module) rank of the ModNTRU lattice
        self.n = n
        # m is the rank of the ModNTRU lattice as a Z-module
        # Useful in later computations
        m = (self.n + 1) * self.d


        # The maximal number of queries is limited to 2 ** 64 as per [NIST]
        self.nb_queries = 2 ** 64

        # The integer modulus q must verify two constraints:
        # - q is a prime number
        # - (q - 1) is a multiple of 2 * d
        self.q = 1024 * 12 + 1
        # assert isPrime(self.q)
        # assert ((self.q - 1) % (2 * self.d) == 0)

        # gs_norm is the Gram-Schmidt norm of the ModNTRU lattice
        # For security, we want to minimize gs_norm
        # For NTRU lattices, it has been shown in [DLP14, Section 3] that
        # one can achieve gs_norm =< 1.17 * sqrt(q) in practice.
        # For ModNTRU lattices, it is generalized in Section 3.1.
        if (n == 1):
            self.gs_slack = 1.17
        elif (n == 2):
            self.gs_slack = 1.17
        elif (n == 3):
            self.gs_slack = 1.24
        else:
            raise ValueError("Not implemented")
        self.gs_norm = self.gs_slack * (self.q ** (1 / (self.n + 1)))

        # sigma is the standard deviation of the signatures:
        # - On one hand, we require sigma small to make forgery harder.
        # - On the other hand, we require sigma enough so that the signatures'
        #   distribution is indistinguishable from an ideal Gaussian.
        # We set sigma according to arguments given in [Pre17]; it allows to
        # argue that we lose O(1) bits of security compared to the ideal case.
        self.eps = 1 / sqrt(4 * target_bitsec * self.nb_queries)
        self.smoothzm = smooth(self.eps, m, normalized=True)
        self.sigma = self.smoothzm * self.gs_norm

        # The tailcut rate intervenes during the signing procedure.
        # The expected value of the signature norm is sigma * sqrt(d * (n + 1)).
        # If the signature norm is larger than its expected value by more than
        # a factor tailcut_rate, it is rejected and the procedure restarts.
        # The max signature norm is also called "ρ" in our submission.
        # The rejection rate is given by [Lyu12, Lemma 4.4].
        self.tailcut_rate = 1.1
        tau = self.tailcut_rate
        self.max_sig_norm = floor(tau * sqrt(m) * self.sigma)
        self.rejection_rate = (tau ** m) * exp(m * (1 - tau ** 2) / 2)

        # Security metrics
        # This is the targeted bit-security.
        self.target_bitsec = target_bitsec

        # Compute the hardness of key-recovery as per Section 4.2
        for beta in range(100, m):
            # Compute the left part of the success condition for key-recovery
            left_kr = self.gs_slack * sqrt(beta / m)
            # Compute the right part of the success condition for key-recovery
            delta_beta = delta_func(beta)
            right_kr = delta_beta ** (2 * beta - m)
            # Break once the right beta is found
            if (left_kr < right_kr):
                break
        # Dimensions for free [Ducas'18]
        beta -= dimensionsforfree(beta)
        self.bkz_keyrec = beta
        self.keyrec_bitsec_c = floor(self.bkz_keyrec * 0.292)
        self.keyrec_bitsec_q = floor(self.bkz_keyrec * 0.265)

        # Compute the hardness of forgery as per Section 4.2
        beta_min = m
        # Outer loop: increment k until finding the k minimizing beta
        for k in range(self.d * self.n):
            # Inner loop: increment beta until forgery is possible
            for beta in range(100, m):
                delta_beta = delta_func(beta)
                right_kr = self.q ** (self.d / (m - k))
                right_kr *= delta_beta ** (m - k)
                # Break once the right beta is found
                if (right_kr < self.max_sig_norm):
                    break
            if beta <= beta_min:
                beta_min = beta
            # Break once beta(k) is not decreasing anymore
            else:
                k -= 1
                break
        # Dimensions for free [Ducas'18]
        beta_min -= dimensionsforfree(beta_min)
        self.bkz_forgery = beta_min
        self.k_forgery = k
        self.forgery_bitsec_c = floor(self.bkz_forgery * 0.292)
        self.forgery_bitsec_q = floor(self.bkz_forgery * 0.265)

    def __repr__(self):
        rep = "Parameters:\n"
        rep += "==========\n"
        rep += "- The degree of the ring ring Z[x]/(x^d + 1) is d.\n"
        rep += "- The (module) rank is n + 1.\n"
        rep += "- The integer modulus is q.\n"
        rep += "- The standard deviation of the signatures is sigma.\n"
        rep += "\n"
        rep += "d       = " + str(self.d) + "\n"
        rep += "n       = " + str(self.n) + "\n"
        rep += "q       = " + str(self.q) + "\n"
        rep += "gs_norm = " + str(self.gs_norm) + "\n"
        rep += "sigma   = " + str(self.sigma) + "\n"
        rep += "\n\n"

        rep += "Metrics:\n"
        rep += "========\n"
        rep += "- The maximal number of signing queries is nb_queries.\n"
        rep += "- Signing's rejection rate is rejection_rate.\n"
        rep += "\n"
        rep += "nb_queries     = 2 ** " + str(int(log(self.nb_queries, 2))) + "\n"
        rep += "rejection_rate = " + str(self.rejection_rate) + "\n"
        rep += "\n\n"

        rep += "Security:\n"
        rep += "=========\n"
        rep += "- The targeted security level is target_bitsec.\n"
        rep += "- The key-recovery BKZ blocksize is bkz_keyrec.\n"
        rep += "- The key-recovery classic bit-security is keyrec_bitsec_c.\n"
        rep += "- The key-recovery quantum bit-security is keyrec_bitsec_q.\n"
        rep += "- The forgery BKZ blocksize is bkz_forgery.\n"
        rep += "- The forgery classic bit-security is forgery_bitsec_c.\n"
        rep += "- The forgery quantum bit-security is forgery_bitsec_q.\n"
        rep += "\n"
        rep += "target_bitsec    = " + str(self.target_bitsec) + "\n\n"
        rep += "bkz_keyrec       = " + str(self.bkz_keyrec) + "\n"
        rep += "keyrec_bitsec_c  = " + str(self.keyrec_bitsec_c) + "\n"
        rep += "keyrec_bitsec_q  = " + str(self.keyrec_bitsec_q) + "\n\n"
        rep += "bkz_forgery      = " + str(self.bkz_forgery) + "\n"
        rep += "forgery_bitsec_c = " + str(self.forgery_bitsec_c) + "\n"
        rep += "forgery_bitsec_q = " + str(self.forgery_bitsec_q) + "\n"
        return rep


if  __name__ == "__main__":
    falcon512 = ModFalcon(d=512, n=1, target_bitsec=128)
    print(falcon512)

    falcon1024 = ModFalcon(d=1024, n=1, target_bitsec=256)
    print(falcon1024)

    modfalcon_2_512 = ModFalcon(d=512, n=2, target_bitsec=192)
    print(modfalcon_2_512)