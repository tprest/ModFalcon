"""
Profile the code with:
> make profile
"""
from test import *

if __name__ == "__main__":
    # test_ffnp(128, 3, 10)
    test_module_ntru_gen(256, 3, 1)