import numpy as np

from MarketData import *

class Correlation:
    def __init__(self):
        self.beta_1 = 1
        self.beta_2 = 2
        self.beta_3 = 3

        self.M = len(forwards)
        self.corr_surf = np.zeros((v_swptn_mkt.shape))

        self.init_correlation_variables()

    def init_correlation_variables(self):
        while True:
            self.beta_1, self.beta_2 = np.random.rand(1, 2)[0]

            if 3 * self.beta_1 >= self.beta_2:
                break

        # Generate beta_3
        self.beta_3 = np.exp(np.random.uniform(-(self.beta_1 + self.beta_2), 0))

        return self

    def rho(self, i, j):
        return np.exp(-abs(j - i) / (self.M - 1) * (-np.log(self.beta_3) + self.beta_1 * (
                i**2 + j**2 + i * j - 3 * self.M * i - 3 * self.M * j + 3 * i + 3 * j + 2 * self.M**2 - self.M - 4) /
                                                    ((self.M - 2) * (self.M - 3)) - self.beta_2 * (i**2 + j**2 + i * j -
                                                                                                   self.M * i - self.M * j - 3 * i - 3 * j + 3 * self.M**2 - 2) /
                                                    ((M - 2) * (self.M - 3))))

    def generate_corr_surface(self):
        for a in range(len(opt_tenor)):
            for b in range(len(swap_length)):
                T_a, T_b = opt_tenor[a], swap_length[b]
                self.corr_surf[a, b] = self.rho(T_a, T_b)