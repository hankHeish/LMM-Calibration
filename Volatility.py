import numpy as np
from scipy.integrate import quad

from MarketData import *

def calc_forward_swap_rate(T_alpha, T_beta):
    start, end = T_alpha, T_alpha + T_beta
    w, F = [], []
    annuity = 0

    for i in range(T_alpha + 1, T_alpha + T_beta + 1):
        w.append(tau * discount_factor[i - 2])
        F.append(forwards[i - 2])
        annuity += tau * discount_factor[i - 2]

    w = w / annuity
    swap_rate = np.sum(w * F)
    # swap_rate = (discount_factor[start - 2] - discount_factor[end - 2]) / annuity

    return w, F, swap_rate, annuity

class Volatility:
    def __init__(self):
        self.alpha_1 = 0.0285
        self.alpha_2 = 0.057
        self.alpha_3 = 0.20004
        self.alpha_4 = 0.11

        self.v_type = 'CV'
        self.psi = np.zeros(len(forwards))

    def I(self, T_x, T_k, T_m):
        result = 0

        if self.v_type == 'TVV':
            v = lambda t: ((self.alpha_1 * (T_k - t) + self.alpha_2) * np.exp(-(T_k - t) * self.alpha_3) + self.alpha_4) * ((self.alpha_1 * (T_m - t) + self.alpha_2) * np.exp(-(T_m - t) * self.alpha_3) + self.alpha_4)
            # v_m = lambda t: (self.alpha_1 * (T_m - t) + self.alpha_2) * np.exp(-(T_m - t) * self.alpha_3) + self.alpha_4

            result = quad(v, 0, T_x)[0]

        elif self.v_type == 'CV':
            result = T_x

        return result

    def Diagonal_Recursive_Calibration(self, co_terminal):

        for i in range(M):
            T_alpha, T_beta = M - i, i + 1
            v_mkt = co_terminal[i]

            w, F, swap_rate, annuity = calc_forward_swap_rate(T_alpha, T_beta)
            F_star = w * F

            # iterative calculate for co-terminal swaptions
            start, end = T_alpha - 1, T_alpha + T_beta - 1

            c1, c2, c3 = 0, 0, 0
            c1 = F_star[0] ** 2 * self.I(opt_tenor[start], T_alpha, T_alpha)

            for k in range(1, len(w)):
                # c2 += F_star[k] * psi[-k] * rho_func(start, start + k) * I(opt_tenor[start], 1, 1)
                c2 += F_star[k] * self.psi[-k] * self.rho(start, start + k) * self.I(opt_tenor[start], start, start + k)
            c2 = 2 * F_star[0] * c2

            for k in range(1, len(w)):
                for l in range(1, len(w)):

                    c3 += F_star[k] * F_star[l] * self.rho(start + k, start + l) * self.psi[-k] * self.psi[-l] * self.I(
                        opt_tenor[start], start + k, start + l)
            c3 = c3 - T_alpha * swap_rate ** 2 * v_mkt ** 2

            self.psi[M - 1 - i] = (-c2 + np.sqrt(c2 ** 2 - 4 * c1 * c3)) / (2 * c1)

            # print(psi)

        return self

