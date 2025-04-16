import numpy as np

import Volatility as Vol
import Correlation as Corr
from MarketData import *


class LiborMarketModel(Vol.Volatility, Corr.Correlation):
    def __init__(self):
        super().__init__()
        Corr.Correlation.__init__(self)

    def Rebonato_formula(self, w, F_star, T_alpha):
        # Rebonato volatility
        result = 0
        for i in range(len(w)):
            for j in range(len(w)):
                # result += F_star[i] * F_star[j] * rho_func(T_alpha + i, T_alpha + j) * psi[T_alpha + i - 1] * psi[T_alpha + j - 1] * I(T_alpha, 1, 1)
                T_k, T_m = T_alpha + i, T_alpha + j

                if i == j:
                    result += F_star[i] * F_star[j] * self.rho(T_alpha + i, T_alpha + j) * self.psi[T_alpha + i - 1] * self.psi[
                        T_alpha + j - 1] * self.I(T_alpha, T_k, T_m)
                else:
                    result += 2 * F_star[i] * F_star[j] * self.rho(T_alpha + i, T_alpha + j) * self.psi[T_alpha + i - 1] * \
                              self.psi[T_alpha + j - 1] * self.I(T_alpha, T_k, T_m)

        return result

    def calibration(self, params):
        if self.v_type == 'CV':
            self.beta_1, self.beta_2, self.beta_3 = params[0], params[1], params[2]
        else:
            self.beta_1, self.beta_2, self.beta_3 = params[0], params[1], params[2]
            self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4 = params[3], params[4], params[5], params[6]

        model_vol, market_vol, errors = [], [], []

        for a in range(upper_triangle.shape[0]):
            for b in range(upper_triangle.shape[1] - a):
                T_alpha, T_beta = opt_tenor[a], swap_length[b]

                w, F, swap_rate, annuity = Vol.calc_forward_swap_rate(T_alpha, T_beta)
                F_star = w * F

                # Rebonato volatility
                result = self.Rebonato_formula(w, F_star, T_alpha)

                # Model Vol vs Market Vol
                model_vol.append(np.sqrt(result / (T_alpha * swap_rate ** 2)))
                market_vol.append(upper_triangle[a, b])
                errors.append(abs(model_vol[-1] - market_vol[-1]) / market_vol[-1])

        return np.sum(errors)

    def calibration_lsq(self, params):
        if self.v_type == 'CV':
            self.beta_1, self.beta_2, self.beta_3 = params[0], params[1], params[2]
        else:
            self.beta_1, self.beta_2, self.beta_3 = params[0], params[1], params[2]
            self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4 = params[3], params[4], params[5], params[6]

        model_vol, market_vol, errors = [], [], []

        for a in range(upper_triangle.shape[0]):
            for b in range(upper_triangle.shape[1] - a):
                T_alpha, T_beta = opt_tenor[a], swap_length[b]

                w, F, swap_rate, annuity = Vol.calc_forward_swap_rate(T_alpha, T_beta)
                F_star = w * F

                # Rebonato volatility
                result = self.Rebonato_formula(w, F_star, T_alpha)

                # Model Vol vs Market Vol
                model_vol.append(np.sqrt(result / (T_alpha * swap_rate ** 2)))
                market_vol.append(upper_triangle[a, b])
                errors.append(abs(model_vol[-1] - market_vol[-1]) / market_vol[-1])

        return errors

    def calibration_lmfit(self, params):
        if self.v_type == 'CV':
            self.beta_1, self.beta_2, self.beta_3 = params['beta_1'].value, params['beta_2'].value, params['beta_3'].value
        else:
            self.beta_1, self.beta_2, self.beta_3 = params[0], params[1], params[2]
            self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4 = params[3], params[4], params[5], params[6]

        model_vol, market_vol, residuals = [], [], []
        for a in range(upper_triangle.shape[0]):
            for b in range(upper_triangle.shape[1] - a):
                T_alpha, T_beta = opt_tenor[a], swap_length[b]

                w, F, swap_rate, annuity = Vol.calc_forward_swap_rate(T_alpha, T_beta)
                F_star = w * F

                # Rebonato volatility
                result = self.Rebonato_formula(w, F_star, T_alpha)

                # Model Vol vs Market Vol
                model_vol.append(np.sqrt(result / (T_alpha * swap_rate ** 2)))
                market_vol.append(upper_triangle[a, b])
                residuals.append(abs(model_vol[-1] - market_vol[-1]) * 100)
                # errors.append((model_vol[-1] - market_vol[-1])**2)

        # residuals = np.array(model_vol) - np.array(market_vol)
        # mse = np.mean(errors)
        # penalty = 0
        # if self.beta_2 < 0 or self.beta_2 > 3 * self.beta_1:
        #     penalty += 1e6 * (max(0, -self.beta_2) + max(0, self.beta_2 - 3 * self.beta_1))**2
        #
        # if -3 * np.log(self.beta_3) < self.beta_1 + self.beta_2:
        #     penalty += 1e6 * (self.beta_1 + self.beta_2 + 3 * np.log(self.beta_3)) ** 2
        #
        # if penalty > 1e-5:
        #     residuals = np.append(residuals, np.sqrt(penalty) * np.ones(len(residuals)))
        penalty_residuals = np.zeros(2)
        penalty_scale = 1e4
        penalty_residuals[0] = penalty_scale * (max(0, -self.beta_2) + max(0, self.beta_2 - 3 * self.beta_1))
        penalty_residuals[1] = penalty_scale * max(0, self.beta_1 + self.beta_2 + 3 * np.log(self.beta_3))

        residuals = np.concatenate((residuals, penalty_residuals))

        return residuals

    def calc_volatility_instruments(self):
        model_vol, market_vol = np.zeros(v_swptn_mkt.shape), np.zeros(v_swptn_mkt.shape)

        for a in range(v_swptn_mkt.shape[0]):
            for b in range(v_swptn_mkt.shape[1] - a):
                T_alpha, T_beta = opt_tenor[a], swap_length[b]

                w, F, swap_rate, annuity = Vol.calc_forward_swap_rate(T_alpha, T_beta)
                F_star = w * F

                # Rebonato volatility
                result = self.Rebonato_formula(w, F_star, T_alpha)
                model_vol[a, b] = np.sqrt(result / (T_alpha * swap_rate ** 2))
                market_vol[a, b] = v_swptn_mkt[a, b]

        return model_vol, market_vol