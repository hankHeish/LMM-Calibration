import numpy as np
import plotly.graph_objects as go

from scipy.optimize import minimize, least_squares
# from lmfit import minimize, Parameters
from scipy.integrate import quad

import lmfit
import LiborMarketModel as LMM
import Volatility as Vol
from MarketData import *


def plot_corr_surface(X, Y, z):
    # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=z, x=X, y=Y)])

    # Customize the layout (optional)
    fig.update_layout(title='3D Surface Plot',
                      scene=dict(xaxis_title='Maturity',
                                 yaxis_title='Tenor',
                                 zaxis_title='Correlation'))

    # Show the plot
    fig.show()


def plot_volatility_surface(z1=None, z2=None):
    X, Y = np.meshgrid(opt_tenor, swap_length)
    # Z1, Z2 = upper_triangle, model_vol

    # Create the figure and add the first surface
    fig = go.Figure(data=[go.Surface(z=z1, x=X, y=Y, opacity=0.8,
                                     colorscale='Viridis',
                                     name='Market Vol',
                                     hovertemplate='Maturity: %{x}<br>Tenor: %{y}<br>Market Vol: %{z:.2f}<extra></extra>')])

    # Add the second surface
    fig.add_trace(go.Surface(z=z2, x=X, y=Y, opacity=0.8,
                             colorscale='Plasma',
                             name='Model Vol',
                             hovertemplate='Maturity: %{x}<br>Tenor: %{y}<br>Model Vol: %{z:.2f}<extra></extra>'))

    # Update the layout (optional)
    fig.update_layout(
        title='Model Vol vs Market Vol (Upper Triangle, without Co-Terminal)',
        scene=dict(
            xaxis_title='Maturity',
            yaxis_title='Tenor',
            zaxis_title='Vol',
        ),
        width=800,
        height=800,
    )

    # Show the plot
    fig.show()

def constraint_1(params):
    beta_1, beta_2, beta_3 = params[0], params[1], params[2]

    return 3 * beta_1 - beta_2


def constraint_2(params):
    beta_1, beta_2, beta_3 = params[0], params[1], params[2]

    return beta_2


def constraint_3(params):
    beta_1, beta_2, beta_3 = params[0], params[1], params[2]

    return beta_1 + beta_2


def constraint_4(params):
    beta_1, beta_2, beta_3 = params[0], params[1], params[2]

    return -3 * np.log(beta_3) - (beta_1 + beta_2)


constraints = [
    {'type': 'ineq', 'fun': constraint_1},  # 3*beta_1 - beta_2 >= 0
    {'type': 'ineq', 'fun': constraint_2},  # beta_2 >= 0
    {'type': 'ineq', 'fun': constraint_3},  # beta_1 + beta_2 >= 0
    {'type': 'ineq', 'fun': constraint_4},  # -ln(beta_3) - (beta_1 + beta_2) >= 0
]

bounds = [(0, 1.0), (0, 1.0), (1e-5, 1.0)]
bounds_ttv = [(0, 1.0), (0, 1.0), (1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0), (1e-5, 1.0)]

lower_bounds = [1e-8, 1e-8, 1e-8]
# upper_bounds = [np.inf, np.inf, np.inf]
upper_bounds = [1.0, 1.0, 1.0]

if __name__ == '__main__':
    clsLMM = LMM.LiborMarketModel()

    # LMM calibration
    errors = []
    clsLMM.Diagonal_Recursive_Calibration(co_terminal)

    # if clsLMM.v_type == 'CV':
    #     params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3]
    # else:
    #     params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3, clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4]
    #
    # error = clsLMM.calibration(params)
    # errors.append(error)
    #
    # # before calibration
    # model_vol, market_vol = clsLMM.calc_volatility_instruments()
    # plot_volatility_surface(market_vol, model_vol)
    #
    # for i in range(15):
    #     if clsLMM.v_type == 'CV':
    #         params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3]
    #
    #         result = minimize(clsLMM.calibration, np.array(params),
    #                           bounds=bounds, constraints=constraints,
    #                           method='SLSQP')
    #         # result = least_squares(clsLMM.calibration, params, bounds=(lower_bounds, upper_bounds),
    #         #                        method='trf')
    #
    #         clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3 = result.x
    #
    #     else:
    #         params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3,
    #                   clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4]
    #
    #         result = minimize(clsLMM.calibration, np.array(params),
    #                           bounds=bounds_ttv, constraints=constraints,
    #                           method='SLSQP')
    #
    #         clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3, clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4 = result.x
    #
    #     params = result.x
    #     clsLMM.generate_corr_surface()
    #     error = clsLMM.calibration(params)
    #
    #     clsLMM.Diagonal_Recursive_Calibration(co_terminal)
    #
    #     if error > errors[-1]:
    #         errors.append(error)
    #         break
    #
    #     errors.append(error)
    #
    # print(errors)
    #
    # # well-calibrated correlation surface
    # clsLMM.generate_corr_surface()
    # plot_corr_surface(opt_tenor, swap_length, clsLMM.corr_surf)
    #
    # # after calibration
    # if clsLMM.v_type == 'CV':
    #     print('beta: ', clsLMM.beta_1, '\t', clsLMM.beta_2, '\t', clsLMM.beta_3)
    # else:
    #     print('beta: ', clsLMM.beta_1, '\t', clsLMM.beta_2, '\t', clsLMM.beta_3)
    #     print('alpha: ', clsLMM.alpha_1, '\t', clsLMM.alpha_2, '\t', clsLMM.alpha_3, '\t', clsLMM.alpha_4)
    #
    # model_vol, market_vol = clsLMM.calc_volatility_instruments()
    # plot_volatility_surface(market_vol, model_vol)

    if clsLMM.v_type == 'CV':
        # params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3]
        params = lmfit.Parameters()
        params.add('beta_1', value=clsLMM.beta_1, min=1e-10, max=1.0)
        params.add('beta_2', value=clsLMM.beta_2, min=1e-10, max=1.0)
        params.add('beta_3', value=clsLMM.beta_3, min=1e-10, max=1.0)
    else:
        params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3, clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4]

    error = clsLMM.calibration_lmfit(params)
    errors.append(np.sum(error))

    # before calibration
    model_vol, market_vol = clsLMM.calc_volatility_instruments()
    plot_volatility_surface(market_vol, model_vol)

    for i in range(15):
        if clsLMM.v_type == 'CV':
            # params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3]
            #
            # result = minimize(clsLMM.calibration, np.array(params),
            #                   bounds=bounds, constraints=constraints,
            #                   method='SLSQP')
            # result = least_squares(clsLMM.calibration, params, bounds=(lower_bounds, upper_bounds),
            #                        method='trf')
            result = lmfit.minimize(clsLMM.calibration_lmfit, params, method='leastsq')

            clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3 = result.params['beta_1'].value, result.params['beta_2'].value, result.params['beta_3'].value

        else:
            params = [clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3,
                      clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4]

            result = minimize(clsLMM.calibration, np.array(params),
                              bounds=bounds_ttv, constraints=constraints,
                              method='SLSQP')

            clsLMM.beta_1, clsLMM.beta_2, clsLMM.beta_3, clsLMM.alpha_1, clsLMM.alpha_2, clsLMM.alpha_3, clsLMM.alpha_4 = result.x

        # params = result.x
        # params = [result.params['beta_1'].value, result.params['beta_2'].value, result.params['beta_3'].value]
        params['beta_1'].value, params['beta_2'].value, params['beta_3'].value = result.params['beta_1'].value, result.params['beta_2'].value, result.params['beta_3'].value
        clsLMM.generate_corr_surface()
        error = clsLMM.calibration_lmfit(params)
        error = np.sum(error)
        print(f'iter: {i + 1}, error: {error}')

        clsLMM.Diagonal_Recursive_Calibration(co_terminal)

        if error > errors[-1]:
            errors.append(error)
            break

        errors.append(error)

    print(errors)

    # well-calibrated correlation surface
    clsLMM.generate_corr_surface()
    plot_corr_surface(opt_tenor, swap_length, clsLMM.corr_surf)

    # after calibration
    if clsLMM.v_type == 'CV':
        print('beta: ', clsLMM.beta_1, '\t', clsLMM.beta_2, '\t', clsLMM.beta_3)
    else:
        print('beta: ', clsLMM.beta_1, '\t', clsLMM.beta_2, '\t', clsLMM.beta_3)
        print('alpha: ', clsLMM.alpha_1, '\t', clsLMM.alpha_2, '\t', clsLMM.alpha_3, '\t', clsLMM.alpha_4)

    model_vol, market_vol = clsLMM.calc_volatility_instruments()
    plot_volatility_surface(market_vol, model_vol)