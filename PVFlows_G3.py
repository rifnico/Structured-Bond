# PVFlows.py

import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from Bootstrap.FinDates.daycount import yearfrac


def equity_linked_coupon_pv(
    r_base: pd.Series, start_date: pd.Timestamp,
    principal: float = 100e6,
    S0_1: float = 100, S0_2: float = 200,
    sigma1: float = .162, sigma2: float = .2,
    d1: float = .025, d2: float = 0.029,
    rho: float = .45,
    T: int = 4, alpha: float = 0.9,
    n_paths: int = 100_000, seed: int = 42,
    dc: str = "ACT/360"
) -> tuple[float, float, float]:
    """
    Monte Carlo pricing of the equity-linked payoff:
    alpha * N * max((0.5 * (S1_T/S0_1 + S2_T/S0_2) - 1), 0)
    """
    maturity_date = start_date + pd.DateOffset(years=T)
    zero_rate_T = r_base.loc[maturity_date]
    #print(f"Maturity Date: {maturity_date}, Zero Rate at Maturity: {zero_rate_T}")

    tau = yearfrac(start_date, maturity_date, dc)
    df_T = np.exp(-zero_rate_T * tau)
    #print(f"Tau: {tau}, Discount Factor: {df_T}")

    np.random.seed(seed)
    cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov_matrix)
    Z = np.random.normal(size=(2, n_paths))
    Z1, Z2 = (L @ Z)

    drift1 = (zero_rate_T - d1 - 0.5 * sigma1**2) * tau
    drift2 = (zero_rate_T - d2 - 0.5 * sigma2**2) * tau

    S1_T = S0_1 * np.exp(drift1 + sigma1 * np.sqrt(tau) * Z1)
    S2_T = S0_2 * np.exp(drift2 + sigma2 * np.sqrt(tau) * Z2)

    avg_return = 0.5 * (S1_T / 100 + S2_T / 200)
    payoff = alpha * np.maximum(avg_return - 1, 0)

    price = principal * df_T * np.mean(payoff)
    S1_mean = np.mean(S1_T)
    S2_mean = np.mean(S2_T)

    return price, S1_mean, S2_mean



def floating_leg_pv(
    curve: pd.Series,
    start_date: pd.Timestamp,
    principal: float = 100e6,
    spread: float = 0.03,
    n_quarters: int = 16,
    dc: str = "ACT/360"
) -> float:
    """
    PV of floating leg: Euribor 3M + spread, paid quarterly.
    Discount factors: P(t) = 1 / (1 + tau * r(t)), with r(t) interpolated.
    """
    zero_curve = curve.sort_index()

    # Payment schedule (quarterly)
    schedule = pd.DatetimeIndex([start_date + DateOffset(months=3 * i)
                                 for i in range(1, n_quarters + 1)])

    # Interpolate zero rates to all schedule dates
    zc_full = zero_curve.reindex(zero_curve.index.union(schedule)).sort_index()
    zc_full = zc_full.interpolate(method='time')
    r_t = zc_full.loc[schedule]

    # Compute taus and discount factors
    taus = np.array([yearfrac(start_date, d, dc) for d in schedule])
    P_t = np.exp(-r_t.values * taus) # type: ignore

    #print(f'Discount Factors: {[f"{x*100:,.2f} %" for x in P_t]}')

    # Compute forward rates using consecutive discount factors

    P_prev = np.concatenate([[1.0], P_t[:-1]])
    full_schedule = [start_date] + list(schedule)
    taus_period = np.array([yearfrac(full_schedule[i], full_schedule[i+1], dc) for i in range(n_quarters)])
    fwds = (P_prev / P_t - 1.0) / taus_period
     
    #print(f'Fordward Rates: {[f"{x*100:,.2f} %" for x in fwds]}')

    # PV calculation (excluding principal at the end)
    pv_coupon = principal * np.sum((fwds + spread) * taus_period * P_t)

    return pv_coupon  

