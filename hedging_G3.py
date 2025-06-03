import sensitivity_analysis_G3 as sens
import PVFlows_G3 as flows
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from Bootstrap.FinDates.daycount import yearfrac

######################################
#this function will create the deltas of each stock, simulating by montecarlo, one with initial price of
#S0, and the other one with S0+epsilon (small number), on the denominator we will have epsilon.
#before this, we have to make sure that the function flows. equity_linkedcoupon returns the values of
#S1 and S2 at maturity through the simulation.

def deltastock(stock,r,T,start_date, epsilon):
    maturity_date = start_date + pd.DateOffset(years=T)
    zero_rate_T = r.loc[maturity_date]

    if stock == "ENEL":
        S1 = flows.equity_linked_coupon_pv(r, start_date)[1]  
        S1e = flows.equity_linked_coupon_pv(r,start_date,100e6,100+epsilon)[1] 
        delta = (S1e-S1)/epsilon
        delta = np.exp(-zero_rate_T * T)*delta

    elif stock == "AXA":
        S2 = flows.equity_linked_coupon_pv(r,start_date)[2]  
        S2e = flows.equity_linked_coupon_pv(r,start_date,100e6,100,200+epsilon)[2] 
        delta = (S2e-S2)/epsilon
        delta = np.exp(-zero_rate_T * T)*delta
 
    else:
        print ("Wrong stock selected")
    return delta # type: ignore

#######################################
#this functions will return the number of stocks to acquire on the short position to hedge for the
#corresponding delta.
#deltastock is the sensitivity of the stock price at maturity (AXA or ENEL). We asume it positive (taking its module)
#deltacontract is the delta obtained on item b
#######################################
def hedgedelta(deltastock, deltacontract):
    num_stocks= deltacontract/deltastock
    return num_stocks

#######################################
#this function will calculate the product of S(fixed rate paid)*delta, that is, the fixed payment with a notional=1.
#B should be a vector containing the discount factors from years 0,1,2,3,4.
# I assume B[0]=B(to,to), B[1]=B(to,t1), etc
#delta should be the year fraction with the corresponding convention: 30/360

def fixedcashflows (P_t):
    numerator = 1 - P_t[-1]
    denominator =  np.sum(P_t)
    par_swap_rate = numerator / denominator
    cf = par_swap_rate   # fixed flow of the period
    return cf

def floating_leg(curve: pd.Series, start_date: pd.Timestamp, n_quarters: int = 16,dc: str = "ACT/360"):
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

    # Compute forward rates using consecutive discount factors

    P_prev = np.concatenate([[1.0], P_t[:-1]])
    full_schedule = [start_date] + list(schedule)
    taus_period = np.array([yearfrac(full_schedule[i], full_schedule[i+1], dc) for i in range(n_quarters)])
    fwds = (P_prev / P_t - 1.0) / taus_period

    return fwds

    

def hedgeDV01 (zero_curve, zero_curve_bumped, floating,floating_shifted, dv01, start_date, n_quarters: int = 16,dc: str = "ACT/360"):
    # Compute taus and discount factors for base case
    schedule = pd.DatetimeIndex([start_date + DateOffset(months=3 * i)
                                 for i in range(1, n_quarters + 1)])

    zc_full = zero_curve.reindex(zero_curve.index.union(schedule)).sort_index()
    zc_full = zc_full.interpolate(method='time')
    r_t = zc_full.loc[schedule]
    taus = np.array([yearfrac(start_date, d, dc) for d in schedule])
    P_t = np.exp(-r_t.values * taus)


    # Compute taus and discount factors with shift
    zc_full_shifted = zero_curve_bumped.reindex(zero_curve.index.union(schedule)).sort_index()
    zc_full_shifted = zc_full_shifted.interpolate(method='time')
    r_t_shifted = zc_full_shifted.loc[schedule]
    taus = np.array([yearfrac(start_date, d, dc) for d in schedule])
    P_t_shifted = np.exp(-r_t_shifted.values * taus)


    dv01_floating=np.sum(floating_shifted*P_t_shifted-floating*P_t)
    dv01_fixed=np.sum(fixedcashflows(P_t)*(P_t_shifted)-fixedcashflows(P_t)*(P_t_shifted))
    DV01_swap_per_eur = dv01_floating-dv01_fixed
    hedge_notional = abs(dv01) / DV01_swap_per_eur

    return hedge_notional, DV01_swap_per_eur

