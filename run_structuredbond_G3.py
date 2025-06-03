import sys
from PVFlows_G3 import floating_leg_pv, equity_linked_coupon_pv
from sensitivity_analysis_G3 import compute_dv01, compute_delta
from pandas.tseries.offsets import DateOffset
from hedging_G3 import hedgedelta, hedgeDV01, deltastock, fixedcashflows, floating_leg
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from Bootstrap.FinDates.daycount import yearfrac

if __name__ == "__main__":

    #Inputs of the contract
    T = 4
    S0_1 = 100 #ENEL
    S0_2 = 200 #AXA
    sigma1 = 0.162
    sigma2 = 0.20
    d1 = 0.025
    d2 = 0.029
    rho = 0.45
    alpha = 0.9
    principal = 100e6
    spread = 0.03 #over libor
    bump_ratio = 0.000001

    if len(sys.argv) > 1:
        start_date = pd.to_datetime(sys.argv[1])
    else:
        start_date = pd.to_datetime("2023-02-02")

    #Load curves and interpolate with payment dates, creation of dt
    #Base Curve
    maturity_date = start_date + DateOffset(years=4)
    zero_curve = pd.read_csv("Bootstrap/zero_curve.csv",index_col=0,parse_dates=True)
    zero_curve = zero_curve.iloc[:, 0] if isinstance(zero_curve, pd.DataFrame) else zero_curve   
    new_index = zero_curve.index.union([start_date, maturity_date])
    zero_curve = zero_curve.reindex(new_index).interpolate(method="time").bfill().ffill() 
    r_base = zero_curve

    #Bumped Curve
    zero_curve_bumped = pd.read_csv("Bootstrap/zero_curve_bumped.csv", index_col=0, parse_dates=True)
    zero_curve_bumped = zero_curve_bumped.iloc[:, 0] if isinstance(zero_curve_bumped, pd.DataFrame) else zero_curve_bumped
    new_index_bumped = zero_curve_bumped.index.union([start_date,maturity_date])
    zero_curve_bumped = zero_curve_bumped.reindex(new_index_bumped).interpolate(method="time").bfill().ffill() 
    r_bumped = zero_curve_bumped

    #Compute the X % based on the PV of the two coupons: Linked to libor + spread & Equity Linked Coupon
    pv_equity_coupon = equity_linked_coupon_pv(r_base,start_date, principal,S0_1, S0_2, sigma1, sigma2,d1,d2, rho, T,alpha)[0]   
    pv_floating = floating_leg_pv(r_base, start_date, principal,spread, int(T * 4))
    x_pct = (pv_floating - pv_equity_coupon) / principal

    #Print results X%
    print(f"Valuation Date: {start_date.date()}")
    print(f"Maturity Date: {maturity_date.date()}")
    print(f"4Y Zero Rate: {r_base.loc[maturity_date]:.3%}")
    print(f"PV of Equity-Linked Coupon: {pv_equity_coupon:,.2f} EUR")
    print(f"PV of Floating Leg: {pv_floating:,.2f} EUR")
    print(f"Fair Upfront X%: {x_pct:.4%}")

    #Compute Sensitivity
    # delta in equity move and dv01 with the zero rates shifted from the bootstrap
    delta1=compute_delta("ENEL", r_base, start_date, principal, 100, 200, bump_ratio * S0_1)
    delta2=compute_delta("AXA", r_base, start_date, principal, 100, 200, bump_ratio * S0_2)
    dv01 = compute_dv01(start_date, r_base, r_bumped, principal, S0_1, S0_2, sigma1, sigma2, d1, d2, rho, T, alpha)

    #Print results sentivity
    print(f"ENEL → delta={delta1:.3f} EUR ")
    print(f"AXA  → delta={delta2:.3f} EUR ")
    print(f"DV01 : {dv01:,.2f} EUR per 1 bp rate shift")


    #Compute the hedging

    # Delta per stock
    delta_enel = deltastock("ENEL", r_base, 4, start_date, bump_ratio * S0_1)
    delta_axa = deltastock("AXA", r_base, 4, start_date, bump_ratio * S0_2)

    # Hedge for stocks
    short_enel = hedgedelta(delta_enel, delta1)
    print(f"We have to short {short_enel:.2f} shares of ENEL")

    short_axa = hedgedelta(delta_axa, delta2)
    print(f"We have to short {short_axa:.2f} shares of AXA")

    # DV01 Hedge
    floating = floating_leg(r_base, start_date)
    floating_shifted = floating_leg(r_bumped, start_date)

    notional, dv01_swap = hedgeDV01(r_base, r_bumped, floating, floating_shifted, dv01, start_date)
    print(f"The notional required to hedge the DV01 is {notional:.2f}")


