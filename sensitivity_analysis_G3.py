# sensitivity_analysis_G3.py
import pandas as pd
import numpy as np
from PVFlows_G3 import floating_leg_pv, equity_linked_coupon_pv
from typing import Literal
from Bootstrap.FinDates.daycount import yearfrac

def compute_dv01(
    start_date: pd.Timestamp,    r_base: pd.Series,    r_bumped: pd.Series,
    principal: float = 100e6,
    S0_1: float = 100,    S0_2: float = 200,
    sigma1: float = 0.162,    sigma2: float = 0.2,
    d1: float = 0.025,    d2: float = 0.029,
    rho: float = 0.45,    T: int = 4,    alpha: float = 0.9
) -> float:   

    PV_eq_base = equity_linked_coupon_pv(r_base,start_date,principal,S0_1,S0_2,sigma1,sigma2,d1,d2,rho,T,alpha)[0]
    PV_eq_bump = equity_linked_coupon_pv(r_bumped,start_date,principal,S0_1,S0_2,sigma1,sigma2,d1,d2,rho,T,alpha)[0]

    PV_float_base = floating_leg_pv(start_date=start_date, curve=r_base, principal=principal)
    PV_float_bump = floating_leg_pv(start_date=start_date, curve=r_bumped, principal=principal)

    delta_eq = PV_eq_bump - PV_eq_base
    delta_float = PV_float_bump - PV_float_base

    return delta_eq - delta_float

def compute_delta(stock, r_base, start_date, principal, S01, S02,bump):
    if stock == "ENEL":
        cupon_base=equity_linked_coupon_pv(r_base, start_date, principal)[0]
        cupon_bump=equity_linked_coupon_pv(r_base, start_date,principal, S01+bump)[0]
        
    elif stock == "AXA":
        cupon_base=equity_linked_coupon_pv(r_base, start_date, principal)[0]
        cupon_bump=equity_linked_coupon_pv(r_base, start_date,principal, S01, S02+bump)[0]
        
    delta= (cupon_bump-cupon_base)/bump # type: ignore
    return delta



