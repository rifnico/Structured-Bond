# Import what you need...
import numpy as np
import pandas as pd  
# The function for dates handling.
from FinDates.daycount import yearfrac, yearfractions
import Utilities as utils 

# These functions are used to convert the data in csv files into what I nedd
depo_converter = lambda x: float(x)/100
future_converter = lambda x: 1-float(x)/100
swap_converter = lambda x: float(x)/100


# Dictionary of day count convenction: better to have a mapping here that directly into the code!
DC_CONV = {"DEPO": "ACT/360"
          , "FUTURE": "ACT/360"
    , "SWAP" : "30E/360"
    , "BOND" : "30E/360"
    , "INTERP" : "ACT/365 FIXED"}

# This function get the zero rates!
def getZeroRates(dates, df):
    effDates, effDf = dates, df
    if isinstance(effDates, pd.DatetimeIndex): 
        effDates = list(yearfractions(list(dates), DC_CONV["INTERP"]))[1:]
        effDf = df[1:]
        print(list(-np.log(effDf)/effDates))
    return list(-np.log(effDf)/effDates)

# This function, interpolates!!
def getRatesLinInterpDiscount(dtSettle, dtRef, xDates, xDf, daycount=DC_CONV["INTERP"]):
# xDates, xDf : available set of dates/discounts on which interpolate, it contains today date-1
# returns discounts on dtRef dates
    assert(len(xDates) == len(xDf))
    yearFracxDates = [yearfrac(dtSettle, T, daycount) for T in xDates[1:]]  
    xRates = getZeroRates(yearFracxDates, xDf[1:])
    yearFracRef = yearfrac(dtSettle, dtRef, daycount)
    rate = np.interp(yearFracRef, yearFracxDates, xRates)
    return np.exp(-yearFracRef*rate)

def extract_mid_rates(df_futures: pd.DataFrame,
    bid_col: str = 'BID',
    ask_col: str = 'ASK',
    bump: float = 0.0
) -> pd.Series:
    """
    Compute the mid rates from bid and ask columns in a futures DataFrame.
    Returns
    pd.Series
        The mid rates for each row, with bump applied.
    """

    mid_rates = df_futures[[bid_col, ask_col]].mean(axis=1)
    return mid_rates + bump

# This function compute the Bootstrap using Depos
def bootstrapDepo(dtSettle, df_depo, df_futures, termDates, discounts,bump = 0.0):
    # NOTE: bump=0.0 in for sensitivity: that happens if the depo are 0.01 higher on -0.01 lower than now? Very useful!

    # Select only some depos, up the the first date of futures indexes
    iDepo = (df_depo.index <= df_futures.index[0]).sum() +1   # count of how many depo rates I need
    
    # Get the selected depo: be carefull when you use dataframe in python: sometimes they are "by reference" and not "by value".
    depoSelected = df_depo[:iDepo].copy()
    # Get the depo dates
    depoDates = depoSelected.index
    # Use "list comprehension" in python to compute the year fract
    depoYearFrac = [yearfrac(dtSettle, t, DC_CONV["DEPO"]) for t in depoDates]
    depoMidRates = extract_mid_rates(depoSelected, bump= bump)
    # This is the formula to use to compute the discount
    depoDiscounts = list(1./(1. + depoYearFrac* depoMidRates))   # B(t0,t1) = 1 /(1+delta(t0,t1)*L(t0,t1))

    # Update the dates and the discount by adding stuff 
    termDates += depoDates.tolist() 
    discounts += depoDiscounts

    # Return the dates and the discount (i.e. the first part of the discount curve)
    return termDates, discounts


def bootstrapFuture(dtSettle, df_futures, termDates, discounts, bump = 0.0):
    # Bootstrap from Futures!
    iFutures = 7 # Take the first 7 futures: actually it is a better idea do no hard code this value here, but prefilted outside!
    
    # Select the future to use: only the first seven!
    futuresSelected = df_futures[:iFutures].copy()
    # Compute the Year Fract: UHM, DC_CONV["DEPO"] or DC_CONV["FURURES"]? I guess the second one!
    futuresYearFrac = [yearfrac(rowFut.Settle, rowFut.Expiry, DC_CONV["DEPO"])  for t,rowFut in
                       futuresSelected.iterrows()]
    # Compute the mid rate, since you have the BID and the ASK price
    futuresMidRates = extract_mid_rates(futuresSelected, bump=bump)
    
    # Coompute the future, i.e. use the formula from the slides
    futuresSelected["Fwd"] = (1./(1. + futuresYearFrac * futuresMidRates))

    # Loop and compute the discount
    for t, rowFut in futuresSelected.iterrows():
        futuresYearFrac # type: ignore
        # Linear interpolation
        startDisc = getRatesLinInterpDiscount(dtSettle, rowFut.Settle, termDates, discounts) #t0, t1, ti 
        # update dates and discount in the discount curve!
        termDates += [rowFut.Expiry]
        discounts += [rowFut.Fwd*startDisc]   #B(t0,t2)=B(t

    # Return the updated curve: observe that termDates, discounts have been passed as parameters from the code, so that they
    # contain also discount curve that you have computed starting from DEPOS.
    return termDates, discounts

def bootstrapSwap(dtSettle, df_swaps, termDates, discounts, bump = 0.0):
    # Select only those swaps with termdates higher that the last data you have in the constructed discount curve.
    iSwaps = (df_swaps.index >= termDates[-1]).sum() 
    # Selected Swaps
    swapsSelected = df_swaps[-iSwaps:].copy()
    # Again, take the average!!
    swapsSelected["Mid"] = extract_mid_rates(swapsSelected,bump=bump)
    # Consider 1Y 
    prevSwapDate = df_swaps.index[0] 
    # Compute the year fraction
    swapYearFrac = [yearfrac(dtSettle, prevSwapDate, DC_CONV["SWAP"])]
    # Interpolate, if needed!
    swapDisc = [getRatesLinInterpDiscount(dtSettle, prevSwapDate, termDates, discounts)]
    
    # Loop over the dataset and compute the curve
    for swapDate, rowSwap in swapsSelected.iterrows():
        # Select the rate
        rate = rowSwap.Mid
        # Compute the year fraction using the right DC_CONV
        yf = yearfrac(prevSwapDate, swapDate, DC_CONV["SWAP"])        
        # Compute BPV (look into the slides): this is a "simple sum"
        bpv = np.sum(np.multiply(swapYearFrac, swapDisc))       
        # Compute df (i.e. the discount factor, using the formula into the slides)
        df = (1. - bpv*rate)/(1. + rate*yf)
        # Update the dates in your Discount curve
        termDates += [swapDate]
        # Update the discount curve
        discounts.append(df)
        # Update this: since you need to compute the BPV using what you have computed so far.
        swapDisc.append(df)
        swapYearFrac.append(yf)
        prevSwapDate = swapDate
    
    # Return what you need! The exercize is done!
    return termDates, discounts



