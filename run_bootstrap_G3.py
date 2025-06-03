import os
from pathlib import Path
import pandas as pd
import numpy as np
from FinDates.daycount import yearfrac
import datetime as dt
from Bootstrap_G3 import (
    DC_CONV,
    depo_converter, future_converter, swap_converter,
    bootstrapDepo, bootstrapFuture, bootstrapSwap,
    getZeroRates, extract_mid_rates
)

INPUT_DIR = Path(__file__).resolve().parent / "market"

def load_settlement_date(file_path: str) -> pd.Timestamp:
    df = pd.read_csv(
        file_path,
        index_col='Market',
        usecols=['Market', 'TARGET'],
        converters={'TARGET': pd.to_datetime}
    )
    return df.loc['Settlement', 'TARGET'] # type: ignore

def preview_dataframes(dataframes: dict):
    for name, df in dataframes.items():
        print(f"\n{name} preview:")
        print(df.iloc[[0, 1, -1]])

def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir())

    dtSettle = load_settlement_date('dt.csv')

    df_depo = pd.read_csv(
        INPUT_DIR /'depos.csv',
        index_col='Depos',
        usecols=['Depos', 'ASK', 'BID'],
        converters={
            'Depos': pd.to_datetime,
            'BID': depo_converter,
            'ASK': depo_converter
        }
    )

    futures = pd.read_csv(
        INPUT_DIR /'futures.csv',
        index_col='Future',
        usecols=['Future', 'ASK', 'BID'],
        converters={
            'Future': pd.to_datetime,
            'BID': future_converter,
            'ASK': future_converter
        }
    )

    settle = pd.read_csv(
        INPUT_DIR /'settles.csv',
        index_col='Future',
        usecols=['Future', 'Settle', 'Expiry'],
        converters={
            'Future': pd.to_datetime,
            'Settle': pd.to_datetime,
            'Expiry': pd.to_datetime
        }
    )
    df_futures = futures.join(settle)

    df_swaps = pd.read_csv(
        INPUT_DIR /'swaps.csv',
        index_col='Swap',
        usecols=['Swap', 'BID', 'ASK'],
        converters={
            'Swap': pd.to_datetime,
            'BID': swap_converter,
            'ASK': swap_converter
        }
    )

    dataset = {
        "depo": df_depo,
        "futures": df_futures,
        "swaps": df_swaps
    }

    preview_dataframes(dataset)

    termDates0, discounts0 = [dtSettle], [1.0]

    # No bump
    termDates_n, discounts_n = bootstrapDepo(dtSettle, df_depo, df_futures, termDates0.copy(), discounts0.copy())
    termDates_n, discounts_n = bootstrapFuture(dtSettle, df_futures, termDates_n, discounts_n)
    termDates_n, discounts_n = bootstrapSwap(dtSettle, df_swaps, termDates_n, discounts_n)

    yearFrac_n = [yearfrac(dtSettle, T, DC_CONV["INTERP"]) for T in termDates_n[1:]]
    zero_n = getZeroRates(yearFrac_n, discounts_n[1:])
    discCurve_n = pd.Series(index=pd.DatetimeIndex(termDates_n), data=discounts_n)
    zeroCurve_n = pd.Series(index=pd.DatetimeIndex(termDates_n[1:]), data=zero_n)


    # With bump
    bump = 0.0001
    depos_b = extract_mid_rates(df_depo, bump=bump)
    futu_b = extract_mid_rates(df_futures, bump=bump)

    termDates_b, discounts_b = bootstrapDepo(dtSettle, df_depo, df_futures, termDates0.copy(), discounts0.copy(), bump=bump)
    termDates_b, discounts_b = bootstrapFuture(dtSettle, df_futures, termDates_b, discounts_b, bump=-bump)
    termDates_b, discounts_b = bootstrapSwap(dtSettle, df_swaps, termDates_b, discounts_b, bump=bump)

    yearFrac_b = [yearfrac(dtSettle, T, DC_CONV["INTERP"]) for T in termDates_b[1:]]
    zero_b = getZeroRates(yearFrac_b, discounts_b[1:])
    discCurve_b = pd.Series(index=pd.DatetimeIndex(termDates_b), data=discounts_b)
    zeroCurve_b = pd.Series(index=pd.DatetimeIndex(termDates_b[1:]), data=zero_b)


    # Save outputs
    #depos_b.to_csv("depos_shifted.csv")
    #futu_b.to_csv("futu_shifted.csv")
    #discCurve_n.to_csv('discCurve.csv')
    #discCurve_b.to_csv('discCurve_bumped.csv')
    zeroCurve_n.to_csv('zero_curve.csv')
    zeroCurve_b.to_csv('zero_curve_bumped.csv')


if __name__ == "__main__":
    main()
