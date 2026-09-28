"""Korean resident tax on overseas stock gains (config `tax`).

One calendar year's realized gains and losses are netted; the net gain in KRW
(USD x tax.fx, a fixed rate) above tax.deduction_krw is taxed at tax.rate.
A loss year pays 0 and the loss is not carried forward; the deduction starts
fresh every year. Year Y's tax is taken from the account on the first trading
day of May (tax.pay_month) of Y+1; the final year's tax, whose May is outside
the sample, on the last day (sizing.account, tax_pay_days). A buy-and-hold is
sold on the last day and taxed once on its whole gain that day
(hold_after_tax). Interest on idle cash is not taxed.
"""
from __future__ import annotations

import pandas as pd


def tax_due_usd(gain_usd: float, tc: dict) -> float:
    gain_krw = float(gain_usd) * float(tc["fx"])
    return max(0.0, gain_krw - float(tc["deduction_krw"])) * float(tc["rate"]) / float(tc["fx"])


def tax_pay_days(cal: pd.DatetimeIndex, month: int) -> dict[int, pd.Timestamp]:
    """{tax year Y: day the tax is paid} for every year in cal: the first day of cal in
    year Y+1 with month >= `month`; the last day of cal when no such day exists."""
    out = {}
    for y in sorted(set(cal.year)):
        later = cal[(cal.year == y + 1) & (cal.month >= month)]
        out[y] = later[0] if len(later) else cal[-1]
    return out


def hold_after_tax(eq: pd.Series, cap: float, tc: dict) -> pd.Series:
    out = eq.astype(float).copy()
    out.iloc[-1] -= tax_due_usd(out.iloc[-1] - cap, tc)
    return out
