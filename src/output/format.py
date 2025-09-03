from __future__ import annotations
import math
import datetime as dt
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style

def format_output_line(
    now: dt.datetime,
    symbol: str,
    spot_now: float,
    vwap_fut: float,
    D: float,
    ATR_D: float,
    snap: Dict,
    mp: int,
    atm_k: Optional[int],
    probs: Dict[str,float],
    top: str,
    tp: Dict,
    oi_flags: Dict[str,bool],
    vwap_spot: float,
    adx5: float,
    div: float,
    iv_pct_hint: float,
    macd_last: float,
    macd_sig_last: float,
    VND: float,
    PIN_NORM: float,
    MPH_NORM_THR: float,
) -> str:
    def pretty_scn(name: str) -> str:
        mapping = {
            "Short-cover reversion up": "Short-Cover Reversion Up",
            "Bear migration": "Bear Migration",
            "Bull migration / gamma carry": "Bull Migration",
            "Pin & decay day (IV crush)": "Pin and Decay (IV crush)",
            "Squeeze continuation (one-way)": "Squeeze Continuation (One-way)",
            "Event knee-jerk then revert": "Event Knee-jerk then Revert",
        }
        return mapping.get(name, name)

    iv_crush = (div == div and div <= 0) and (iv_pct_hint == iv_pct_hint and iv_pct_hint < 33)
    step_sz = 100 if "BANK" in symbol.upper() else 50
    def suggest_spread(action: str):
        if action == "BUY_CE":
            return ("BUY", (int(atm_k + step_sz), int(atm_k + 2*step_sz)))
        if action == "BUY_PE":
            return ("BUY", (int(atm_k - step_sz), int(atm_k - 2*step_sz)))
        return ("NO-TRADE", None)

    side, spread = suggest_spread(tp.get("action", "NO-TRADE"))
    _sorted = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    alt_name = pretty_scn(_sorted[1][0]) if len(_sorted) > 1 else ""
    alt_name = alt_name.replace(" (IV crush)", "") if alt_name else ""
    alt_pct  = int(_sorted[1][1]*100) if len(_sorted) > 1 else 0
    top_disp = pretty_scn(top)
    if iv_crush and "(IV crush)" not in top_disp:
        top_disp = f"{top_disp} (IV crush)"

    above_vwap = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now > vwap_spot)
    below_vwap = (spot_now == spot_now) and (vwap_spot == vwap_spot) and (spot_now < vwap_spot)
    macd_up = macd_last > macd_sig_last
    macd_dn = macd_last < macd_sig_last
    trend_strong = adx5 >= 18
    bull_flow = oi_flags.get("pe_write_above", False) and oi_flags.get("ce_unwind_below", False)
    bear_flow = oi_flags.get("ce_write_above", False) and oi_flags.get("pe_unwind_below", False)
    bull_pcr = snap.get("dpcr_z", 0) >= 0.5
    bear_pcr = snap.get("dpcr_z", 0) <= -0.5
    bull_drift = (snap.get("mph_norm", float('nan')) == snap.get("mph_norm", float('nan'))) and (snap.get("mph_norm", 0) >= MPH_NORM_THR)
    bear_drift = (snap.get("mph_norm", float('nan')) == snap.get("mph_norm", float('nan'))) and (snap.get("mph_norm", 0) <= -MPH_NORM_THR)

    conds: List[Tuple[str,bool]] = []
    if side == "BUY":
        if tp.get("action") == "BUY_CE":
            conds = [
                ("Price above VWAP", above_vwap),
                ("ADX >= 18 (trend strength)", trend_strong),
                ("OI flow supports up (PE write, CE unwind)", bull_flow),
                ("PCR momentum up (dz >= 0.5)", bull_pcr),
                ("MACD(5m) cross up", macd_up),
                ("MaxPain drift up (norm >= thr)", bull_drift),
            ]
        elif tp.get("action") == "BUY_PE":
            conds = [
                ("Price below VWAP", below_vwap),
                ("ADX >= 18 (trend strength)", trend_strong),
                ("OI flow supports down (CE write, PE unwind)", bear_flow),
                ("PCR momentum down (dz <= -0.5)", bear_pcr),
                ("MACD(5m) cross down", macd_dn),
                ("MaxPain drift down (norm <= -thr)", bear_drift),
            ]
    else:
        conds = [
            ("VND < pin norm (range-bound)", VND < PIN_NORM),
            ("ADX < low (chop)", adx5 < 15),
            ("Two-sided OI near ATM", oi_flags.get("two_sided_adjacent", False)),
            ("IV crushing (div<=0, pctile<33)", iv_crush),
        ]

    total = len(conds)
    threshold = max(3, (total * 3)//5) if total else 0
    met = sum(1 for _, ok in conds if ok)

    def tick(ok: bool) -> str:
        # Use ASCII characters instead of Unicode to avoid cp1252 encoding issues on Windows
        return (Fore.GREEN + "[Y]" + Style.RESET_ALL) if ok else (Fore.RED + "[N]" + Style.RESET_ALL)

    cond_lines = []
    for idx, (name, ok) in enumerate(conds, start=1):
        cond_lines.append(f"{idx}. {name} - {tick(ok)}")

    # Preserve index spot precision rather than truncating to integer. The
    # dashboard relies on the exact spot value to compute the day-over-day
    # change. Casting to ``int`` caused the displayed spot to drop the
    # fractional part (e.g. ``24426.85`` became ``24426``) which in turn led to
    # incorrect difference calculations.  Format to two decimals instead.
    spot_print = f"{spot_now:.2f}" if not math.isnan(spot_now) else "NA"
    vwap_fut_print = int(vwap_fut) if not math.isnan(vwap_fut) else "NA"
    header = f"{now.strftime('%H:%M')} IST | {symbol} {spot_print} | VWAP(fut) {vwap_fut_print}"
    l1 = f"D={int(D)} | ATR_D={int(ATR_D)} | VND={snap.get('vnd')} | SSD={snap.get('ssd')} | PD={snap.get('pdist_pct')}%"
    l2 = f"PCR {snap.get('pcr')} (dz={snap.get('dpcr_z')}) | MaxPain {mp} | Drift {snap.get('mph_pts_per_hr')}/hr (norm {snap.get('mph_norm')})"
    l3 = f"ATM {atm_k} IV {snap.get('atm_iv')}% (dIV_z={snap.get('iv_z')}) | Basis {snap.get('basis')}"
    l4 = f"Scenario: {top_disp} {int(probs[top]*100)}%" + (f" (alt: {alt_name} {alt_pct}%)" if alt_name else "")

    if side == "BUY" and spread:
        k1, k2 = spread
        cepe = 'CE' if tp.get('action')=='BUY_CE' else 'PE'
        action_line = Style.BRIGHT + Fore.CYAN + f"Action: TRADE | BUY {k1} {cepe} / {k2} {cepe}" + Style.RESET_ALL
    else:
        reason = tp.get('why', 'No favorable setup')
        action_line = Style.BRIGHT + Fore.YELLOW + f"Action: NO-TRADE | {reason}" + Style.RESET_ALL

    entry_gate = Style.BRIGHT + Fore.MAGENTA + f"Enter when atleast {threshold} of below {total} are satisfied:" + Style.RESET_ALL
    verdict_ok = (side == "BUY") and (met >= threshold)
    verdict = (Style.BRIGHT + Fore.GREEN + "Final Verdict: Enter Now" + Style.RESET_ALL) if verdict_ok else (Style.BRIGHT + Fore.YELLOW + "Final Verdict: Hold" + Style.RESET_ALL)

    # Exit checklist (only when we have a trade suggestion)
    exit_lines: List[str] = []
    if side == "BUY" and spread:
        if tp.get('action') == 'BUY_CE':
            exit_conds = [
                ("Close back below VWAP", below_vwap),
                ("ADX weakening (<16)", adx5 < 16),
                ("OI flips bearish (CE write, PE unwind)", bear_flow),
                ("PCR momentum down (dz <= -0.5)", bear_pcr),
                ("MACD(5m) cross down", macd_dn),
                ("MaxPain drift down (norm <= -thr)", bear_drift),
            ]
        else:  # BUY_PE
            exit_conds = [
                ("Close back above VWAP", above_vwap),
                ("ADX weakening (<16)", adx5 < 16),
                ("OI flips bullish (PE write, CE unwind)", bull_flow),
                ("PCR momentum up (dz >= 0.5)", bull_pcr),
                ("MACD(5m) cross up", macd_up),
                ("MaxPain drift up (norm >= thr)", bull_drift),
            ]
        exit_hdr = Style.BRIGHT + Fore.MAGENTA + "Exit when atleast one or more satisfies:" + Style.RESET_ALL
        exit_lines.append(exit_hdr)
        for idx, (name, ok) in enumerate(exit_conds, start=1):
            exit_lines.append(f"{idx}. {name}")

    out_lines = [header, l1, l2, l3, l4, action_line, entry_gate] + cond_lines + exit_lines + [verdict]
    return "\n".join(out_lines)
