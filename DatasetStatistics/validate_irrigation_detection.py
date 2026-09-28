"""
Validation of the irrigation event detector (v4) against field-documented
irrigation records stored in sensor_data.csv (DOC|ENV__SOIL__IRRIGATION column).

Outputs
-------
Console: categorisation summary, recall/precision, breakdowns by volume and month
figures/appendix/table_irrigation_validation.tex : LaTeX table for the manuscript
DatasetStatistics/irrigation_validation_detail.csv : per-event categorisation
DatasetStatistics/irrigation_validation_by_tree.csv : per-tree TP/FN/FP summary

Run
---
    cd DatasetStatistics
    python validate_irrigation_detection.py

Requires bsc_events_all.csv (from src/_poc_irrigation_v3.py) and
irrigation_events_all.csv (from src/_poc_irrigation_v4.py) to exist.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# -- Paths ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
TREES_DIR  = REPO_ROOT / "TreeTabularData" / "trees"
BSC_CSV    = REPO_ROOT / "TreeTabularData" / "bsc_events_all.csv"
EVENTS_CSV = REPO_ROOT / "TreeTabularData" / "irrigation_events_all.csv"
LATEX_OUT  = REPO_ROOT / "figures" / "appendix" / "table_irrigation_validation.tex"
DETAIL_OUT = SCRIPT_DIR / "irrigation_validation_detail.csv"
TREE_OUT   = SCRIPT_DIR / "irrigation_validation_by_tree.csv"

# -- Constants ------------------------------------------------------------------
IRR_COL   = "DOC|ENV__SOIL__IRRIGATION"
VWC_COLS  = ["-10|ENV__SOIL__VWC", "-30|ENV__SOIL__VWC", "-45|ENV__SOIL__VWC"]

MATCH_WINDOW_H  = 48   # hours: documented event ↔ detection match tolerance
VWC_DATA_WIN_H  = 3    # hours before/after documented event to look for VWC readings
VWC_MIN_ROWS    = 3    # minimum non-null VWC readings to consider data available
BSC_BUFFER_H    = 0    # hours of extra buffer around BSC rain windows (same as v4)

VOL_BINS  = [0, 100, 150, 200, 9999]
VOL_LBLS  = ["<100 L", "100 L", "150 L", ">200 L"]


# -- Helpers -------------------------------------------------------------------

def load_bsc_index(path: Path) -> dict:
    """Load BSC rain windows indexed by EUI → list of (start, end) Timestamps."""
    bsc = pd.read_csv(path, usecols=["eui", "ev_start", "ev_end"],
                      parse_dates=["ev_start", "ev_end"])
    buf = pd.Timedelta(hours=BSC_BUFFER_H)
    idx = {}
    for eui, grp in bsc.groupby("eui"):
        windows = [(pd.Timestamp(r.ev_start) - buf, pd.Timestamp(r.ev_end) + buf)
                   for r in grp.itertuples()]
        idx[str(eui)] = windows
    return idx


def in_bsc_window(t: pd.Timestamp, windows: list) -> bool:
    for s, e in windows:
        if s <= t <= e:
            return True
    return False


def has_vwc_data(t: pd.Timestamp, sensor_df: pd.DataFrame | None) -> bool:
    """Return True if VWC data exists near the documented event time."""
    if sensor_df is None:
        return False
    win = pd.Timedelta(hours=VWC_DATA_WIN_H)
    sub = sensor_df.loc[(sensor_df["datetime"] >= t - win) &
                        (sensor_df["datetime"] <= t + win * 8)]
    for col in VWC_COLS:
        if col in sub.columns and sub[col].dropna().shape[0] >= VWC_MIN_ROWS:
            return True
    return False


def load_sensor_df(eui: str) -> pd.DataFrame | None:
    p = TREES_DIR / eui / "sensor_data.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, low_memory=False)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df
    except Exception:
        return None


def collect_documented_events() -> pd.DataFrame:
    """Scan every tree folder for documented irrigation events."""
    rows = []
    dirs = sorted(d for d in TREES_DIR.iterdir() if d.is_dir())
    print(f"Scanning {len(dirs)} tree folders for documented events…", flush=True)
    for d in dirs:
        csv = d / "sensor_data.csv"
        if not csv.exists():
            continue
        try:
            df = pd.read_csv(csv, usecols=lambda c: c in ["datetime", IRR_COL],
                             low_memory=False)
        except Exception:
            continue
        if IRR_COL not in df.columns:
            continue
        sub = df[["datetime", IRR_COL]].dropna(subset=[IRR_COL]).copy()
        if sub.empty:
            continue
        sub["eui"]      = d.name
        sub["datetime"] = pd.to_datetime(sub["datetime"], errors="coerce")
        sub.rename(columns={"datetime": "doc_time", IRR_COL: "amount_L"}, inplace=True)
        rows.append(sub[["eui", "doc_time", "amount_L"]])
    if not rows:
        sys.exit("No documented irrigation events found. "
                 "Check that sensor_data.csv files contain the DOC|ENV__SOIL__IRRIGATION column.")
    return pd.concat(rows, ignore_index=True)


def categorise_events(doc_all: pd.DataFrame,
                      irr_ev: pd.DataFrame,
                      bsc_idx: dict) -> pd.DataFrame:
    """
    Assign each documented event to one of four categories:
      no_data         – no VWC time-series available
      mixed_excluded  – event falls inside a BSC rain window
      TP              – matched to an IRRIGATION detection within ±MATCH_WINDOW_H
      FN              – assessable but no matching detection found
    """
    match_win = pd.Timedelta(hours=MATCH_WINDOW_H)
    det = irr_ev.reset_index(drop=True).copy()
    det["used"] = False

    sensor_cache: dict[str, pd.DataFrame | None] = {}
    cats = []

    n = len(doc_all)
    for i, row in enumerate(doc_all.itertuples(), 1):
        if i % 500 == 0 or i == n:
            print(f"  Categorising {i}/{n}…", flush=True)

        eui = str(row.eui)
        t   = row.doc_time
        amt = row.amount_L

        if eui not in sensor_cache:
            sensor_cache[eui] = load_sensor_df(eui)

        has_data = has_vwc_data(t, sensor_cache[eui])
        is_mixed = in_bsc_window(t, bsc_idx.get(eui, []))

        # find best unused detection within match window
        cands = det.loc[
            (det["eui"] == eui) &
            ~det["used"] &
            ((det["event_time"] - t).abs() <= match_win)
        ]

        if not has_data:
            cat = "no_data"
        elif is_mixed:
            cat = "mixed_excluded"
        elif not cands.empty:
            best = (cands["event_time"] - t).abs().idxmin()
            det.at[best, "used"] = True
            cat = "TP"
        else:
            cat = "FN"

        cats.append({"eui": eui, "doc_time": t, "amount_L": amt,
                     "cat": cat, "has_data": has_data, "is_mixed": is_mixed})

    return pd.DataFrame(cats), det


def count_fp(irr_ev_annotated: pd.DataFrame, doc_all: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """
    Count detections that have no documented event within ±MATCH_WINDOW_H for
    that tree -- regardless of which detections were consumed as TPs.
    """
    match_win = pd.Timedelta(hours=MATCH_WINDOW_H)
    fp_rows = []
    for eui, grp in irr_ev_annotated.groupby("eui"):
        docs = doc_all.loc[doc_all["eui"] == eui, "doc_time"]
        for row in grp.itertuples():
            if docs.empty:
                fp_rows.append(row)
                continue
            if (docs - row.event_time).abs().min() > match_win:
                fp_rows.append(row)
    fp_df = pd.DataFrame(fp_rows) if fp_rows else pd.DataFrame(columns=irr_ev_annotated.columns)
    return len(fp_rows), fp_df


def build_latex(total, n_trees_doc, n_no_data, n_mixed, n_assessable,
                n_tp, n_fn, fp_count, recall, precision) -> str:
    pct = lambda n, d: f"{n/d:.0%}" if d else "---"
    return (
        r"\begin{table}[h!]" + "\n"
        r"\centering" + "\n"
        r"\caption{Validation of the irrigation event detector (v4) against"
        " field-documented irrigation records"
        r" (N\,=\," + str(total) + " events across "
        + str(n_trees_doc) + r" trees)."
        " Events are categorised by data availability and by overlap with"
        " independently detected rain periods (BSC method, v3)."
        r" Recall and precision are computed over the assessable subset only.}" + "\n"
        r"\label{tab:irrigation_validation}" + "\n"
        r"\begin{tabular}{llrr}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Category} & \textbf{Description} & \textbf{N} & \textbf{\%} \\" + "\n"
        r"\midrule" + "\n"
        r"\multicolumn{4}{l}{\textit{Documented events ("
        + str(total) + r" total)}} \\" + "\n"
        r"\quad A.\ No sensor data & No VWC time-series available during event window & "
        + str(n_no_data) + r" & " + pct(n_no_data, total) + r" \\" + "\n"
        r"\quad B.\ Mixed signal & Irrigation coincides with BSC-detected rain; excluded & "
        + str(n_mixed) + r" & " + pct(n_mixed, total) + r" \\" + "\n"
        r"\quad C.\ Assessable & Clean window, VWC data present & "
        + str(n_assessable) + r" & " + pct(n_assessable, total) + r" \\" + "\n"
        r"\midrule" + "\n"
        r"\multicolumn{4}{l}{\textit{Detection performance (assessable subset,"
        r" N\,=\," + str(n_assessable) + r")}} \\" + "\n"
        r"\quad True Positive (TP) & Irrigation detected within $\pm$48\,h of documentation & "
        + str(n_tp) + r" & " + pct(n_tp, n_assessable) + r" \\" + "\n"
        r"\quad False Negative (FN) & Documented event not detected & "
        + str(n_fn) + r" & " + pct(n_fn, n_assessable) + r" \\" + "\n"
        r"\midrule" + "\n"
        r"False Positive (FP) & Detection without documented counterpart & "
        + str(fp_count) + r" & --- \\" + "\n"
        r"\midrule" + "\n"
        r"\textbf{Recall} & TP / (TP + FN) & \multicolumn{2}{c}{$"
        + f"{recall:.2f}" + r"$} \\" + "\n"
        r"\textbf{Precision} & TP / (TP + FP) & \multicolumn{2}{c}{$"
        + f"{precision:.2f}" + r"$} \\" + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )


def print_report(doc_all, cat_df, n_no_data, n_mixed, n_assessable,
                 n_tp, n_fn, fp_count, recall, precision, irr_total, rain_total):
    W = 72
    print("\n" + "=" * W)
    print("IRRIGATION DETECTION VALIDATION REPORT")
    print("=" * W)

    print(f"\n  Dataset summary")
    print(f"  {'Documented irrigation events:':<40} {len(doc_all):>6}")
    print(f"  {'Trees with documented events:':<40} {doc_all['eui'].nunique():>6}")
    print(f"  {'IRRIGATION detections (v4):':<40} {irr_total:>6}")
    print(f"  {'RAIN detections (BSC v3):':<40} {rain_total:>6}")

    print(f"\n  Categorisation of documented events  (N = {len(doc_all)})")
    print(f"  {'-'*60}")
    print(f"  {'A. No VWC time-series data:':<40} {n_no_data:>5}  ({n_no_data/len(doc_all):.0%})  -- not assessable")
    print(f"  {'B. Mixed signal (BSC rain window):':<40} {n_mixed:>5}  ({n_mixed/len(doc_all):.0%})  -- excluded")
    print(f"  {'C. Assessable:':<40} {n_assessable:>5}  ({n_assessable/len(doc_all):.0%})")
    print(f"     {'C1. True Positive  (TP):':<37} {n_tp:>5}  ({n_tp/n_assessable:.0%} of assessable)")
    print(f"     {'C2. False Negative (FN):':<37} {n_fn:>5}  ({n_fn/n_assessable:.0%} of assessable)")
    print(f"\n  {'False Positive (FP):':<40} {fp_count:>5}")

    print(f"\n  {'-'*60}")
    print(f"  {'Recall   (TP / assessable):':<40} {recall:.3f}  ({recall:.0%})")
    print(f"  {'Precision (TP / TP+FP):':<40} {precision:.3f}  ({precision:.0%})")
    print(f"  Note: precision is a lower bound -- FP partly reflects undocumented events")

    # Volume breakdown
    print(f"\n  {'-'*60}")
    print("  Recall by documented irrigation volume (assessable events only)")
    cat_df["vol_bin"] = pd.cut(cat_df["amount_L"], bins=VOL_BINS,
                               labels=VOL_LBLS, right=True)
    vt = (cat_df[cat_df["cat"].isin(["TP", "FN"])]
          .groupby(["vol_bin", "cat"], observed=False)
          .size().unstack(fill_value=0))
    vt["recall"] = (vt.get("TP", 0) /
                    (vt.get("TP", 0) + vt.get("FN", 0))).round(3)
    print("  " + vt.to_string().replace("\n", "\n  "))

    # Monthly breakdown
    print(f"\n  {'-'*60}")
    print("  Recall by calendar month (assessable events only)")
    MONTHS = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    cat_df["month"] = pd.to_datetime(cat_df["doc_time"]).dt.month
    mt = (cat_df[cat_df["cat"].isin(["TP", "FN"])]
          .groupby(["month", "cat"], observed=False)
          .size().unstack(fill_value=0))
    mt.index = [MONTHS.get(m, m) for m in mt.index]
    mt["recall"] = (mt.get("TP", 0) /
                    (mt.get("TP", 0) + mt.get("FN", 0))).round(3)
    print("  " + mt.to_string().replace("\n", "\n  "))
    print()


def save_detail_csv(cat_df: pd.DataFrame, fp_df: pd.DataFrame, path: Path):
    out = cat_df.copy()
    out["type"] = "documented"
    if not fp_df.empty:
        fp_out = fp_df[["eui", "event_time"]].copy()
        fp_out.rename(columns={"event_time": "doc_time"}, inplace=True)
        fp_out["amount_L"]  = np.nan
        fp_out["cat"]       = "FP"
        fp_out["has_data"]  = np.nan
        fp_out["is_mixed"]  = False
        fp_out["type"]      = "detection"
        out = pd.concat([out, fp_out], ignore_index=True)
    out.to_csv(path, index=False)


def save_tree_summary(cat_df: pd.DataFrame, fp_df: pd.DataFrame,
                      irr_ev: pd.DataFrame, path: Path):
    doc_summary = (cat_df.groupby(["eui", "cat"])
                   .size().unstack(fill_value=0)
                   .rename_axis(None, axis=1))
    for col in ["no_data", "mixed_excluded", "TP", "FN"]:
        if col not in doc_summary.columns:
            doc_summary[col] = 0
    doc_summary["total_doc"] = doc_summary[["no_data","mixed_excluded","TP","FN"]].sum(axis=1)
    assessable = doc_summary["TP"] + doc_summary["FN"]
    doc_summary["recall"] = (doc_summary["TP"] / assessable).where(assessable > 0)

    fp_counts = fp_df.groupby("eui").size().rename("FP") if not fp_df.empty else pd.Series(dtype=int, name="FP")
    det_counts = irr_ev.groupby("eui").size().rename("total_detections")

    summary = (doc_summary
               .join(fp_counts, how="outer")
               .join(det_counts, how="outer")
               .fillna(0))
    summary["FP"] = summary["FP"].astype(int)
    summary["total_detections"] = summary["total_detections"].astype(int)
    summary.sort_values("total_doc", ascending=False).to_csv(path)


# -- Main ----------------------------------------------------------------------

def main():
    for p, label in [(BSC_CSV, "bsc_events_all.csv"),
                     (EVENTS_CSV, "irrigation_events_all.csv")]:
        if not p.exists():
            sys.exit(f"Required file not found: {p}\n"
                     f"  → Run src/_poc_irrigation_v3.py then src/_poc_irrigation_v4.py first.")

    # Load inputs
    print("Loading BSC rain windows…", flush=True)
    bsc_idx = load_bsc_index(BSC_CSV)

    print("Loading detected events…", flush=True)
    all_ev  = pd.read_csv(EVENTS_CSV,
                          usecols=["eui", "event_time", "label"],
                          parse_dates=["event_time"],
                          low_memory=False)
    irr_ev  = all_ev[all_ev["label"] == "IRRIGATION"].copy()
    rain_ev = all_ev[all_ev["label"] == "RAIN"]

    # Documented events
    doc_all = collect_documented_events()

    # Categorise
    print("\nCategorising documented events…", flush=True)
    cat_df, irr_ev_annotated = categorise_events(doc_all, irr_ev, bsc_idx)

    # FP count
    print("Counting false positives…", flush=True)
    fp_count, fp_df = count_fp(irr_ev_annotated, doc_all)

    # Aggregate metrics
    counts      = cat_df["cat"].value_counts()
    total       = len(cat_df)
    n_no_data   = int(counts.get("no_data", 0))
    n_mixed     = int(counts.get("mixed_excluded", 0))
    n_tp        = int(counts.get("TP", 0))
    n_fn        = int(counts.get("FN", 0))
    n_assessable = n_tp + n_fn
    recall      = n_tp / n_assessable if n_assessable else 0.0
    precision   = n_tp / (n_tp + fp_count) if (n_tp + fp_count) else 0.0
    n_trees_doc = doc_all["eui"].nunique()

    # Console report
    print_report(doc_all, cat_df, n_no_data, n_mixed, n_assessable,
                 n_tp, n_fn, fp_count, recall, precision,
                 len(irr_ev), len(rain_ev))

    # Save LaTeX
    latex = build_latex(total, n_trees_doc, n_no_data, n_mixed, n_assessable,
                        n_tp, n_fn, fp_count, recall, precision)
    LATEX_OUT.write_text(latex, encoding="utf-8")
    print(f"  Saved: {LATEX_OUT.relative_to(REPO_ROOT)}")

    # Save detail CSV
    save_detail_csv(cat_df, fp_df, DETAIL_OUT)
    print(f"  Saved: {DETAIL_OUT.relative_to(REPO_ROOT)}")

    # Save per-tree summary
    save_tree_summary(cat_df, fp_df, irr_ev, TREE_OUT)
    print(f"  Saved: {TREE_OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
