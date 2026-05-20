"""
generate_data.py — v1 synthetic dataset for the agentic analytics project.

Generates three Parquet files representing a coffee shop chain network and one
coordinated intervention (a mobile order pilot) for causal-inference evaluation:

  - merchants.parquet      : ~500 individual store locations
  - transactions.parquet   : merchant-day grain, ~2 years of history
  - interventions.parquet  : one intervention, with hidden ground-truth lift

Generation is deterministic given the seed. The ground-truth `true_effect`
column in interventions.parquet is the oracle for the evaluation harness —
the agent must NEVER see this column. Isolate it accordingly when loading
into Unity Catalog.

Methodology framing: standard difference-in-differences setup with a single
treatment date (canonical 2x2 case). Staggered adoption is a phase-2 extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SEED = 42
N_MERCHANTS = 500
HISTORY_START = date(2023, 1, 1)
HISTORY_END = date(2024, 12, 31)  # ~2 years, 731 days inclusive

# Intervention configuration
INTERVENTION_NAME = "mobile_order_pilot"
INTERVENTION_START = date(2024, 6, 3)  # Monday, well into the history window
N_TREATED = 25
TRUE_EFFECT = 0.05  # +5% lift on daily revenue at treated stores post-start
PRE_WINDOW_DAYS = 90
POST_WINDOW_DAYS = 60

# Categorical domains
LOCATION_TYPES = ["urban", "suburban", "highway", "mall", "campus"]
REGIONS = ["northeast", "southeast", "midwest", "west"]
SIZE_BANDS = ["small", "mid", "large"]
BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD"]

# Baseline daily revenue by size_band (mean dollars/day)
SIZE_BASELINE = {"small": 1200.0, "mid": 2400.0, "large": 4500.0}

# Average ticket size (used to derive txn_count from amount)
SIZE_AVG_TICKET = {"small": 6.50, "mid": 7.25, "large": 8.00}

# Output directory
OUT_DIR = Path("./data")

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def daterange(start: date, end: date) -> list[date]:
    """Inclusive date range."""
    n_days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n_days)]


def weekday_factor(d: date, location_type: str) -> float:
    """
    Day-of-week multiplier. Coffee shops have distinct weekday vs. weekend
    rhythms that vary by location type:
      - urban / campus: weekday commuter peak, weekend drop
      - suburban / mall: more even, slight weekend uplift
      - highway: weekend-leaning (travel)
    Returns a multiplier centered roughly on 1.0.
    """
    dow = d.weekday()  # 0=Mon, 6=Sun
    is_weekend = dow >= 5

    if location_type in ("urban", "campus"):
        return 0.70 if is_weekend else 1.12
    if location_type == "suburban":
        return 1.05 if is_weekend else 0.98
    if location_type == "mall":
        return 1.10 if is_weekend else 0.96
    if location_type == "highway":
        return 1.15 if is_weekend else 0.94
    return 1.0


def yearly_factor(d: date, location_type: str) -> float:
    """
    Yearly seasonality. Coffee demand has a mild summer iced-drink uplift
    in commuter areas and a Q4 holiday bump in malls/urban. Modeled as a
    smooth sinusoid plus a December bump.
    """
    # Day of year, 0..1
    doy = (d.timetuple().tm_yday - 1) / 365.0

    # Mild annual sinusoid; phase shifts by location
    if location_type in ("urban", "campus"):
        # Slight summer dip (people on vacation), recovery in fall
        annual = 1.0 + 0.04 * np.sin(2 * np.pi * (doy - 0.5))
    elif location_type == "highway":
        # Summer travel peak
        annual = 1.0 + 0.08 * np.sin(2 * np.pi * (doy - 0.25))
    else:
        # Suburban / mall: relatively flat with mild seasonal sway
        annual = 1.0 + 0.03 * np.sin(2 * np.pi * (doy - 0.3))

    # December bump for mall & urban (holiday foot traffic)
    if location_type in ("mall", "urban") and d.month == 12:
        annual *= 1.10

    return annual


def trend_factor(d: date, start: date, annual_growth: float) -> float:
    """Slow linear growth/decline. annual_growth e.g. 0.03 = +3%/year."""
    years_in = (d - start).days / 365.0
    return 1.0 + annual_growth * years_in


# ----------------------------------------------------------------------------
# Step 1: merchants
# ----------------------------------------------------------------------------

def generate_merchants(rng: np.random.Generator) -> pd.DataFrame:
    """Generate ~500 individual coffee shop store locations."""
    merchant_ids = [f"M{idx:05d}" for idx in range(1, N_MERCHANTS + 1)]

    # Sample categorical attributes with realistic mix
    location_type = rng.choice(
        LOCATION_TYPES,
        size=N_MERCHANTS,
        p=[0.30, 0.30, 0.10, 0.15, 0.15],  # mostly urban/suburban
    )
    region = rng.choice(REGIONS, size=N_MERCHANTS, p=[0.28, 0.25, 0.22, 0.25])
    size_band = rng.choice(SIZE_BANDS, size=N_MERCHANTS, p=[0.40, 0.45, 0.15])
    brand = rng.choice(BRANDS, size=N_MERCHANTS, p=[0.40, 0.25, 0.20, 0.15])

    # Onboarded date — well before the history window so all stores have full
    # pre-period coverage. Spread across the year before HISTORY_START.
    onboarded_offsets = rng.integers(low=-720, high=-30, size=N_MERCHANTS)
    onboarded_date = [HISTORY_START + timedelta(days=int(o)) for o in onboarded_offsets]

    # Per-merchant random effect: a multiplicative factor centered on 1.0 with
    # modest spread. This is what makes two stores in the same segment/region/
    # size *similar but not identical* — essential for matching to be a real
    # problem rather than trivial or impossible.
    merchant_effect = rng.normal(loc=1.0, scale=0.12, size=N_MERCHANTS)
    merchant_effect = np.clip(merchant_effect, 0.65, 1.40)

    # Per-merchant slow growth rate (annualized). Some stores trending up,
    # some flat, some slowly declining. Mostly mild.
    annual_growth = rng.normal(loc=0.02, scale=0.04, size=N_MERCHANTS)
    annual_growth = np.clip(annual_growth, -0.05, 0.10)

    merchants = pd.DataFrame({
        "merchant_id": merchant_ids,
        "location_type": location_type,
        "region": region,
        "size_band": size_band,
        "brand": brand,
        "onboarded_date": onboarded_date,
        # The two below are generation parameters, not analyst-visible attributes.
        # Keeping them on the merchant row makes the generator self-contained;
        # when loading into Unity Catalog, you can drop these columns from the
        # agent-visible view if you want strict realism.
        "_merchant_effect": merchant_effect,
        "_annual_growth": annual_growth,
    })

    return merchants


# ----------------------------------------------------------------------------
# Step 2: baseline transactions (no intervention yet)
# ----------------------------------------------------------------------------

def generate_transactions(merchants: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate merchant-day rows for the full history window.
    Vectorized per-merchant for speed.
    """
    dates = daterange(HISTORY_START, HISTORY_END)
    n_days = len(dates)
    dates_arr = np.array(dates)

    # Precompute day-of-week and day-of-year arrays
    weekdays = np.array([d.weekday() for d in dates])
    is_weekend = weekdays >= 5
    doy = np.array([(d.timetuple().tm_yday - 1) / 365.0 for d in dates])
    months = np.array([d.month for d in dates])
    years_in = np.array([(d - HISTORY_START).days / 365.0 for d in dates])

    rows: list[pd.DataFrame] = []

    for _, m in merchants.iterrows():
        size = m["size_band"]
        loc = m["location_type"]
        base_amount = SIZE_BASELINE[size]
        avg_ticket = SIZE_AVG_TICKET[size]

        # Weekday factor for this location type (vectorized)
        if loc in ("urban", "campus"):
            wf = np.where(is_weekend, 0.70, 1.12)
        elif loc == "suburban":
            wf = np.where(is_weekend, 1.05, 0.98)
        elif loc == "mall":
            wf = np.where(is_weekend, 1.10, 0.96)
        elif loc == "highway":
            wf = np.where(is_weekend, 1.15, 0.94)
        else:
            wf = np.ones(n_days)

        # Yearly factor (vectorized)
        if loc in ("urban", "campus"):
            yf = 1.0 + 0.04 * np.sin(2 * np.pi * (doy - 0.5))
        elif loc == "highway":
            yf = 1.0 + 0.08 * np.sin(2 * np.pi * (doy - 0.25))
        else:
            yf = 1.0 + 0.03 * np.sin(2 * np.pi * (doy - 0.3))

        # December bump for mall & urban
        if loc in ("mall", "urban"):
            yf = np.where(months == 12, yf * 1.10, yf)

        # Trend factor
        tf = 1.0 + m["_annual_growth"] * years_in

        # Combine deterministic components
        deterministic = base_amount * m["_merchant_effect"] * wf * yf * tf

        # Daily noise — multiplicative, lognormal-ish for positivity
        noise = rng.normal(loc=1.0, scale=0.10, size=n_days)
        noise = np.clip(noise, 0.65, 1.40)

        amount = deterministic * noise
        # Derive txn_count from amount and avg_ticket with small per-day jitter
        ticket_jitter = rng.normal(loc=1.0, scale=0.04, size=n_days)
        ticket_jitter = np.clip(ticket_jitter, 0.85, 1.15)
        daily_avg_ticket = avg_ticket * ticket_jitter
        txn_count = np.maximum(1, np.round(amount / daily_avg_ticket)).astype(int)

        df = pd.DataFrame({
            "merchant_id": m["merchant_id"],
            "txn_date": dates_arr,
            "amount": np.round(amount, 2),
            "txn_count": txn_count,
        })
        rows.append(df)

    transactions = pd.concat(rows, ignore_index=True)
    return transactions


# ----------------------------------------------------------------------------
# Step 3: inject the intervention
# ----------------------------------------------------------------------------

def inject_intervention(
    transactions: pd.DataFrame,
    merchants: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pick treated merchants and multiply their post-start amount by (1 + lift)
    with a small dose of noise so the effect isn't artificially clean.

    Treated merchants are sampled to look like a plausible pilot cohort —
    we bias slightly toward urban/suburban stores of mid-or-large size, which
    is the realistic profile for a mobile-order rollout. This is a
    *selection on observables* mechanism; matching on those observables in
    the control-selection tool will still recover comparable controls.
    """
    # Score each merchant for "likelihood of being selected into the pilot"
    loc_weight = merchants["location_type"].map(
        {"urban": 3.0, "campus": 2.0, "suburban": 2.0, "mall": 1.0, "highway": 0.5}
    )
    size_weight = merchants["size_band"].map({"small": 0.5, "mid": 2.0, "large": 3.0})
    score = loc_weight.values * size_weight.values
    probs = score / score.sum()

    treated_idx = rng.choice(
        len(merchants), size=N_TREATED, replace=False, p=probs
    )
    treated_ids = merchants.iloc[treated_idx]["merchant_id"].tolist()

    # Apply the lift to treated merchants for txn_date >= INTERVENTION_START
    # Ensure consistent date types for comparison. We store as datetime64
    # for downstream queryability, then compare against INTERVENTION_START.
    transactions["txn_date"] = pd.to_datetime(transactions["txn_date"])
    mask_treated = transactions["merchant_id"].isin(treated_ids)
    mask_post = transactions["txn_date"] >= pd.Timestamp(INTERVENTION_START)
    mask = mask_treated & mask_post

    # Lift with small noise (~1% sd) so post-period isn't a flat shift
    n_affected = int(mask.sum())
    lift_noise = rng.normal(loc=1.0, scale=0.01, size=n_affected)
    lift_noise = np.clip(lift_noise, 0.97, 1.03)
    multiplier = (1.0 + TRUE_EFFECT) * lift_noise

    transactions.loc[mask, "amount"] = np.round(
        transactions.loc[mask, "amount"].values * multiplier, 2
    )
    # Recompute txn_count proportionally so basket size stays sane
    transactions.loc[mask, "txn_count"] = np.maximum(
        1,
        np.round(transactions.loc[mask, "txn_count"].values * multiplier),
    ).astype(int)

    interventions = pd.DataFrame([{
        "intervention_id": "INT_001",
        "name": INTERVENTION_NAME,
        "treated_merchant_ids": treated_ids,
        "start_date": INTERVENTION_START,
        "pre_window_days": PRE_WINDOW_DAYS,
        "post_window_days": POST_WINDOW_DAYS,
        "true_effect": TRUE_EFFECT,  # ORACLE — never expose to the agent
    }])

    return transactions, interventions


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Generating merchants...")
    merchants = generate_merchants(rng)
    print(f"  -> {len(merchants)} merchants")

    print("Generating transactions (this may take ~10-30s)...")
    transactions = generate_transactions(merchants, rng)
    print(f"  -> {len(transactions):,} merchant-day rows")

    print("Injecting intervention...")
    transactions, interventions = inject_intervention(transactions, merchants, rng)
    treated = interventions.iloc[0]["treated_merchant_ids"]
    print(f"  -> intervention '{INTERVENTION_NAME}' applied to {len(treated)} merchants")
    print(f"  -> start_date={INTERVENTION_START}, true_effect={TRUE_EFFECT:+.2%}")

    # Strip generator-only columns from merchants before writing
    merchants_out = merchants.drop(columns=["_merchant_effect", "_annual_growth"])

    merchants_path = OUT_DIR / "merchants.parquet"
    transactions_path = OUT_DIR / "transactions.parquet"
    interventions_path = OUT_DIR / "interventions.parquet"

    merchants_out.to_parquet(merchants_path, index=False)
    transactions.to_parquet(transactions_path, index=False)
    interventions.to_parquet(interventions_path, index=False)

    print()
    print("Written:")
    print(f"  {merchants_path}  ({merchants_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {transactions_path}  ({transactions_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  {interventions_path}  ({interventions_path.stat().st_size / 1024:.1f} KB)")
    print()
    print("REMINDER: interventions.parquet contains `true_effect`, the oracle.")
    print("Isolate this column from any view the agent's tools can query.")


if __name__ == "__main__":
    main()
