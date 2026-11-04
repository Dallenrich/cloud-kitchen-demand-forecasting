#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Simulated cloud-kitchen order generator.

Writes one CSV per day (like a real source system) plus reference files:
  - calendar (festivals, IPL days), simulated weather, and a "truth" file
    with the true expected demand, used only to validate the model later.

Usage (from the project root):
  python data_generator/generate_orders.py --start 2024-01-01 --end 2024-01-31 --out data/orders_test
  python data_generator/generate_orders.py --start 2024-01-01 --end 2025-12-31 --out data/orders
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOURS = np.arange(11, 24)  # kitchen open 11:00 to 23:59
HOUR_CURVE = np.array([0.4, 1.3, 1.7, 1.2, 0.6, 0.5, 0.7, 1.0, 1.5, 1.9, 1.6, 0.9, 0.4])
HOUR_CURVE = HOUR_CURVE / HOUR_CURVE.mean()  # lunch and dinner peaks
WEEKDAY = np.array([0.90, 0.90, 0.92, 0.95, 1.10, 1.25, 1.20])  # Mon..Sun
BASE_RATE = 0.30
PAYMENT_MODE_FROM = pd.Timestamp("2025-07-01")  # new column appears from this file date

# Approximate festival windows. Verify exact dates before presenting them as real facts.
EVENTS = [
    ("pongal", "2024-01-14", "2024-01-16"),
    ("holi", "2024-03-24", "2024-03-26"),
    ("eid", "2024-04-10", "2024-04-12"),
    ("navratri", "2024-10-03", "2024-10-12"),
    ("diwali", "2024-10-30", "2024-11-02"),
    ("christmas_newyear", "2024-12-24", "2025-01-01"),
    ("pongal", "2025-01-14", "2025-01-16"),
    ("holi", "2025-03-13", "2025-03-15"),
    ("eid", "2025-03-30", "2025-04-01"),
    ("navratri", "2025-09-22", "2025-10-01"),
    ("diwali", "2025-10-19", "2025-10-23"),
    ("christmas_newyear", "2025-12-24", "2025-12-31"),
]
IPL_SEASONS = [("2024-03-22", "2024-05-26"), ("2025-03-22", "2025-06-03")]


def build_calendar(start, end, rng):
    days = pd.date_range(start, end)
    cal = pd.DataFrame({"date": days, "event": "none"})
    for name, s, e in EVENTS:
        cal.loc[(cal["date"] >= s) & (cal["date"] <= e), "event"] = name
    cal["is_ipl_day"] = False
    for s, e in IPL_SEASONS:
        in_season = ((cal["date"] >= s) & (cal["date"] <= e)).values
        cal.loc[in_season, "is_ipl_day"] = rng.random(in_season.sum()) < 0.7
    return cal


def build_weather(days, cities, rng):
    frames = []
    months = days.month.values
    for city in cities:
        if city == "Chennai":
            p = np.where(np.isin(months, [10, 11, 12]), 0.40, 0.10)
        elif city in ("Mumbai", "Pune", "Bengaluru"):
            p = np.where(np.isin(months, [6, 7, 8, 9]), 0.50, 0.05)
        else:
            p = np.where(np.isin(months, [6, 7, 8, 9]), 0.35, 0.06)
        rain = rng.random(len(days)) < p
        base_t = 30 + 4 * np.sin(2 * np.pi * (days.day_of_year.values - 90) / 365)
        temp = base_t - 3 * rain + rng.normal(0, 1.5, len(days))
        frames.append(pd.DataFrame({
            "city": city, "date": days,
            "rain_flag": rain.astype(int), "temp_c": temp.round(1)}))
    return pd.concat(frames, ignore_index=True)

def event_multiplier(cal, dishes):
    """Festival effect, shape (days, dishes)."""
    k = len(dishes)
    ev = cal["event"].values
    ids = dishes["dish_id"].values
    veg = dishes["is_veg"].values == 1
    cuisine = dishes["cuisine"].values
    mult = np.ones((len(cal), k))

    def apply(mask, per_dish):
        nonlocal mult
        mult = mult * np.where(mask[:, None], per_dish[None, :], 1.0)

    apply(ev == "navratri", np.where(veg, 1.15, 0.60))
    apply(ev == "eid", np.where(np.isin(ids, [1, 2]), 1.5, 1.0))
    apply(ev == "diwali", np.where(cuisine == "Sweets", 2.2, 1.1))
    apply(ev == "holi", np.where(cuisine == "Sweets", 1.5, np.where(ids == 20, 1.4, 1.0)))
    apply(ev == "pongal", np.where(cuisine == "South Indian", 1.3, 1.0))
    apply(ev == "christmas_newyear", np.full(k, 1.15))
    nye = ((cal["date"].dt.month == 12) & (cal["date"].dt.day == 31)).values
    apply(nye, np.full(k, 1.4))
    return mult

def inject_problems(df, rng):
    """Add realistic data-quality problems for the Silver layer to fix."""
    n = len(df)
    m = rng.random(n) < 0.06
    df.loc[m, "city"] = df.loc[m, "city"].str.upper()
    m = rng.random(n) < 0.04
    df.loc[m, "city"] = df.loc[m, "city"].str.lower()
    m = rng.random(n) < 0.002
    df.loc[m, "quantity"] = rng.choice([0, -1], size=m.sum())
    dup = df.sample(frac=0.01, random_state=int(rng.integers(1_000_000)))
    return pd.concat([df, dup], ignore_index=True)


def generate(start, end, dims_dir, out_dir, truth_dir, seed):
    rng = np.random.default_rng(seed)
    outlets = pd.read_csv(Path(dims_dir) / "outlets.csv")
    dishes = pd.read_csv(Path(dims_dir) / "dishes.csv")

    cal = build_calendar(start, end, rng)
    days = pd.DatetimeIndex(cal["date"])
    weather = build_weather(days, outlets["city"].unique(), rng)
    d_n, o_n, k_n, h_n = len(cal), len(outlets), len(dishes), len(HOURS)

    
    # rain flag per (day, outlet)
    rain = np.zeros((d_n, o_n))
    for j, city in enumerate(outlets["city"]):
        rain[:, j] = weather[weather["city"] == city].sort_values("date")["rain_flag"].values

    ids = dishes["dish_id"].values
    rain_factor = np.full(k_n, 1.12)
    rain_factor[np.isin(ids, [18, 19])] = 1.5   # chai and coffee up in rain
    rain_factor[np.isin(ids, [17, 20])] = 0.75  # cold items down in rain
    rain_mult = np.where(rain[:, :, None] == 1, rain_factor[None, None, :], 1.0)

    ipl_factor = np.where(dishes["category"] == "Starter", 1.25,
                          np.where(dishes["category"] == "Drink", 1.10, 1.0))
    evening = HOURS >= 19
    ipl_day = cal["is_ipl_day"].values
    ipl_mult = np.where(ipl_day[:, None, None, None] & evening[None, None, None, :],
                        ipl_factor[None, None, :, None], 1.0)

    weekday = WEEKDAY[cal["date"].dt.day_of_week.values]
    salary = np.where(cal["date"].dt.day.values <= 7, 1.08, 1.0)
    growth = 1 + 0.10 * np.arange(d_n) / 365
    ev_mult = event_multiplier(cal, dishes)
    noise = rng.gamma(20, 1 / 20, size=(d_n, o_n, k_n))  # daily randomness, mean 1

    lam = (BASE_RATE
           * dishes["popularity"].values[None, None, :, None]
           * outlets["size_factor"].values[None, :, None, None]
           * HOUR_CURVE[None, None, None, :]
           * weekday[:, None, None, None]
           * salary[:, None, None, None]
           * growth[:, None, None, None]
           * ev_mult[:, None, :, None]
           * rain_mult[:, :, :, None]
           * ipl_mult
           * noise[:, :, :, None])

    counts = rng.poisson(lam)
    d, o, k, h = np.nonzero(counts)
    reps = counts[d, o, k, h]
    d, o, k, h = (np.repeat(x, reps) for x in (d, o, k, h))
    n = len(d)

    secs = HOURS[h] * 3600 + rng.integers(0, 3600, n)
    ts = pd.to_datetime(cal["date"].values[d]) + pd.to_timedelta(secs, unit="s")
    id_start = int(pd.Timestamp(start).strftime("%Y%m%d")) * 10_000_000

    rating = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.04, 0.06, 0.15, 0.35, 0.40]).astype(float)
    rating[rng.random(n) < 0.20] = np.nan

    orders = pd.DataFrame({
        "order_id": id_start + np.arange(1, n + 1),
        "order_ts": ts,
        "outlet_id": outlets["outlet_id"].values[o],
        "city": outlets["city"].values[o],
        "dish_id": ids[k],
        "quantity": rng.choice([1, 2, 3], size=n, p=[0.75, 0.20, 0.05]),
        "unit_price": dishes["base_price"].values[k],
        "discount_pct": rng.choice([0, 5, 10, 15, 20], size=n, p=[0.55, 0.15, 0.15, 0.10, 0.05]),
        "platform": rng.choice(["Swiggy", "Zomato", "Direct App", "Own Website"],
                               size=n, p=[0.40, 0.35, 0.15, 0.10]),
        "rating": rating,
        "status": rng.choice(["delivered", "cancelled", "refunded"], size=n, p=[0.95, 0.04, 0.01]),
        "payment_mode": rng.choice(["UPI", "Card", "COD", "Wallet"], size=n, p=[0.60, 0.20, 0.15, 0.05]),
    })
    orders = inject_problems(orders, rng)

    # late-arriving rows: event happened on day X, file arrives on day X+1
    file_date = orders["order_ts"].dt.normalize()
    late = rng.random(len(orders)) < 0.015
    file_date[late] = file_date[late] + pd.Timedelta(days=1)
    orders["file_date"] = file_date

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = 0
    for fd, g in orders.groupby("file_date"):
        g = g.drop(columns="file_date")
        if fd < PAYMENT_MODE_FROM:
            g = g.drop(columns="payment_mode")  # schema evolution test for Auto Loader
        g.to_csv(out / f"orders_{fd:%Y-%m-%d}.csv", index=False)
        files += 1

    # reference files
    dims = Path(dims_dir)
    cal.to_csv(dims / "calendar_events.csv", index=False)
    weather.to_csv(dims / "weather_sim.csv", index=False)

    truth = pd.DataFrame({
        "date": np.repeat(cal["date"].values, o_n * k_n),
        "outlet_id": np.tile(np.repeat(outlets["outlet_id"].values, k_n), d_n),
        "dish_id": np.tile(ids, d_n * o_n),
        "expected_orders": lam.sum(axis=3).ravel().round(2),
    })
    Path(truth_dir).mkdir(parents=True, exist_ok=True)
    truth.to_csv(Path(truth_dir) / "expected_demand_truth.csv", index=False)

    print(f"Order rows written: {len(orders):,} across {files} daily files -> {out}")
    print(f"Duplicates injected (approx 1 percent), late rows: {int(late.sum()):,}")


    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--dims", default="data/dimensions")
    p.add_argument("--out", default="data/orders")
    p.add_argument("--truth", default="data/truth")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.start, a.end, a.dims, a.out, a.truth, a.seed)


# In[ ]:




