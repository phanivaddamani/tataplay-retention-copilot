"""features.py -- Shared feature engineering functions for Tata Play Copilot."""
import pandas as pd
import numpy as np

NUMERIC_FEATURES = [
    "tenure_months", "age", "add_on_count", "monthly_recharge_inr",
    "hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week",
    "channels_watched", "support_calls_90d", "n_success_6m",
    "n_failed_6m", "recency_days"
]

CATEGORICAL_FEATURES = ["city", "plan_id", "payment_mode", "auto_renew"]

def build_features(subscribers_df, recharges_df, snapshot_date="2026-03-31"):
    """
    Computes recharge aggregate features as of snapshot_date and joins them to subscribers.
    Features:
      - n_success_6m: count of successful recharges
      - n_failed_6m: count of failed recharges
      - recency_days: days from snapshot_date to last successful recharge (NaN if none)
    """
    snap = pd.to_datetime(snapshot_date)
    rech = recharges_df.copy()
    rech['recharge_date'] = pd.to_datetime(rech['recharge_date'])
    
    # Filter on or before snapshot date
    rech = rech[rech['recharge_date'] <= snap]
    
    # Success recharges
    succ = rech[rech['status'] == 'success']
    succ_agg = succ.groupby('subscriber_id').agg(
        n_success_6m=('recharge_id', 'count'),
        last_success=('recharge_date', 'max')
    ).reset_index()
    succ_agg['recency_days'] = (snap - succ_agg['last_success']).dt.days
    
    # Failed recharges
    fail = rech[rech['status'] == 'failed']
    fail_agg = fail.groupby('subscriber_id').agg(
        n_failed_6m=('recharge_id', 'count')
    ).reset_index()
    
    # Merge with subscribers
    df = subscribers_df.copy()
    df = df.merge(succ_agg[['subscriber_id', 'n_success_6m', 'recency_days']], on='subscriber_id', how='left')
    df = df.merge(fail_agg[['subscriber_id', 'n_failed_6m']], on='subscriber_id', how='left')
    
    # Fill counts with 0 for subscribers with no recharges
    df['n_success_6m'] = df['n_success_6m'].fillna(0).astype(int)
    df['n_failed_6m'] = df['n_failed_6m'].fillna(0).astype(int)
    # recency_days remains NaN for subscribers with no successful recharges
    
    return df
