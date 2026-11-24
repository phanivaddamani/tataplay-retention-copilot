"""part_a_data.py -- Audit, clean, store, and query Tata Play subscriber data."""
import os
import shutil
import sqlite3
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")
DB_PATH = os.path.join("data", "tataplay.db")
SQL_DIR = "sql"
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(SQL_DIR, exist_ok=True)

def audit_raw():
    sub_raw = pd.read_csv(os.path.join(RAW_DIR, "subscribers.csv"))
    rech_raw = pd.read_csv(os.path.join(RAW_DIR, "recharges.csv"))
    
    print("=== AUDIT: RAW SUBSCRIBERS ===")
    print(f"Shape: {sub_raw.shape}")
    print(f"Exact duplicates: {sub_raw.duplicated().sum()}")
    print(f"Distinct subscriber_id: {sub_raw['subscriber_id'].nunique()}")
    print(f"Blank age: {sub_raw['age'].isna().sum()}")
    num_age = pd.to_numeric(sub_raw['age'], errors='coerce')
    age_out = ((num_age < 10) | (num_age > 100)).sum()
    print(f"Age outside 10-100: {age_out}")
    print(f"Blank payment_mode: {sub_raw['payment_mode'].isna().sum()}")
    print(f"Distinct raw city spellings: {sub_raw['city'].nunique()}")
    print(f"Distinct raw auto_renew spellings: {sub_raw['auto_renew'].nunique()}")
    print(f"Blank next_cycle_recharge_inr: {sub_raw['next_cycle_recharge_inr'].isna().sum()}")
    print("\nMissing values per column in subscribers:")
    for col in sub_raw.columns:
        n_miss = sub_raw[col].isna().sum()
        pct = (n_miss / len(sub_raw)) * 100
        print(f"  {col}: {n_miss} ({pct:.2f}%) dtype={sub_raw[col].dtype}")
        
    print("\n=== AUDIT: RAW RECHARGES ===")
    print(f"Shape: {rech_raw.shape}")
    print(f"Exact duplicates: {rech_raw.duplicated().sum()}")
    print(f"Distinct recharge_id: {rech_raw['recharge_id'].nunique()}")
    non_num_amt = pd.to_numeric(rech_raw['amount_inr'], errors='coerce').isna().sum()
    print(f"Non-numeric amount_inr: {non_num_amt}")
    print(f"Distinct raw status spellings: {rech_raw['status'].nunique()}")
    valid_sids = set(sub_raw['subscriber_id'])
    orphan_rech = (~rech_raw['subscriber_id'].isin(valid_sids)).sum()
    print(f"Orphan recharges (subscriber_id not in subscribers): {orphan_rech}")
    print("\nMissing values per column in recharges:")
    for col in rech_raw.columns:
        n_miss = rech_raw[col].isna().sum()
        pct = (n_miss / len(rech_raw)) * 100
        print(f"  {col}: {n_miss} ({pct:.2f}%) dtype={rech_raw[col].dtype}")

def clean_data():
    sub = pd.read_csv(os.path.join(RAW_DIR, "subscribers.csv"))
    rech = pd.read_csv(os.path.join(RAW_DIR, "recharges.csv"))
    
    # Clean subscribers
    # 1. Remove exact duplicate rows
    sub = sub.drop_duplicates().copy()
    
    # 2. Standardise city
    city_map = {
        "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
        "bombay": "Mumbai", "mumbai": "Mumbai",
        "madras": "Chennai", "chennai": "Chennai",
        "calcutta": "Kolkata", "kolkata": "Kolkata",
        "new delhi": "Delhi", "delhi": "Delhi",
        "hyd": "Hyderabad", "hyderabad": "Hyderabad",
        "pune": "Pune"
    }
    sub['city'] = sub['city'].astype(str).str.strip().str.lower().map(city_map)
    
    # 3. Standardise auto_renew to Yes / No
    auto_map = {"y": "Yes", "yes": "Yes", "n": "No", "no": "No"}
    sub['auto_renew'] = sub['auto_renew'].astype(str).str.strip().str.lower().map(auto_map)
    
    # 4. Turn age outside 10-100 into missing
    sub['age'] = pd.to_numeric(sub['age'], errors='coerce')
    sub.loc[(sub['age'] < 10) | (sub['age'] > 100), 'age'] = np.nan
    
    # Clean recharges
    # 1. Remove exact duplicates
    rech = rech.drop_duplicates().copy()
    
    # 2. Remove rows whose subscriber does not exist in subscribers
    valid_sids = set(sub['subscriber_id'])
    rech = rech[rech['subscriber_id'].isin(valid_sids)].copy()
    
    # 3. Lower-case status
    rech['status'] = rech['status'].astype(str).str.strip().str.lower()
    
    # 4. Convert amount_inr to numbers; rebuild from monthly_recharge_inr where needed
    num_amt = pd.to_numeric(rech['amount_inr'], errors='coerce')
    rebuilt_mask = num_amt.isna()
    n_rebuilt = rebuilt_mask.sum()
    
    # Map subscriber monthly recharge
    sub_monthly = dict(zip(sub['subscriber_id'], sub['monthly_recharge_inr']))
    num_amt.loc[rebuilt_mask] = rech.loc[rebuilt_mask, 'subscriber_id'].map(sub_monthly)
    rech['amount_inr'] = num_amt.astype(int)
    
    # 5. Parse recharge_date as date (YYYY-MM-DD string)
    rech['recharge_date'] = pd.to_datetime(rech['recharge_date']).dt.strftime('%Y-%m-%d')
    
    print("\n=== CLEANING VERIFICATION ===")
    print(f"Clean subscribers shape: {sub.shape}")
    print("City distribution:\n", sub['city'].value_counts())
    print("Auto-renew distribution:\n", sub['auto_renew'].value_counts())
    print(f"Missing age: {sub['age'].isna().sum()}")
    print(f"Missing payment_mode: {sub['payment_mode'].isna().sum()}")
    print(f"Churned count: {sub['churned'].sum()} ({sub['churned'].mean()*100:.2f}%)")
    
    print(f"\nClean recharges shape: {rech.shape}")
    print(f"Amounts rebuilt: {n_rebuilt}")
    print("Status counts:\n", rech['status'].value_counts())
    succ_amt = rech.loc[rech['status'] == 'success', 'amount_inr'].sum()
    print(f"Total successful recharge amount: INR {succ_amt:,}")
    
    # Save clean files
    sub.to_csv(os.path.join(CLEAN_DIR, "subscribers.csv"), index=False)
    rech.to_csv(os.path.join(CLEAN_DIR, "recharges.csv"), index=False)
    shutil.copy(os.path.join(RAW_DIR, "plans.csv"), os.path.join(CLEAN_DIR, "plans.csv"))
    print("Clean CSVs saved to data/clean/")
    
    return sub, rech

def store_and_query():
    sub = pd.read_csv(os.path.join(CLEAN_DIR, "subscribers.csv"))
    rech = pd.read_csv(os.path.join(CLEAN_DIR, "recharges.csv"))
    plans = pd.read_csv(os.path.join(CLEAN_DIR, "plans.csv"))
    
    conn = sqlite3.connect(DB_PATH)
    sub.to_sql("subscribers", conn, if_exists="replace", index=False)
    rech.to_sql("recharges", conn, if_exists="replace", index=False)
    plans.to_sql("plans", conn, if_exists="replace", index=False)
    print("Stored tables into SQLite data/tataplay.db")
    
    queries = {
        "Q1_revenue_by_plan": """
SELECT 
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue,
    ROUND(AVG(r.amount_inr), 2) AS avg_recharge,
    COUNT(DISTINCT s.subscriber_id) AS distinct_subscribers
FROM recharges r
INNER JOIN subscribers s ON r.subscriber_id = s.subscriber_id
INNER JOIN plans p ON s.plan_id = p.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue DESC;
""",
        "Q2_never_successfully_recharged": """
SELECT 
    s.subscriber_id,
    s.tenure_months,
    s.monthly_recharge_inr
FROM subscribers s
LEFT JOIN recharges r 
    ON s.subscriber_id = r.subscriber_id 
    AND r.status = 'success'
WHERE r.recharge_id IS NULL;
""",
        "Q3_ranking_within_city": """
WITH subscriber_spend AS (
    SELECT 
        s.subscriber_id,
        s.city,
        SUM(r.amount_inr) AS total_spend
    FROM subscribers s
    INNER JOIN recharges r ON s.subscriber_id = r.subscriber_id
    WHERE r.status = 'success'
    GROUP BY s.subscriber_id, s.city
),
ranked_spend AS (
    SELECT 
        subscriber_id,
        city,
        total_spend,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, subscriber_id ASC) AS row_num
    FROM subscriber_spend
)
SELECT * FROM ranked_spend;
""",
        "Q4_running_total": """
SELECT 
    recharge_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (ORDER BY recharge_date ASC, recharge_id ASC) AS running_total
FROM recharges
WHERE subscriber_id = 'TP00010' AND status = 'success'
ORDER BY recharge_date ASC, recharge_id ASC;
"""
    }
    
    # Save queries to sql/queries.sql
    with open(os.path.join(SQL_DIR, "queries.sql"), "w") as f:
        for qname, qsql in queries.items():
            f.write(f"-- {qname}\n{qsql.strip()}\n\n")
            
    print("\n=== RUNNING SQL QUERIES ===")
    print("\n--- Q1: Revenue by Plan ---")
    df_q1 = pd.read_sql_query(queries["Q1_revenue_by_plan"], conn)
    print(df_q1)
    
    # Confirm one row with pandas groupby
    merged = rech[rech['status'] == 'success'].merge(sub, on='subscriber_id').merge(plans, on='plan_id')
    grp = merged.groupby('plan_name').agg(
        total_rev=('amount_inr', 'sum'),
        avg_rev=('amount_inr', 'mean'),
        sub_count=('subscriber_id', 'nunique')
    ).loc['Premium']
    print("\nPandas groupby confirmation for Premium plan:")
    print(f"Total: {grp['total_rev']}, Avg: {grp['avg_rev']:.2f}, Subs: {grp['sub_count']}")
    
    print("\n--- Q2: Subscribers Who Never Successfully Recharged ---")
    df_q2 = pd.read_sql_query(queries["Q2_never_successfully_recharged"], conn)
    print(f"Total never recharged: {len(df_q2)}")
    print(f"Count with tenure_months = 0: {(df_q2['tenure_months'] == 0).sum()}")
    
    print("\n--- Q3: Ranking Within City ---")
    df_q3 = pd.read_sql_query(queries["Q3_ranking_within_city"], conn)
    rank1 = df_q3[df_q3['rnk'] == 1].groupby('city').size()
    print("Subscribers with RANK() = 1 per city:\n", rank1)
    rownum1 = df_q3[df_q3['row_num'] == 1].groupby('city').size()
    print("Subscribers with ROW_NUMBER() = 1 per city:\n", rownum1)
    print("\nKolkata first 4 rows:")
    kolkata_top4 = df_q3[df_q3['city'] == 'Kolkata'].sort_values('row_num').head(4)
    print(kolkata_top4[['subscriber_id', 'total_spend', 'rnk', 'dense_rnk', 'row_num']])
    
    print("\n--- Q4: Running Total for TP00010 ---")
    df_q4 = pd.read_sql_query(queries["Q4_running_total"], conn)
    print(df_q4)
    
    conn.close()

if __name__ == "__main__":
    audit_raw()
    clean_data()
    store_and_query()
