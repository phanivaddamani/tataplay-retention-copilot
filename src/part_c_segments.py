"""part_c_segments.py -- K-Means segmentation, spend regression, and revenue-at-risk prioritization."""
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_part_c():
    print("=== PART C: SEGMENTATION & SPEND MODEL ===")
    sub = pd.read_csv(os.path.join(CLEAN_DIR, "subscribers.csv"))
    
    # -------------------------------------------------------------
    # Task 1: K-Means Clustering Sweep (K=2 to 8)
    # -------------------------------------------------------------
    cluster_features = ["tenure_months", "hours_watched_per_week", "monthly_recharge_inr"]
    X_clus = sub[cluster_features].copy()
    
    scaler_clus = StandardScaler()
    X_clus_scaled = scaler_clus.fit_transform(X_clus)
    
    print("\n--- K-Means Clustering Sweep ---")
    print(" K |  Inertia  | Silhouette")
    print("---|-----------|-----------")
    sweep_results = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_clus_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_clus_scaled, labels)
        sweep_results[k] = (inertia, sil)
        print(f" {k} | {inertia:9.1f} |   {sil:.3f}")
        
    # -------------------------------------------------------------
    # Task 2: Profile, Name, and Assign for K=4
    # -------------------------------------------------------------
    K_CHOSEN = 4
    km_final = KMeans(n_clusters=K_CHOSEN, n_init=10, random_state=42)
    sub_clusters = km_final.fit_predict(X_clus_scaled)
    sub['cluster_id'] = sub_clusters
    
    print(f"\n--- Cluster Profiling (K={K_CHOSEN}) ---")
    profiles = sub.groupby('cluster_id').agg(
        size=('subscriber_id', 'count'),
        mean_tenure=('tenure_months', 'mean'),
        mean_hours=('hours_watched_per_week', 'mean'),
        mean_monthly=('monthly_recharge_inr', 'mean'),
        churn_rate=('churned', 'mean')
    ).reset_index()
    
    # Identify Premium and At-Risk clusters
    prem_idx = profiles['mean_monthly'].idxmax()
    prem_cluster = profiles.loc[prem_idx, 'cluster_id']
    
    at_risk_idx = profiles['churn_rate'].idxmax()
    at_risk_cluster = profiles.loc[at_risk_idx, 'cluster_id']
    
    # Assign names based on cluster characteristics
    # For K=4:
    # - highest monthly spend -> Premium
    # - highest churn / lowest tenure -> At-Risk
    # - remaining two: one is high hours/moderate spend (e.g., Heavy Viewers/Family), one is low spend/moderate tenure (Budget)
    cluster_names = {}
    for _, row in profiles.iterrows():
        cid = int(row['cluster_id'])
        if cid == prem_cluster:
            cluster_names[cid] = "Premium"
        elif cid == at_risk_cluster:
            cluster_names[cid] = "At-Risk"
        elif row['mean_hours'] > profiles['mean_hours'].median():
            cluster_names[cid] = "Family Streamers"
        else:
            cluster_names[cid] = "Budget Linear"
            
    profiles['segment_name'] = profiles['cluster_id'].map(cluster_names)
    print(profiles.to_string(index=False))
    
    # Save outputs/segments.csv
    sub['segment_name'] = sub['cluster_id'].map(cluster_names)
    segments_df = sub[['subscriber_id', 'cluster_id', 'segment_name']].rename(columns={'cluster_id': 'segment_id'})
    segments_path = os.path.join(OUTPUT_DIR, "segments.csv")
    segments_df.to_csv(segments_path, index=False)
    print(f"\nSaved segments to {segments_path} ({len(segments_df)} rows)")
    
    # Hand assignment of new subscriber:
    # tenure 4 months, 15 hours per week, monthly recharge INR 520
    new_sub = np.array([[4, 15, 520]])
    new_sub_scaled = (new_sub - scaler_clus.mean_) / scaler_clus.scale_
    centroids = km_final.cluster_centers_
    
    distances = np.linalg.norm(centroids - new_sub_scaled, axis=1)
    nearest_idx = np.argmin(distances)
    predicted_cluster = km_final.predict(new_sub_scaled)[0]
    
    print("\n--- Hand Assignment for New Subscriber (4 mo, 15 hrs/wk, INR 520) ---")
    print(f"Scaled vector: {new_sub_scaled[0]}")
    for cid, dist in enumerate(distances):
        print(f"  Cluster {cid} ({cluster_names[cid]}): distance = {dist:.4f}")
    print(f"Nearest Centroid by distance: Cluster {nearest_idx} ({cluster_names[nearest_idx]}) with distance {distances[nearest_idx]:.4f}")
    print(f"KMeans predict() result:      Cluster {predicted_cluster} ({cluster_names[predicted_cluster]})")
    assert nearest_idx == predicted_cluster, "Hand distance and predict() mismatch!"
    print("Verification: Hand calculation matches kmeans.predict() perfectly.")
    
    # -------------------------------------------------------------
    # Task 3: Collinearity & Spend Model (Ridge / Lasso)
    # -------------------------------------------------------------
    print("\n--- Collinearity Analysis ---")
    view_cols = ["hours_watched_per_week", "hd_hours_per_week", "sports_hours_per_week", "channels_watched"]
    corr_matrix = sub[view_cols].corr()
    print("Correlation Matrix:\n", corr_matrix.round(3))
    
    print("\nFeature pairs with |correlation| > 0.90:")
    for i in range(len(view_cols)):
        for j in range(i + 1, len(view_cols)):
            c_val = corr_matrix.iloc[i, j]
            if abs(c_val) > 0.90:
                print(f"  {view_cols[i]} -- {view_cols[j]}: {c_val:.3f}")
                
    # Spend Model on non-churners (churned == 0)
    non_churners = sub[sub['churned'] == 0].copy()
    spend_features = [
        "monthly_recharge_inr", "tenure_months", "hours_watched_per_week",
        "hd_hours_per_week", "sports_hours_per_week", "channels_watched",
        "support_calls_90d"
    ]
    X_spend = non_churners[spend_features]
    y_spend = non_churners['next_cycle_recharge_inr']
    
    X_sp_train, X_sp_test, y_sp_train, y_sp_test = train_test_split(
        X_spend, y_spend, test_size=0.2, random_state=42
    )
    print(f"\nSpend Model Splits: Train={len(X_sp_train)} rows, Test={len(X_sp_test)} rows")
    
    # Grid search for Ridge and Lasso
    alphas = [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 10, 30, 100]
    
    # 1. Linear Regression
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
    lr_pipe.fit(X_sp_train, y_sp_train)
    y_sp_pred_lr = lr_pipe.predict(X_sp_test)
    
    # 2. Ridge
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
    ridge_gs = GridSearchCV(ridge_pipe, {"reg__alpha": alphas}, cv=5, scoring="r2")
    ridge_gs.fit(X_sp_train, y_sp_train)
    best_ridge = ridge_gs.best_estimator_
    y_sp_pred_ridge = best_ridge.predict(X_sp_test)
    
    # 3. Lasso
    lasso_pipe = Pipeline([("scaler", StandardScaler()), ("reg", Lasso(max_iter=10000, random_state=42))])
    lasso_gs = GridSearchCV(lasso_pipe, {"reg__alpha": alphas}, cv=5, scoring="r2")
    lasso_gs.fit(X_sp_train, y_sp_train)
    best_lasso = lasso_gs.best_estimator_
    y_sp_pred_lasso = best_lasso.predict(X_sp_test)
    
    print("\n--- Regression Model Comparison ---")
    def report_reg(name, best_a, cv_r2, y_pred, y_true):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        print(f"{name:18s} | Alpha: {str(best_a):5s} | CV R2: {cv_r2:.4f} | Test R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
        return r2, mae, rmse

    # For OLS CV R2:
    from sklearn.model_selection import cross_val_score
    lr_cv_r2 = cross_val_score(lr_pipe, X_sp_train, y_sp_train, cv=5, scoring="r2").mean()
    
    report_reg("Linear Regression", "N/A", lr_cv_r2, y_sp_pred_lr, y_sp_test)
    report_reg("Ridge Regression", ridge_gs.best_params_['reg__alpha'], ridge_gs.best_score_, y_sp_pred_ridge, y_sp_test)
    report_reg("Lasso Regression", lasso_gs.best_params_['reg__alpha'], lasso_gs.best_score_, y_sp_pred_lasso, y_sp_test)
    
    # Lasso Zero Coefficients
    lasso_coefs = pd.Series(best_lasso.named_steps['reg'].coef_, index=spend_features)
    print("\nLasso Coefficients (standardized features):")
    for feat, coef in lasso_coefs.items():
        print(f"  {feat:24s}: {coef:+.4f}")
    zero_coefs = lasso_coefs[lasso_coefs == 0].index.tolist()
    print(f"Lasso dropped features (coef == 0): {zero_coefs}")
    
    # Stability Experiment (5 random 80% subsamples)
    print("\n--- Stability Experiment (5 random 80% subsamples) ---")
    ols_hours_coefs = []
    ridge30_hours_coefs = []
    
    for k in range(5):
        sample = X_sp_train.sample(frac=0.8, random_state=k)
        sample_y = y_sp_train.loc[sample.index]
        
        # OLS fit
        p_ols = Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())])
        p_ols.fit(sample, sample_y)
        coef_ols = p_ols.named_steps['reg'].coef_[spend_features.index('hours_watched_per_week')]
        ols_hours_coefs.append(coef_ols)
        
        # Ridge alpha=30 fit
        p_ridge30 = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=30))])
        p_ridge30.fit(sample, sample_y)
        coef_ridge30 = p_ridge30.named_steps['reg'].coef_[spend_features.index('hours_watched_per_week')]
        ridge30_hours_coefs.append(coef_ridge30)
        
    std_ols_pop = np.std(ols_hours_coefs, ddof=0)
    std_ridge_pop = np.std(ridge30_hours_coefs, ddof=0)
    std_ols_sample = np.std(ols_hours_coefs, ddof=1)
    std_ridge_sample = np.std(ridge30_hours_coefs, ddof=1)
    
    print("Coefficient of hours_watched_per_week across 5 subsamples:")
    print(f"  OLS fits:      {[round(c, 3) for c in ols_hours_coefs]}")
    print(f"  Ridge(30) fits:{[round(c, 3) for c in ridge30_hours_coefs]}")
    print(f"  Population std (ddof=0): OLS = {std_ols_pop:.2f} vs Ridge(30) = {std_ridge_pop:.2f}")
    print(f"  Sample std     (ddof=1): OLS = {std_ols_sample:.2f} vs Ridge(30) = {std_ridge_sample:.2f}")
    
    # -------------------------------------------------------------
    # Task 4: Revenue at Risk for Test Subscribers
    # -------------------------------------------------------------
    print("\n--- Revenue at Risk Prioritization ---")
    test_scores = pd.read_csv(os.path.join(OUTPUT_DIR, "test_scores.csv"))
    
    # Join spend features from clean subscribers
    test_full = test_scores.merge(sub, on='subscriber_id', suffixes=('', '_sub'))
    
    # Predict next cycle recharge using best Ridge pipeline
    X_test_spend = test_full[spend_features]
    predicted_next_recharge = best_ridge.predict(X_test_spend)
    test_full['predicted_next_recharge_inr'] = np.round(predicted_next_recharge, 2)
    test_full['revenue_at_risk_inr'] = np.round(test_full['churn_probability'] * test_full['predicted_next_recharge_inr'], 2)
    
    # Sort top 15 by revenue at risk
    top15 = test_full.sort_values('revenue_at_risk_inr', ascending=False).head(15).copy()
    priority_cols = [
        "subscriber_id", "churn_probability", "predicted_next_recharge_inr",
        "revenue_at_risk_inr", "segment_name", "churned"
    ]
    top15_out = top15[priority_cols]
    
    priority_path = os.path.join(OUTPUT_DIR, "retention_priority.csv")
    top15_out.to_csv(priority_path, index=False)
    print(f"Saved top 15 priority list to {priority_path}")
    print(top15_out.to_string(index=False))
    
    tot_rar = top15_out['revenue_at_risk_inr'].sum()
    n_churned_top15 = top15_out['churned'].sum()
    print(f"\nTop 15 Total Revenue at Risk: INR {tot_rar:,.2f}")
    print(f"Top 15 Actual Churners: {n_churned_top15} of 15 ({n_churned_top15/15*100:.1f}%)")

if __name__ == "__main__":
    run_part_c()
