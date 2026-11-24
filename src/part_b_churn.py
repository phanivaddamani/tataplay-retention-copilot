"""part_b_churn.py -- Churn prediction pipeline, tuning, thresholding, and scoring."""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_features

CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COST_FN = 1200  # missed churner
COST_FP = 300   # unnecessary retention offer

def run_part_b():
    print("=== PART B: CHURN PREDICTION ===")
    sub = pd.read_csv(os.path.join(CLEAN_DIR, "subscribers.csv"))
    rech = pd.read_csv(os.path.join(CLEAN_DIR, "recharges.csv"))
    
    # 1. Feature Engineering
    df = build_features(sub, rech, snapshot_date="2026-03-31")
    n_recency_missing = df['recency_days'].isna().sum()
    print(f"Subscribers shape: {df.shape}")
    print(f"Recency days missing count: {n_recency_missing}")
    
    # Class balance
    n_churn = df['churned'].sum()
    print(f"Class balance: {n_churn} of {len(df)} ({df['churned'].mean()*100:.2f}%)")
    
    # 2. Three-way split (Train 60%, Val 20%, Test 20%)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df['churned'].copy()
    sids = df['subscriber_id'].copy()
    
    # Hold out 20% test set
    X_temp, X_test, y_temp, y_test, sids_temp, sids_test = train_test_split(
        X, y, sids, test_size=0.2, stratify=y, random_state=42
    )
    # Split remaining into train (60% of total = 75% of temp) and validation (20% of total = 25% of temp)
    X_train, X_val, y_train, y_val, sids_train, sids_val = train_test_split(
        X_temp, y_temp, sids_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    print("\n--- Data Splits ---")
    print(f"Train: {len(X_train)} rows ({y_train.sum()} churners, {y_train.mean()*100:.2f}%)")
    print(f"Val:   {len(X_val)} rows ({y_val.sum()} churners, {y_val.mean()*100:.2f}%)")
    print(f"Test:  {len(X_test)} rows ({y_test.sum()} churners, {y_test.mean()*100:.2f}%)")
    
    # 3. Leakage Experiment
    print("\n--- Leakage Experiment ---")
    leaky_num = NUMERIC_FEATURES + ["next_cycle_recharge_inr"]
    leaky_cat = CATEGORICAL_FEATURES + ["closure_request_logged"]
    
    X_train_leaky = df.loc[X_train.index, leaky_num + leaky_cat]
    X_val_leaky = df.loc[X_val.index, leaky_num + leaky_cat]
    
    leaky_preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), leaky_num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), leaky_cat)
    ])
    leaky_pipe = Pipeline([
        ("prep", leaky_preprocessor),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    leaky_pipe.fit(X_train_leaky, y_train)
    y_val_leaky_pred = leaky_pipe.predict(X_val_leaky)
    print(f"With Leakage    - Val Accuracy: {accuracy_score(y_val, y_val_leaky_pred):.3f}, Val F1: {f1_score(y_val, y_val_leaky_pred):.3f}")
    
    # 4. Clean Preprocessor (Fit on train only)
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES)
    ])
    preprocessor.fit(X_train)
    
    X_train_trans = preprocessor.transform(X_train)
    print("\n--- Preprocessor Verification ---")
    print(f"Transformed output columns: {X_train_trans.shape[1]}")
    med_age = preprocessor.named_transformers_['num'].named_steps['imp'].statistics_[NUMERIC_FEATURES.index('age')]
    print(f"Learned median age: {med_age:.1f}")
    scaler_tenure_mean = preprocessor.named_transformers_['num'].named_steps['scaler'].mean_[NUMERIC_FEATURES.index('tenure_months')]
    full_tenure_mean = df['tenure_months'].mean()
    print(f"Scaler mean tenure_months: {scaler_tenure_mean:.2f} vs Full-data mean: {full_tenure_mean:.2f}")
    print(f"Remaining missing values in transformed train: {np.isnan(X_train_trans).sum()}")
    
    # 5. Baseline and Simple Models
    print("\n--- Baseline & Simple Models (Validation Set) ---")
    # Majority baseline
    maj_pred = np.zeros(len(y_val), dtype=int)
    print(f"Majority Baseline - Acc: {accuracy_score(y_val, maj_pred):.3f}, F1: {f1_score(y_val, maj_pred):.3f}")
    
    # Logistic Regression
    lr_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    lr_pipe.fit(X_train, y_train)
    lr_val_pred = lr_pipe.predict(X_val)
    print(f"Logistic Regression - Acc: {accuracy_score(y_val, lr_val_pred):.3f}, "
          f"Prec: {precision_score(y_val, lr_val_pred):.3f}, "
          f"Recall: {recall_score(y_val, lr_val_pred):.3f}, "
          f"F1: {f1_score(y_val, lr_val_pred):.3f}")
    
    # Unrestricted Decision Tree
    dt_pipe = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=42))])
    dt_pipe.fit(X_train, y_train)
    dt_train_acc = accuracy_score(y_train, dt_pipe.predict(X_train))
    dt_val_f1 = f1_score(y_val, dt_pipe.predict(X_val))
    print(f"Unrestricted Decision Tree - Train Acc: {dt_train_acc:.3f}, Val F1: {dt_val_f1:.3f}")
    
    # Default Random Forest
    rf_pipe = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_pipe.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_pipe.predict(X_train))
    rf_val_f1 = f1_score(y_val, rf_pipe.predict(X_val))
    print(f"Default Random Forest     - Train Acc: {rf_train_acc:.3f}, Val F1: {rf_val_f1:.3f}")
    print(f"Without Leakage RF        - Val Acc: {accuracy_score(y_val, rf_pipe.predict(X_val)):.3f}, Val F1: {rf_val_f1:.3f}")
    
    # Sweep Decision Tree max_depth 1 to 10
    print("\nDecision Tree Depth Sweep:")
    print("Depth | Train Acc | Val F1")
    for d in range(1, 11):
        d_pipe = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(max_depth=d, random_state=42))])
        d_pipe.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, d_pipe.predict(X_train))
        v_f1 = f1_score(y_val, d_pipe.predict(X_val))
        print(f"  {d:2d}  |   {tr_acc:.3f}   |  {v_f1:.3f}")
        
    # 6. Cross-Validation Tuning
    print("\n--- Cross-Validation Tuning (Training Set Only) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tune Logistic Regression
    lr_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=cv, scoring="f1", n_jobs=-1)
    lr_gs.fit(X_train, y_train)
    print(f"Logistic Regression Best Params: {lr_gs.best_params_}")
    print(f"Logistic Regression Best CV F1: {lr_gs.best_score_:.3f} (std: {lr_gs.cv_results_['std_test_score'][lr_gs.best_index_]:.3f})")
    
    # Tune Random Forest
    rf_grid = {
        "clf__n_estimators": [50, 100, 150],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__max_features": [3, 4, 5]
    }
    rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=cv, scoring="f1", n_jobs=-1)
    rf_gs.fit(X_train, y_train)
    n_combos = len(rf_gs.cv_results_['params'])
    print(f"Random Forest Grid: {n_combos} combos, {n_combos * 5} fits")
    print(f"Random Forest Best Params: {rf_gs.best_params_}")
    print(f"Random Forest Best CV F1: {rf_gs.best_score_:.3f} (std: {rf_gs.cv_results_['std_test_score'][rf_gs.best_index_]:.3f})")
    
    best_lr = lr_gs.best_estimator_
    best_rf = rf_gs.best_estimator_
    
    # 7. Threshold Search by Cost on Validation Set
    print("\n--- Threshold Optimization by Cost on Validation Set ---")
    thresholds = np.round(np.arange(0.10, 0.95, 0.05), 2)
    
    val_probs_lr = best_lr.predict_proba(X_val)[:, 1]
    val_probs_rf = best_rf.predict_proba(X_val)[:, 1]
    
    def find_best_threshold(y_true, probs):
        best_th, min_cost = 0.5, float('inf')
        for th in thresholds:
            preds = (probs >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
            cost = COST_FN * fn + COST_FP * fp
            if cost < min_cost:
                min_cost = cost
                best_th = th
        return best_th, min_cost

    best_th_lr, min_cost_lr = find_best_threshold(y_val, val_probs_lr)
    best_th_rf, min_cost_rf = find_best_threshold(y_val, val_probs_rf)
    print(f"Logistic Regression: Optimal Threshold = {best_th_lr:.2f}, Val Cost = INR {min_cost_lr:,}")
    print(f"Random Forest:       Optimal Threshold = {best_th_rf:.2f}, Val Cost = INR {min_cost_rf:,}")
    
    # 8. Test Set Evaluation (Once!)
    print("\n--- Test Set Evaluation ---")
    test_churners = y_test.sum()
    cost_no_action = COST_FN * test_churners
    cost_offer_all = COST_FP * (len(y_test) - test_churners)
    print(f"Reference Baseline - No Action Cost:  INR {cost_no_action:,} (misses {test_churners} churners)")
    print(f"Reference Baseline - Offer All Cost:  INR {cost_offer_all:,}")
    
    def eval_test(name, model, th, X_t, y_t):
        probs = model.predict_proba(X_t)[:, 1]
        preds = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, preds).ravel()
        cost = COST_FN * fn + COST_FP * fp
        prec = precision_score(y_t, preds)
        rec = recall_score(y_t, preds)
        f1 = f1_score(y_t, preds)
        print(f"\n{name} (Threshold {th:.2f}):")
        print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
        print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
        print(f"  Total Cost: INR {cost:,} ({(cost/cost_no_action)*100:.1f}% of No-Action)")
        return tn, fp, fn, tp, prec, rec, f1, cost, probs
        
    eval_test("Logistic Regression", best_lr, best_th_lr, X_test, y_test)
    tn_rf, fp_rf, fn_rf, tp_rf, prec_rf, rec_rf, f1_rf, cost_rf, test_probs_rf = eval_test(
        "Random Forest", best_rf, best_th_rf, X_test, y_test
    )
    
    # 9. Feature Interpretability
    print("\n--- Feature Interpretability ---")
    # Feature names
    feature_names = preprocessor.get_feature_names_out()
    
    # RF Feature Importances
    rf_importances = best_rf.named_steps['clf'].feature_importances_
    rf_fi = pd.Series(rf_importances, index=feature_names).sort_values(ascending=False)
    print("Top 5 Random Forest Features:")
    for feat, imp in rf_fi.head(5).items():
        print(f"  {feat}: {imp:.4f}")
        
    # LR Coefficients
    lr_coefs = pd.Series(best_lr.named_steps['clf'].coef_[0], index=feature_names)
    print("\nKey Logistic Regression Coefficients:")
    for feat in ["num__support_calls_90d", "num__tenure_months", "num__n_failed_6m"]:
        if feat in lr_coefs:
            print(f"  {feat}: {lr_coefs[feat]:+.4f}")
            
    # 10. Persist Model and Test Predictions
    print("\n--- Persisting Model and Test Scores ---")
    # Select Random Forest based on validation cost
    selected_model = best_rf
    model_path = os.path.join(OUTPUT_DIR, "churn_model.joblib")
    joblib.dump(selected_model, model_path)
    print(f"Saved selected model to {model_path}")
    
    # Verify reload
    loaded_model = joblib.load(model_path)
    loaded_probs = loaded_model.predict_proba(X_test)[:, 1]
    assert np.allclose(test_probs_rf, loaded_probs), "Reloaded model probabilities do not match!"
    print("Verified: Reloaded model reproduces predictions exactly.")
    
    # Write outputs/test_scores.csv
    test_scores_df = pd.DataFrame({
        "subscriber_id": sids_test.values,
        "churn_probability": np.round(test_probs_rf, 4),
        "churned": y_test.values
    })
    test_scores_path = os.path.join(OUTPUT_DIR, "test_scores.csv")
    test_scores_df.to_csv(test_scores_path, index=False)
    print(f"Saved test scores to {test_scores_path} ({len(test_scores_df)} rows)")

if __name__ == "__main__":
    run_part_b()
