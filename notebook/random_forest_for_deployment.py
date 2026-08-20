"""
Final deployment save step for Random Forest (best_rf_s2).

Run this AFTER stage2_selfcontained_rf_vs_xgb.py has produced best_rf_s2
in your kernel (or paste this logic into cells right after it — either
way, run top to bottom in a fresh kernel to avoid the drift issues
from earlier in this session).

This refits Stage2Preprocessor properly as a real, saveable object
(replacing the loose df_s2/X_train_s2 script variables) and pairs it
with best_rf_s2 for deployment.
"""

import joblib
from stage2_preprocessing import Stage2Preprocessor

assert 'best_rf_s2' in dir(), (
    "best_rf_s2 not found — run stage2_selfcontained_rf_vs_xgb.py first "
    "(fresh kernel, top to bottom)."
)
assert 'df_s2' in dir() and 'y_s2' in dir(), (
    "df_s2/y_s2 not found — these come from the same script."
)

# Refit the preprocessor as a real object using the SAME train/test split
# indices the model was actually trained on, so nothing drifts between
# what best_rf_s2 learned and what preprocessor.transform() produces.
from sklearn.model_selection import train_test_split

raw_cols_needed = [
    'timestamp', 'amount_usd', 'fee', 'new_device', 'location_mismatch',
    'ip_risk_score', 'kyc_tier', 'account_age_days', 'device_trust_score',
    'chargeback_history_count', 'risk_score_internal', 'txn_velocity_1h',
    'txn_velocity_24h', 'device_id', 'ip_address', 'customer_id',
]
if 'corridor_risk' in df_s2.columns:
    raw_cols_needed.append('corridor_risk')

df_train_raw, df_test_raw, y_train_raw, y_test_raw = train_test_split(
    df_s2[raw_cols_needed], y_s2, test_size=0.2, stratify=y_s2, random_state=42
)

preprocessor = Stage2Preprocessor()
X_train_final = preprocessor.fit_transform(df_train_raw, y_train_raw)
X_test_final = preprocessor.transform(df_test_raw)

# Sanity check: this should score close to what you already saw in the
# confusion matrix. If it's meaningfully different, the refit split
# doesn't match what best_rf_s2 was originally trained on — flag it
# rather than deploying a mismatched pair.
from sklearn.metrics import average_precision_score
check_proba = best_rf_s2.predict_proba(X_test_final)[:, 1]
check_prauc = average_precision_score(y_test_raw, check_proba)
print(f"Sanity-check PR-AUC on refit preprocessor: {check_prauc:.4f}")
print("(Compare this to the PR-AUC printed when best_rf_s2 was first trained — "
      "should be close. If it's way off, don't deploy this pair yet.)")

# --- save both as real fitted objects ---
assert hasattr(preprocessor, 'transform'), f"preprocessor is a {type(preprocessor)}, not fitted!"
assert hasattr(best_rf_s2, 'predict'), f"best_rf_s2 is a {type(best_rf_s2)}, not fitted!"

joblib.dump(preprocessor, 'notebook/preprocessor.joblib')
joblib.dump(best_rf_s2, 'notebook/Nova_Model.joblib')

# reload check in a clean load, exactly like app.py will do
reloaded_prep = joblib.load('notebook/preprocessor.joblib')
reloaded_model = joblib.load('notebook/Nova_Model.joblib')
assert hasattr(reloaded_prep, 'transform')
sample_pred = reloaded_model.predict(reloaded_prep.transform(df_test_raw.iloc[:3]))
print("Reload sanity check passed. Sample predictions:", sample_pred)
print("Saved: notebook/preprocessor.joblib, notebook/Nova_Model.joblib")