"""
NovaPreprocessor — a single, joblib-saveable object that wraps every
feature-engineering step (bucketing, target encoding, imputation, and
model-specific encoding) so training and the FastAPI service use the
EXACT same logic. This is what should be saved as preprocessor.joblib
instead of a bare list/dict.

Why this exists: the earlier scripts did feature engineering as loose
pandas code across several helper functions. That's fine inside a
notebook, but it can't be joblib.dump()'d as a working object — there's
nothing with a .transform() method to save. This class fixes that by
being a real scikit-learn transformer: fit() on training data, then
transform() on new data (in the API) reuses the exact stats learned
at training time (bin edges, target-encoding rate maps, medians).

Usage at TRAINING time:
    prep = NovaPreprocessor(model_type='catboost')   # or 'logreg'
    X_train_processed = prep.fit_transform(X_train_raw, y_train)
    X_test_processed  = prep.transform(X_test_raw)
    model.fit(X_train_processed, y_train, cat_features=prep.cat_feature_idx_)
    joblib.dump(prep, 'notebook/preprocessor.joblib')
    joblib.dump(model, 'notebook/Nova_Model.joblib')

Usage at SERVE time (inside FastAPI):
    df_processed = preprocessor.transform(raw_input_df)
    pred = model.predict(df_processed)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


AGE_BUCKET_BINS   = [-1, 30, 90, 180, 365, np.inf]
AGE_BUCKET_LABELS = ['<30d', '30-90d', '91-180d', '181-365d', '>1yr']

RATE_ENCODE_COLUMNS = {
    'age_bucket': 30,
    'kyc_tier': 30,
    'amount_bucket': 30,
    'chargeback_history_count': 20,
    'hour': 20,
}

NUMERIC_COLUMNS = [
    'amount_usd', 'fee', 'new_device', 'location_mismatch', 'ip_risk_score',
    'account_age_days', 'device_trust_score', 'chargeback_history_count',
    'risk_score_internal', 'txn_velocity_1h', 'txn_velocity_24h', 'hour',
]

CAT_COLUMNS = ['age_bucket', 'amount_bucket', 'kyc_tier']


def _clean_numeric_string(series):
    """Handles values like '9,998.85' — strips thousands separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(',', '', regex=False).str.strip(),
        errors='coerce'
    )


class NovaPreprocessor(BaseEstimator, TransformerMixin):
    """
    model_type: 'catboost' -> categoricals stay as raw strings
                              (returns a DataFrame; pass cat_feature_idx_
                              to CatBoostClassifier's .fit())
                'logreg'   -> categoricals one-hot encoded + numeric
                              columns scaled (returns a numpy array
                              ready for LogisticRegression)
    """

    def __init__(self, model_type='catboost', target_recall=0.80):
        self.model_type = model_type
        self.target_recall = target_recall

    # ------------------------------------------------------------
    def _engineer_raw(self, df):
        df = df.copy()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['hour'] = df['timestamp'].dt.hour
        # if hour was already provided directly (no timestamp), leave as-is

        for col in ['amount_src', 'amount_usd', 'fee']:
            if col in df.columns:
                df[col] = _clean_numeric_string(df[col])

        df['age_bucket'] = pd.cut(
            df['account_age_days'], bins=AGE_BUCKET_BINS,
            labels=AGE_BUCKET_LABELS, include_lowest=True
        ).astype(str)

        return df

    # ------------------------------------------------------------
    def fit(self, df, y):
        df = self._engineer_raw(df)
        y = pd.Series(np.asarray(y), index=df.index)

        # amount_bucket: qcut edges learned here, reused at transform time
        _, self.amount_bin_edges_ = pd.qcut(
            df['amount_usd'], q=6, duplicates='drop', retbins=True
        )
        df['amount_bucket'] = pd.cut(
            df['amount_usd'], bins=self.amount_bin_edges_, include_lowest=True
        ).astype(str)

        # medians for numeric imputation — computed on TRAIN only
        self.medians_ = {col: df[col].median() for col in NUMERIC_COLUMNS if col in df.columns}

        # smoothed target-encoding maps — computed on TRAIN only
        self.rate_maps_ = {}
        self.global_mean_ = y.mean()
        for col, m in RATE_ENCODE_COLUMNS.items():
            stats = y.groupby(df[col].fillna('missing').astype(str)).agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + self.global_mean_ * m) / (stats['count'] + m)
            self.rate_maps_[col] = smooth.to_dict()

        df_transformed = self._apply_transform(df, fitting=True)

        if self.model_type == 'logreg':
            self.scaler_ = StandardScaler()
            self.scaler_.fit(df_transformed[self.numeric_output_cols_])

        return self

    # ------------------------------------------------------------
    def _apply_transform(self, df, fitting=False):
        df = df.copy()

        for col, m in RATE_ENCODE_COLUMNS.items():
            key_series = df[col].fillna('missing').astype(str)
            df[f'rate_{col}'] = key_series.map(self.rate_maps_[col]).fillna(self.global_mean_)

        for col, med in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        if self.model_type == 'catboost':
            for col in CAT_COLUMNS:
                df[col] = df[col].fillna('missing').astype(str)

            feature_cols = (
                NUMERIC_COLUMNS + CAT_COLUMNS + [f'rate_{c}' for c in RATE_ENCODE_COLUMNS]
            )
            feature_cols = [c for c in feature_cols if c in df.columns]

            if fitting:
                self.feature_columns_ = feature_cols
                self.cat_feature_idx_ = [feature_cols.index(c) for c in CAT_COLUMNS]

            missing = set(self.feature_columns_) - set(df.columns)
            if missing:
                raise ValueError(f"Input is missing required columns: {missing}")

            return df[self.feature_columns_]

        else:  # logreg
            dummy_df = pd.get_dummies(df[CAT_COLUMNS], columns=CAT_COLUMNS, drop_first=False)
            numeric_cols = NUMERIC_COLUMNS + [f'rate_{c}' for c in RATE_ENCODE_COLUMNS]
            numeric_cols = [c for c in numeric_cols if c in df.columns]

            if fitting:
                self.numeric_output_cols_ = numeric_cols
                self.dummy_columns_ = list(dummy_df.columns)

            for col in self.dummy_columns_:
                if col not in dummy_df.columns:
                    dummy_df[col] = 0
            dummy_df = dummy_df[self.dummy_columns_]

            combined = pd.concat([df[numeric_cols].reset_index(drop=True),
                                   dummy_df.reset_index(drop=True)], axis=1)
            return combined

    # ------------------------------------------------------------
    def transform(self, df):
        df = self._engineer_raw(df)
        df['amount_bucket'] = pd.cut(
            df['amount_usd'], bins=self.amount_bin_edges_, include_lowest=True
        ).astype(str)

        out = self._apply_transform(df, fitting=False)

        if self.model_type == 'logreg':
            out[self.numeric_output_cols_] = self.scaler_.transform(out[self.numeric_output_cols_])

        return out