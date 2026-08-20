"""
Stage2Preprocessor — wraps the FULL Stage 2 feature pipeline (the one
behind your best RF/XGBoost results: rate encoding, velocity/fee
interaction features, device/IP sharing signals) as a single
fit/transform object, so it can be joblib.dump()'d and used correctly
in app.py — same reasoning as NovaPreprocessor, but matching Stage 2's
actual (more sophisticated) feature set instead of the simpler LR/CatBoost one.

IMPORTANT — this needs MORE raw columns at inference time than the
earlier NovaPreprocessor did: device_id, ip_address, and customer_id
are required (not just the numeric/categorical transaction fields),
because device_shared_accounts / ip_shared_accounts / ip_txn_count are
built from those. Your FastAPI Transaction schema needs device_id and
ip_address fields for this to work — customer_id is optional (only
used at fit time to count shared devices/IPs).

Usage at TRAINING time:
    prep = Stage2Preprocessor()
    X_train_processed = prep.fit_transform(df_train_raw, y_train)
    X_test_processed  = prep.transform(df_test_raw)
    model.fit(X_train_processed, y_train, sample_weight=sample_weight_train)
    joblib.dump(prep, 'notebook/preprocessor.joblib')
    joblib.dump(model, 'notebook/Nova_Model.joblib')

Usage at SERVE time (inside FastAPI):
    df_processed = preprocessor.transform(raw_input_df)   # raw_input_df needs
                                                            # device_id, ip_address, etc.
    pred = model.predict(df_processed)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


AGE_BINS = [-1, 30, 90, 180, 365, np.inf]
AGE_LABELS = ['<30d', '30-90d', '91-180d', '181-365d', '>1yr']

RAW_FEATURES = [
    'amount_usd', 'fee', 'new_device', 'location_mismatch', 'ip_risk_score',
    'kyc_tier', 'account_age_days', 'device_trust_score',
    'chargeback_history_count', 'risk_score_internal',
    'txn_velocity_1h', 'txn_velocity_24h', 'hour', 'age_bucket',
]

RATE_ENCODE_COLS = ['age_bucket', 'kyc_tier', 'amount_bucket', 'chargeback_history_count']
NUMERIC_IMPUTE_COLS = ['amount_usd', 'fee', 'device_trust_score', 'ip_risk_score']
CAT_CODE_COLS = ['age_bucket', 'kyc_tier']


class Stage2Preprocessor(BaseEstimator, TransformerMixin):

    def _base_engineer(self, df):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['age_bucket'] = pd.cut(
            df['account_age_days'], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
        ).astype(str)
        if 'amount_src' in df.columns:
            df['amount_src'] = pd.to_numeric(
                df['amount_src'].astype(str).str.replace(',', '', regex=False), errors='coerce'
            )
        return df

    # ------------------------------------------------------------
    def fit(self, df_raw, y):
        df = self._base_engineer(df_raw)
        y = pd.Series(np.asarray(y), index=df.index)

        missing = [c for c in RAW_FEATURES if c not in df.columns]
        assert not missing, f"Missing raw columns: {missing}"

        # amount_bucket edges — TRAIN only
        _, self.amount_bin_edges_ = pd.qcut(df['amount_usd'], q=6, duplicates='drop', retbins=True)
        df['amount_bucket'] = pd.cut(
            df['amount_usd'], bins=self.amount_bin_edges_, include_lowest=True
        ).astype(str)

        # rate encoding maps — TRAIN only, plain mean encoding (matches
        # your validated Stage 2 notebook exactly — not Bayesian-smoothed,
        # to avoid changing behavior from what you already tested)
        self.global_mean_ = y.mean()
        self.rate_maps_ = {}
        for col in RATE_ENCODE_COLS:
            self.rate_maps_[col] = y.groupby(df[col]).mean().to_dict()
        self._apply_rate_encoding(df)

        # numeric medians — TRAIN only
        self.medians_ = {c: df[c].median() for c in NUMERIC_IMPUTE_COLS}
        for c, med in self.medians_.items():
            df[c] = df[c].fillna(med)

        # interaction features (deterministic, no fit stats needed)
        self._apply_interaction_features(df)

        # sanitize inf -> per-column train median (store medians for reuse)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        self.sanitize_medians_ = {}
        for col in numeric_cols:
            if df[col].isna().any():
                med = df[col].median()
                self.sanitize_medians_[col] = med
                df[col] = df[col].fillna(med)

        # device/IP sharing lookups — TRAIN only, unseen defaults to 1 at transform time
        assert 'device_id' in df_raw.columns and 'ip_address' in df_raw.columns and 'customer_id' in df_raw.columns, \
            "Stage2Preprocessor.fit() requires device_id, ip_address, customer_id in the input."
        self.device_account_counts_ = df_raw.loc[df.index].groupby('device_id')['customer_id'].nunique().to_dict()
        self.ip_account_counts_ = df_raw.loc[df.index].groupby('ip_address')['customer_id'].nunique().to_dict()
        self.ip_txn_counts_ = df_raw.loc[df.index].groupby('ip_address').size().to_dict()

        self._apply_id_features(df, df_raw.loc[df.index])

        if 'corridor_risk' in df_raw.columns:
            self.has_corridor_risk_ = True
            df['corridor_risk'] = df_raw.loc[df.index, 'corridor_risk'].values
            self.corridor_risk_median_ = df['corridor_risk'].median()
            df['corridor_risk'] = df['corridor_risk'].fillna(self.corridor_risk_median_)
        else:
            self.has_corridor_risk_ = False

        # category -> codes, consistent mapping stored for transform time
        self.cat_dtypes_ = {}
        for col in CAT_CODE_COLS:
            cats = df[col].astype('category').cat.categories
            self.cat_dtypes_[col] = pd.CategoricalDtype(categories=cats)
            df[col] = df[col].astype(self.cat_dtypes_[col]).cat.codes

        self.feature_columns_ = df.drop(columns=[c for c in df.columns if c not in self._final_columns(df)]).columns.tolist() \
            if False else self._final_columns(df)

        return self

    def _final_columns(self, df):
        # Everything engineered EXCEPT: timestamp (not model input), and the
        # raw ID columns (device_id/ip_address/customer_id) which are only
        # needed to DERIVE device_shared_accounts/ip_shared_accounts/etc —
        # feeding the raw IDs themselves to the model would crash it
        # (they're high-cardinality strings, not real signal on their own).
        drop_always = {'timestamp', 'device_id', 'ip_address', 'customer_id'}
        return [c for c in df.columns if c not in drop_always]

    # ------------------------------------------------------------
    def _apply_rate_encoding(self, df):
        for col in RATE_ENCODE_COLS:
            df[f'rate_{col}'] = df[col].map(self.rate_maps_[col]).astype(float).fillna(self.global_mean_)
        # Match your notebook's original naming exactly (rate_amount_src,
        # rate_chargeback), not the generic f'rate_{col}' pattern.
        df.rename(columns={
            'rate_amount_bucket': 'rate_amount_src',
            'rate_chargeback_history_count': 'rate_chargeback',
        }, inplace=True)
        df.drop(columns=['amount_bucket'], errors='ignore', inplace=True)

    def _apply_interaction_features(self, df):
        df['velocity_ratio'] = (df['txn_velocity_1h'] / df['txn_velocity_24h'].replace(0, np.nan)).fillna(0)
        denom = (df['fee'] + 1).replace(0, np.nan)
        df['amount_to_fee_ratio'] = (df['amount_usd'] / denom).fillna(0)
        established = (df['account_age_days'] >= 90).astype(int)
        df['established_high_velocity'] = established * df['txn_velocity_1h']
        df['established_location_mismatch'] = established * df['location_mismatch']

    def _apply_id_features(self, df, df_raw_aligned):
        df['device_shared_accounts'] = df_raw_aligned['device_id'].map(self.device_account_counts_).fillna(1)
        df['ip_shared_accounts'] = df_raw_aligned['ip_address'].map(self.ip_account_counts_).fillna(1)
        df['ip_txn_count'] = df_raw_aligned['ip_address'].map(self.ip_txn_counts_).fillna(1)
        df['kyc_tier_low'] = df_raw_aligned['kyc_tier'].astype(str).str.lower().str.contains('low').astype(int)

    # ------------------------------------------------------------
    def transform(self, df_raw):
        df = self._base_engineer(df_raw)
        df['amount_bucket'] = pd.cut(
            df['amount_usd'], bins=self.amount_bin_edges_, include_lowest=True
        ).astype(str)

        self._apply_rate_encoding(df)

        for c, med in self.medians_.items():
            df[c] = df[c].fillna(med)

        self._apply_interaction_features(df)

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        for col, med in self.sanitize_medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)

        assert 'device_id' in df_raw.columns and 'ip_address' in df_raw.columns, \
            "Input is missing device_id/ip_address — required by this preprocessor."
        self._apply_id_features(df, df_raw)

        if self.has_corridor_risk_:
            if 'corridor_risk' in df_raw.columns:
                df['corridor_risk'] = df_raw['corridor_risk'].values
                df['corridor_risk'] = df['corridor_risk'].fillna(self.corridor_risk_median_)
            else:
                df['corridor_risk'] = self.corridor_risk_median_

        for col in CAT_CODE_COLS:
            df[col] = df[col].astype(self.cat_dtypes_[col]).cat.codes

        missing = set(self.feature_columns_) - set(df.columns)
        if missing:
            raise ValueError(f"Transform output is missing expected columns: {missing}")

        return df[self.feature_columns_]