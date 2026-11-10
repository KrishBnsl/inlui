"""Small, serializable research baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LastObservationRegressor(RegressorMixin, BaseEstimator):
    """Predict the most recent strictly-prior-year cutoff.

    ``last_closing_rank`` is produced by the past-only feature builder.  The
    estimator intentionally has no learned parameters; ``fit`` validates the
    contract and records the sklearn fitted-state marker for serialization.
    """

    feature_name = "last_closing_rank"

    def fit(self, X: pd.DataFrame, y: object = None) -> "LastObservationRegressor":
        self._values(X)
        self.is_fitted_ = True
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.feature_name], dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        return self._values(X)

    def _values(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LastObservationRegressor requires a pandas DataFrame")
        if self.feature_name not in X.columns:
            raise ValueError(f"Missing required feature: {self.feature_name}")
        values = pd.to_numeric(X[self.feature_name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("last_closing_rank must be finite for every forecast row")
        return values
