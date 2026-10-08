from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


def correlation_filter(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Select features with low pairwise correlation."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in X.columns if col not in to_drop]


def mutual_information_selection(X: pd.DataFrame, y: pd.Series, k: int = 20) -> list[str]:
    """Select the top-k features using mutual information."""
    mi = mutual_info_classif(X, y, random_state=42)
    return X.columns[mi.argsort()[::-1][:k]].tolist()


def recursive_feature_elimination(X: pd.DataFrame, y: pd.Series, k: int = 20) -> list[str]:
    """Select features with recursive feature elimination."""
    estimator = LogisticRegression(max_iter=1000, solver="liblinear")
    selector = RFE(estimator, n_features_to_select=k)
    selector.fit(X, y)
    return X.columns[selector.support_].tolist()


def permutation_importance_selection(X: pd.DataFrame, y: pd.Series, k: int = 20) -> list[str]:
    """Select features based on permutation importance."""
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    estimator.fit(X, y)
    importance = permutation_importance(estimator, X, y, n_repeats=5, random_state=42)
    return X.columns[importance.importances_mean.argsort()[::-1][:k]].tolist()


def tree_importance_selection(X: pd.DataFrame, y: pd.Series, k: int = 20) -> list[str]:
    """Select features using tree-based importance."""
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    estimator.fit(X, y)
    importances = pd.Series(estimator.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(k).index.tolist()
