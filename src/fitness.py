"""Wrapper fitness function: decode a particle into (shared feature subset,
active classifiers), cross-validate the resulting soft-voting ensemble, and
combine balanced accuracy with a feature-parsimony penalty."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.optimizers.base import CLASSIFIER_NAMES, decode_bits, decode_bits_top1
from src.optimizers.transfer import N_TF_CANDIDATES, TF_CANDIDATES, decode_and_binarize_searched_tf


def build_classifiers(seed, names=None):
    """n_jobs/nthread are pinned to 1 everywhere -- this project always runs
    many independent fitness evaluations in parallel across processes (see
    run_fs_experiment.py), and letting individual models also spawn their
    own internal thread pools oversubscribes the machine's cores (N worker
    processes x M threads each) rather than cleanly using one core/worker."""
    names = names if names is not None else CLASSIFIER_NAMES
    factory = {
        "svm": lambda: make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                SVC(kernel="rbf", random_state=seed), ensemble=False, cv=3
            ),
        ),
        "rf": lambda: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
        "xgb": lambda: XGBClassifier(
            n_estimators=100, eval_metric="logloss", random_state=seed, verbosity=0, n_jobs=1
        ),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
        ),
    }
    return {name: factory[name]() for name in names}


class FitnessEvaluator:
    def __init__(self, X, y, groups, omega=0.9, cv_folds=5, seed=0):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.groups = np.asarray(groups)
        self.omega = omega
        self.cv_folds = cv_folds
        self.seed = seed
        self.n_features = self.X.shape[1]
        self.n_evaluations = 0
        self._cv_splits = list(
            StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(
                self.X, self.y, self.groups
            )
        )

    def evaluate_searched_tf(self, position, prev_bits, rng, classifier_encoding="multi_hot", fixed_tf_index=None):
        """Proposed-method-only pathway (`src/optimizers/cso.py`): each
        particle's *leading* 5-dim segment selects which transfer function
        binarizes its own *trailing* feature+classifier segment
        (`transfer.TF_CANDIDATES`) -- position layout
        `[t_1..t_5 | f_1..f_D | c_1..c_5]` -- rather than every algorithm
        sharing one hard-coded transfer function the way the 7 reproduced
        baselines do. `fixed_tf_index` forces a single candidate for every
        particle/generation instead of letting each particle search its own
        choice; unused by the registered `CSO_searched_tf` algorithm (always
        `None` there), available for ad-hoc analysis.

        Returns `(bits, fitness, info)`, matching
        `transfer.binarize_and_eval`'s tuple order, so `cso.py`'s call site
        reads like every other optimizer's."""
        bits, tf_idx = decode_and_binarize_searched_tf(position, prev_bits, rng, fixed_tf_index)
        seg_bits = bits[N_TF_CANDIDATES:]
        seg_position = position[N_TF_CANDIDATES:]
        if classifier_encoding == "top1":
            feature_mask, clf_mask = decode_bits_top1(seg_bits, seg_position, self.n_features, rng)
        else:
            feature_mask, clf_mask = decode_bits(seg_bits, self.n_features, rng)
        fitness, info = self.evaluate_masks(feature_mask, clf_mask)
        info["tf_index"] = tf_idx
        info["tf_name"] = TF_CANDIDATES[tf_idx].name
        return bits, fitness, info

    def evaluate_masks(self, feature_mask, clf_mask):
        active = [name for name, on in zip(CLASSIFIER_NAMES, clf_mask) if on]
        X_sub = self.X[:, feature_mask]

        oof_proba = np.zeros(len(self.y))
        for train_idx, val_idx in self._cv_splits:
            X_train, X_val = X_sub[train_idx], X_sub[val_idx]
            y_train = self.y[train_idx]
            classifiers = build_classifiers(self.seed, active)
            proba_sum = np.zeros(len(val_idx))
            for name in active:
                clf = classifiers[name]
                clf.fit(X_train, y_train)
                proba_sum += clf.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = proba_sum / len(active)

        oof_pred = (oof_proba >= 0.5).astype(int)
        balanced_acc = balanced_accuracy_score(self.y, oof_pred)
        feature_ratio = feature_mask.sum() / self.n_features
        fitness = self.omega * (1 - balanced_acc) + (1 - self.omega) * feature_ratio

        # Standard classification metrics, computed from the same out-of-fold
        # predictions used for the fitness above -- not just balanced_accuracy,
        # so downstream comparisons/tables have Accuracy/Precision/Recall/F1/
        # Specificity/ROC-AUC available per run without re-evaluating anything.
        tn, fp, fn, tp = confusion_matrix(self.y, oof_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        try:
            roc_auc = roc_auc_score(self.y, oof_proba)
        except ValueError:
            roc_auc = float("nan")  # oof_pred/proba collapsed to a single class

        self.n_evaluations += 1
        info = {
            "balanced_accuracy": balanced_acc,
            "accuracy": accuracy_score(self.y, oof_pred),
            "precision": precision_score(self.y, oof_pred, zero_division=0),
            "recall": recall_score(self.y, oof_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(self.y, oof_pred, zero_division=0),
            "roc_auc": roc_auc,
            "feature_ratio": feature_ratio,
            "n_features": int(feature_mask.sum()),
            "active_classifiers": active,
            "feature_mask": feature_mask,
            "clf_mask": clf_mask,
        }
        return fitness, info
