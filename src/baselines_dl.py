"""Non-evolutionary-computation baselines: neural networks trained directly
on the FULL feature set -- no swarm-based feature selection, no classifier-
pool switching. Uses the same grouped StratifiedGroupKFold protocol as
FitnessEvaluator so balanced accuracy is directly comparable to the
EOACSO_Paper/baseline optimizer results, even though there is no feature-count to
report for a no-FS baseline.

Two architectures, both reproduced per Oseni, Obanla & Jimoh (2026),
"Attention-Based Deep Learning for Early Parkinson's Disease Detection with
Tabular Biomedical Data" (arXiv:2602.07933) -- which evaluates exactly the
classic UCI Oxford voice dataset (195 samples, 22 features) our project
already uses:

  MLP    - generic feedforward net (Eq. 7-9 of that paper): the paper gives
           no specific layer count/hidden width, only that training used
           100 epochs; hidden=(64,32) below is this project's own choice.
  SAINT  - Somepalli et al. (2021)'s attention-based tabular transformer
           (self-attention across a sample's features + intersample
           attention across the batch), the paper's best performer
           (weighted F1 0.97, MCC 0.9990). The paper's Eq. 10-12 describe
           only generic embedding + single-head self-attention, not SAINT's
           actual dual self+intersample block structure or any of embed_dim/
           num_heads/num_layers/dropout -- those are filled in below with
           reasonable defaults sized for a small tabular dataset, per the
           original SAINT paper's typical small-dataset configuration, not
           values the PD paper itself states.

Note on the PD paper's own evaluation protocol: it uses a single stratified
80/20 random train/test split (seed 42) with no subject-level grouping --
for the Oxford dataset (31 subjects, 6-7 repeated recordings each) that
risks the same-subject-in-train-and-test leakage this project's grouped CV
is designed to avoid elsewhere, so its headline numbers (SAINT F1=0.97)
aren't directly apples-to-apples with this module's grouped-CV output.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SAINTBlock(nn.Module):
    """One (self-attention, intersample-attention) pair, Fig. 3(a) of
    Somepalli et al. (2021) / Fig. 3 of the PD paper."""

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)

        self.inter_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff2 = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, embed_dim))
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim), seq_len = n_features + 1 (CLS token)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff1(x))

        # intersample attention: attend across the BATCH at each feature position
        x_t = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        attn_out2, _ = self.inter_attn(x_t, x_t, x_t)
        x_t = self.norm3(x_t + attn_out2)
        x_t = self.norm4(x_t + self.ff2(x_t))
        return x_t.transpose(0, 1)


class SAINT(nn.Module):
    def __init__(self, n_features, embed_dim=16, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.feature_embeds = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_features)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([SAINTBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, x):
        # x: (batch, n_features)
        b = x.shape[0]
        toks = torch.stack([emb(x[:, i : i + 1]) for i, emb in enumerate(self.feature_embeds)], dim=1)
        seq = torch.cat([self.cls_token.expand(b, -1, -1), toks], dim=1)
        for block in self.blocks:
            seq = block(seq)
        return self.head(seq[:, 0, :]).squeeze(-1)


def _train_torch_model(model, X_train, y_train, X_val, seed, lr, epochs, weight_decay):
    torch.manual_seed(seed)
    scaler = StandardScaler().fit(X_train)
    X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_val_t = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_val_t)).numpy()


def _run_grouped_cv(model_factory, X, y, groups, cv_folds, seed, lr, epochs, weight_decay):
    torch.set_num_threads(1)  # avoid oversubscribing cores when run in a process pool
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    oof_proba = np.zeros(len(y))
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        fold_seed = seed + fold_idx
        torch.manual_seed(fold_seed)
        model = model_factory(X.shape[1])
        oof_proba[val_idx] = _train_torch_model(
            model, X[train_idx], y[train_idx], X[val_idx], fold_seed, lr, epochs, weight_decay
        )

    oof_pred = (oof_proba >= 0.5).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, oof_pred),
        "f1": f1_score(y, oof_pred),
        "auc": roc_auc_score(y, oof_proba),
        "n_features": X.shape[1],
    }


def run_mlp_baseline(
    X, y, groups, cv_folds=5, seed=0, hidden=(64, 32), dropout=0.3, lr=1e-3, epochs=200, weight_decay=1e-4
):
    return _run_grouped_cv(
        lambda n_features: MLP(n_features, hidden=hidden, dropout=dropout), X, y, groups, cv_folds, seed, lr, epochs, weight_decay
    )


def run_saint_baseline(
    X,
    y,
    groups,
    cv_folds=5,
    seed=0,
    embed_dim=16,
    num_heads=4,
    num_layers=1,
    dropout=0.1,
    lr=1e-3,
    epochs=200,
    weight_decay=1e-4,
):
    return _run_grouped_cv(
        lambda n_features: SAINT(n_features, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout),
        X,
        y,
        groups,
        cv_folds,
        seed,
        lr,
        epochs,
        weight_decay,
    )
