"""Matched numeric/Attention NN, fixed architecture and development-only stopping."""

import os

from mqf_research.modeling._validation import (
    ensure_new_checkpoints,
    require,
    validate_history,
    validate_preprocessing,
)

os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

torch.set_num_threads(1)
SEEDS = [42, 43]
RECIPE = dict(
    numeric_width=32,
    head_width=64,
    dropout=0.2,
    weight_decay=0.05,
    learning_rate=0.002,
    projection_width=32,
    gate_width=16,
    batch_size=128,
    max_epochs=40,
    patience=5,
    minimum_search_epochs=5,
)


class Network(nn.Module):
    """Matched numeric/attention network preserving the archived parameter initialization order."""

    def __init__(self, numeric_dim, use_text):
        """Create the model/helper in the fixed parameter order used by the experiment."""
        super().__init__()
        require(isinstance(numeric_dim, int) and numeric_dim > 0, "numeric_dim must be positive")
        self.use_text = bool(use_text)
        # Preserve the original model's parameter creation order for both arms.
        self.proj = nn.Linear(384, 32)
        self.v = nn.Linear(32, 16)
        self.u = nn.Linear(32, 16)
        self.a = nn.Linear(16, 1, bias=False)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, 32), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def text_branch(self, embeddings, mask):
        """Pool valid signals with gated attention; empty bags return exactly zero.

        Mask padding before softmax and guard all-empty rows to avoid NaN gradients.
        """
        require(
            embeddings.ndim == 3 and embeddings.shape[-1] == 384,
            "Signal batch must have shape (events, signals, 384)",
        )
        require(
            mask.dtype == torch.bool and tuple(mask.shape) == tuple(embeddings.shape[:2]),
            "Signal mask must be boolean and match event/signal axes",
        )
        require(embeddings.shape[1] >= 1, "Use a masked slot for an empty signal bag")
        h = self.proj(embeddings)  # Linear projection; tanh is applied after pooling.
        logits = self.a(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        empty = ~mask.any(dim=1)
        # Avoid all--inf softmax, including its otherwise undefined gradient.
        logits = torch.where(empty[:, None], torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1) * mask.to(logits.dtype)
        return (h * weights[..., None]).sum(dim=1), weights

    def forward(self, x, embeddings=None, mask=None):
        """Return one signed-AR forecast per event from numeric features and optional signal bags."""
        require(
            x.ndim == 2 and x.shape[1] == self.numeric[0].in_features,
            "Numeric input must have shape (events, fitted features)",
        )
        if self.use_text:
            require(
                (embeddings is not None and mask is not None),
                "Invariant failed: embeddings is not None and mask is not None",
            )
            require(
                len(x) == len(embeddings), "Numeric and signal batches must contain the same events"
            )
            pooled, _ = self.text_branch(embeddings, mask)
        else:
            pooled = x.new_zeros((len(x), 32))
        return self.head(torch.cat([self.numeric(x), torch.tanh(pooled)], dim=1)).squeeze(1)


def inner_split(history):
    """Split earlier history into a core fit and the most recent validation quarters.

    Require at least 200 core and 100 validation events, with two validation quarters.
    The split is chronological and uses no later evaluation outcome.
    """
    validate_history(history)
    cut = pd.Period(history.quarter.max(), freq="Q") - 1
    while len(history[history.quarter.ge(str(cut))]) < 100:
        cut -= 1
    core = history[history.quarter.lt(str(cut))].copy()
    valid = history[history.quarter.ge(str(cut))].copy()
    require(
        (len(core) >= 200 and len(valid) >= 100 and (valid.quarter.nunique() >= 2)),
        "Invariant failed: len(core) >= 200 and len(valid) >= 100 and (valid.quarter.nunique() >= 2)",
    )
    require(
        (core.reaction_session.max() < valid.reaction_session.min()),
        "Invariant failed: core.reaction_session.max() < valid.reaction_session.min()",
    )
    return core, valid


def prepare(train, test, numeric, categorical):
    """Fit imputation, scaling, categories, and target standardization on training rows only.

    Return float32 model matrices plus fitted preprocessing objects. Test features are
    transformed without fitting; test targets do not affect any returned quantity.
    """
    validate_preprocessing(train, test, numeric, categorical)
    # Learn every transform from history. Future magnitudes and unseen sectors
    # may change their transformed rows, never the learned statistics/vocabulary.
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    a = np.clip(scaler.fit_transform(imp.fit_transform(train[numeric])), -10, 10)
    b = np.clip(scaler.transform(imp.transform(test[numeric])), -10, 10)
    if categorical:
        hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        cats = hot.fit_transform(train[categorical].fillna("UNKNOWN").astype(str))
        cat_test = hot.transform(test[categorical].fillna("UNKNOWN").astype(str))
    else:
        hot = None
        cats = np.empty((len(train), 0), dtype=np.float32)
        cat_test = np.empty((len(test), 0), dtype=np.float32)
    x = np.column_stack([a, cats]).astype("float32")
    xt = np.column_stack([b, cat_test]).astype("float32")
    mu = float(train.y.mean())
    # Population SD and the fixed floor are part of saved-checkpoint compatibility.
    sd = max(float(train.y.std(ddof=0)), 1e-6)
    y = ((train.y.to_numpy() - mu) / sd).astype("float32")
    require(
        (np.isfinite(x).all() and np.isfinite(xt).all() and np.isfinite(y).all()),
        "Invariant failed: np.isfinite(x).all() and np.isfinite(xt).all() and np.isfinite(y).all()",
    )
    prep = dict(
        imputer=imp,
        scaler=scaler,
        onehot=hot,
        numeric_features=numeric,
        categorical_features=categorical,
        fit_event_ids=train.event_id.tolist(),
    )
    return dict(
        X_train=x,
        X_test=xt,
        y_train_std=y,
        target_mean=mu,
        target_sd=sd,
        train_ids=train.event_id.to_numpy(str),
        test_ids=test.event_id.to_numpy(str),
    ), prep


def bag_batch(ids, vectors, bags):
    """Pad individual signal vectors into a batch and return an explicit validity mask.

    Every signal is retained. Empty events receive a masked zero slot, not a synthetic signal.
    """
    require(
        getattr(vectors, "ndim", None) == 2 and vectors.shape[1] == 384,
        "Signal vectors must have shape (signals, 384)",
    )
    missing = [str(event) for event in ids if str(event) not in bags]
    require(not missing, f"Missing signal bags for events: {missing[:5]}")
    rows = [bags[str(event)] for event in ids]
    width = max(1, max(map(len, rows), default=0))
    emb = np.zeros((len(rows), width, 384), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=bool)
    for j, indices in enumerate(rows):
        if len(indices):
            indices = np.asarray(indices)
            require(
                indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer),
                "Signal indices must be a one-dimensional integer sequence",
            )
            require(
                (indices >= 0).all() and (indices < len(vectors)).all(),
                "Signal index outside vector array",
            )
            require(np.isfinite(vectors[indices]).all(), "Selected signal vectors must be finite")
            emb[j, : len(indices)] = vectors[indices]
            mask[j, : len(indices)] = True
    require(
        (int(mask.sum()) == sum(map(len, rows))),
        "Invariant failed: int(mask.sum()) == sum(map(len, rows))",
    )  # No truncation or discarded signals.
    return torch.from_numpy(emb), torch.from_numpy(mask)


def initialize(dim, use_text):
    """Initialize both seeded networks and their independent optimizer/RNG states.

    Parameter creation order is part of numerical compatibility with saved models.
    """
    states = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Network(dim, use_text)
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.05)
        states.append(
            dict(
                seed=seed,
                model=model,
                opt=opt,
                rng=np.random.default_rng(seed),
                torch_state=torch.get_rng_state(),
            )
        )
    return states


def epoch_step(state, x, y, ids, vectors, bags):
    """Advance one shuffled training epoch and preserve the seed-specific RNG state.

    Loss is mean squared standardized AR; gradients are clipped to norm five.
    """
    torch.set_rng_state(state["torch_state"])
    model = state["model"]
    model.train()
    order = state["rng"].permutation(len(x))
    total = 0.0
    for start in range(0, len(order), 128):
        ix = order[start : start + 128]
        state["opt"].zero_grad(set_to_none=True)
        emb, mask = bag_batch(ids[ix], vectors, bags) if model.use_text else (None, None)
        loss = (model(x[ix], emb, mask) - y[ix]).square().mean()
        require((torch.isfinite(loss)), "Invariant failed: torch.isfinite(loss)")
        loss.backward()
        require(
            (all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))),
            "Invariant failed: all((p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))",
        )
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        state["opt"].step()
        total += float(loss.detach()) * len(ix)
    state["torch_state"] = torch.get_rng_state()
    return total / len(x)


def predict(model, x, ids, vectors, bags, mu, sd):
    """Return raw-AR forecasts with dropout disabled and no gradient tracking.

    Invert target standardization in float32 before constructing the result array,
    matching the archived prediction arithmetic.
    """
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            ix = np.arange(start, min(start + 128, len(x)))
            emb, mask = bag_batch(ids[ix], vectors, bags) if model.use_text else (None, None)
            p = model(x[ix], emb, mask)
            values.extend((p.numpy() * sd + mu).tolist())
    return np.asarray(values)


def select_epochs(matrix, y_valid, use_text, vectors, bags):
    """Choose epoch count by ensemble MSE on strictly earlier validation events.

    Use the fixed 40-epoch cap, five-epoch patience, and earliest exact MSE tie.
    Return the full stopping curve and the fit/validation event identifiers.
    """
    x, xt = torch.from_numpy(matrix["X_train"]), torch.from_numpy(matrix["X_test"])
    y = torch.from_numpy(matrix["y_train_std"])
    states = initialize(x.shape[1], use_text)
    mu, sd = float(matrix["target_mean"]), float(matrix["target_sd"])
    curve = []
    # Stopping selects a duration, not a checkpoint to deploy. The later refit
    # starts from the same seeds and uses all eligible earlier training history.
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, 41):
        loss = [epoch_step(s, x, y, matrix["train_ids"], vectors, bags) for s in states]
        p = np.mean(
            [predict(s["model"], xt, matrix["test_ids"], vectors, bags, mu, sd) for s in states],
            axis=0,
        )
        mse = float(np.mean((p - y_valid) ** 2))
        require((np.isfinite(mse)), "Invariant failed: np.isfinite(mse)")
        if mse < best:
            best, best_epoch, stale = mse, epoch, 0
        else:
            stale += 1
        curve.append(
            dict(
                epoch=epoch,
                training_standardized_loss=float(np.mean(loss)),
                validation_mse=mse,
                validation_rmse_pp=float(np.sqrt(mse) * 100),
            )
        )
        if epoch >= 5 and stale >= 5:
            break
    require(
        (best_epoch == min(curve, key=lambda row: row["validation_mse"])["epoch"]),
        "Invariant failed: best_epoch == min(curve, key=lambda row: row['validation_mse'])['epoch']",
    )
    return dict(
        selected_epoch=best_epoch,
        stopped_epoch=epoch,
        hit_cap=epoch == 40,
        curve=curve,
        best_inner_mse=best,
        core_train_ids=matrix["train_ids"].tolist(),
        inner_validation_ids=matrix["test_ids"].tolist(),
        patience=5,
        minimum_search_epochs=5,
        max_epochs=40,
    )


def refit(matrix, epochs, use_text, vectors, bags, destination):
    """Fit the locked epoch count on earlier history and save both seeded checkpoints.

    Return their mean raw-AR prediction; the destination belongs to this new run.
    """
    require((1 <= epochs <= 40), "Invariant failed: 1 <= epochs <= 40")
    destination = ensure_new_checkpoints(destination, SEEDS)
    x, xt = torch.from_numpy(matrix["X_train"]), torch.from_numpy(matrix["X_test"])
    y = torch.from_numpy(matrix["y_train_std"])
    mu, sd = float(matrix["target_mean"]), float(matrix["target_sd"])
    states = initialize(x.shape[1], use_text)
    for _ in range(epochs):
        for state in states:
            epoch_step(state, x, y, matrix["train_ids"], vectors, bags)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    preds = []
    for state in states:
        preds.append(predict(state["model"], xt, matrix["test_ids"], vectors, bags, mu, sd))
        torch.save(
            dict(
                state_dict=state["model"].state_dict(),
                numeric_dim=x.shape[1],
                recipe=RECIPE,
                seed=state["seed"],
                epochs=epochs,
                use_text=use_text,
                target_mean=mu,
                target_sd=sd,
            ),
            destination / f"seed{state['seed']}.pt",
        )
    np.save(destination / "SEED_PREDICTIONS.npy", np.asarray(preds))
    return np.mean(preds, axis=0)


def replay(path, matrix, vectors, bags):
    """Reconstruct predictions from a trusted local checkpoint without training.

    PyTorch checkpoints can contain pickle objects; never pass an untrusted file.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Network(checkpoint["numeric_dim"], checkpoint["use_text"])
    model.load_state_dict(checkpoint["state_dict"])
    return predict(
        model,
        torch.from_numpy(matrix["X_test"]),
        matrix["test_ids"],
        vectors,
        bags,
        checkpoint["target_mean"],
        checkpoint["target_sd"],
    )


def self_check():
    # Exact numeric path equivalence, and invariants essential to variable-size bags.
    """Check numeric-path parity and essential attention invariants using synthetic data."""
    from mqf_research.modeling.tuning.engine import NumericNet

    torch.manual_seed(42)
    old = NumericNet(7, dict(numeric_width=32, head_width=64, dropout=0.2))
    torch.manual_seed(42)
    new = Network(7, False)
    old.eval()
    new.eval()
    x = torch.arange(28, dtype=torch.float32).reshape(4, 7) / 28
    torch.testing.assert_close(old(x), new(x), rtol=0, atol=0)
    net = Network(7, True)
    net.eval()
    vectors = np.random.default_rng(42).normal(size=(6, 384)).astype("float32")
    bags = {"a": [0, 1, 2], "b": [], "c": [3], "d": [4, 5]}
    ids = np.asarray(["a", "b", "c", "d"])
    emb, mask = bag_batch(ids, vectors, bags)
    pooled, weights = net.text_branch(emb, mask)
    torch.testing.assert_close(weights.sum(1), torch.tensor([1.0, 0.0, 1.0, 1.0]))
    require(
        (torch.equal(pooled[1], torch.zeros(32))),
        "Invariant failed: torch.equal(pooled[1], torch.zeros(32))",
    )
    expected = net(x, emb, mask)
    reverse = {k: list(reversed(v)) for k, v in bags.items()}
    emb2, mask2 = bag_batch(ids, vectors, reverse)
    torch.testing.assert_close(expected, net(x, emb2, mask2), rtol=1e-6, atol=1e-7)
    padded = torch.cat([emb, torch.full((4, 2, 384), 999.0)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros((4, 2), dtype=torch.bool)], dim=1)
    torch.testing.assert_close(expected, net(x, padded, padded_mask), rtol=1e-6, atol=1e-7)
    expected.square().mean().backward()
    require(
        (all((p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))),
        "Invariant failed: all((p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))",
    )
    return dict(
        passed=True,
        numeric_path_exact=True,
        all_signals_retained=True,
        empty_bag_exact_zero=True,
        attention_sums=True,
        finite_empty_bag_gradients=True,
        padding_and_permutation_invariant=True,
    )
