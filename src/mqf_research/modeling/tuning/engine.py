"""Small numeric NN. Same original initialization, optimizer and seed policy."""

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


def recipes():
    """Return the twelve prespecified numeric-network settings in their original order.

    The original recipe appears exactly once; adding candidates changes the research protocol.
    """
    result = []
    for width, head in [(8, 16), (16, 32), (32, 64)]:
        for dropout in [0.2, 0.4]:
            for decay in [0.05, 0.2]:
                result.append(
                    dict(
                        recipe_id=f"n{width}_h{head}_d{int(dropout * 100)}_w{int(decay * 100)}",
                        numeric_width=width,
                        head_width=head,
                        dropout=dropout,
                        weight_decay=decay,
                        is_original=(width == 16 and dropout == 0.2 and decay == 0.05),
                    )
                )
    require(
        (len(result) == 12 and sum((r["is_original"] for r in result)) == 1),
        "Invariant failed: len(result) == 12 and sum((r['is_original'] for r in result)) == 1",
    )
    return result


class NumericNet(nn.Module):
    """Two-branch-compatible numeric network with an intentionally zero text branch."""

    def __init__(self, numeric_dim, recipe):
        """Create the model/helper in the fixed parameter order used by the experiment."""
        super().__init__()
        require(isinstance(numeric_dim, int) and numeric_dim > 0, "numeric_dim must be positive")
        # These inert modules preserve the original numeric-only RNG initialization.
        self.proj = nn.Linear(384, 32)
        self.v = nn.Linear(32, 16)
        self.u = nn.Linear(32, 16)
        self.a = nn.Linear(16, 1, bias=False)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, recipe["numeric_width"]), nn.Tanh())
        self.head = nn.Sequential(
            nn.Linear(recipe["numeric_width"] + 32, recipe["head_width"]),
            nn.ReLU(),
            nn.Dropout(recipe["dropout"]),
            nn.Linear(recipe["head_width"], 1),
        )

    def forward(self, x):
        """Return one signed-AR forecast per event from numeric features and optional signal bags."""
        require(
            x.ndim == 2 and x.shape[1] == self.numeric[0].in_features,
            "Numeric input must have shape (events, fitted features)",
        )
        zeros = x.new_zeros((len(x), 32))
        return self.head(torch.cat([self.numeric(x), torch.tanh(zeros)], dim=1)).squeeze(1)


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
    # The test frame is transformed only; its distribution cannot fit imputation
    # medians, scaling moments, category membership, or target standardization.
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    a = np.clip(scaler.fit_transform(imp.fit_transform(train[numeric])), -10, 10)
    b = np.clip(scaler.transform(imp.transform(test[numeric])), -10, 10)
    if categorical:
        hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        cat = hot.fit_transform(train[categorical].fillna("UNKNOWN").astype(str))
        catt = hot.transform(test[categorical].fillna("UNKNOWN").astype(str))
    else:
        # A market-only input has no category vocabulary to fit.
        hot = None
        cat = np.empty((len(train), 0), dtype=np.float32)
        catt = np.empty((len(test), 0), dtype=np.float32)
    x = np.column_stack([a, cat]).astype("float32")
    xt = np.column_stack([b, catt]).astype("float32")
    mu = float(train.y.mean())
    # Keep population SD and its original floor for exact checkpoint replay.
    sd = max(float(train.y.std(ddof=0)), 1e-6)
    y = ((train.y.to_numpy() - mu) / sd).astype("float32")
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


def initialize(dim, recipe):
    """Initialize both seeded networks and their independent optimizer/RNG states.

    Parameter creation order is part of numerical compatibility with saved models.
    """
    states = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = NumericNet(dim, recipe)
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=recipe["weight_decay"])
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


def epoch_step(state, x, y):
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
        loss = (model(x[ix]) - y[ix]).square().mean()
        require((torch.isfinite(loss)), "Invariant failed: torch.isfinite(loss)")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        state["opt"].step()
        total += float(loss.detach()) * len(ix)
    state["torch_state"] = torch.get_rng_state()
    return total / len(x)


def predict(model, x, mu, sd):
    """Return raw-AR forecasts with dropout disabled and no gradient tracking.

    Invert target standardization in float32 before constructing the result array,
    matching the archived prediction arithmetic.
    """
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            ix = np.arange(start, min(start + 128, len(x)))
            p = model(x[ix])
            # Retain float32 inversion before conversion, exactly as original.
            values.extend((p.numpy() * sd + mu).tolist())
    return np.array(values)


def select_epochs(matrix, y_valid, recipe):
    """Choose epoch count by ensemble MSE on strictly earlier validation events.

    Use the fixed 40-epoch cap, five-epoch patience, and earliest exact MSE tie.
    Return the full stopping curve and the fit/validation event identifiers.
    """
    x = torch.from_numpy(matrix["X_train"])
    xt = torch.from_numpy(matrix["X_test"])
    y = torch.from_numpy(matrix["y_train_std"])
    states = initialize(x.shape[1], recipe)
    mu = float(matrix["target_mean"])
    sd = float(matrix["target_sd"])
    curve = []
    best = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, 41):
        losses = [epoch_step(s, x, y) for s in states]
        p = np.mean([predict(s["model"], xt, mu, sd) for s in states], axis=0)
        mse = float(np.mean((p - y_valid) ** 2))
        require((np.isfinite(mse)), "Invariant failed: np.isfinite(mse)")
        if mse < best:
            best = mse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        curve.append(
            dict(
                epoch=epoch,
                training_standardized_loss=float(np.mean(losses)),
                validation_mse=mse,
                validation_rmse_pp=float(np.sqrt(mse) * 100),
            )
        )
        if epoch >= 5 and stale >= 5:
            break
    require(
        (best_epoch == min(curve, key=lambda r: r["validation_mse"])["epoch"]),
        "Invariant failed: best_epoch == min(curve, key=lambda r: r['validation_mse'])['epoch']",
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
        min_epochs=5,
        max_epochs=40,
    )


def refit(matrix, recipe, epochs, dest):
    """Fit the locked epoch count on earlier history and save both seeded checkpoints.

    Return their mean raw-AR prediction; the destination belongs to this new run.
    """
    require((1 <= epochs <= 40), "Invariant failed: 1 <= epochs <= 40")
    dest = ensure_new_checkpoints(dest, SEEDS)
    x = torch.from_numpy(matrix["X_train"])
    xt = torch.from_numpy(matrix["X_test"])
    y = torch.from_numpy(matrix["y_train_std"])
    mu = float(matrix["target_mean"])
    sd = float(matrix["target_sd"])
    states = initialize(x.shape[1], recipe)
    for _ in range(epochs):
        for s in states:
            epoch_step(s, x, y)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    predictions = []
    for state in states:
        p = predict(state["model"], xt, mu, sd)
        predictions.append(p)
        torch.save(
            dict(
                state_dict=state["model"].state_dict(),
                numeric_dim=x.shape[1],
                recipe=recipe,
                seed=state["seed"],
                epochs=epochs,
                target_mean=mu,
                target_sd=sd,
            ),
            dest / f"seed{state['seed']}.pt",
        )
    seedpred = np.array(predictions)
    np.save(dest / "SEED_PREDICTIONS.npy", seedpred)
    return seedpred.mean(axis=0)


def replay(path, matrix):
    """Reconstruct predictions from a trusted local checkpoint without training.

    PyTorch checkpoints can contain pickle objects; never pass an untrusted file.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = NumericNet(checkpoint["numeric_dim"], checkpoint["recipe"])
    model.load_state_dict(checkpoint["state_dict"])
    return predict(
        model,
        torch.from_numpy(matrix["X_test"]),
        checkpoint["target_mean"],
        checkpoint["target_sd"],
    )
