"""Deep comparison methods of related work, rerun on this study's protocol (Stage 1 features, fit_train -> fit_holdout, seeds 3-5).

Every network trains on the inner-training part of fit_train and stops early on the macro-F1 of the inner-validation part
(the latest 20 % per day and class, the rule of fit_holdout). The weights of the best inner-validation epoch score
fit_holdout. Scalers are fitted on the inner-training rows. Class balance: balanced class weights in the loss, as in Stage 3
and baselines.py; FlowTransformer keeps the class-balanced batches of its code, the same balance by sampling.

- tabm: Gorishniy, Kotelnikov, Babenko, ICLR 2025. Package tabm, TabM.make defaults (k 32, 3 blocks of 512, dropout 0.1);
  AdamW 2e-3, weight decay 3e-4, gradient clip 1.0, patience 16, quantile-normal features (paper App. D).
- ft_transformer: Gorishniy, Rubachev, Khrulkov, Babenko, NeurIPS 2021. Package rtdl_revisiting_models, default 3-block
  configuration (Table 12) and default AdamW (1e-4, weight decay 1e-5 except embeddings, norms and biases); stop after
  patience + 1 = 17 epochs without improvement (Sect. 4.3), batch 1024 (Table 7), quantile-normal features.
- cnn1d: Donoso et al., J. Cybersecur. Priv. 6(5):183, 2026, Sect. 3.4-3.7 and their repository: signed log on the columns
  with |Fisher kurtosis| > 10 (computed on the inner-training rows; the authors used all rows), RobustScaler on 500,000
  rows, clip to 10; Conv1d(1, 64, 3) + BN + ReLU, Conv1d(64, 128, 3) + BN + ReLU + MaxPool(2), Linear 128 + ReLU
  + Dropout 0.3; Adam 1e-3, ReduceLROnPlateau on validation macro-F1 (factor 0.5, patience 2), patience 4, 20 epochs, batch 512.
- flow_transformer: Manocchio et al., ESWA 2024, and their repository: each flow with its 7 predecessors in the time order
  of its capture day (the predecessors of a training flow are training flows only); log min-max scaling; record-level projection to 64; 2 decoder blocks of the official code (the
  paper's recommendation, Sect. VIII; 2 heads, inner size 128, dropout 0.1; Keras Glorot-uniform weights, zero biases); last token; MLP 128 with dropout 0.1; Adam with the learning rate of their grid that is best on inner
  validation; batches of 128 with equal rows per class, 128 batches per epoch, 20 epochs, patience 5.
- tabnet: Zegarra Rodriguez et al., PLOS ONE 18(10):e0286652, 2023, Table 4. Package pytorch-tabnet; no SMOTE (the protocol
  has no oversampling); the paper gives no learning rate or epoch limit, so the library learning rate and TABNET_MAX_EPOCHS apply.
"""
import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from rtdl_revisiting_models import FTTransformer
from scipy.stats import kurtosis
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from tabm import TabM

from idsvae import files, metrics
from idsvae.data import splits
from idsvae.experiments import baselines
from idsvae.framework import vae

# The papers of TabM, FT-Transformer and TabNet stop only by patience; these limits are far above every stop epoch of the runs,
# so patience alone ends the training (at a limit of 50, TabNet reached the limit in 4 of 12 runs).
MAX_EPOCHS = 300
# TabNet needs about 25 s per epoch on 1 million rows, so its limit is lower.
TABNET_MAX_EPOCHS = 200
PREDICT_BATCH = 8192
WINDOW_SIZE = 8
FLOW_TRANSFORMER_LEARNING_RATES = (1e-2, 1e-3, 5e-4, 1e-4, 1e-5)
CNN_SCALER_ROWS = 500_000
CNN_CLIP = 10.0
# Donoso et al. repository, _identificar_colas_pesadas: a column with |Fisher kurtosis| above this is heavy-tailed.
CNN_KURTOSIS_LIMIT = 10.0


@dataclass
class Recipe:
    batch_size: int
    max_epochs: int
    patience: int
    gradient_clip: float | None = None
    steps_per_epoch: int | None = None
    balanced_batches: bool = False
    plateau_scheduler: bool = False


class TabularData:
    """Model inputs of a split: one row per flow, or one window of flows per flow (FlowTransformer)."""

    def __init__(self, features, labels, windows=None):
        self.features = features
        self.labels = labels
        self.windows = windows

    def __len__(self):
        return len(self.labels)

    def inputs(self, rows):
        if self.windows is None:
            return self.features[rows]
        return self.features[self.windows[rows]]


# ---------------------------------------------------------------- networks: forward(x) -> logits (rows, members, classes)

class TabMNetwork(torch.nn.Module):
    def __init__(self, feature_count, class_count):
        super().__init__()
        self.model = TabM.make(n_num_features=feature_count, cat_cardinalities=[], d_out=class_count)

    def forward(self, inputs):
        return self.model(inputs)


class FTTransformerNetwork(torch.nn.Module):
    def __init__(self, feature_count, class_count):
        super().__init__()
        self.model = FTTransformer(n_cont_features=feature_count, cat_cardinalities=[], d_out=class_count, **FTTransformer.get_default_kwargs())

    def forward(self, inputs):
        return self.model(inputs, None)[:, None]


class CnnNetwork(torch.nn.Module):
    def __init__(self, feature_count, class_count):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=3, padding=1), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.MaxPool1d(2),
            torch.nn.Flatten(), torch.nn.Linear(128 * (feature_count // 2), 128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(128, class_count),
        )

    def forward(self, inputs):
        return self.layers(inputs[:, None, :])[:, None]


class FlowTransformerDecoderBlock(torch.nn.Module):
    """Decoder block of the FlowTransformer code (implementations/transformers/basic/decoder_block.py): one Keras
    MultiHeadAttention with key_dim = width per head, applied twice without a causal mask; the second LayerNorm and
    dropout are shared, as in that file."""

    def __init__(self, width, inner, heads, dropout):
        super().__init__()
        self.heads, self.key_size = heads, width
        self.query, self.key, self.value = (torch.nn.Linear(width, heads * width) for _ in range(3))
        self.output = torch.nn.Linear(heads * width, width)
        self.dropout1, self.dropout2 = torch.nn.Dropout(dropout), torch.nn.Dropout(dropout)
        self.norm1, self.norm2 = torch.nn.LayerNorm(width, eps=1e-6), torch.nn.LayerNorm(width, eps=1e-6)
        self.feed_forward = torch.nn.Sequential(torch.nn.Linear(width, inner), torch.nn.ReLU(), torch.nn.Linear(inner, width))

    def attend(self, query_input, context):
        rows, length, _ = query_input.shape

        def split(projection, values):
            return projection(values).view(rows, length, self.heads, self.key_size).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(split(self.query, query_input), split(self.key, context), split(self.value, context))
        return self.output(attended.transpose(1, 2).reshape(rows, length, self.heads * self.key_size))

    def forward(self, inputs):
        first = self.norm1(inputs + self.dropout1(self.attend(inputs, inputs)))
        second = self.norm2(first + self.dropout2(self.attend(first, inputs)))
        return self.norm2(second + self.dropout2(self.feed_forward(second)))


class FlowTransformerNetwork(torch.nn.Module):
    def __init__(self, feature_count, class_count, width=64, inner=128, heads=2, blocks=2, dropout=0.1):
        super().__init__()
        self.projection = torch.nn.Linear(feature_count, width, bias=False)
        self.blocks = torch.nn.Sequential(*(FlowTransformerDecoderBlock(width, inner, heads, dropout) for _ in range(blocks)))
        self.head = torch.nn.Sequential(torch.nn.Linear(width, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, class_count))
        # Keras Dense defaults of the official code: Glorot-uniform weights and zero biases.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, windows):
        return self.head(self.blocks(self.projection(windows))[:, -1])[:, None]


# ---------------------------------------------------------------- preprocessing (fitted on the inner-training rows)

class QuantileNormal:
    """Quantile-normal features; Gorishniy et al. 2021, App. B.2: N(0, 1e-3) noise only when the quantiles are fitted."""

    def __init__(self, train_values, seed):
        generator = np.random.default_rng(seed)
        noise_std = 1e-3 / np.maximum(train_values.std(axis=0, keepdims=True), 1e-3)
        self.transformer = QuantileTransformer(n_quantiles=max(min(len(train_values) // 30, 1000), 10), output_distribution="normal", subsample=10**9, random_state=seed)
        self.transformer.fit(train_values + noise_std * generator.standard_normal(train_values.shape))

    def transform(self, values):
        return self.transformer.transform(values)


class CnnScaler:
    """Signed log of the heavy-tailed columns, robust scaling and clipping (Donoso et al.)."""

    def __init__(self, train_values, seed):
        generator = np.random.default_rng(seed)
        self.heavy_tailed = np.abs(np.nan_to_num(kurtosis(train_values, axis=0, fisher=True))) > CNN_KURTOSIS_LIMIT
        sample = self.heavy_tail_log(train_values)[generator.choice(len(train_values), size=min(CNN_SCALER_ROWS, len(train_values)), replace=False)]
        self.scaler = RobustScaler().fit(sample)

    def heavy_tail_log(self, values):
        return np.where(self.heavy_tailed, vae.signed_log(values), values)

    def transform(self, values):
        return np.clip(self.scaler.transform(self.heavy_tail_log(values)), -CNN_CLIP, CNN_CLIP)


def log_min_max_parameters(train_values):
    """FlowTransformer StandardPreProcessing: log(x - min + 1) / log(range + 1) with the training minimum and range."""
    minimum = train_values.min(axis=0)
    value_range = train_values.max(axis=0) - minimum
    return minimum, np.where(value_range > 0, np.log1p(np.maximum(value_range, 0.0)), np.inf)


def apply_log_min_max(values, minimum, scale, out=None):
    """The same float64 operations as log1p(maximum(values - minimum, 0)) / scale; out=values saves one copy per step."""
    out = np.subtract(values, minimum, out=out)
    np.maximum(out, 0.0, out=out)
    np.log1p(out, out=out)
    out /= scale
    return out


# Fitted input transform of each method that scores one flow at a time; FlowTransformer scales its stream in run_method.
TRANSFORMS = {"tabm": QuantileNormal, "ft_transformer": QuantileNormal, "cnn1d": CnnScaler, "tabnet": lambda train_values, seed: baselines.Standardizer(train_values)}


class DeployedNetwork:
    """A network method in its deployment form on the CPU: raw flow features -> class probabilities (transform included)."""

    def __init__(self, transform, network):
        self.transform, self.network = transform, network.to("cpu").eval()

    def single_thread(self):
        torch.set_num_threads(1)

    def predict_proba(self, values):
        with torch.no_grad():
            inputs = torch.as_tensor(self.transform.transform(values), dtype=torch.float32)
            return torch.softmax(self.network(inputs), dim=-1).mean(dim=1).numpy()


class DeployedTabNet(DeployedNetwork):
    def __init__(self, transform, model):
        model.device = torch.device("cpu")
        model.network.to("cpu")
        # pytorch-tabnet keeps its group attention matrices as plain tensor attributes, which Module.to does not move.
        for module in model.network.modules():
            for name, value in list(vars(module).items()):
                if name.endswith("matrix") and isinstance(value, torch.Tensor):
                    setattr(module, name, value.to("cpu"))
        self.transform, self.model = transform, model

    def predict_proba(self, values):
        return self.model.predict_proba(self.transform.transform(values))


class DeployedFlowTransformer(DeployedNetwork):
    """FlowTransformer on a stream: each flow with its WINDOW_SIZE - 1 predecessors in arrival order; the first flows repeat
    the earliest flow, as day_stream_windows does at the start of a day."""

    def __init__(self, minimum, scale, network):
        super().__init__(None, network)
        self.minimum, self.scale = minimum, scale

    def stream(self):
        """values -> probabilities of a stream that keeps the last WINDOW_SIZE - 1 scaled flows between calls."""
        history = []

        def predict(values):
            scaled = torch.as_tensor(apply_log_min_max(np.asarray(values, dtype=np.float64), self.minimum, self.scale), dtype=torch.float32)
            context = torch.cat([*history, scaled]) if history else torch.cat([scaled[:1].expand(WINDOW_SIZE - 1, -1), scaled])
            positions = torch.arange(len(scaled))[:, None] + torch.arange(WINDOW_SIZE)[None, :] + (len(context) - len(scaled) - WINDOW_SIZE + 1)
            history[:] = [context[-(WINDOW_SIZE - 1):]]
            with torch.no_grad():
                return torch.softmax(self.network(context[positions]), dim=-1).mean(dim=1).numpy()
        return predict

    def predict_proba(self, values):
        return self.stream()(values)


# ---------------------------------------------------------------- training

def macro_f1(labels, predicted, class_count):
    classes = list(range(class_count))
    return metrics.macro_f1_with_min_support(metrics.per_class_label_metrics(labels, predicted, classes))["value"]


def predict_probabilities(network, split, device):
    network.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(split), PREDICT_BATCH):
            rows = torch.arange(start, min(start + PREDICT_BATCH, len(split)), device=device)
            outputs.append(torch.softmax(network(split.inputs(rows)), dim=-1).mean(dim=1).cpu())
    return torch.cat(outputs).numpy()


def batch_rows(split, recipe, generator, device):
    """Row index batches of one epoch: a shuffled pass, or class-balanced draws (equal rows per class in each batch)."""
    if not recipe.balanced_batches:
        order = torch.randperm(len(split), generator=generator, device="cpu").to(device)
        batches = order.split(recipe.batch_size)
        return batches if recipe.steps_per_epoch is None else batches[:recipe.steps_per_epoch]
    labels = split.labels.cpu().numpy()
    by_class = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    per_class = math.ceil(recipe.batch_size / len(by_class))
    numpy_generator = np.random.default_rng(int(torch.randint(0, 2**31, (1,), generator=generator)))
    return [
        torch.as_tensor(np.concatenate([numpy_generator.choice(rows, size=per_class, replace=len(rows) < per_class) for rows in by_class]), device=device)
        for _ in range(recipe.steps_per_epoch)
    ]


def train_network(network, optimizer, train, validation, class_weights, recipe, seed, device, log):
    """Best inner-validation state of the network, and its epoch and macro-F1."""
    generator = torch.Generator().manual_seed(seed)
    loss_weights = None if recipe.balanced_batches else class_weights
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2) if recipe.plateau_scheduler else None
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, recipe.max_epochs + 1):
        network.train()
        for rows in batch_rows(train, recipe, generator, device):
            logits = network(train.inputs(rows))
            members = logits.shape[1]
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), train.labels[rows].repeat_interleave(members), weight=loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if recipe.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(network.parameters(), recipe.gradient_clip)
            optimizer.step()
        predicted = predict_probabilities(network, validation, device).argmax(axis=1)
        score = macro_f1(validation.labels.cpu().numpy(), predicted, len(class_weights))
        if scheduler is not None:
            scheduler.step(score)
        log(f"epoch {epoch}: inner macro-F1 {score:.4f}")
        if score > best_score:
            best_score, best_epoch, best_state, stale = score, epoch, copy.deepcopy(network.state_dict()), 0
            continue
        stale += 1
        if stale >= recipe.patience:
            break
    network.load_state_dict(best_state)
    return best_epoch, best_score


class InnerMacroF1(Metric):
    """pytorch-tabnet early-stopping metric: macro-F1 over classes with min support, as for the other networks."""

    def __init__(self):
        self._name = "macro_f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        return macro_f1(y_true, y_score.argmax(axis=1), y_score.shape[1])


# ---------------------------------------------------------------- per method

def day_stream_windows(times, days, tie_key):
    """(rows, WINDOW_SIZE) indices of each row and its predecessors in the time order of its day (splits.stream_order);
    the first flows of a day repeat the day's earliest flow in place of the missing predecessors."""
    windows = np.empty((len(times), WINDOW_SIZE), dtype=np.int64)
    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        order = rows[splits.stream_order(times[rows], tie_key[rows])]
        positions = np.arange(len(order))[:, None] + np.arange(-WINDOW_SIZE + 1, 1)[None, :]
        windows[order] = order[np.clip(positions, 0, None)]
    return windows


def stream_windows(times, days, tie_key, training_count):
    """(windows of every row, windows of the first training_count rows, the training rows, built from those rows only).

    The context of a training flow holds only training flows, so no hold-out flow and no flow of a left-out class enters
    training; an evaluation window keeps every flow of its day, as a deployed model sees the stream."""
    training = slice(0, training_count)
    return day_stream_windows(times, days, tie_key), day_stream_windows(times[training], days[training], tie_key[training])


def run_method(method, seed, train_values, train_labels, train_frame, evaluation, class_count, log):
    """(probabilities for each (values, frame) part of evaluation, dict of method details, deployed model on the CPU); one
    trained model predicts every part."""
    evaluation_values = [values for values, _ in evaluation]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("no CUDA device: the network trains on the CPU")
    torch.manual_seed(seed)
    np.random.seed(seed)
    inner_validation = splits.assign_seen_holdout(train_frame)
    inner_train = ~inner_validation
    class_weights = torch.as_tensor(compute_class_weight("balanced", classes=np.arange(class_count), y=train_labels[inner_train]), dtype=torch.float32, device=device)

    if method != "flow_transformer":
        fitted = TRANSFORMS[method](train_values[inner_train], seed)
    if method == "tabnet":
        inner_values, validation_values, *evaluation_scaled = (fitted.transform(values) for values in (train_values[inner_train], train_values[inner_validation], *evaluation_values))
        model = TabNetClassifier(
            n_d=60, n_a=60, n_steps=3, gamma=1.5, n_independent=2, n_shared=3, lambda_sparse=1e-3, momentum=0.02, epsilon=1e-15,
            mask_type="entmax", optimizer_fn=torch.optim.Adam, scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"gamma": 0.95, "step_size": 10}, seed=seed, device_name=device, verbose=1,
        )
        model.fit(
            inner_values, train_labels[inner_train], eval_set=[(validation_values, train_labels[inner_validation])], eval_name=["inner"],
            eval_metric=[InnerMacroF1], max_epochs=TABNET_MAX_EPOCHS, patience=20, batch_size=1024, virtual_batch_size=128,
            loss_fn=torch.nn.CrossEntropyLoss(weight=class_weights),
        )
        probabilities = [model.predict_proba(values) for values in evaluation_scaled]
        return probabilities, {"best_epoch": int(model.best_epoch), "inner_macro_f1": float(model.best_cost)}, DeployedTabNet(fitted, model)

    if method != "flow_transformer":
        inner_values, validation_values, *evaluation_scaled = (
            torch.as_tensor(fitted.transform(values), dtype=torch.float32, device=device)
            for values in (train_values[inner_train], train_values[inner_validation], *evaluation_values)
        )
        labels = torch.as_tensor(train_labels, device=device)
        train = TabularData(inner_values, labels[torch.as_tensor(inner_train, device=device)])
        validation = TabularData(validation_values, labels[torch.as_tensor(inner_validation, device=device)])
        evaluation_data = [TabularData(values, torch.zeros(len(values), dtype=torch.long, device=device)) for values in evaluation_scaled]
        network_class, recipe = {
            "tabm": (TabMNetwork, Recipe(batch_size=1024, max_epochs=MAX_EPOCHS, patience=16, gradient_clip=1.0)),
            "ft_transformer": (FTTransformerNetwork, Recipe(batch_size=1024, max_epochs=MAX_EPOCHS, patience=17)),
            "cnn1d": (CnnNetwork, Recipe(batch_size=512, max_epochs=20, patience=4, plateau_scheduler=True)),
        }[method]
        network = network_class(train_values.shape[1], class_count).to(device)
        if method == "tabm":
            optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=3e-4)
        elif method == "ft_transformer":
            optimizer = network.model.make_default_optimizer()
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        best_epoch, best_score = train_network(network, optimizer, train, validation, class_weights, recipe, seed, device, log)
        probabilities = [predict_probabilities(network, part, device) for part in evaluation_data]
        return probabilities, {"best_epoch": best_epoch, "inner_macro_f1": best_score}, DeployedNetwork(fitted, network)

    # FlowTransformer: one stream per capture day over all rows (features only).
    # The stream has millions of rows (the held-out days of the detector), so each step below avoids a full float64 copy.
    stream_values = np.concatenate([train_values, *evaluation_values]).astype(np.float64, copy=False)
    times = np.concatenate([train_frame["time"].to_numpy(), *(frame["time"].to_numpy() for _, frame in evaluation)])
    days = times.astype("datetime64[D]")
    stream_frame = pd.DataFrame(stream_values, copy=False)
    feature_hash = splits.row_hashes(stream_frame, list(stream_frame.columns))
    del stream_frame
    windows, training_windows = (torch.as_tensor(values, device=device) for values in stream_windows(times, days, feature_hash, len(train_values)))
    other_rows = len(stream_values) - len(train_values)
    inner_rows = np.flatnonzero(np.concatenate([inner_train, np.zeros(other_rows, dtype=bool)]))
    validation_rows = np.flatnonzero(np.concatenate([inner_validation, np.zeros(other_rows, dtype=bool)]))
    part_starts = np.cumsum([len(train_values), *(len(values) for values in evaluation_values)])
    evaluation_rows = [np.arange(start, end) for start, end in itertools.pairwise(part_starts)]
    minimum, scale = log_min_max_parameters(train_values[inner_train])
    stream_float32 = apply_log_min_max(stream_values, minimum, scale, out=stream_values).astype(np.float32)
    del stream_values
    stream = torch.as_tensor(stream_float32, device=device)
    del stream_float32
    all_labels = torch.as_tensor(np.concatenate([train_labels, np.zeros(other_rows, dtype=train_labels.dtype)]), device=device)

    def windowed(rows, row_windows):
        rows = torch.as_tensor(rows, device=device)
        return TabularData(stream, all_labels[rows], row_windows[rows])
    train, validation = windowed(inner_rows, training_windows), windowed(validation_rows, training_windows)
    evaluation_data = [windowed(rows, windows) for rows in evaluation_rows]
    recipe = Recipe(batch_size=128, max_epochs=20, patience=5, steps_per_epoch=128, balanced_batches=True)
    best = None
    for learning_rate in FLOW_TRANSFORMER_LEARNING_RATES:
        torch.manual_seed(seed)
        network = FlowTransformerNetwork(train_values.shape[1], class_count).to(device)
        best_epoch, best_score = train_network(network, torch.optim.Adam(network.parameters(), lr=learning_rate), train, validation, class_weights, recipe, seed, device, log)
        log(f"learning rate {learning_rate}: inner macro-F1 {best_score:.4f} at epoch {best_epoch}")
        if best is None or best_score > best[0]:
            best = (best_score, best_epoch, learning_rate, [predict_probabilities(network, part, device) for part in evaluation_data], copy.deepcopy(network))
    best_score, best_epoch, learning_rate, probabilities, best_network = best
    details = {"best_epoch": best_epoch, "inner_macro_f1": best_score, "learning_rate": learning_rate}
    return probabilities, details, DeployedFlowTransformer(minimum, scale, best_network)


def run_on_gpu(method, seed, train_values, train_labels, train_frame, evaluation, class_count, log):
    with files.gpu_lock():
        return run_method(method, seed, train_values, train_labels, train_frame, evaluation, class_count, log)
