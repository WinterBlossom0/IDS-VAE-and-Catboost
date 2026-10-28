"""Stage 2: product-of-experts VAE with a Student-t or Gaussian likelihood on the selected features.

A flow statistic that CICFlowMeter cannot compute for a flow (too few packets, no idle period) is a missing entry: its expert
is absent from the encoder product and its term is absent from the reconstruction likelihood.
"""

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional

from idsvae import config, files
from idsvae.data import splits
from idsvae.framework import features

ENCODED_SPLITS = ["fit_train", "fit_holdout", "vae_val", "calib", "test"]
# Burda, Grosse, Salakhutdinov, ICLR 2016, Sect. 5.2: a latent unit is active if the variance over the data of its posterior mean exceeds 0.01.
ACTIVE_UNIT_VARIANCE_THRESHOLD = 0.01
LIKELIHOODS = ("student_t", "gaussian")
LOG_SCALE_MIN = -6.0
# Margin in log space: a log scale below LOG_SCALE_MIN + NEAR_FLOOR_MARGIN counts as at the floor.
NEAR_FLOOR_MARGIN = 0.05
DECODER_STATISTICS_ROWS = 20_000
# Ma et al. (EDDI), ICML 2019, App. B.2.1: "a 20 dimensional feature mapping h parameterized by a single layer neural network".
EXPERT_WIDTH = 20
# EDDI App. B.1.1: "We first draw a missing rate parameter from a uniform distribution U(0, 0.7)", redrawn for each row and step.
SUBSET_MAX_MISSING_RATE = 0.7
# The subset masks use their own random stream, so that they do not shift the latent samples of the other loss terms.
MASK_SEED_OFFSET = 1_000_003
# Rows per batch for the topological loss (Moor et al. use mini-batches of this order); the spanning tree runs on the GPU.
TOPOLOGY_ROWS = 256
COUNT_COLUMNS = ("Tot Fwd Pkts", "Tot Bwd Pkts")

IDLE_COLUMN, ACTIVE_COLUMN = "Idle Max", "Active Max"
# CICFlowMeter BasicFlow.java (dumpFlowBasedFeatures) writes 0 for a statistic the flow cannot define; the longest matching prefix decides.
UNDEFINED_RULES = {
    "Flow IAT": lambda forward, backward, source: forward + backward <= 1,
    "Flow IAT Std": lambda forward, backward, source: forward + backward <= 2,
    "Fwd IAT": lambda forward, backward, source: forward <= 1,
    "Fwd IAT Std": lambda forward, backward, source: forward <= 2,
    "Bwd IAT": lambda forward, backward, source: backward <= 1,
    "Bwd IAT Std": lambda forward, backward, source: backward <= 2,
    "Fwd Pkt Len Std": lambda forward, backward, source: forward <= 1,
    "Bwd Pkt Len": lambda forward, backward, source: backward == 0,
    "Bwd Pkt Len Std": lambda forward, backward, source: backward <= 1,
    "Bwd Seg Size Avg": lambda forward, backward, source: backward == 0,
    "Pkt Len Std": lambda forward, backward, source: forward + backward <= 1,
    "Pkt Len Var": lambda forward, backward, source: forward + backward <= 1,
    # Known limit: Idle Std and Active Std of a flow with exactly one period are 0 but count as defined; the period count is not in the data.
    "Idle": lambda forward, backward, source: source[IDLE_COLUMN] == 0,
    "Active": lambda forward, backward, source: source[ACTIVE_COLUMN] == 0,
}
# A TCP window cannot be negative; the flow tool of the dataset writes -1 when it saw no window in that direction.
WINDOW_PLACEHOLDER = -1
RULE_COLUMNS = {"Idle": IDLE_COLUMN, "Active": ACTIVE_COLUMN}


@dataclass
class VaeSettings:
    likelihood: str = "student_t"
    latent_dim: int = 5
    hidden_dim: int = 128
    beta: float = 0.5
    # Sinha & Dieng, NeurIPS 2021 (CR-VAE): the default consistency weight of a beta-VAE is 0.1 * beta.
    consistency_ratio: float = 0.1
    # Weight lambda of the topological loss (Moor et al., ICML 2020, Eq. 1-3); 0 turns it off.
    topology_weight: float = 0.0
    # Linear learning-rate scaling for 8 times the batch of 4096 (Goyal et al., 2017); 1e-3 times 1, sqrt(8) and 8 compared on
    # the 02-28 validation loss, and only 8e-3 converged. The cap leaves the stop to the early-stopping rule.
    learning_rate: float = 8e-3
    batch_size: int = 32768
    max_epochs: int = 1000
    patience: int = 10
    # 1e-4 is the documented default threshold of torch ReduceLROnPlateau in threshold_mode="rel".
    relative_tolerance: float = 1e-4
    lr_patience: int = 3
    lr_factor: float = 0.5
    feature_count: int = 0
    # A leave-one-out VAE must not see the left-out attack class, in its training rows or in its standardizer.
    left_out_class: str = ""
    seed: int = 0
    # "main" uses the Stage 1 ranking without the TCP fingerprint columns; "fingerprints" uses the ranking that keeps them.
    feature_set: str = "main"

    def run_name(self):
        base = f"{self.likelihood}_k{self.feature_count}_L{self.latent_dim}_s{self.seed}"
        # A weight of 0 adds no suffix, so that the stored runs without the topological loss keep their directory names.
        if self.topology_weight != 0.0:
            base = f"{base}_topo{self.topology_weight:g}"
        return base if self.feature_set == "main" else f"{base}_{self.feature_set}"


def latent_columns(latent_dim):
    return [f"z{index}" for index in range(latent_dim)]


def signed_log(values):
    array_module = torch if isinstance(values, torch.Tensor) else np
    return array_module.sign(values) * array_module.log1p(array_module.abs(values))


def undefined_rule(column):
    matches = [prefix for prefix in UNDEFINED_RULES if column.startswith(prefix)]
    return max(matches, key=len) if matches else None


def is_window(column):
    return column.startswith("Init ") and column.endswith("Win Byts")


def can_be_undefined(column):
    return undefined_rule(column) is not None or is_window(column)


def mask_source_columns(columns):
    """Raw columns that decide which entries are undefined: the packet counts and the period columns of the selected features."""
    extra = {RULE_COLUMNS[rule] for rule in map(undefined_rule, columns) if rule in RULE_COLUMNS}
    return sorted(set(columns) | set(COUNT_COLUMNS) | extra)


def column_undefined(source, column):
    """Undefined mask of one column, or None for a column that every flow defines; source values may be NumPy arrays or torch tensors."""
    if is_window(column):
        return source[column] == WINDOW_PLACEHOLDER
    rule = undefined_rule(column)
    if rule is None:
        return None
    forward, backward = (source[count_column] for count_column in COUNT_COLUMNS)
    return UNDEFINED_RULES[rule](forward, backward, source)


def undefined_entries(source, columns):
    """Boolean matrix (rows, columns); source maps a column name to its raw values (a DataFrame or a dict of arrays)."""
    source = {column: np.asarray(source[column]) for column in {*columns, *mask_source_columns(columns)}}
    missing = np.zeros((len(source[COUNT_COLUMNS[0]]), len(columns)), dtype=bool)
    for index, column in enumerate(columns):
        undefined = column_undefined(source, column)
        if undefined is not None:
            missing[:, index] = undefined
    return missing


def error_feature_mask(columns):
    """The features that every flow defines; recon_nll sums over these only, so that it compares rows with different missing entries."""
    mask = np.array([not can_be_undefined(column) for column in columns])
    if not mask.any():
        raise ValueError("no selected feature is defined for every flow, so recon_nll has no terms")
    return mask


class Standardizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, values, missing):
        """Mean and std of each feature over its defined entries."""
        defined = np.where(missing, np.nan, values)
        mean = np.nan_to_num(np.nanmean(defined, axis=0))
        std = np.nan_to_num(np.nanstd(defined, axis=0))
        return cls(mean, np.where(std > 0, std, 1.0))

    def transform(self, values):
        return ((values - self.mean) / self.std).astype("float32")

    def to_json(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def model_inputs(source, columns, standardizer):
    """Standardized signed-log inputs with 0 at the missing entries, and the observed mask (1.0 defined, 0.0 missing)."""
    values = np.column_stack([np.asarray(source[column], dtype="float64") for column in columns])
    missing = undefined_entries(source, columns)
    inputs = standardizer.transform(signed_log(values))
    inputs[missing] = 0.0
    return inputs, (~missing).astype("float32")


def product_of_experts(expert_mean, expert_log_variance, observed):
    """Gaussian product of the N(0, I) prior expert and the observed experts (Wu & Goodman, NeurIPS 2018, Sect. 2.1)."""
    log_precision = (-expert_log_variance).masked_fill(observed[..., None] == 0, float("-inf"))
    all_log_precision = torch.cat([torch.zeros_like(log_precision[:, :1]), log_precision], dim=1)
    log_total_precision = torch.logsumexp(all_log_precision, dim=1)
    weights = torch.exp(all_log_precision[:, 1:] - log_total_precision[:, None])
    return (weights * expert_mean).sum(dim=1), -log_total_precision


class Vae(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, likelihood):
        super().__init__()
        if likelihood not in LIKELIHOODS:
            raise ValueError(f"unknown likelihood: {likelihood}")
        self.likelihood = likelihood
        self.latent_dim = latent_dim
        # Each expert is a one-hidden-layer network on one scalar feature; the bounds are the nn.Linear defaults (fan_in 1, then EXPERT_WIDTH).
        self.expert_weight = nn.Parameter(torch.empty(input_dim, EXPERT_WIDTH).uniform_(-1.0, 1.0))
        self.expert_bias = nn.Parameter(torch.empty(input_dim, EXPERT_WIDTH).uniform_(-1.0, 1.0))
        bound = 1.0 / math.sqrt(EXPERT_WIDTH)
        self.expert_output_weight = nn.Parameter(torch.empty(input_dim, EXPERT_WIDTH, 2 * latent_dim).uniform_(-bound, bound))
        self.expert_output_bias = nn.Parameter(torch.empty(input_dim, 2 * latent_dim).uniform_(-bound, bound))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.location_head = nn.Linear(hidden_dim, input_dim)
        self.log_scale_head = nn.Linear(hidden_dim, input_dim)
        if likelihood == "student_t":
            self.degrees_of_freedom_head = nn.Linear(hidden_dim, input_dim)

    def experts(self, inputs):
        """Mean and log variance of each feature's expert, shape (rows, features, latent)."""
        hidden = torch.relu(inputs[..., None] * self.expert_weight + self.expert_bias)
        output = torch.einsum("rfh,fho->rfo", hidden, self.expert_output_weight) + self.expert_output_bias
        return output[..., :self.latent_dim], output[..., self.latent_dim:]

    def encode(self, inputs, observed):
        return product_of_experts(*self.experts(inputs), observed)

    def sample_latent(self, mean, log_variance):
        return mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)

    def decode(self, latent):
        """Per-feature location, log scale and (Student-t only, else None) degrees of freedom."""
        hidden = self.decoder(latent)
        log_scale = soft_clip_log_scale(self.log_scale_head(hidden))
        if self.likelihood == "gaussian":
            return self.location_head(hidden), log_scale, None
        return self.location_head(hidden), log_scale, degrees_of_freedom_from(self.degrees_of_freedom_head(hidden))


def is_new_best(loss, best_loss, relative_tolerance):
    # The start value inf has no relative margin: inf - tolerance * inf is NaN.
    if math.isinf(best_loss):
        return loss < best_loss
    return loss < best_loss - relative_tolerance * abs(best_loss)


class RelativePlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    # Override of a private torch method: the torch rule loss < best * (1 - threshold) accepts a worse loss when best is negative.
    def _is_better(self, loss, best_loss):
        return is_new_best(loss, best_loss, self.threshold)


class EarlyStopping:
    def __init__(self, patience, relative_tolerance):
        self.patience = patience
        self.relative_tolerance = relative_tolerance
        self.best_loss = float("inf")
        self.stale_epochs = 0

    def update(self, loss):
        """Return (improved, should_stop)."""
        if is_new_best(loss, self.best_loss, self.relative_tolerance):
            self.best_loss = loss
            self.stale_epochs = 0
            return True, False
        self.stale_epochs += 1
        return False, self.stale_epochs >= self.patience


def soft_clip_log_scale(raw):
    # Rybkin, Daniilidis, Levine, ICML 2021, Sect. 3.3: a soft lower bound on log sigma stops the scale collapse to zero.
    return LOG_SCALE_MIN + functional.softplus(raw - LOG_SCALE_MIN)


def degrees_of_freedom_from(raw):
    # nu > 1 keeps the mean of each feature finite.
    return 1.0 + functional.softplus(raw)


# Per-feature full negative log-densities in nats, shape (rows, features): Takahashi et al., IJCAI 2018, Eq. 7 (Gaussian) and Eq. 11 (Student-t, univariate per feature).
def gaussian_nll_terms(inputs, location, log_scale):
    standardized = (inputs - location) * torch.exp(-log_scale)
    return 0.5 * math.log(2 * math.pi) + log_scale + 0.5 * standardized.pow(2)


def student_t_nll_terms(inputs, location, log_scale, degrees_of_freedom, log_gamma=torch.lgamma):
    standardized = (inputs - location) * torch.exp(-log_scale)
    log_normalizer = log_gamma((degrees_of_freedom + 1) / 2) - log_gamma(degrees_of_freedom / 2) - 0.5 * torch.log(math.pi * degrees_of_freedom)
    return -log_normalizer + log_scale + ((degrees_of_freedom + 1) / 2) * torch.log1p(standardized.pow(2) / degrees_of_freedom)


def kl_per_dimension(mean, log_variance):
    return 0.5 * (mean.pow(2) + log_variance.exp() - 1.0 - log_variance)


def gaussian_kl(mean_q, log_variance_q, mean_p, log_variance_p):
    """KL(N(mean_q, var_q) || N(mean_p, var_p)) summed over the latent dimensions."""
    return 0.5 * ((log_variance_q.exp() + (mean_q - mean_p).pow(2)) / log_variance_p.exp() - 1.0 + log_variance_p - log_variance_q).sum(dim=1)


def reconstruction_nll(model, inputs, latent, weights):
    """Sum of the per-feature NLL terms times weights: the observed mask in training, the error-feature mask for recon_nll."""
    location, log_scale, degrees_of_freedom = model.decode(latent)
    if model.likelihood == "gaussian":
        terms = gaussian_nll_terms(inputs, location, log_scale)
    else:
        terms = student_t_nll_terms(inputs, location, log_scale, degrees_of_freedom)
    return (terms * weights).sum(dim=1)


def subset_mask(observed, generator):
    """EDDI random subsets of the defined features, restricted to the missing pattern of another row of the batch."""
    # VAEAC (Ivanov et al., ICLR 2019, Sect. 3): the mask distribution p(b) "may be defined by the user depending on the problem".
    rows = len(observed)
    missing_rate = torch.rand(rows, 1, generator=generator, device=observed.device) * SUBSET_MAX_MISSING_RATE
    kept = (torch.rand(observed.shape, generator=generator, device=observed.device) >= missing_rate).float()
    donors = torch.randperm(rows, generator=generator, device=observed.device)
    return observed * kept * observed[donors]


def distance_matrix(points):
    """Pairwise Euclidean distances divided by their mean, so that the input space and the latent space, of different dimensions, compare."""
    # The matrix-product mode of cdist gives rounding noise instead of 0 for equal rows, which changes the spanning tree.
    distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    return distances / distances.mean().clamp(min=1e-12)


def spanning_tree_masks(distances):
    """Upper-triangle masks of the minimum spanning trees of a stack of distance matrices (graphs, n, n).

    The tree edges are the 0-dimensional persistence pairing of the Vietoris-Rips filtration (Moor et al., Sect. 3.1).
    Boruvka's algorithm with a fixed count of rounds runs on the device of the matrices without a host synchronization.
    As in the reference code of Moor et al. (github.com/BorgwardtLab/topological-autoencoders, src/topology.py), an edge
    of weight 0 is an edge, and edges of equal weight are ordered by their row-major upper-triangle index (their stable
    sort), so that the tree equals the tree of their union-find.
    """
    graphs, nodes = distances.shape[0], distances.shape[-1]
    device = distances.device
    directed = distances.detach()
    # Each edge must have one weight; with two different weights the order of equal weights is not consistent and the
    # tree can close a cycle. The reference code reads the upper triangle.
    weights = torch.triu(directed, 1) + torch.triu(directed, 1).transpose(1, 2)
    index = torch.arange(nodes, device=device)
    no_edge = nodes * nodes
    edge_id = torch.minimum(index[:, None], index[None, :]) * nodes + torch.maximum(index[:, None], index[None, :])
    labels = index.expand(graphs, nodes)
    component = labels.clone()
    tree = torch.zeros(graphs, no_edge + 1, dtype=torch.bool, device=device)
    # Each round at least halves the count of components, and pointer doubling halves the depth of the parent forest.
    rounds = max(1, math.ceil(math.log2(nodes)))
    for _ in range(rounds):
        outgoing = weights.masked_fill(component[:, :, None] == component[:, None, :], math.inf)
        node_weight = outgoing.amin(dim=2)
        node_edge = torch.where(outgoing == node_weight[:, :, None], edge_id, no_edge).amin(dim=2)
        node_edge = node_edge.masked_fill(torch.isinf(node_weight), no_edge)
        component_weight = torch.full_like(node_weight, math.inf).scatter_reduce(1, component, node_weight, "amin")
        is_best = node_weight == component_weight.gather(1, component)
        component_edge = torch.full_like(node_edge, no_edge).scatter_reduce(1, component, torch.where(is_best, node_edge, no_edge), "amin")
        tree.scatter_(1, component_edge, True)
        has_edge = component_edge < no_edge
        ends = component_edge.clamp(max=no_edge - 1)
        first_end = component.gather(1, ends // nodes)
        second_end = component.gather(1, ends % nodes)
        parent = torch.where(has_edge, torch.where(first_end == labels, second_end, first_end), labels)
        # Two components that chose the same edge point to each other; the smaller label becomes the root.
        mutual = (parent.gather(1, parent) == labels) & (labels < parent)
        parent = torch.where(mutual, labels, parent)
        for _ in range(rounds):
            parent = parent.gather(1, parent)
        component = parent.gather(1, component)
    return tree[:, :no_edge].view(graphs, nodes, nodes)


def topological_loss(inputs, latent, rows=TOPOLOGY_ROWS):
    """L_X->Z + L_Z->X of Moor et al. (ICML 2020, Eq. 2-3) on the first rows of a shuffled batch, divided by the row count."""
    input_distances, latent_distances = distance_matrix(inputs[:rows]), distance_matrix(latent[:rows])
    input_tree, latent_tree = spanning_tree_masks(torch.stack([input_distances, latent_distances]))
    squared_difference = (input_distances - latent_distances).pow(2)
    input_to_latent = (squared_difference * input_tree).sum() / 2
    latent_to_input = (squared_difference * latent_tree).sum() / 2
    return (input_to_latent + latent_to_input) / len(inputs[:rows])


def training_loss(model, inputs, observed, settings, mask_generator):
    """Full ELBO + EDDI partial ELBO on a subset (Ma et al., ICML 2019, Eq. 4) + CR-VAE consistency KL(q(z | subset) || q(z | x)) (Sinha & Dieng, NeurIPS 2021, Algorithm 1)."""
    # The experts do not depend on the mask, so one expert pass serves both posteriors.
    expert_mean, expert_log_variance = model.experts(inputs)
    mean, log_variance = product_of_experts(expert_mean, expert_log_variance, observed)
    recon = reconstruction_nll(model, inputs, model.sample_latent(mean, log_variance), observed)
    kl = kl_per_dimension(mean, log_variance).sum(dim=1)
    subset = subset_mask(observed, mask_generator)
    subset_mean, subset_log_variance = product_of_experts(expert_mean, expert_log_variance, subset)
    subset_recon = reconstruction_nll(model, inputs, model.sample_latent(subset_mean, subset_log_variance), subset)
    subset_kl = kl_per_dimension(subset_mean, subset_log_variance).sum(dim=1)
    consistency = gaussian_kl(subset_mean, subset_log_variance, mean, log_variance)
    elbo_terms = (recon + settings.beta * kl + subset_recon + settings.beta * subset_kl + settings.consistency_ratio * settings.beta * consistency).mean()
    if settings.topology_weight == 0.0:
        return elbo_terms
    # model_inputs already writes 0 at the missing entries, so the inputs need no mask here.
    return elbo_terms + settings.topology_weight * topological_loss(inputs, mean)


def mean_loss(model, inputs, observed, settings):
    """Validation loss: the full ELBO at the posterior mean, without the subset and consistency terms."""
    # Early stopping uses this negative beta-ELBO with z at the posterior mean, without sampling, so it has no sampling noise.
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(inputs), settings.batch_size):
            batch, batch_observed = inputs[start:start + settings.batch_size], observed[start:start + settings.batch_size]
            mean, log_variance = model.encode(batch, batch_observed)
            recon = reconstruction_nll(model, batch, mean, batch_observed)
            total += float((recon + settings.beta * kl_per_dimension(mean, log_variance).sum(dim=1)).sum())
    return total / len(inputs)


def train(model, train_inputs, train_observed, val_inputs, val_observed, settings, generator, mask_generator):
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = RelativePlateauScheduler(optimizer, mode="min", factor=settings.lr_factor, patience=settings.lr_patience, threshold=settings.relative_tolerance, threshold_mode="rel")
    stopping = EarlyStopping(settings.patience, settings.relative_tolerance)
    best_state, best_epoch, stopped_early, history = None, 0, False, []
    for epoch in range(settings.max_epochs):
        model.train()
        permutation = torch.randperm(len(train_inputs), generator=generator, device="cpu").to(train_inputs.device)
        for start in range(0, len(train_inputs), settings.batch_size):
            rows = permutation[start:start + settings.batch_size]
            loss = training_loss(model, train_inputs[rows], train_observed[rows], settings, mask_generator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = mean_loss(model, val_inputs, val_observed, settings)
        if not math.isfinite(val_loss):
            raise RuntimeError(f"validation loss is not finite at epoch {epoch + 1} with likelihood {settings.likelihood}: {val_loss}")
        history.append({"epoch": epoch + 1, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch + 1}: val_loss {val_loss:.4f}", flush=True)
        scheduler.step(val_loss)
        improved, should_stop = stopping.update(val_loss)
        if improved:
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
        if should_stop:
            stopped_early = True
            break
    model.load_state_dict(best_state)
    return history, best_epoch, stopped_early


def encode_batch(model, inputs, observed, error_weights):
    """Posterior mean, reconstruction NLL over the error features at the posterior mean, and KL per latent dimension."""
    mean, log_variance = model.encode(inputs, observed)
    return mean, reconstruction_nll(model, inputs, mean, error_weights), kl_per_dimension(mean, log_variance)


def device_description(device):
    return f"cuda: {torch.cuda.get_device_name()}" if device == "cuda" else "cpu"


def encode_frame(model, frame, columns, standardizer, settings, device):
    inputs, observed = (torch.from_numpy(array).to(device) for array in model_inputs(frame, columns, standardizer))
    error_weights = torch.from_numpy(error_feature_mask(columns).astype("float32")).to(device)
    means, recons, kls = [], [], []
    with torch.no_grad():
        for start in range(0, len(inputs), settings.batch_size):
            mean, recon, kl = encode_batch(model, inputs[start:start + settings.batch_size], observed[start:start + settings.batch_size], error_weights)
            means.append(mean.cpu())
            recons.append(recon.cpu())
            kls.append(kl.cpu())
    mean = torch.cat(means).numpy()
    kl = torch.cat(kls).numpy()
    encoded = pd.DataFrame(mean, columns=latent_columns(settings.latent_dim))
    encoded["recon_nll"] = torch.cat(recons).numpy()
    encoded["kl"] = kl.sum(axis=1)
    for column in splits.METADATA_COLUMNS:
        if column in frame.columns:
            encoded[column] = frame[column].to_numpy()
    return encoded, kl.mean(axis=0)


def latent_mean_variance(encoded, settings):
    """Variance over the VAE training rows of each posterior-mean dimension."""
    rows = vae_rows(encoded, settings)
    return rows[latent_columns(settings.latent_dim)].to_numpy(dtype="float64").var(axis=0)


def vae_rows(frame, settings):
    """Rows of a split that the VAE and its standardizer see: all classes except the left-out attack class."""
    features.require_attack_class(settings.left_out_class)
    if not settings.left_out_class:
        return frame
    return frame[frame["label"] != settings.left_out_class]


def decoder_statistics(model, inputs, observed):
    """Medians of the decoder scale and degrees of freedom over the defined entries at the posterior mean, and the share of scales near the floor."""
    with torch.no_grad():
        _, log_scale, degrees_of_freedom = model.decode(model.encode(inputs, observed)[0])
    defined = observed > 0
    log_scale = log_scale[defined]
    statistics = {
        "rows": len(inputs),
        "scale_median": float(torch.exp(log_scale).median()),
        "share_scale_near_floor": float((log_scale < (LOG_SCALE_MIN + NEAR_FLOOR_MARGIN)).float().mean()),
    }
    if model.likelihood == "student_t":
        statistics["degrees_of_freedom_median"] = float(degrees_of_freedom[defined].median())
    return statistics


def run_vae(settings, output_root, processed_dir=config.PROCESSED_DIR, ranking_dir=features.STAGE1_DIR):
    ranking = features.load_ranking(ranking_dir)
    if settings.feature_count == 0:
        settings.feature_count = ranking["selected_count"]
    columns = features.top_features(ranking, settings.feature_count)
    source_columns = mask_source_columns(columns)
    error_features = [column for column, keep in zip(columns, error_feature_mask(columns)) if keep]
    output_dir = output_root / settings.run_name()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(settings.seed)
    generator = torch.Generator().manual_seed(settings.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_generator = torch.Generator(device=device).manual_seed(settings.seed + MASK_SEED_OFFSET)
    print(f"VAE device: {device_description(device)}", flush=True)

    train_frame = vae_rows(splits.load_split("fit_train", processed_dir, columns=source_columns), settings)
    # The 02-28 attack rows stay in the validation split too, because the loss does not use labels.
    val_frame = vae_rows(splits.load_split("vae_val", processed_dir, columns=source_columns), settings)
    train_missing = undefined_entries(train_frame, columns)
    standardizer = Standardizer.fit(signed_log(train_frame[columns].to_numpy(dtype="float64")), train_missing)
    train_inputs, train_observed = (torch.from_numpy(array).to(device) for array in model_inputs(train_frame, columns, standardizer))
    val_inputs, val_observed = (torch.from_numpy(array).to(device) for array in model_inputs(val_frame, columns, standardizer))

    model = Vae(len(columns), settings.hidden_dim, settings.latent_dim, settings.likelihood).to(device)
    start_time = time.perf_counter()
    history, best_epoch, stopped_early = train(model, train_inputs, train_observed, val_inputs, val_observed, settings, generator, mask_generator)
    training = {"seconds": time.perf_counter() - start_time, "device": device_description(device), "epochs_run": len(history), "best_epoch": best_epoch, "stopped_early": stopped_early}
    model.eval()
    sample_rows = torch.randperm(len(train_inputs), generator=generator)[:DECODER_STATISTICS_ROWS].to(device)
    decoder = decoder_statistics(model, train_inputs[sample_rows], train_observed[sample_rows])

    torch.save(model.state_dict(), output_dir / "model.pt")
    kl_by_split = {}
    for split in ENCODED_SPLITS:
        encoded, kl_by_split[split] = encode_frame(model, splits.load_split(split, processed_dir, columns=source_columns), columns, standardizer, settings, device)
        files.write_parquet_atomic(encoded, output_dir / f"{split}.parquet", index=False)
        if split == "fit_train":
            train_mean_variance = latent_mean_variance(encoded, settings)

    summary = {
        "settings": asdict(settings),
        "features": columns,
        "error_features": error_features,
        "undefined_share_by_feature": dict(zip(columns, train_missing.mean(axis=0).round(6).tolist())),
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "history": history,
        "training": training,
        "decoder": decoder,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_file_bytes": (output_dir / "model.pt").stat().st_size,
        "mean_kl_per_dimension": {split: values.tolist() for split, values in kl_by_split.items()},
        "latent_mean_variance": train_mean_variance.tolist(),
        "active_units": int((train_mean_variance > ACTIVE_UNIT_VARIANCE_THRESHOLD).sum()),
        "standardizer": standardizer.to_json(),
    }
    files.write_json_atomic(output_dir / "summary.json", summary)
    return output_dir, summary


def load_model(run_dir):
    """Trained model (CPU, eval mode), standardizer and selected features of a stored VAE run."""
    summary = json.loads((run_dir / "summary.json").read_text())
    settings = VaeSettings(**summary["settings"])
    model = Vae(len(summary["features"]), settings.hidden_dim, settings.latent_dim, settings.likelihood)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()
    standardizer = Standardizer(np.array(summary["standardizer"]["mean"]), np.array(summary["standardizer"]["std"]))
    return model, standardizer, summary["features"]


