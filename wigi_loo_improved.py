"""
Wi-Gitation leave-one-out (location 1, receiver 0), IMPROVED training protocol.

Same script as the paper-protocol version, with only the training procedure changed:
  * Validation on VALIDATION_COUNT (3) participants instead of one: the highest IDs other than the
    test participant. Every participant except the last ID is the test participant once (22 folds).
  * Validate once per epoch, on the validation loss.
  * Early stopping cannot trigger before MIN_EPOCHS (5) epochs; patience is PATIENCE (3) epochs
    without a new best validation loss.
  * The checkpoint with the lowest validation loss is the one tested (not the last one).
  * Batch-norm momentum 0.1 (= MATLAB MeanDecay 0.1).
  * Results go to their own folder (OUT_DIR), so they never mix with the paper-protocol results.

Protocol (Sharma et al., Data 2024; 9(1):9)
  * Nearest-neighbour interpolation of every recording to 12,000 samples (120 s x 100 Hz) so
    packet loss does not change the number of samples.
  * Windows of 2 s (200 samples) with 50% overlap, input 270 x 200 x 1, ResNet50 (from ONNX).
  * Metric: macro-F1 over the 7 classes (accuracy is stored too).

Idle windows (activity-period detection)
  Add "+idle" to any method name, e.g. "none+idle" or "pca10+idle". Windows whose activity (std or
  MAD of the signal, averaged over the 270 channels) is below IDLE_FACTOR x the median window of the
  SAME recording are dropped. The score is computed on the raw recording, so every method drops exactly
  the same windows. By default (IDLE_SCOPE = "train") only training windows are dropped; validation
  and test windows are kept, so every method is tested on the same windows as the baseline.

Extra rules for this study
  * Participants are matched by ID across activities (never by position in a list).
  * Anything learned from data (PCA, normalization statistics, the zerocenter mean) is fitted on
    the training participants of that fold only.
  * Every method is run with several seeds; results are stored per fold, so an interrupted run
    resumes where it stopped.

Usage
  python wigi_loo_improved.py           run the experiments (resumable), then write the report
  python wigi_loo_improved.py --report  only rebuild the report from the stored results
"""

import argparse
import copy
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx2torch import convert
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_DIR = r"S:\6th year\honors\project\WiGi_processed_data"    # change to your local directory
ONNX_PATH = r"S:\6th year\honors\project\resnet50_7act.onnx"
OUT_DIR = Path("loo_gpu_results")
LOG_DIR = OUT_DIR / "logs"
RESULTS_FILE = OUT_DIR / "fold_results.jsonl"     # one JSON line per finished fold
REPORT_FILE = OUT_DIR / "report.md"

NO_OF_PARTICIPANTS = 25
RECEIVER = 0                                      # receiver folder R0
LOCATION = "L1"

# (folder name, class label)
ACTIVITIES = [
    ("Disturbed walking_Free", 0),
    ("Sitting standing_L1",    1),
    ("Kicking_L1",             2),
    ("Tapping_L1",             3),
    ("Wringing hands_L1",      4),
    ("Rubbing table_L1",       5),
    ("Flipping objects_L1",    6),
]
NUM_CLASSES = len(ACTIVITIES)

# Methods to run. "none" is the baseline. Remove any you do not want.
#   butter         Butterworth low-pass            (per recording, no fitting)
#   dwt            wavelet soft-threshold denoising (per recording, no fitting; needs PyWavelets)
#   pca10          PCA projection, 10 components   (fitted on the training participants only)
#   zscore_window  z-score of every 270 x 200 window on its own (no fitting)
#   zscore_channel z-score per channel (270), mean/std fitted on the training participants only
#   norm_recording each recording on its own: per-channel mean subtracted, divided by its std (no fitting)
#   pca_drop       PCA that REMOVES the top PCA_DROP_COMPONENTS components (fitted on the training participants only)
#   <method>+adabn the same method, but the batch-norm statistics used for the test are recomputed on the
#                  test participant's own (unlabeled) windows instead of the training windows
# 


#   <method>+idle  the same method, with idle windows dropped (see IDLE_* below), e.g. "none+idle"
METHODS = [ "none+adabn", "norm_recording+adabn"]                   # run the no-preprocessing baseline first
BASELINE = "none"
SEEDS = [5]                           # one seed first
TEST_IDS = None                       # None = every participant except the last ID (22 folds)
#TEST_IDS = [1, 11, 14]               # pilot: 3 folds first
#SEEDS = [5, 6, 7]
#TEST_IDS = None            # None = every participant except the validation one; or e.g. [1, 3, 5]

# Idle-window dropping, used by every method whose name ends in "+idle"
IDLE_FACTOR = 0.7          # drop windows with activity < IDLE_FACTOR x the recording's median (0 < factor <= 1)
IDLE_SCORE = "std"         # activity score: "std" or "mad" (1.4826 x median absolute deviation)
IDLE_SCOPE = "train"       # "train": drop from training windows only (validation and test keep every window)
                           # "all"  : drop from training, validation and test windows

# Signal / windowing
FS = 100                                   # packets per second
RECORDING_SECONDS = 120
EXPECTED_SAMPLES = FS * RECORDING_SECONDS  # 12,000 samples per recording
NUM_CHANNELS = 270                         # 3 x 3 antenna pairs x 30 subcarriers
WIN_LEN = 2                                # seconds -> 200 samples
OVERLAP_PCT = 50

# Preprocessing parameters (single source of truth, also written into the report)
BUTTER_CUTOFF_HZ = 3.0
BUTTER_ORDER = 4
DWT_WAVELET = "db4"
DWT_LEVEL = 3
PCA_COMPONENTS = 10
PCA_DROP_COMPONENTS = 3     # "pca_drop": remove this many top-variance components (fitted on the training participants)
ZSCORE_EPS = 1e-8

# Training (identical for every method)
MINI_BATCH_SIZE = 8
MAX_EPOCHS = 50
INITIAL_LR = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
VALIDATION_COUNT = 3       # validation participants per fold (highest IDs other than the test participant)
MIN_EPOCHS = 5             # early stopping cannot trigger before this many epochs
PATIENCE = 3               # epochs without a new best validation loss before stopping
BN_MOMENTUM = 0.1          # PyTorch batch-norm momentum = MATLAB MeanDecay 0.1

# ---------------------------------------------------------------------------
# Console + log file output
# ---------------------------------------------------------------------------
_LOG_PATH = None


def set_log(path):
    global _LOG_PATH
    _LOG_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def emit(text=""):
    """Print to the console and append the same text to the current log file."""
    print(text)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(text) + "\n")


# ---------------------------------------------------------------------------
# Data loading (participants are stored by ID)
# ---------------------------------------------------------------------------
def load_dat(path):
    """Read one CSI .dat file (a MAT-file) as a float32 array (time, 30)."""
    with open(path, "rb") as f:
        if f.read(6) != b"MATLAB":
            raise ValueError(f"{path} does not look like a MAT-file")
    try:
        mat = loadmat(path)
        arr = mat[[k for k in mat if not k.startswith("__")][0]]
    except NotImplementedError:      # v7.3 files need h5py
        with h5py.File(path, "r") as f:
            arr = np.array(f[list(f.keys())[0]]).T
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return arr.astype(np.float32)


def nearest_resample(x, target=EXPECTED_SAMPLES):
    """Nearest-neighbour interpolation along time to a fixed number of samples (lost packets)."""
    m = x.shape[0]
    if m == target:
        return x
    idx = np.rint(np.linspace(0, m - 1, target)).astype(np.int64)
    return x[idx]


def load_recording(folder):
    """Read the 9 CSI files of one recording, interpolate each to 12,000 samples, join to (12000, 270)."""
    parts, lengths = [], []
    for tx in (1, 2, 3):
        for rx in (1, 2, 3):
            x = load_dat(os.path.join(folder, f"CSI_{tx}{rx}.dat"))
            lengths.append(x.shape[0])
            parts.append(nearest_resample(x))
    return np.concatenate(parts, axis=1), lengths


def load_all_recordings():
    """recs[label][participant_id] = (12000, 270) float32. Participants are keyed by their ID."""
    recs = {label: {} for _, label in ACTIVITIES}
    interpolated, total = 0, 0
    for activity_dir, label in ACTIVITIES:
        base = os.path.join(TOP_DIR, activity_dir)
        for pid in range(1, NO_OF_PARTICIPANTS + 1):
            folder = os.path.join(base, f"R{RECEIVER}", f"P{pid}")
            if not os.path.isdir(folder):
                print(f"Participant = {pid}  ; {activity_dir} CSI data not available. SKIPPING TO NEXT PARTICIPANT")
                continue
            x, lengths = load_recording(folder)
            recs[label][pid] = x
            total += 1
            if any(n != EXPECTED_SAMPLES for n in lengths):
                interpolated += 1
        print(f"Loaded {len(recs[label])} participants for class {label} ({activity_dir})")
    print(f"Recordings that needed nearest-neighbour interpolation (length != {EXPECTED_SAMPLES}): "
          f"{interpolated} of {total}")
    return recs


def common_participants(recs):
    """IDs that have data for every activity (matched by ID, not by position)."""
    sets = [set(recs[label]) for _, label in ACTIVITIES]
    ids = sorted(set.intersection(*sets))
    dropped = sorted(set.union(*sets) - set(ids))
    if dropped:
        print(f"Participants missing in at least one activity (dropped): {dropped}")
    return ids


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def overlap(x, freq=FS, win_len=WIN_LEN, overlap_pct=OVERLAP_PCT):
    """
    Split x (time x 270) into overlapping windows.
    Returns array of shape (frame_count, 270, window_length).
    """
    m = x.shape[0]
    window_length = int(win_len * freq)
    overlap_length = int((overlap_pct / 100) * window_length)

    # -1 because last frame would require zero padding
    frame_count = int(m / overlap_length - 1)

    frames = []
    for i in range(frame_count):
        start = i * overlap_length
        temp = x[start:start + window_length, :]
        frames.append(temp.T)            # (270, window_length)
    return np.stack(frames, axis=0)


FRAMES_PER_RECORDING = int(EXPECTED_SAMPLES / int((OVERLAP_PCT / 100) * WIN_LEN * FS) - 1)   # 119


# ---------------------------------------------------------------------------
# Idle-window dropping (activity-period detection)
# ---------------------------------------------------------------------------
IDLE_KEEP = {}     # (label, participant ID) -> boolean mask over the windows of that recording


def split_method(method):
    """'pca10+idle' -> ('pca10', True); 'pca10' -> ('pca10', False). A trailing '+adabn' is ignored here."""
    method = method[:-len("+adabn")] if method.endswith("+adabn") else method
    return (method[:-len("+idle")], True) if method.endswith("+idle") else (method, False)


def uses_adabn(method):
    """'none+adabn' / 'none+idle+adabn' -> True."""
    return method.endswith("+adabn")


def activity_scores(frames, score=IDLE_SCORE):
    """One number per window: spread of each channel over the window, averaged over the 270 channels."""
    if score == "std":
        per_channel = frames.std(axis=2)
    elif score == "mad":
        med = np.median(frames, axis=2, keepdims=True)
        per_channel = 1.4826 * np.median(np.abs(frames - med), axis=2)
    else:
        raise ValueError(f"IDLE_SCORE must be 'std' or 'mad', got {score!r}")
    return per_channel.mean(axis=1)


def compute_idle_masks(recs):
    """
    For every recording, decide which windows to keep: score >= IDLE_FACTOR x the recording's median score.
    Computed once from the raw recordings, so all methods and seeds drop the same windows.
    """
    if not 0 < IDLE_FACTOR <= 1:
        raise ValueError("IDLE_FACTOR must be in (0, 1]")
    if IDLE_SCOPE not in ("train", "all"):
        raise ValueError("IDLE_SCOPE must be 'train' or 'all'")
    masks, ratios, per_class = {}, [], {}
    for activity_dir, label in ACTIVITIES:
        total = dropped = 0
        for pid, x in recs[label].items():
            frames = overlap(x)
            assert len(frames) == FRAMES_PER_RECORDING, (label, pid, len(frames))
            sc = activity_scores(frames)
            med = float(np.median(sc))
            ratio = sc / med if med > 0 else np.ones_like(sc)
            keep = ratio >= IDLE_FACTOR
            masks[(label, pid)] = keep
            ratios.append(ratio)
            total += keep.size
            dropped += int((~keep).sum())
        per_class[activity_dir] = (dropped, total)
    all_total = sum(t for _, t in per_class.values())
    all_dropped = sum(d for d, _ in per_class.values())
    print(f"\nIdle windows (score < {IDLE_FACTOR:g} x recording median, {IDLE_SCORE}, scope {IDLE_SCOPE}): "
          f"{all_dropped} of {all_total} windows ({100 * all_dropped / max(all_total, 1):.1f}%)")
    for activity_dir, (d, t) in per_class.items():
        print(f"    {activity_dir:28s} {d:5d} of {t:5d} ({100 * d / max(t, 1):.1f}%)")
    q = np.percentile(np.concatenate(ratios), [5, 25, 50, 75, 95])
    print("    activity / recording median, percentiles 5/25/50/75/95: " + "  ".join(f"{v:.2f}" for v in q))
    if all_dropped == 0:
        print("    Nothing is dropped; raise IDLE_FACTOR (up to 1.0).")
    return masks


# ---------------------------------------------------------------------------
# Preprocessing
#   stage 1  : per recording, no fitting (filters)
#   fit      : statistics learned from the TRAINING participants of the fold only
#   stream   : applied to a recording before windowing, using the fitted state
#   window   : applied to the windows of a recording
# ---------------------------------------------------------------------------
def butterworth_lowpass(x):
    """Zero-phase Butterworth low-pass along time."""
    b, a = butter(BUTTER_ORDER, BUTTER_CUTOFF_HZ / (0.5 * FS), btype="low")
    return filtfilt(b, a, x, axis=0)


def dwt_denoise(x):
    """Wavelet soft-threshold denoising, per channel (needs: pip install PyWavelets)."""
    import pywt
    out = np.empty_like(x)
    for col in range(x.shape[1]):
        coeffs = pywt.wavedec(x[:, col], DWT_WAVELET, level=DWT_LEVEL)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeffs[1:] = [pywt.threshold(d, value=thresh, mode="soft") for d in coeffs[1:]]
        out[:, col] = pywt.waverec(coeffs, DWT_WAVELET)[:len(x)]
    return out


def normalize_recording(x):
    """Per recording and channel: subtract the recording's own mean, divide by its own std."""
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + ZSCORE_EPS)


STAGE1 = {"butter": butterworth_lowpass, "dwt": dwt_denoise, "norm_recording": normalize_recording}
FIT_METHODS = {"pca10", "zscore_channel", "pca_drop"}
VALID_METHODS = {"none", "butter", "dwt", "pca10", "zscore_window", "zscore_channel", "norm_recording", "pca_drop"}


def apply_stage1(method, recs):
    """Run the per-recording filter of a method once over all recordings."""
    if method not in STAGE1:
        return recs
    fn = STAGE1[method]
    return {label: {pid: fn(x).astype(np.float32) for pid, x in per_pid.items()}
            for label, per_pid in recs.items()}


def compute_stats(recs):
    """Per recording: sample count, channel sums and cross-products (used to fit PCA / z-score)."""
    stats = {}
    for label, per_pid in recs.items():
        for pid, x in per_pid.items():
            x64 = x.astype(np.float64)
            stats[(label, pid)] = (x64.shape[0], x64.sum(axis=0), x64.T @ x64)
    return stats


def fit_state(method, stats, train_ids):
    """Fit anything learned from data on the training participants only."""
    if method not in FIT_METHODS:
        return None
    n_total = 0
    s_total = np.zeros(NUM_CHANNELS)
    c_total = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
    for _, label in ACTIVITIES:
        for pid in train_ids:
            n, s, c = stats[(label, pid)]
            n_total += n
            s_total += s
            c_total += c
    mean = s_total / n_total
    cov = c_total / n_total - np.outer(mean, mean)
    if method == "pca10":
        _, evecs = np.linalg.eigh(cov)                              # ascending eigenvalues
        comps = evecs[:, ::-1][:, :PCA_COMPONENTS].T                # (components, 270)
        return {"mean": mean.astype(np.float32), "comps": comps.astype(np.float32)}
    if method == "pca_drop":
        _, evecs = np.linalg.eigh(cov)
        comps = evecs[:, ::-1][:, :PCA_DROP_COMPONENTS].T           # the top-variance directions to remove
        return {"mean": mean.astype(np.float32), "comps": comps.astype(np.float32)}
    std = np.sqrt(np.maximum(np.diag(cov), 0.0)) + ZSCORE_EPS       # zscore_channel
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_stream(method, x, state):
    """Apply the fitted transform to one recording (time, 270) before windowing."""
    if method == "pca10":
        return ((x - state["mean"]) @ state["comps"].T) @ state["comps"] + state["mean"]
    if method == "pca_drop":
        return x - ((x - state["mean"]) @ state["comps"].T) @ state["comps"]   # remove the top components
    if method == "zscore_channel":
        return (x - state["mean"]) / state["std"]
    return x


def apply_window(method, frames):
    """Apply a window-level transform to (frames, 270, 200)."""
    if method == "zscore_window":
        mean = frames.mean(axis=(1, 2), keepdims=True)
        std = frames.std(axis=(1, 2), keepdims=True) + ZSCORE_EPS
        return ((frames - mean) / std).astype(np.float32)
    return frames


def build_split(recs, ids, method, state, drop_idle=False):
    """
    All 7 activities of the given participant IDs, shuffled, as (N, 1, 270, 200) tensors.
    With drop_idle=True the idle windows (IDLE_KEEP masks) of every recording are left out.
    """
    def keep_mask(label, pid):
        return IDLE_KEEP[(label, pid)] if drop_idle else np.ones(FRAMES_PER_RECORDING, dtype=bool)

    n_total = sum(int(keep_mask(label, pid).sum()) for _, label in ACTIVITIES for pid in ids)
    X = np.empty((n_total, NUM_CHANNELS, int(WIN_LEN * FS)), dtype=np.float32)
    y = np.empty(n_total, dtype=np.int64)
    perm = np.random.permutation(n_total)            # random position of every frame
    k = 0
    for _, label in ACTIVITIES:
        for pid in ids:
            x = apply_stream(method, recs[label][pid], state)
            frames = apply_window(method, overlap(x))
            assert len(frames) == FRAMES_PER_RECORDING, (label, pid, len(frames))
            frames = frames[keep_mask(label, pid)]
            pos = perm[k:k + len(frames)]
            X[pos] = frames
            y[pos] = label
            k += len(frames)
    assert k == n_total
    return torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(net, loader, criterion, device):
    """Return (loss, accuracy, predictions, labels) with the net in eval mode."""
    net.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = net(xb)
            total_loss += criterion(out, yb).item() * xb.size(0)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            n += xb.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    return (total_loss / n, correct / n,
            torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy())


def print_header():
    emit("|======================================================================================================================|")
    emit("|  Epoch  |  Iteration  |  Time Elapsed  |   Training   |  Validation  |   Training   |  Validation  |  Base Learning  |")
    emit("|         |             |   (hh:mm:ss)   |   Accuracy   |   Accuracy   |     Loss     |     Loss     |      Rate       |")
    emit("|======================================================================================================================|")


def log_row(epoch, it, t0, tr_acc, tr_loss, val_acc=None, val_loss=None, note=""):
    """One table row. Training accuracy / loss are running averages over the epoch so far; the validation
    columns are filled only on the last row of an epoch (validation runs once per epoch)."""
    h, rem = divmod(int(time.time() - t0), 3600)
    m, s = divmod(rem, 60)
    time_str = f"{h:02d}:{m:02d}:{s:02d}"
    va = f"{val_acc*100:11.2f}%" if val_acc is not None else f"{'-':>12}"
    vl = f"{val_loss:12.4f}" if val_loss is not None else f"{'-':>12}"
    emit(f"| {epoch:7d} | {it:11d} | {time_str:>14} | {tr_acc*100:11.2f}% | {va} "
         f"| {tr_loss:12.4f} | {vl} | {INITIAL_LR:15.4f} |{note}")


def format_matrix(cm):
    w = len(str(int(cm.max()))) + 3
    return "\n".join("".join(f"{int(v):>{w}d}" for v in row) for row in cm)


def run_fold(method, seed, recs, stats, train_ids, val_ids, test_pid, device, onnx_model, criterion):
    """Train on train_ids, validate on val_ids, test on test_pid. Returns a result dict."""
    fold_start = time.time()
    emit(f"\n[{method} | seed {seed}] test participant P{test_pid}, validation "
         + ", ".join(f"P{p}" for p in val_ids) + f", training on {len(train_ids)} participants")
    full_name = method                               # name stored in the results ("pca10+idle")
    method, idle = split_method(method)              # method used for preprocessing ("pca10")
    set_seed(seed * 1000 + test_pid)                 # same shuffles for every method (fair comparison)

    state = fit_state(method, stats, train_ids)      # fitted on the training participants only
    Xtr, ytr = build_split(recs, train_ids, method, state, drop_idle=idle)
    Xva, yva = build_split(recs, val_ids, method, state, drop_idle=idle and IDLE_SCOPE == "all")
    Xte, yte = build_split(recs, [test_pid], method, state, drop_idle=idle and IDLE_SCOPE == "all")
    if idle:
        emit(f"Windows: train {len(ytr)}, validation {len(yva)}, test {len(yte)}")

    # MATLAB 'zerocenter' (channel-wise, one scalar): mean of the TRAINING data only
    mean = Xtr.mean()
    Xtr.sub_(mean)
    Xva.sub_(mean)
    Xte.sub_(mean)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=MINI_BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=128)

    net = convert(onnx_model).to(device)             # fresh network each fold

    # MATLAB MeanDecay/VarianceDecay 0.1 = PyTorch momentum 0.1 (epsilon 0.001 kept)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = BN_MOMENTUM
    optimizer = torch.optim.SGD(net.parameters(), lr=INITIAL_LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)

    emit("Training on single CPU." if device.type == "cpu" else "Training on GPU.")
    print_header()

    t0 = time.time()
    best_val_loss, best_val_acc, best_epoch, best_state = float("inf"), 0.0, 0, None
    val_lag, it, epochs_run = 0, 0, 0
    stopped_early = False
    train_acc = 0.0
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        epochs_run = epoch
        run_loss, run_correct, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            it += 1
            run_loss += loss.item() * xb.size(0)
            run_correct += (out.argmax(1) == yb).sum().item()
            seen += xb.size(0)
            if it % 250 == 0:
                log_row(epoch, it, t0, run_correct / seen, run_loss / seen)

        # validate once per epoch; the row also shows the epoch's training average
        train_acc, train_loss = run_correct / seen, run_loss / seen
        val_loss, val_acc, _, _ = evaluate(net, val_loader, criterion, device)

        # Early stopping: PATIENCE epochs without a new best validation loss, but not before MIN_EPOCHS
        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch, val_lag = val_loss, val_acc, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
            note = "  <- best"
        else:
            val_lag += 1
            note = ""
        log_row(epoch, it, t0, train_acc, train_loss, val_acc, val_loss, note)
        if epoch >= MIN_EPOCHS and val_lag >= PATIENCE:
            emit(f"\nEarly stopping: validation loss has not improved for {PATIENCE} epochs "
                 f"(minimum {MIN_EPOCHS} epochs trained).")
            stopped_early = True
            break
    emit("|======================================================================================================================|")

    # test the checkpoint with the lowest validation loss, not the last network
    net.load_state_dict(best_state)
    emit(f"Restoring the best checkpoint: epoch {best_epoch} (validation loss {best_val_loss:.4f}, "
         f"accuracy {best_val_acc:.4f}).")

    # MATLAB BatchNormalizationStatistics = 'population': after training, pass over the whole
    # training set once and use that mean/variance for prediction
    # with "+adabn" the statistics come from the test participant's own windows (no labels used)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    stats_set = (Xte, yte) if uses_adabn(full_name) else (Xtr, ytr)
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(*stats_set), batch_size=128):
            net(xb.to(device))

    _, accuracy, y_pred, y_true = evaluate(net, test_loader, criterion, device)
    labels = np.arange(NUM_CLASSES)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    emit(f"Test participant P{test_pid}: macro-F1 {macro_f1:.4f} | accuracy {accuracy:.4f}")
    emit(format_matrix(cm))

    del Xtr, ytr, Xva, yva, Xte, yte, train_loader, val_loader, test_loader, net, optimizer
    gc.collect()
    return {
        "method": full_name, "seed": seed, "test_pid": test_pid, "val_pids": list(val_ids),
        "n_train_participants": len(train_ids),
        "macro_f1": float(macro_f1), "accuracy": float(accuracy),
        "per_class_f1": [float(v) for v in per_class],
        "iterations": it, "epochs": epochs_run, "best_epoch": best_epoch, "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc), "train_acc_last": float(train_acc), "stopped_early": stopped_early,
        "minutes": (time.time() - fold_start) / 60, "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Stored results
# ---------------------------------------------------------------------------
def load_rows():
    rows = []
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_row(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Report: summary table, per-fold macro-F1 and parameter settings
# ---------------------------------------------------------------------------
METHOD_INFO = {
    "none": ("Baseline (no preprocessing)", "-", "-", "-"),
    "butter": ("Butterworth low-pass", "each recording, along time, before windowing (zero-phase filtfilt)",
               "no", f"order {BUTTER_ORDER}, cutoff {BUTTER_CUTOFF_HZ} Hz, sampling rate {FS} Hz"),
    "dwt": ("DWT denoising", "each recording, per channel, before windowing", "no",
            f"wavelet {DWT_WAVELET}, level {DWT_LEVEL}, soft threshold sigma*sqrt(2 ln n), "
            f"sigma = median(|finest detail|)/0.6745"),
    "pca10": ("PCA projection (10 components)", "each recording, across the 270 channels, before windowing",
              "training participants only", f"{PCA_COMPONENTS} components, reconstruction kept in 270 channels"),
    "zscore_window": ("Z-score per window", "each 270 x 200 window, after windowing", "no",
                      f"mean/std of the window itself, eps {ZSCORE_EPS}"),
    "norm_recording": ("Per-recording normalization", "each recording, before windowing", "no",
                       f"per-channel mean subtracted and divided by the recording's own std, eps {ZSCORE_EPS}"),
    "pca_drop": ("PCA, top components removed", "each recording, across the 270 channels, before windowing",
                 "training participants only", f"top {PCA_DROP_COMPONENTS} components removed, 270 channels kept"),
    "zscore_channel": ("Z-score per channel (train statistics)", "each recording, before windowing",
                       "training participants only", f"per-channel mean/std of 270 channels, eps {ZSCORE_EPS}"),
}


def method_info(name):
    """(description, applied to, fitted on, parameters) of a method, including the '+idle' variants."""
    base, idle = split_method(name)
    desc, applied, fitted, params = METHOD_INFO[base]
    if uses_adabn(name):
        desc += " + test-participant batch-norm statistics"
        params = (params + "; " if params != "-" else "") + "batch-norm statistics recomputed on the test participant's unlabeled windows"
    if not idle:
        return desc, applied, fitted, params
    rule = (f"drop windows with {IDLE_SCORE} activity < {IDLE_FACTOR:g} x the median of the same recording "
            f"(scope: {'training windows only' if IDLE_SCOPE == 'train' else 'training, validation and test windows'})")
    return (desc + " + idle-window dropping",
            ("idle windows removed after windowing" if applied == "-" else applied + "; idle windows removed after windowing"),
            fitted,
            (params + "; " if params != "-" else "") + rule)


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def _std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def _fmt(v, digits=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def summarize(by_method):
    """by_method[seed][test_pid] = row  ->  dict of statistics."""
    seeds = sorted(by_method)
    pids = sorted({p for s in seeds for p in by_method[s]})
    fold_f1 = {p: _mean([by_method[s][p]["macro_f1"] for s in seeds if p in by_method[s]]) for p in pids}
    fold_acc = {p: _mean([by_method[s][p]["accuracy"] for s in seeds if p in by_method[s]]) for p in pids}
    seed_f1 = {s: _mean([r["macro_f1"] for r in by_method[s].values()]) for s in seeds}
    pooled = [r["macro_f1"] for s in seeds for r in by_method[s].values()]
    return {"seeds": seeds, "pids": pids, "fold_f1": fold_f1, "fold_acc": fold_acc, "seed_f1": seed_f1,
            "mean": _mean(list(fold_f1.values())), "std_folds": _std(list(fold_f1.values())),
            "std_seeds": _std(list(seed_f1.values())), "std_pooled": _std(pooled),
            "acc": _mean(list(fold_acc.values())), "n_runs": len(pooled)}


def make_report():
    rows = load_rows()
    if not rows:
        print("No results yet.")
        return
    by = {}
    for r in rows:
        by.setdefault(r["method"], {}).setdefault(r["seed"], {})[r["test_pid"]] = r
    methods = [m for m in METHODS if m in by] + [m for m in by if m not in METHODS]
    summ = {m: summarize(by[m]) for m in methods}
    base = summ.get(BASELINE)
    n_planned_folds = max(len(s["pids"]) for s in summ.values())

    out = ["# Leave-one-out results, paper protocol",
           f"Location {LOCATION}, receiver {RECEIVER}. Generated {datetime.now():%Y-%m-%d %H:%M}.", "",
           "Macro-F1 over the 7 classes on the held-out participant. 'Std (folds)' is the standard deviation "
           "across held-out participants of the seed-averaged macro-F1; 'Std (seeds)' is the standard deviation "
           "across seeds of the per-seed mean macro-F1. Both use n-1.", "",
           "## Summary", ""]
    header = ["Method", "Macro-F1 mean", "Std (folds)", "Std (seeds)", "Accuracy mean",
              "Delta vs baseline", "Folds improved", "Wilcoxon p", "Folds x seeds"]
    table_rows = []
    for m in methods:
        s = summ[m]
        delta = improved = p_value = "-"
        if base is not None and m != BASELINE:
            common = [p for p in s["pids"] if p in base["fold_f1"]]
            diffs = np.array([s["fold_f1"][p] - base["fold_f1"][p] for p in common])
            if len(diffs):
                delta = f"{diffs.mean():+.4f}"
                improved = f"{int((diffs > 0).sum())}/{len(diffs)}"
                try:
                    p_value = f"{wilcoxon(diffs).pvalue:.3f}"
                except ValueError:
                    p_value = "n/a"
        table_rows.append([m, _fmt(s["mean"]), _fmt(s["std_folds"]), _fmt(s["std_seeds"]), _fmt(s["acc"]),
                           delta, improved, p_value, f"{len(s['pids'])} x {len(s['seeds'])} = {s['n_runs']}"])
    out += [_table(header, table_rows), "",
            f"Complete protocol: {n_planned_folds} folds per method. Rows with fewer runs are still in progress.", "",
            "## Macro-F1 per fold (mean over seeds)", ""]

    all_pids = sorted({p for s in summ.values() for p in s["pids"]})
    fold_rows = [[f"P{p}"] + [_fmt(summ[m]["fold_f1"].get(p, float("nan"))) for m in methods] for p in all_pids]
    fold_rows.append(["**Mean**"] + [_fmt(summ[m]["mean"]) for m in methods])
    fold_rows.append(["**Std (folds)**"] + [_fmt(summ[m]["std_folds"]) for m in methods])
    out += [_table(["Test participant"] + methods, fold_rows), "", "## Macro-F1 per seed (mean over folds)", ""]
    all_seeds = sorted({s for m in methods for s in summ[m]["seeds"]})
    seed_rows = [[f"seed {sd}"] + [_fmt(summ[m]["seed_f1"].get(sd, float("nan"))) for m in methods]
                 for sd in all_seeds]
    out += [_table(["Seed"] + methods, seed_rows), "", "## Parameter settings", "", "### Preprocessing methods", ""]
    out += [_table(["Method", "Description", "Applied to", "Fitted on", "Parameters"],
                   [[m, *method_info(m)] for m in methods if split_method(m)[0] in METHOD_INFO]), "",
            "### Protocol and training (identical for every method)", ""]
    common_rows = [
        ["Data", f"Wi-Gitation, location {LOCATION}, receiver R{RECEIVER}, 7 classes, 270 channels (3x3x30)"],
        ["Lost packets", f"nearest-neighbour interpolation of every recording to {EXPECTED_SAMPLES} samples "
                         f"({RECORDING_SECONDS} s x {FS} Hz)"],
        ["Windows", f"{WIN_LEN} s = {int(WIN_LEN * FS)} samples, {OVERLAP_PCT}% overlap, "
                    f"{FRAMES_PER_RECORDING} windows per recording, input 270 x 200 x 1"],
        ["Split", f"train on all participants except {VALIDATION_COUNT} validation participants (the highest IDs "
                  f"other than the test participant) and the test participant; every participant except the last ID "
                  f"is the test participant once"],
        ["Participant matching", "by participant ID; only IDs present in all 7 activities are used"],
        ["Network", "ResNet50 (from the MATLAB layer graph via ONNX), 7 outputs"],
        ["Normalization", "scalar mean of the training windows subtracted (MATLAB zerocenter, channel-wise)"],
        ["Optimizer", f"SGD, momentum {MOMENTUM}, learning rate {INITIAL_LR}, weight decay {WEIGHT_DECAY}, "
                      f"mini-batch {MINI_BATCH_SIZE}, max {MAX_EPOCHS} epochs, shuffled every epoch"],
        ["Early stopping", f"validate once per epoch; not before epoch {MIN_EPOCHS}; stop after {PATIENCE} epochs "
                           f"without a new best validation loss; the best-validation-loss checkpoint is tested"],
        ["Batch norm", f"momentum {BN_MOMENTUM} during training (MATLAB decay 0.1), epsilon 0.001; population "
                       f"statistics recomputed over the training set before testing"],
        ["Seeds", ", ".join(str(s) for s in SEEDS)],
        ["Metric", "macro-F1 (unweighted mean of the 7 per-class F1 scores), accuracy also stored"],
    ]
    if any(split_method(m)[1] for m in methods):
        common_rows.insert(-1, ["Idle windows (methods ending +idle)",
                                f"score = {IDLE_SCORE} of each channel over the window, averaged over 270 channels, computed on the raw "
                                f"recording; a window is dropped when its score is below {IDLE_FACTOR:g} x the median score of the same "
                                f"recording; scope: {'training windows only' if IDLE_SCOPE == 'train' else 'training, validation and test windows'}; "
                                f"PCA / z-score statistics are still fitted on the full training recordings"])
    out += [_table(["Item", "Setting"], common_rows), ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")

    csv_lines = ["method,seed,test_participant,macro_f1,accuracy,iterations,epochs,best_epoch,best_val_acc,minutes"]
    for r in sorted(rows, key=lambda r: (r["method"], r["seed"], r["test_pid"])):
        csv_lines.append(f"{r['method']},{r['seed']},{r['test_pid']},{r['macro_f1']:.6f},{r['accuracy']:.6f},"
                         f"{r['iterations']},{r.get('epochs', '')},{r.get('best_epoch', '')},{r['best_val_acc']:.6f},{r['minutes']:.2f}")
    (OUT_DIR / "all_fold_results.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="only rebuild the report from stored results")
    args = parser.parse_args()
    if args.report:
        make_report()
        return

    for method in METHODS:
        assert split_method(method)[0] in VALID_METHODS, f"Unknown method: {method}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("This is for leave one out data analysis for Wi-Gitation dataset by using "
          "ResNet50 for location 1 Receiver 0 analysis (improved protocol)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    recs = load_all_recordings()
    ids = common_participants(recs)
    last_id = max(ids)
    test_ids = [p for p in ids if p != last_id] if TEST_IDS is None else list(TEST_IDS)
    print(f"Participants used ({len(ids)}): {ids}")
    print(f"Validation participants per fold: the {VALIDATION_COUNT} highest IDs other than the test participant; "
          f"test participants ({len(test_ids)}): {test_ids}")

    done = {(r["method"], r["seed"], r["test_pid"]) for r in load_rows()}
    pending = sum((m, s, t) not in done for m in METHODS for s in SEEDS for t in test_ids)
    print(f"Fold runs still to do: {pending} of {len(METHODS) * len(SEEDS) * len(test_ids)}")

    if any(split_method(m)[1] for m in METHODS):
        IDLE_KEEP.update(compute_idle_masks(recs))           # once, from the raw recordings

    onnx_model = onnx.load(ONNX_PATH)
    criterion = nn.CrossEntropyLoss()

    for seed in SEEDS:                                       # seed outermost: a full single-seed table comes first
        for method in METHODS:
            todo = [t for t in test_ids if (method, seed, t) not in done]
            if not todo:
                continue
            set_log(LOG_DIR / f"{method}_seed{seed}.txt")
            emit("\n" + "=" * 118)
            emit(f"Method: {method} | seed {seed} | {len(todo)} folds to run")
            emit("=" * 118)

            base = split_method(method)[0]                   # "pca10+idle" -> "pca10"
            recs_m = apply_stage1(base, recs)                # per-recording filters, once per method and seed
            stats = compute_stats(recs_m) if base in FIT_METHODS else None
            for test_pid in todo:
                val_ids = [p for p in sorted(ids, reverse=True) if p != test_pid][:VALIDATION_COUNT]
                train_ids = [p for p in ids if p != test_pid and p not in val_ids]
                row = run_fold(method, seed, recs_m, stats, train_ids, val_ids, test_pid,
                               device, onnx_model, criterion)
                append_row(row)
                done.add((method, seed, test_pid))
            if recs_m is not recs:
                del recs_m
            del stats
            gc.collect()
            set_log(None)
            make_report()

    make_report()
    print("\nAll runs finished.")


if __name__ == "__main__":
    main()
