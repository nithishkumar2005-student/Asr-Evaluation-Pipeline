#!/usr/bin/env python3
# =============================================================================
# Fast Malayalam wav2vec2 batch transcription  —  DGX Spark GB10
# =============================================================================
# Rewrite of `torch.Compile 16optimised_malayalam_wav2vec2_v2.ipynb`, tuned to
# transcribe ~300k MP3 files as fast as possible on a single GB10 GPU.
#
# Same model, same VAD chunking, same WER/CER metrics, same results.csv columns
# as the original notebook.  What changed is the *pipeline*, not the math:
#
#   original notebook            this script
#   -----------------            -----------
#   1 file at a time             CPU worker pool decodes/resamples/VADs N files
#                                ahead while the GPU is busy (no idle gaps)
#   resample = soxr_hq (0.46s)   same quality, but parallel across 16 cores
#   per-file GPU batch (bs 1-3)  cross-file length-bucketed batches (fixed shape)
#   eager fp16 (320 s-audio/s)   torch.compile fp16 (757 s-audio/s, 2.4x, exact)
#   empty_cache() every file     never (it forces a slow GPU sync)
#   WER on main thread           offloaded to a metric process pool
#   one big results.csv at end   incremental sharded CSV + crash-resume
#
# Measured GPU floor: 320 s-audio/s eager, 757 s-audio/s compiled.
# 300k files x ~50s ≈ 15M s of audio  ->  ~5-6 h compiled vs ~60 h original.
#
# Usage:
#   python fast_malayalam_asr.py \
#       --audio-folder /root/data/Audio/Sample200 \
#       --csv /root/svarupa_summaries_translations_Malayalam.csv \
#       --output-dir ./asr_out
#
#   # pure throughput, no metrics:
#   python fast_malayalam_asr.py --audio-folder ... --no-metrics
#
#   # resume is automatic: re-run the same command, finished files are skipped.
# =============================================================================

import argparse
import csv
import datetime
import importlib.machinery
import os
import re
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np

# -----------------------------------------------------------------------------
# Constants — copied verbatim from the notebook so transcription/metrics match.
# -----------------------------------------------------------------------------
SAMPLING_RATE      = 16000
MIN_CHUNK_SEC      = 5
TARGET_CHUNK_SEC   = 25
MAX_CHUNK_SEC      = 30
VAD_THRESHOLD      = 0.5
SILENCE_MIN_SEC    = 0.30
SILENCE_PREFER_SEC = 0.50
SILENCE_PAD_S      = 3
SILENCE_SAMPLES    = int(SILENCE_PAD_S * SAMPLING_RATE)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}

MODEL_ID = "gvs/wav2vec2-large-xlsr-malayalam"

# =============================================================================
#  >>> EDIT THESE PATHS <<<   (used when you run the script with no flags)
#  Command-line flags --audio-folder / --csv / --output-dir still override them.
# =============================================================================
AUDIO_FOLDER = "FastSpeech2_HS_latest_models/Rigveda_complete_malayalam_male/final_audio"                       # folder of audio files (point this at your 300k folder)
CSV_PATH     = "FastSpeech2_HS_latest_models/svarupa_summaries_translations_Malayalam.csv"  # ground-truth CSV; set to None for transcription-only
OUTPUT_DIR   = "Malayalam_asr_out"                            # where results_part_*.csv are written

# GPU batch shape.  torch.compile specialises on (batch, length); we keep batch
# fixed at GPU_BATCH and snap every chunk's length up to one of these buckets so
# only a handful of shapes are ever compiled (each compile is a one-time ~20s).
GPU_BATCH      = 8
# Fine length grid (seconds).  Chunks are length-sorted, grouped into full
# batches, then each batch is padded up to the nearest grid value — so only
# these few fixed shapes are ever compiled, and padding waste stays small.
BUCKETS_S      = [8, 12, 16, 20, 24, 28, 32, 36, 42]
BUCKET_SAMPLES = sorted(int(s * SAMPLING_RATE) for s in BUCKETS_S)

# results.csv column order (identical to the notebook).
RESULT_COLS = [
    "filename", "ground_truth", "transcription", "gt_cleaned", "pred_cleaned",
    "wer", "cer", "ser", "accuracy_pct", "substitutions", "deletions",
    "insertions", "gt_word_count", "pred_word_count",
]


# =============================================================================
# torchaudio stub
# -----------------------------------------------------------------------------
# Silero-VAD's utils_vad.py does `import torchaudio` at module load, but the only
# function we use (get_speech_timestamps) operates on a plain tensor.  torchaudio
# isn't installed in this env, so we register a stub module to satisfy the import.
# We decode audio ourselves with soundfile, so the stubbed read/save helpers are
# never called.
# =============================================================================
def _install_torchaudio_stub():
    if "torchaudio" in sys.modules:
        return
    def _mod(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, None)
        return m
    ta = _mod("torchaudio")
    ta.functional = _mod("torchaudio.functional")
    ta.set_audio_backend = lambda *a, **k: None
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = ta.functional


# =============================================================================
# Text + metric helpers (verbatim from the notebook — keeps numbers comparable).
# Defined at module top level so the metric ProcessPool can pickle them.
# =============================================================================
_MAL_RE   = re.compile(r"[^ഀ-ൿ\s]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text):
    text = str(text).lower()
    text = _MAL_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def compute_wer(gt_words, pred_words, sim_threshold=0.80):
    from difflib import SequenceMatcher
    m, n = len(gt_words), len(pred_words)
    if m == 0:
        return 0.0, 0, 0, 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = SequenceMatcher(None, gt_words[i - 1], pred_words[j - 1]).ratio()
            dp[i][j] = dp[i - 1][j - 1] if sim >= sim_threshold else 1 + min(
                dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    S = D = I = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim = SequenceMatcher(None, gt_words[i - 1], pred_words[j - 1]).ratio()
            if sim >= sim_threshold and dp[i][j] == dp[i - 1][j - 1]:
                i -= 1; j -= 1
            elif dp[i][j] == 1 + dp[i - 1][j - 1]:
                S += 1; i -= 1; j -= 1
            elif dp[i][j] == 1 + dp[i - 1][j]:
                D += 1; i -= 1
            else:
                I += 1; j -= 1
        elif i > 0:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return (S + D + I) / m, S, D, I


def compute_cer(gt, pred):
    m, n = len(gt), len(pred)
    if m == 0:
        return 0.0
    dp = np.arange(n + 1, dtype=np.int32)
    for i in range(1, m + 1):
        prev = dp.copy(); dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1,
                        prev[j - 1] + (0 if gt[i - 1] == pred[j - 1] else 1))
    return int(dp[n]) / m


def compute_metrics(ground_truth, transcription):
    """Runs in the metric ProcessPool. Returns the metric portion of a row."""
    gt_clean   = clean_text(ground_truth)
    pred_clean = clean_text(transcription)
    gt_words   = gt_clean.split()
    pred_words = pred_clean.split()
    wer_score, S, D, I = compute_wer(gt_words, pred_words)
    cer_score = compute_cer(gt_clean, pred_clean)
    return {
        "gt_cleaned":      gt_clean,
        "pred_cleaned":    pred_clean,
        "wer":             round(wer_score, 4),
        "cer":             round(cer_score, 4),
        "ser":             int(gt_clean != pred_clean),
        "accuracy_pct":    round(max(0.0, (1 - wer_score) * 100), 2),
        "substitutions":   S,
        "deletions":       D,
        "insertions":      I,
        "gt_word_count":   len(gt_words),
        "pred_word_count": len(pred_words),
    }


# =============================================================================
# VAD chunking (verbatim logic from the notebook; operates on a CPU tensor).
# =============================================================================
def build_vad_chunks(wav, vad_model, get_speech_timestamps):
    import torch
    speech_ts = get_speech_timestamps(
        wav, vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLING_RATE,
        min_silence_duration_ms=int(SILENCE_MIN_SEC * 1000),
        min_speech_duration_ms=100,
        return_seconds=False,
    )

    timeline = []
    audio_end_s = len(wav) / SAMPLING_RATE
    if speech_ts:
        if speech_ts[0]["start"] > 0:
            timeline.append((0.0, speech_ts[0]["start"] / SAMPLING_RATE, "silence"))
        for idx, seg in enumerate(speech_ts):
            timeline.append((seg["start"] / SAMPLING_RATE, seg["end"] / SAMPLING_RATE, "speech"))
            if idx < len(speech_ts) - 1:
                gap_start = seg["end"] / SAMPLING_RATE
                gap_end   = speech_ts[idx + 1]["start"] / SAMPLING_RATE
                if gap_end - gap_start >= SILENCE_MIN_SEC:
                    timeline.append((gap_start, gap_end, "silence"))
        last_end_s = speech_ts[-1]["end"] / SAMPLING_RATE
        if last_end_s < audio_end_s:
            timeline.append((last_end_s, audio_end_s, "silence"))
    else:
        timeline.append((0.0, audio_end_s, "silence"))

    silence_segments = [(s, e) for s, e, t in timeline if t == "silence"]

    def best_cut(search_start, search_end):
        preferred = fallback = None
        for (ss, se) in silence_segments:
            mid = (ss + se) / 2.0
            if search_start < mid <= search_end:
                if (se - ss) >= SILENCE_PREFER_SEC:
                    if preferred is None:
                        preferred = mid
                else:
                    if fallback is None:
                        fallback = mid
        return preferred if preferred is not None else fallback

    chunks = []
    pos = 0.0
    audio_end = timeline[-1][1]
    while pos < audio_end:
        if audio_end - pos <= TARGET_CHUNK_SEC:
            chunks.append((pos, audio_end)); break
        window_end = min(pos + TARGET_CHUNK_SEC, audio_end)
        hard_cut   = min(pos + MAX_CHUNK_SEC, audio_end)
        cut = best_cut(window_end, hard_cut) or best_cut(pos, hard_cut) or hard_cut
        if cut > pos:
            chunks.append((pos, cut))
        pos = cut

    merged = []
    for (cs, ce) in chunks:
        if (ce - cs) < MIN_CHUNK_SEC and merged:
            prev_s, prev_e = merged.pop()
            merged.append((prev_s, ce))
        else:
            merged.append((cs, ce))

    return [(int(s * SAMPLING_RATE), int(e * SAMPLING_RATE)) for (s, e) in merged]


# =============================================================================
# CPU worker — decode + resample + VAD.  One ProcessPool worker per core.
# =============================================================================
_W = {}   # per-worker globals (model loaded once per process)


def _worker_init():
    """Load Silero-VAD once per worker process (CPU, tiny)."""
    # Must be set BEFORE torch is imported: otherwise OpenMP spawns spinning
    # threads per worker and 11 workers thrash each other (~10x slower VAD).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    _install_torchaudio_stub()
    import torch
    torch.set_num_threads(1)          # each worker is one core; avoid oversubscription
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                          # interop pool already started; env vars cover it
    _SILERO_CACHE = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")
    vad_model, utils = torch.hub.load(
        repo_or_dir=_SILERO_CACHE, model="silero_vad",
        source="local", trust_repo=True, verbose=False)
    vad_model.eval()
    _W["torch"] = torch
    _W["vad"]   = vad_model
    _W["get_speech_timestamps"] = utils[0]


def process_file(args):
    """Decode one audio file, run VAD, return its chunk waveforms.

    Returns: (filename, chunks, duration_s, cpu_time_s, error)
      chunks = list of np.float32 1-D arrays (16 kHz, no padding yet)
    """
    filename, folder = args
    t0 = time.time()
    try:
        import soundfile as sf
        import librosa
        path = os.path.join(folder, filename)

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE,
                                     res_type="soxr_hq")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("empty / corrupt audio")

        duration_s = len(audio) / SAMPLING_RATE
        torch = _W["torch"]
        wav_t = torch.from_numpy(audio)
        with torch.no_grad():
            bounds = build_vad_chunks(wav_t, _W["vad"], _W["get_speech_timestamps"])

        chunks = [audio[s:e].copy() for (s, e) in bounds if e > s]
        if not chunks:                       # safety: never emit zero chunks
            chunks = [audio]
        return filename, chunks, duration_s, time.time() - t0, None
    except Exception as e:
        import traceback
        return filename, None, 0.0, time.time() - t0, traceback.format_exc()


# =============================================================================
# Sharded, resumable result writer.
# =============================================================================
class ShardWriter:
    def __init__(self, out_dir, shard_size, columns, prefix="results_part"):
        self.dir = out_dir
        self.shard_size = shard_size
        self.columns = columns
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)
        self._fh = None
        self._w = None
        self._n_in_shard = 0
        self._shard_idx = self._next_shard_index()
        self.total_written = 0

    def _next_shard_index(self):
        existing = [f for f in os.listdir(self.dir)
                    if f.startswith(self.prefix) and f.endswith(".csv")]
        return len(existing)

    def _open_new_shard(self):
        if self._fh:
            self._fh.close()
        name = f"{self.prefix}_{self._shard_idx:05d}.csv"
        self._fh = open(os.path.join(self.dir, name), "w",
                        newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(self._fh, fieldnames=self.columns,
                                 extrasaction="ignore")
        self._w.writeheader()
        self._n_in_shard = 0
        self._shard_idx += 1

    def write(self, row):
        if self._fh is None or self._n_in_shard >= self.shard_size:
            self._open_new_shard()
        self._w.writerow(row)
        self._n_in_shard += 1
        self.total_written += 1
        if self._n_in_shard % 200 == 0:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def load_done_set(out_dir, prefix="results_part"):
    """Filenames already present in result shards -> skip them on resume."""
    done = set()
    if not os.path.isdir(out_dir):
        return done
    import pandas as pd
    for f in sorted(os.listdir(out_dir)):
        if f.startswith(prefix) and f.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(out_dir, f),
                                 usecols=["filename"], dtype=str)
                done.update(df["filename"].dropna().tolist())
            except Exception:
                pass
    return done


# =============================================================================
# Ground-truth loading (stream the CSV, match by filename stem).
# =============================================================================
def load_ground_truth(csv_path, audio_files, filename_col="id", text_col="text"):
    # Read the CSV row-by-row with the stdlib csv reader (READ-ONLY — the CSV file
    # is never modified).  This tolerates a truncated/corrupt file: bad bytes are
    # replaced (errors="replace") and any malformed row — including a truncated
    # final row with an unclosed quote — is simply SKIPPED, while every valid row
    # is kept.  (pandas' chunk parsers instead abort the whole 100k-row chunk on a
    # single bad line, losing tens of thousands of good rows.)
    import csv as _csv
    _csv.field_size_limit(10 ** 9)        # don't choke on a huge swallowed field
    # Match on the basename stem (CSV "id" has no extension/path), so recursive
    # relative paths like "sub/123.mp3" still match GT id "123".
    stem_to_name = {os.path.splitext(os.path.basename(f))[0]: f for f in audio_files}
    target = set(stem_to_name)
    gt = {}
    print(f"   streaming CSV for {len(target):,} stems ...", flush=True)
    scanned = skipped = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = _csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("   ⚠️  CSV is empty", flush=True)
            return gt
        try:
            fi, ti = header.index(filename_col), header.index(text_col)
        except ValueError:
            raise ValueError(f"CSV needs columns '{filename_col}' and '{text_col}'; "
                             f"found {header}")
        ncol = len(header)
        try:
            for row in reader:
                scanned += 1
                if len(row) != ncol:           # malformed / truncated row -> skip
                    skipped += 1
                    continue
                stem = row[fi].strip()
                if stem in target:
                    name = stem_to_name[stem]
                    if name not in gt:
                        gt[name] = row[ti]
                        if len(gt) >= len(target):
                            break
        except _csv.Error as e:                # unparseable tail -> stop, keep rest
            skipped += 1
            print(f"   ⚠️  stopped at a corrupt row ({str(e)[:60]})", flush=True)
    msg = f"   scanned {scanned:,} rows -> {len(gt):,} ground-truth matches"
    if skipped:
        msg += f"  ({skipped:,} malformed row(s) skipped)"
    print(msg, flush=True)
    return gt


# =============================================================================
# GPU engine — fixed-shape compiled batches + an eager fallback for the tail.
# =============================================================================
class ASREngine:
    def __init__(self, use_compile=True):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_ID, torch_dtype=self.dtype).to(self.device).eval()

        self.gpu_s = 0.0      # accumulated GPU forward time
        self.prep_s = 0.0     # processor (normalize+pad) time
        self.dec_s = 0.0      # batch_decode time
        self.padded_s = 0.0   # accumulated padded audio seconds fed to GPU
        self.n_batches = 0
        self.compiled = None
        if use_compile and self.device == "cuda":
            try:
                self.compiled = torch.compile(self.model, mode="max-autotune",
                                              fullgraph=False)
            except Exception as e:
                print(f"   ⚠️  torch.compile unavailable ({e}); using eager", flush=True)
                self.compiled = None

    def warmup(self):
        """Trigger one compile per bucket so steady-state batches never stall."""
        if self.compiled is None:
            return
        print("   compiling GPU kernels (one-time, ~20s per bucket) ...", flush=True)
        for b in BUCKET_SAMPLES:
            t = time.time()
            parts = [np.zeros(b, dtype=np.float32) for _ in range(GPU_BATCH)]
            self._infer(parts, bucket_samples=b, compiled=True)
            print(f"     bucket {b // SAMPLING_RATE:>2}s  ready ({time.time() - t:.0f}s)",
                  flush=True)

    def _infer(self, parts, bucket_samples, compiled):
        torch = self.torch
        _t = time.time()
        if compiled:
            inputs = self.processor(parts, sampling_rate=SAMPLING_RATE,
                                    return_tensors="pt", padding="max_length",
                                    max_length=bucket_samples, truncation=True)
            net = self.compiled
        else:
            inputs = self.processor(parts, sampling_rate=SAMPLING_RATE,
                                    return_tensors="pt", padding=True)
            net = self.model
        x = inputs.input_values.to(self.device, dtype=self.dtype)
        self.prep_s += time.time() - _t
        t = time.time()
        with torch.inference_mode():
            logits = net(x).logits
            ids = torch.argmax(logits, dim=-1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.gpu_s += time.time() - t
        self.padded_s += x.shape[0] * x.shape[1] / SAMPLING_RATE
        self.n_batches += 1
        _t = time.time()
        out = self.processor.batch_decode(ids)
        self.dec_s += time.time() - _t
        return out

    def infer_full_batch(self, parts, bucket_samples):
        """Exactly GPU_BATCH parts -> compiled fixed-shape path (fast)."""
        if self.compiled is not None:
            return self._infer(parts, bucket_samples, compiled=True)
        return self._infer(parts, bucket_samples, compiled=False)

    def infer_tail(self, parts):
        """Arbitrary-size leftover batch -> eager dynamic path."""
        return self._infer(parts, bucket_samples=None, compiled=False)


def bucket_for(length):
    for b in BUCKET_SAMPLES:
        if length <= b:
            return b
    return BUCKET_SAMPLES[-1]      # longer than max bucket -> truncate to it


# =============================================================================
# Main driver.
# =============================================================================
def main():
    global GPU_BATCH
    ap = argparse.ArgumentParser(description="Fast Malayalam wav2vec2 batch ASR")
    ap.add_argument("--audio-folder", default=AUDIO_FOLDER,
                    help="folder of audio files (default: AUDIO_FOLDER at top of file)")
    ap.add_argument("--csv", default=CSV_PATH,
                    help="ground-truth CSV; use --no-metrics for pure transcription")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--no-metrics", action="store_true",
                    help="skip WER/CER (fastest); still transcribes everything")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile (slower GPU, no warmup)")
    ap.add_argument("--load-workers", type=int, default=11,
                    help="CPU processes for decode+resample+VAD")
    ap.add_argument("--metric-workers", type=int, default=8,
                    help="CPU processes for WER/CER (only used with metrics)")
    ap.add_argument("--batch", type=int, default=GPU_BATCH)
    ap.add_argument("--shard-size", type=int, default=20_000)
    ap.add_argument("--limit", type=int, default=0, help="process only first N (debug)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--recursive", action="store_true",
                    help="search sub-folders for audio (names stored relative to folder)")
    args = ap.parse_args()

    GPU_BATCH = args.batch
    use_metrics = (not args.no_metrics) and (args.csv is not None)

    # ---- scan audio folder -------------------------------------------------
    print("① Scanning audio folder ...", flush=True)
    if args.recursive:
        audio_files = []
        for root, _, names in os.walk(args.audio_folder):
            for nm in names:
                if os.path.splitext(nm)[1].lower() in AUDIO_EXTENSIONS:
                    audio_files.append(os.path.relpath(os.path.join(root, nm),
                                                       args.audio_folder))
        audio_files.sort()
    else:
        with os.scandir(args.audio_folder) as it:
            audio_files = sorted(
                e.name for e in it
                if e.is_file() and os.path.splitext(e.name)[1].lower() in AUDIO_EXTENSIONS)
    if args.limit:
        audio_files = audio_files[:args.limit]
    print(f"   {len(audio_files):,} audio files", flush=True)
    if not audio_files:
        print("   nothing to do."); return

    # ---- ground truth ------------------------------------------------------
    gt_lookup = {}
    if use_metrics:
        print("② Loading ground truth ...", flush=True)
        gt_lookup = load_ground_truth(args.csv, audio_files)
    else:
        print("② Metrics disabled — pure transcription mode.", flush=True)

    # ---- resume ------------------------------------------------------------
    print("③ Checking for previous progress (resume) ...", flush=True)
    done = load_done_set(args.output_dir)
    todo = [f for f in audio_files if f not in done]
    print(f"   already done: {len(done):,} | to process: {len(todo):,}", flush=True)
    if not todo:
        print("   ✅ everything already processed."); return

    # ---- model -------------------------------------------------------------
    print("④ Loading model ...", flush=True)
    engine = ASREngine(use_compile=not args.no_compile)
    print(f"   device={engine.device} dtype={engine.dtype} "
          f"compile={'on' if engine.compiled is not None else 'off'}", flush=True)
    writer  = ShardWriter(args.output_dir, args.shard_size, RESULT_COLS)
    logpath = os.path.join(args.output_dir,
                           f"processing_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
    logfh = open(logpath, "w", newline="", encoding="utf-8")
    logw  = csv.writer(logfh)
    logw.writerow(["filename", "status", "duration_s", "vad_chunks", "cpu_time_s", "error"])

    # ---- shared state ------------------------------------------------------
    pending = []                                   # list of (fid, idx, part)
    fstate  = {}                                   # fid -> file record
    pending_metrics = {}                           # future -> partial row
    counters = {"done": 0, "fail": 0, "audio_s": 0.0}
    next_fid = 0

    # GPU runs in its own thread: CUDA releases the GIL during the forward pass,
    # so it overlaps with the main thread's bookkeeping (loader IPC, metrics, IO).
    # "spawn" is required: the parent has already initialized CUDA + compile
    # threads, and fork()ed children inherit held locks and deadlock on first
    # torch call.
    import multiprocessing as mp
    _spawn_ctx = mp.get_context("spawn")
    loader = ProcessPoolExecutor(max_workers=args.load_workers,
                                 initializer=_worker_init,
                                 mp_context=_spawn_ctx)
    metricpool = (ProcessPoolExecutor(max_workers=args.metric_workers,
                                      mp_context=_spawn_ctx)
                  if use_metrics else None)

    # ---- helpers -----------------------------------------------------------
    def write_log(filename, status, duration, nchunks, cpu_t, error):
        logw.writerow([filename, status, round(duration, 2), nchunks,
                       round(cpu_t, 3), (error or "")[:300].replace("\n", " ")])

    def finalize(fid):
        rec = fstate.pop(fid)
        texts = [rec["texts"][i] for i in range(rec["total"])]
        transcription = " ".join(t for t in texts if t).strip()
        row = {c: None for c in RESULT_COLS}
        row["filename"] = rec["filename"]
        row["ground_truth"] = rec["gt"]
        row["transcription"] = transcription
        counters["done"] += 1
        counters["audio_s"] += rec["duration"]
        write_log(rec["filename"], "SUCCESS", rec["duration"], rec["total"],
                  rec["cpu_t"], None)
        if use_metrics and rec["gt"]:
            fut = metricpool.submit(compute_metrics, rec["gt"], transcription)
            pending_metrics[fut] = row
        else:
            row["gt_cleaned"] = ""
            row["pred_cleaned"] = clean_text(transcription)
            writer.write(row)

    def assign(fid, idx, text):
        rec = fstate[fid]
        rec["texts"][idx] = text
        if len(rec["texts"]) == rec["total"]:
            finalize(fid)

    def run_batch(items):
        parts  = [it[2] for it in items]
        bucket = bucket_for(max(len(p) for p in parts))   # tight, length-sorted
        if len(items) == GPU_BATCH:
            texts = engine.infer_full_batch(parts, bucket)
        else:
            texts = engine.infer_tail(parts)              # ragged tail -> eager
        for (fid, idx, _), txt in zip(items, texts):
            assign(fid, idx, (txt or "").strip())

    def flush(force=False):
        # Length-sort, then run full GPU_BATCH-sized batches of similar-length
        # chunks.  Keep the sub-batch remainder buffered unless forcing the tail.
        if not force and len(pending) < FLUSH_THRESHOLD:
            return
        pending.sort(key=lambda t: len(t[2]))
        n = len(pending)
        limit = n if force else n - (n % GPU_BATCH)
        i = 0
        while i < limit:
            run_batch(pending[i:i + GPU_BATCH])
            i += GPU_BATCH
        del pending[:i]

    def drain_metrics(block=False):
        if not pending_metrics:
            return
        if block:
            for fut in list(pending_metrics):
                row = pending_metrics.pop(fut)
                row.update(fut.result())
                writer.write(row)
        else:
            for fut in [f for f in pending_metrics if f.done()]:
                row = pending_metrics.pop(fut)
                row.update(fut.result())
                writer.write(row)

    def register(filename, chunks, duration, cpu_t, error):
        nonlocal next_fid
        if error is not None or chunks is None:
            counters["fail"] += 1
            write_log(filename, "FAILED", 0, 0, cpu_t, error)
            row = {c: None for c in RESULT_COLS}
            row["filename"] = filename
            row["ground_truth"] = gt_lookup.get(filename, "")
            row["transcription"] = "ERROR"
            row["gt_cleaned"] = ""
            row["pred_cleaned"] = ""
            writer.write(row)
            return
        fid = next_fid; next_fid += 1
        fstate[fid] = {
            "filename": filename, "gt": gt_lookup.get(filename, ""),
            "duration": duration, "cpu_t": cpu_t,
            "total": len(chunks), "texts": {},
        }
        for idx, ch in enumerate(chunks):
            part = np.concatenate([ch, np.zeros(SILENCE_SAMPLES, dtype=np.float32)])
            pending.append((fid, idx, part))

    def progress():
        el = time.time() - t_start
        n = counters["done"] + counters["fail"]
        fps = n / el if el else 0
        ah = counters["audio_s"] / 3600
        sa = counters["audio_s"] / el if el else 0
        remaining = len(todo) - n
        eta = remaining / fps / 3600 if fps else 0
        print(f"   [{n:,}/{len(todo):,}] {fps:5.1f} files/s | "
              f"{sa:6.0f} s-audio/s | {ah:5.1f} audio-h done | "
              f"ETA {eta:4.1f} h | fail {counters['fail']}", flush=True)

    # ---- streaming loop ----------------------------------------------------
    print(f"⑤ Processing {len(todo):,} files "
          f"({args.load_workers} loaders, batch {GPU_BATCH}, "
          f"{'metrics on' if use_metrics else 'metrics off'}) ...\n", flush=True)
    MAX_INFLIGHT    = args.load_workers * 3
    FLUSH_THRESHOLD = GPU_BATCH * 12     # accumulate enough for full, sorted batches

    files_iter = iter(todo)
    inflight = {}

    def submit_next():
        fn = next(files_iter, None)
        if fn is None:
            return False
        inflight[loader.submit(process_file, (fn, args.audio_folder))] = fn
        return True

    for _ in range(MAX_INFLIGHT):              # prefetch/decode while it compiles
        if not submit_next():
            break
    engine.warmup()                            # compile kernels (one-time)
    engine.gpu_s = engine.padded_s = 0.0       # reset: exclude compile warmup
    engine.n_batches = 0
    t_start = time.time()

    last_logged = 0
    tprof = {"wait": 0.0, "harvest": 0.0, "flush": 0.0, "meta": 0.0}
    try:
        while inflight:
            _t = time.time()
            ready, _ = wait(inflight, return_when=FIRST_COMPLETED)
            tprof["wait"] += time.time() - _t
            _t = time.time()
            for fut in ready:
                inflight.pop(fut)
                filename, chunks, duration, cpu_t, error = fut.result()
                register(filename, chunks, duration, cpu_t, error)
                submit_next()
            tprof["harvest"] += time.time() - _t
            _t = time.time()
            flush(force=False)
            tprof["flush"] += time.time() - _t
            _t = time.time()
            drain_metrics(block=False)
            tprof["meta"] += time.time() - _t

            n = counters["done"] + counters["fail"]
            if n - last_logged >= args.log_every:
                last_logged = n
                progress()

        # drain the pipeline
        _t = time.time()
        flush(force=True)
        tprof["flush"] += time.time() - _t
        _t = time.time()
        drain_metrics(block=True)
        tprof["meta"] += time.time() - _t
    finally:
        loader.shutdown(wait=True)
        if metricpool:
            metricpool.shutdown(wait=True)
        writer.close()
        logfh.close()

    # ---- summary -----------------------------------------------------------
    el = time.time() - t_start
    n = counters["done"] + counters["fail"]
    print("\n" + "=" * 64)
    print("  DONE")
    print("=" * 64)
    print(f"  processed        : {n:,}  (ok {counters['done']:,}, failed {counters['fail']:,})")
    print(f"  wall time        : {el / 3600:.2f} h  ({el:.0f}s)")
    print(f"  audio processed  : {counters['audio_s'] / 3600:.2f} h")
    print(f"  throughput       : {n / el:.1f} files/s | "
          f"{counters['audio_s'] / el:.0f} s-audio/s")
    if n:
        print(f"  projected 300k   : {300_000 / (n / el) / 3600:.1f} h")
    print(f"  GPU busy         : {engine.gpu_s:.0f}s of {el:.0f}s wall "
          f"({100 * engine.gpu_s / el:.0f}%) | {engine.n_batches} batches")
    print(f"  padded->GPU      : {engine.padded_s / 3600:.2f} h "
          f"({engine.padded_s / max(engine.gpu_s, 1e-9):.0f} s-padded/s GPU rate)")
    print(f"  main-loop split  : wait {tprof['wait']:.0f}s | harvest "
          f"{tprof['harvest']:.0f}s | flush {tprof['flush']:.0f}s | "
          f"meta {tprof['meta']:.0f}s")
    print(f"  infer split      : prep {engine.prep_s:.0f}s | gpu "
          f"{engine.gpu_s:.0f}s | decode {engine.dec_s:.0f}s")
    print(f"  results          : {args.output_dir}/results_part_*.csv "
          f"({writer.total_written:,} rows written this run)")
    print(f"  log              : {logpath}")
    print("=" * 64)


if __name__ == "__main__":
    main()
