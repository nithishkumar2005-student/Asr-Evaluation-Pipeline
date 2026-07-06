# ASR Evaluation Pipeline — Malayalam (wav2vec2)

Fast, GPU-optimized batch transcription and scoring pipeline for validating
synthetically generated (TTS) audio against its source ground-truth text.

Built to transcribe ~300k Malayalam audio files as fast as possible on a
single NVIDIA GB10 (DGX Spark) GPU using `wav2vec2-large-xlsr-malayalam`,
while reproducing the exact WER / CER / accuracy metrics of the original
research notebook.

## Why this exists

The original workflow processed one file at a time in a notebook
(~320 s-audio/s eager). This script keeps the model and metrics identical
but rebuilds the pipeline around the GPU:

| | Original notebook | This script |
|---|---|---|
| Loading | 1 file at a time | CPU worker pool decodes/resamples/VADs N files ahead so the GPU never idles |
| Batching | Per-file GPU batch (bs 1–3) | Cross-file, length-bucketed batches (fixed shapes) |
| Compilation | Eager fp16 (~320 s-audio/s) | `torch.compile` fp16 (~757 s-audio/s, 2.4x, same outputs) |
| Memory | `empty_cache()` every file | Never (forces a slow GPU sync) |
| Metrics | Computed on main thread | Offloaded to a metric process pool |
| Output | One results.csv at the end | Incremental sharded CSVs with crash-resume |

Measured GPU floor: 320 s-audio/s eager, 757 s-audio/s compiled.
At ~50s average clip length, 300k files (~15M s of audio) run in roughly
5–6 hours compiled vs. ~60 hours on the original notebook.

## Pipeline architecture

```
Ground-Truth Text
      |
      v
[ TTS Generation ]  --------> Audio File (.wav)
      |
      v
[ Silero VAD ]  -------------> Chunked audio segments
                                (split on speech/silence boundaries)
      |
      v
[ ASR Model ]  --------------> Transcribed text (per chunk)
      |
      v
[ Metric Scoring ]  ---------> CER / WER / Accuracy
                                (vs. ground-truth text)
```

1. **TTS Generation** — ground-truth text is synthesized into audio.
2. **Silero VAD chunking** — audio is split into 5–30s chunks on
   speech/silence boundaries so no chunk cuts mid-word.
3. **ASR transcription** — each chunk is transcribed by
   `gvs/wav2vec2-large-xlsr-malayalam`, batched by length bucket for
   `torch.compile`-friendly fixed shapes.
4. **Metric scoring** — transcription is compared against ground truth to
   produce WER, CER, sentence error rate (SER), and accuracy.

## Metrics

Computed per audio sample, comparing cleaned ASR transcription to cleaned
ground truth (Malayalam Unicode range only, whitespace-normalized):

- **WER** (Word Error Rate) — substitutions, deletions, insertions over
  ground-truth word count, using fuzzy word matching (similarity ≥ 0.80)
  to avoid penalizing near-identical transcription variants.
- **CER** (Character Error Rate) — standard Levenshtein edit distance over
  characters, normalized by ground-truth length.
- **Accuracy %** — `max(0, (1 − WER) × 100)`.
- **SER** (Sentence Error Rate) — 1 if the cleaned strings differ at all,
  else 0.

## Requirements

- Python 3.10+
- CUDA-capable GPU (validated on NVIDIA GB10 / DGX Spark; will run on any
  CUDA GPU, `torch.compile` speedup will vary)
- Model weights for `gvs/wav2vec2-large-xlsr-malayalam` (downloaded via
  Hugging Face on first run)
- Silero VAD, cached locally via `torch.hub`

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full pinned dependency list
(`torch`, `transformers`, `librosa`, `soundfile`, `numpy`, `pandas`).

## Usage

```bash
python fast_malayalam_asr.py \
    --audio-folder /path/to/audio \
    --csv /path/to/ground_truth.csv \
    --output-dir ./asr_out
```

Pure throughput run, no WER/CER scoring:

```bash
python fast_malayalam_asr.py --audio-folder /path/to/audio --no-metrics
```

Resume is automatic — re-running the same command skips files that already
have a result row in `asr_out/results_part_*.csv`.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--audio-folder` | (set in script) | Folder of audio files to transcribe |
| `--csv` | (set in script) | Ground-truth CSV; omit for transcription-only |
| `--output-dir` | `Malayalam_asr_out` | Where `results_part_*.csv` shards are written |
| `--no-metrics` | off | Skip WER/CER scoring, transcription only |
| `--no-compile` | off | Disable `torch.compile` (eager mode fallback) |
| `--load-workers` | 11 | CPU worker processes for decode/resample/VAD |
| `--metric-workers` | 8 | Worker processes for WER/CER scoring |
| `--batch` | 8 | GPU batch size |
| `--shard-size` | 20,000 | Rows per output CSV shard |
| `--limit` | 0 (no limit) | Process only first N files (debugging) |
| `--recursive` | off | Recurse into subfolders of `--audio-folder` |

Output columns (`results_part_*.csv`): `filename, ground_truth,
transcription, gt_cleaned, pred_cleaned, wer, cer, ser, accuracy_pct,
substitutions, deletions, insertions, gt_word_count, pred_word_count`.

## Status

Malayalam has completed full-scale evaluation (~6.3 lakh male + ~6.3 lakh
female samples). Eight additional languages (Bengali, English, Gujarati,
Kannada, Punjabi, Sanskrit, Tamil, Telugu) have been validated on pilot
sample sets and are pending large-scale runs on this same pipeline design.

## Next steps

- Evaluate a faster inference backend for large-scale rollout (e.g.
  TensorRT / TensorRT-LLM, ONNX Runtime, or a batched serving layer such as
  vLLM/Triton) against the current PyTorch baseline.
- Benchmark candidate backend(s) on a like-for-like sample set, tracking
  files/minute alongside CER/WER/Accuracy to confirm no quality regression
  from optimization (e.g. quantization).
- Roll out full-scale evaluation to the remaining 8 languages once a
  backend is selected.
- Document the finalized high-throughput setup (config, dependencies, run
  commands) once the backend decision is made.

## License

Add a license here if you plan to make this repo public (MIT is a common
default for portfolio projects).
