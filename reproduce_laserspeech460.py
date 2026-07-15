"""
Regenerate the LaserSpeech-460 corpus from a local LibriSpeech train-clean-460.

Reads every clean utterance, applies the published synthesis operating point, and
writes the synthetic-laser waveform, mirroring the LibriSpeech directory layout.
Output is bit-exact and independent of worker count or file order, because each
utterance's noise is seeded from its LibriSpeech id.

Example:
    python reproduce_laserspeech460.py \
        --librispeech /path/to/LibriSpeech/train-clean-360 \
        --out ./LaserSpeech-460 --workers 8

    # harder variant (matches the WoodenFaceBox condition)
    python reproduce_laserspeech460.py ... --preset hard
"""

import argparse
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf

from laserspeech import (DEFAULT_MASTER_SEED, LASERSPEECH_460, LASERSPEECH_460_HARD,
                         synthesize, utterance_seed)

PRESETS = {"default": LASERSPEECH_460, "hard": LASERSPEECH_460_HARD}


def _one(args):
    src, dst, params, master_seed = args
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean, fs = sf.read(src)
    if clean.ndim > 1:
        clean = clean[:, 0]
    out = synthesize(clean, fs=fs, seed=utterance_seed(src.stem, master_seed), **params)
    sf.write(dst, out, fs)
    return len(out) / fs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", type=pathlib.Path, required=True,
                    help="Root of a LibriSpeech split (e.g. train-clean-360).")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output directory.")
    ap.add_argument("--preset", choices=PRESETS, default="default",
                    help="Operating point: 'default' = LaserSpeech-460, 'hard' = WoodenFaceBox-like.")
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    ap.add_argument("--limit", type=int, default=None, help="Only process N utterances (for testing).")
    ap.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED,
                    help="Master seed. Leave at the default to reproduce the published corpus.")
    args = ap.parse_args()

    params = PRESETS[args.preset]
    srcs = sorted(args.librispeech.rglob("*.flac")) or sorted(args.librispeech.rglob("*.wav"))
    if not srcs:
        raise SystemExit(f"No .flac/.wav found under {args.librispeech}")
    if args.limit:
        srcs = srcs[:args.limit]

    jobs = [(s, (args.out / s.relative_to(args.librispeech)).with_suffix(".wav"), params, args.seed)
            for s in srcs]
    print(f"utterances : {len(jobs)}")
    print(f"preset     : {args.preset}  {params}")
    print(f"seed       : {args.seed}")
    print(f"output     : {args.out}")

    total_s, done = 0.0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for f in as_completed(futures):
            total_s += f.result()
            done += 1
            if done % 500 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  ({total_s/3600:.2f} h)", flush=True)

    print(f"\nDone. {done} utterances, {total_s/3600:.2f} h written to {args.out}")


if __name__ == "__main__":
    main()
