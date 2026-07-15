"""
LaserSpeech: physics-based synthesis of laser Doppler vibrometry (LDV) speech.

Transforms clean speech into laser-like recordings by modelling the five principal
degradation mechanisms of optical vibrometry (see the paper, Algorithm 1):

  1. bandwidth limiting            -- Butterworth LPF at f_lpf
  2. frequency-dependent noise     -- purple (violet) noise, PSD proportional to f^2
  3. broadband measurement noise   -- white Gaussian noise
  4. Nyquist anti-aliasing         -- Butterworth LPF at f_nyq
  5. multipath frequency smearing  -- Gaussian blur of the STFT magnitude

Synthesis is deterministic: each utterance's noise is drawn from a generator seeded
from (master_seed, utterance key), so the corpus regenerates bit-exactly regardless
of file order or parallelism.
"""

import hashlib

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, istft, sosfilt, stft

# LaserSpeech-460 operating point (matches the Batch-2 / LargeKappa target).
LASERSPEECH_460 = {
    "lpf_cutoff": 2000.0,     # f_lpf  [Hz]
    "nyquist_cutoff": 2500.0, # f_nyq  [Hz]
    "alpha": 0.5,             # purple-noise weight
    "beta": 0.5,              # Gaussian-noise weight
    "speech_weight": 0.4,     # w, speech-preservation weight
    "smear_sigma": 0.5,       # sigma, frequency smearing [bins]
}

# Harder variant, closest match to the Batch-1 / WoodenFaceBox target.
LASERSPEECH_460_HARD = {**LASERSPEECH_460, "alpha": 1.5, "speech_weight": 0.3, "smear_sigma": 1.0}

TARGET_RMS = 0.065          # ~ -23.7 dBFS, matches the mean RMS of the real recordings
DEFAULT_MASTER_SEED = 20260  # published seed for LaserSpeech-460


def utterance_seed(key, master_seed=DEFAULT_MASTER_SEED):
    """Deterministic 32-bit seed for one utterance.

    Derived from the utterance key (e.g. its LibriSpeech id) rather than from a
    counter, so regeneration does not depend on file ordering or worker count.
    """
    digest = hashlib.sha256(f"{master_seed}:{key}".encode()).hexdigest()
    return int(digest[:8], 16)


def butterworth_lpf(x, cutoff_hz, fs, order=5):
    sos = butter(order, cutoff_hz / (fs / 2.0), btype="low", output="sos")
    return sosfilt(sos, x)


def robust_normalize(x, threshold_db=-25.0):
    """Zero-mean/unit-variance using only frames above threshold_db of the peak.

    Silence-dominated utterances would otherwise be normalised by their noise floor.
    """
    energy = x ** 2
    if energy.size == 0:
        return x
    active = energy > energy.max() * 10 ** (threshold_db / 10.0)
    ref = x[active] if active.any() else x
    std = ref.std()
    return x if std == 0 else (x - ref.mean()) / std


def purple_noise(n, rng):
    """Violet/purple noise: first difference of white noise, PSD ~ f^2."""
    noise = np.diff(rng.standard_normal(n), prepend=0.0)
    std = noise.std()
    return noise if std == 0 else noise / std


def freq_smear(x, fs, sigma, nperseg=400, hop=160):
    """Blur the STFT magnitude along frequency; phase is preserved."""
    _, _, zxx = stft(x, fs=fs, nperseg=nperseg, noverlap=nperseg - hop)
    smeared = gaussian_filter1d(np.abs(zxx), sigma=sigma, axis=0)
    _, out = istft(smeared * np.exp(1j * np.angle(zxx)), fs=fs,
                   nperseg=nperseg, noverlap=nperseg - hop)
    return out[:len(x)]


def synthesize(clean, fs=16000, seed=None, **params):
    """Apply the synthesis pipeline to a clean waveform.

    Args:
        clean: 1-D float array, the clean speech.
        fs:    sample rate (the published corpus uses 16 kHz).
        seed:  int seed for this utterance's noise (see `utterance_seed`).
        **params: overrides for LASERSPEECH_460 (alpha, beta, speech_weight,
                  lpf_cutoff, nyquist_cutoff, smear_sigma).

    Returns:
        1-D float array, the synthetic-laser waveform.
    """
    p = {**LASERSPEECH_460, **params}
    rng = np.random.default_rng(seed)
    n = len(clean)

    x = butterworth_lpf(clean, p["lpf_cutoff"], fs)          # 1. bandwidth limiting
    x = robust_normalize(x)                                   # 2. level normalisation
    x = (p["speech_weight"] * x                               # 3-4. noise injection
         + p["alpha"] * purple_noise(n, rng)
         + p["beta"] * rng.standard_normal(n))
    x = butterworth_lpf(x, p["nyquist_cutoff"], fs)           # 5. anti-aliasing cleanup
    if p["smear_sigma"] > 0:
        x = freq_smear(x, fs, p["smear_sigma"])               # 6. multipath smearing

    rms = np.sqrt(np.mean(x ** 2))                            # 7. RMS normalisation
    return x * (TARGET_RMS / rms) if rms > 1e-8 else x


def synthesize_file(in_path, out_path, key=None, master_seed=DEFAULT_MASTER_SEED, **params):
    """Synthesize one WAV file. `key` defaults to the input filename stem."""
    clean, fs = sf.read(in_path)
    if clean.ndim > 1:
        clean = clean[:, 0]
    key = key if key is not None else str(in_path).split("/")[-1].rsplit(".", 1)[0]
    out = synthesize(clean, fs=fs, seed=utterance_seed(key, master_seed), **params)
    sf.write(out_path, out, fs)
    return out, fs
