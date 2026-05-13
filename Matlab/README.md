# MATLAB CGMM-MVDR Beamformer

Line-by-line MATLAB port of [`cgmm_mvdr.py`](../cgmm_mvdr.py).

## Requirements

| Toolbox | Usage |
|---|---|
| Signal Processing Toolbox | `stft`, `istft`, `hann`, `medfilt1` |
| Audio Toolbox (optional) | `audioread`, `audiowrite` |

MATLAB **R2019a** or later is required (`stft`/`istft` were introduced in R2019a).

## Files

| File | Description |
|---|---|
| `complex_gaussian.m` | Zero-mean complex Gaussian PDF — Eq. (15) in \[2\] |
| `OnlineCGMMMVDR.m` | MATLAB class — exact port of `OnlineCGMMMVDR` in Python |
| `compute_steering_covariance.m` | Build prior SCMs from array geometry |
| `cgmm_mvdr_beamform.m` | Main entry-point function `(data, mic_pos, focus_point, fs, c)` |
| `plot_power_comparison.m` | Power spectrum + mask plots (called automatically) |
| `run_demo.m` | Full demo: notebook replication + 4×4 array, 9 focus points |

## Quick Start

```matlab
addpath('path/to/Matlab');

% Load your multi-channel audio (N_samples × N_mics)
[data, fs] = audioread('your_audio.wav');

% Define microphone positions (N_mics × 3) [metres]
mic_pos = [...];

% Define target focus point (1 × 3) [metres]
focus_point = [1.0, 0.0, 1.5];

% Speed of sound [m/s]
c = 343;

% Run the beamformer (online, frame-by-frame)
[s_target, power_data] = cgmm_mvdr_beamform(data, mic_pos, focus_point, fs, c);

% Play back
soundsc(s_target, fs);
```

## Optional Parameters

```matlab
[s_target, pd] = cgmm_mvdr_beamform(data, mic_pos, focus_point, fs, c, ...
    'nperseg',   200, ...   % STFT window length (default 200)
    'noverlap',  100, ...   % STFT overlap length (default 100)
    'ny_k',       10, ...   % CGMM hyperparameter, target (default 10)
    'ny_n',       10, ...   % CGMM hyperparameter, noise  (default 10)
    'beg_noise', 160, ...   % initial noise-only frames (default 0)
    'end_noise',  80, ...   % final   noise-only frames (default 0)
    'refs',    [1 2], ...   % reference microphone indices (default [1 2])
    'priorRk', R_tgt, ...   % custom target prior SCM (F,C,C)
    'priorRn', R_nse);      % custom noise  prior SCM (F,C,C)
```

## Use Pre-estimated Priors (matches Python notebook)

```matlab
% Estimate priors from training recordings (same as Python notebook)
window = hann(200);
[spec_tgt,   f] = stft_multichannel(s_tgt,   fs, window, 100, 200);
[spec_noise, ~] = stft_multichannel(s_noise, fs, window, 100, 200);

[~, F, T] = size(spec_tgt);
R_tgt_prior = zeros(F, C, C, 'like', 1i);
for f = 1:F
    St = squeeze(spec_tgt(:, f, :));
    R_tgt_prior(f,:,:) = (St * St') / T;
end
% ... similarly for R_noise_prior

[s_enh, pd] = cgmm_mvdr_beamform(data, mic_pos, focus_point, fs, c, ...
    'priorRk', R_tgt_prior, 'priorRn', R_noise_prior, ...
    'beg_noise', 160, 'end_noise', 80);
```

## Output: `power_data` struct

| Field | Shape | Description |
|---|---|---|
| `freqs` | (F,1) | Frequency axis \[Hz\] |
| `input` | (F,1) | Mean power spectrum of input channel 1 |
| `enhanced1` | (F,1) | Mean power of beamformed output, ref mic 1 |
| `enhanced2` | (F,1) | Mean power of beamformed output, ref mic 2 |
| `spec_enh` | (2,F,T) | Full enhanced STFT |
| `masks` | (T,F) | Estimated target mask λ\_kn |

## Algorithm Summary

The algorithm follows these steps **per STFT frame** (online / real-time):

1. **Mask estimation** — posterior probability that frame belongs to target vs noise (Eq. 19, 25 in \[2\])
2. **φ update** — time-dependent variance per frequency bin (Eq. 20 in \[2\])
3. **R update** — normalized spatial covariance matrices via closed-form MAP (Eq. 33 in \[2\])
4. **RR\_k update** — mask-weighted target SCM accumulation (Eq. 8 in \[1\])
5. **RR\_y\_inv update** — Sherman-Morrison rank-1 inverse update (Eq. 7 in \[1\])
6. **MVDR filter** — w = (RR\_y\_inv · RR\_k)\[:, ref\] / trace(RR\_y\_inv · RR\_k) (Eq. 9 in \[1\])
7. **Beamforming** — s(f) = w(f)^H · y(f)

## References

\[1\] T. Higuchi, K. Kinoshita, N. Ito, S. Karita and T. Nakatani,
"Frame-by-Frame Closed-Form Update for Mask-Based Adaptive MVDR Beamforming,"
ICASSP 2018, doi:[10.1109/ICASSP.2018.8461850](https://doi.org/10.1109/ICASSP.2018.8461850)

\[2\] T. Higuchi, N. Ito, S. Araki, T. Yoshioka, M. Delcroix and T. Nakatani,
"Online MVDR Beamformer Based on Complex Gaussian Mixture Model With Spatial Prior
for Noise Robust ASR," IEEE/ACM TASLP, vol. 25, no. 4, pp. 780-793, Apr. 2017,
doi:[10.1109/TASLP.2017.2665341](https://doi.org/10.1109/TASLP.2017.2665341)
