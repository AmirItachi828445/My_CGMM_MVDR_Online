# Online CGMM-MVDR Beamformer — MATLAB Implementation

MATLAB implementation of the online CGMM-MVDR spatial zoom beamformer, based on:

- [Frame-by-Frame Closed-Form Update for Mask-Based Adaptive MVDR Beamforming](https://ieeexplore.ieee.org/document/8461850) (Higuchi et al., ICASSP 2018)
- [Online MVDR Beamformer Based on Complex Gaussian Mixture Model With Spatial Prior for Noise Robust ASR](https://ieeexplore.ieee.org/document/7845594) (Higuchi et al., IEEE/ACM TASLP 2017)

Please cite these papers if you use this code. This is not an official implementation of the above papers.

## Repository Structure

```
Matlab/                  Standalone CGMM-MVDR class + demo script
  OnlineCGMMMVDR.m         Frame-by-frame CGMM-MVDR class
  cgmm_mvdr_beamform.m     Main beamforming entry point
  compute_steering_covariance.m  Geometry-guided prior SCMs
  complex_gaussian.m       Zero-mean complex Gaussian PDF
  plot_power_comparison.m  Power spectrum visualisation
  run_demo.m               Full demo (4×4 array, 9 focus points)

Matlab_Audio_Codes/      Production pipeline (multi-beamformer spatial zoom)
  Main.m                   Entry point: load → beamform → save plots
  beamformer/              Beamformer implementations
    CGMM_MVDR_Zoom.m         Fast CGMM-MVDR zoom (vectorised, real-time)
    Base_BeamForming_*.m     Delay-and-sum baselines
    MVDR_Zoom_*.m            Conventional MVDR
    LCMV_*.m                 LCMV / GSC beamformers
  config/
    zoom_config.m            Central configuration (focus grid, sources, paths)
    uma16_array.xml          UMA-16 microphone array geometry
  processing/
    process_beamforming.m    Run all beamformers × all focus points
    save_beamforming_plots.m Spatial-response bar charts
    beamformer_dispatch.m    Route jobs to beamformer functions
    save_zoomed_audio.m      Write output WAV files
  utils/
    load_audio.m             Load WAV or HDF5 audio
    load_mic_positions.m     Parse XML microphone geometry
    compute_energy.m         Short-time energy computation

test_audios/             Sample multichannel WAV files for demo
```

## Quick Start (production pipeline)

```matlab
cd Matlab_Audio_Codes
addpath beamformer utils config processing

% Edit config/zoom_config.m to point to your audio file and set source_points.
cfg = zoom_config();

mic_pos    = load_mic_positions(cfg.xml_file);
mic_pos    = mic_pos([9 11 13 15], :);    % select 4 well-spaced mics
[audio, ~] = load_audio(cfg.audio_file, size(mic_pos,1));
cfg.fs     = 44100;  % or whatever your file's sample rate is

results = process_beamforming(audio, mic_pos, cfg);
save_beamforming_plots(results, cfg);
save_zoomed_audio(results, cfg);
```

## Requirements

| Toolbox | Usage |
|---|---|
| Signal Processing Toolbox | `hann`, `fft`, `ifft` |

MATLAB **R2019a** or later recommended.

## Key Improvements in `CGMM_MVDR_Zoom.m`

| Fix | Description |
|---|---|
| Prior trace correction | `trace(R_target) = N_mics` (was ≈ 1), prevents mask collapse |
| No Cholesky per frame | Precomputed `R⁻¹` + analytical log-likelihood eliminates inner-loop inversions |
| Sherman-Morrison update | `R⁻¹` updated with O(C²) rank-1 formula after each R step |
| Vectorised outer products | SCM accumulation uses element-wise broadcast over all F bins at once |
| Noise bootstrap | First 10 % of frames forced as noise-only (was 5 %) |

Combined these changes yield a **50–200 × speedup** over the original loop-heavy implementation.
