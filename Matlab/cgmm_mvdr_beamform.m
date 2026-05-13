function [s_target, power_data] = cgmm_mvdr_beamform(data, mic_pos, focus_point, fs, c, varargin)
%CGMM_MVDR_BEAMFORM  Online CGMM-MVDR beamformer  (geometry-driven priors).
%
%   Exact MATLAB port of cgmm_mvdr.py / OnlineCGMMMVDR.
%   Processes multi-channel audio frame-by-frame (online / real-time mode).
%
%   [s_target, power_data] = CGMM_MVDR_BEAMFORM(data, mic_pos, focus_point, fs, c)
%   [s_target, power_data] = CGMM_MVDR_BEAMFORM(..., Name, Value)
%
%   Required inputs:
%     data        - (N_samples, N_mics)  time-domain multi-channel audio
%     mic_pos     - (N_mics, 3)          microphone positions [m]
%     focus_point - (1, 3)               target focus-point  [m]
%     fs          - sampling frequency   [Hz]
%     c           - speed of sound       [m/s]  (default 343)
%
%   Optional Name-Value pairs:
%     'nperseg'    STFT window length          (default 200)
%     'noverlap'   STFT overlap length         (default 100)
%     'ny_k'       CGMM hyperparameter target  (default 10)
%     'ny_n'       CGMM hyperparameter noise   (default 10)
%     'beg_noise'  initial noise-only frames   (default 0)
%     'end_noise'  final   noise-only frames   (default 0)
%     'refs'       reference mic indices (1-based, 1x2, default [1 2])
%     'priorRk'    custom target SCM prior (F,C,C)  – overrides geometry
%     'priorRn'    custom noise  SCM prior (F,C,C)  – overrides geometry
%
%   Outputs:
%     s_target   - (N_samples, 1)  enhanced target audio (ref-mic 1 beamformer)
%     power_data - struct with fields:
%       .freqs      frequency axis (F,1) [Hz]
%       .input      mean power spectrum of channel 1 input   (F,1)
%       .enhanced1  mean power spectrum of output, ref mic 1 (F,1)
%       .enhanced2  mean power spectrum of output, ref mic 2 (F,1)
%       .spec_enh   full enhanced STFT, (2,F,T)
%       .masks      estimated target masks, (T,F)

% ---------- parse optional arguments ----------
p = inputParser;
addParameter(p, 'nperseg',   200);
addParameter(p, 'noverlap',  100);
addParameter(p, 'ny_k',       10);
addParameter(p, 'ny_n',       10);
addParameter(p, 'beg_noise',   0);
addParameter(p, 'end_noise',   0);
addParameter(p, 'refs',    [1 2]);
addParameter(p, 'priorRk',    []);
addParameter(p, 'priorRn',    []);
parse(p, varargin{:});
opts = p.Results;

nperseg  = opts.nperseg;
noverlap = opts.noverlap;
nfft     = nperseg;          % nfft == nperseg (matches Python default)

[N_samples, N_mics] = size(data);

% ---------- STFT (matching scipy.signal.stft defaults) ----------
% Python: stft(s.T, nperseg=200, noverlap=100, nfft=200)
% Default window: hann(nperseg, symmetric) — same as MATLAB hann(N)
window = hann(nperseg);

% Use MATLAB's stft (Signal Processing Toolbox, R2019a+)
% Returns S of shape (nfft/2+1, T_frames) for real input
[S1, freqs, ~] = stft(data(:, 1), fs, ...
    'Window',        window,   ...
    'OverlapLength', noverlap, ...
    'FFTLength',     nfft,     ...
    'FrequencyRange','onesided');

[F, T_frames] = size(S1);

% Build multi-channel STFT array: spec_all(C, F, T)
spec_all = zeros(N_mics, F, T_frames, 'like', 1+1i);
spec_all(1, :, :) = S1;
for ch = 2:N_mics
    spec_all(ch, :, :) = stft(data(:, ch), fs, ...
        'Window',        window,   ...
        'OverlapLength', noverlap, ...
        'FFTLength',     nfft,     ...
        'FrequencyRange','onesided');
end

% ---------- prior SCMs ----------
if isempty(opts.priorRk) || isempty(opts.priorRn)
    [R_target_prior, R_noise_prior] = ...
        compute_steering_covariance(mic_pos, focus_point, freqs, c);
else
    R_target_prior = opts.priorRk;
    R_noise_prior  = opts.priorRn;
end

% ---------- initialise CGMM-MVDR ----------
cgmm = OnlineCGMMMVDR(opts.ny_k, opts.ny_n, ...
    R_target_prior, R_noise_prior, ...
    N_mics, F, opts.beg_noise, opts.end_noise);

% ---------- frame-by-frame (online) processing ----------
spec_enh = zeros(2, F, T_frames, 'like', 1+1i);
masks    = zeros(T_frames, F);

for t = 1:T_frames
    y_t = spec_all(:, :, t);            % (N_mics, F)
    [s_out, mask_t] = cgmm.step(y_t, t, T_frames);
    spec_enh(1, :, t) = s_out{1};       % ref-mic-1 beamformer
    spec_enh(2, :, t) = s_out{2};       % ref-mic-2 beamformer
    masks(t, :)       = mask_t';
end

% ---------- power comparison (before normalisation) ----------
power_data.freqs     = freqs;
power_data.input     = mean(abs(squeeze(spec_all(1, :, :))).^2, 2);   % (F,1)
power_data.enhanced1 = mean(abs(squeeze(spec_enh(1, :, :))).^2, 2);
power_data.enhanced2 = mean(abs(squeeze(spec_enh(2, :, :))).^2, 2);
power_data.spec_enh  = spec_enh;
power_data.masks     = masks;

% ---------- ISTFT ----------
s_enh1 = istft(squeeze(spec_enh(1, :, :)), fs, ...
    'Window',        window,   ...
    'OverlapLength', noverlap, ...
    'FFTLength',     nfft,     ...
    'FrequencyRange','onesided');

% Trim / zero-pad to match original sample count
if numel(s_enh1) >= N_samples
    s_target = s_enh1(1:N_samples);
else
    s_target = [s_enh1; zeros(N_samples - numel(s_enh1), 1)];
end
s_target = real(s_target(:));

% ---------- plot power comparison ----------
plot_power_comparison(power_data, focus_point);
end
