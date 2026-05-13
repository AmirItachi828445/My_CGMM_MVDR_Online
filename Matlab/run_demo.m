%RUN_DEMO  Demonstration of MATLAB CGMM-MVDR beamformer.
%
%   This script mirrors the Python Example.ipynb notebook step-by-step
%   and also demonstrates a 4×4 microphone array with 9 focus points,
%   two sources (right-centre and left-centre).
%
%   Run sections independently with Ctrl+Enter, or the full script at once.
%
%   Requirements:
%     - Signal Processing Toolbox  (stft / istft / hann)
%     - Audio Toolbox              (audioread / audiowrite)  [optional]
%     - MATLAB R2019a or later
%
%   All .m files must be on the MATLAB path (add this folder):
%     addpath(fileparts(mfilename('fullpath')))

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

%% ================================================================
%% PART 1 — Replicate Python notebook  (mixture1 / mixture2 files)
%% ================================================================
fprintf('=== Part 1: replicating Python notebook ===\n');

% Paths relative to the repository root (adjust if needed)
repo_root  = fileparts(fileparts(mfilename('fullpath')));
audio_dir  = fullfile(repo_root, 'test_audios');

% ---- load training files for prior estimation ----
[s_tgt,   fs] = audioread(fullfile(audio_dir, 'mixture2_target.wav'));
[s_noise,  ~] = audioread(fullfile(audio_dir, 'mixture2_noise.wav'));

fprintf('mixture2_target : %d samples × %d ch,  fs=%d Hz\n', ...
    size(s_tgt,1), size(s_tgt,2), fs);

% ---- STFT settings (identical to Python notebook) ----
nperseg  = 200;
noverlap = 100;
nfft     = 200;
window   = hann(nperseg);

% ---- STFT of training signals ----
[spec_tgt,   freqs, ~] = stft_multichannel(s_tgt,   fs, window, noverlap, nfft);
[spec_noise, ~,     ~] = stft_multichannel(s_noise, fs, window, noverlap, nfft);

% ---- estimate priors from sample SCM (Eq. notebook) ----
% Python: R_tgt_prior = einsum('cft,dft->fcd', spec, spec.conj()) / T
[C, F, T_tgt] = size(spec_tgt);
R_tgt_prior   = zeros(F, C, C, 'like', 1+1i);
R_noise_prior = zeros(F, C, C, 'like', 1+1i);

for f = 1:F
    St = squeeze(spec_tgt(:, f, :));          % (C, T)
    Sn = squeeze(spec_noise(:, f, :));
    R_tgt_prior(f, :, :)   = (St * St') / T_tgt;
    R_noise_prior(f, :, :) = (Sn * Sn') / size(Sn, 2);
end

fprintf('Prior SCMs computed.  F=%d, C=%d\n', F, C);

% ---- load mixture to enhance ----
[s_mix, ~] = audioread(fullfile(audio_dir, 'mixture1.wav'));
fprintf('mixture1         : %d samples × %d ch\n', size(s_mix,1), size(s_mix,2));

[spec_mix, ~, ~] = stft_multichannel(s_mix, fs, window, noverlap, nfft);
[~, F, T_mix]    = size(spec_mix);

% ---- initialise CGMM-MVDR (identical hyperparameters to Python) ----
cgmm = OnlineCGMMMVDR( ...
    10, 10, ...                          % ny_k, ny_n
    R_tgt_prior, R_noise_prior, ...
    C, F, ...
    160, 80);                            % beg_noise ≈ 2 s, end_noise ≈ 1 s

% ---- online frame-by-frame processing ----
fprintf('Processing %d frames online...\n', T_mix);
spec_enh = zeros(2, F, T_mix, 'like', 1+1i);
masks    = zeros(T_mix, F);

for t = 1:T_mix
    y_t = spec_mix(:, :, t);            % (C, F)
    [s_out, mask_t] = cgmm.step(y_t, t, T_mix);
    spec_enh(1, :, t) = s_out{1};
    spec_enh(2, :, t) = s_out{2};
    masks(t, :)       = mask_t';
end
fprintf('Done.\n');

% ---- reconstruct time-domain output ----
s_enh1 = istft(squeeze(spec_enh(1,:,:)), fs, ...
    'Window', window, 'OverlapLength', noverlap, 'FFTLength', nfft, ...
    'FrequencyRange', 'onesided');
s_enh2 = istft(squeeze(spec_enh(2,:,:)), fs, ...
    'Window', window, 'OverlapLength', noverlap, 'FFTLength', nfft, ...
    'FrequencyRange', 'onesided');

% trim to input length
N = size(s_mix, 1);
s_enh1 = real(s_enh1(1:min(end,N)));
s_enh2 = real(s_enh2(1:min(end,N)));

% ---- power comparison: before normalisation ----
pwr_input = zeros(F, 1);
pwr_enh1  = zeros(F, 1);
pwr_enh2  = zeros(F, 1);
for f = 1:F
    pwr_input(f) = mean(abs(squeeze(spec_mix(1, f, :))).^2);
    pwr_enh1(f)  = mean(abs(squeeze(spec_enh(1, f, :))).^2);
    pwr_enh2(f)  = mean(abs(squeeze(spec_enh(2, f, :))).^2);
end

figure('Name','Part1 — Power before normalisation','Position',[50 50 900 500]);
subplot(2,1,1);
plot(freqs, 10*log10(pwr_input+eps), 'k', 'LineWidth',1.5, 'DisplayName','Input ch1'); hold on;
plot(freqs, 10*log10(pwr_enh1+eps),  'b', 'LineWidth',1.5, 'DisplayName','Enhanced ref-1');
plot(freqs, 10*log10(pwr_enh2+eps),  'r--','LineWidth',1.5,'DisplayName','Enhanced ref-2');
hold off; grid on; legend('Location','best');
xlabel('Frequency [Hz]'); ylabel('Power [dBfs²]');
title('Part 1 – Power spectrum comparison (before normalisation)');

subplot(2,1,2);
imagesc(1:T_mix, freqs, masks'); axis xy; colorbar; colormap(gca,'hot'); clim([0 1]);
xlabel('Frame'); ylabel('Frequency [Hz]'); title('Estimated target mask \lambda_{kn}');
drawnow;

% ---- save output ----
out_path1 = fullfile(audio_dir, 'mixture1_enhanced_ref1.wav');
out_path2 = fullfile(audio_dir, 'mixture1_enhanced_ref2.wav');
audiowrite(out_path1, s_enh1 / max(abs(s_enh1) + eps), fs);
audiowrite(out_path2, s_enh2 / max(abs(s_enh2) + eps), fs);
fprintf('Saved: %s\n', out_path1);
fprintf('Saved: %s\n', out_path2);


%% ================================================================
%% PART 2 — 4×4 array, 9 focus points, two sources
%% ================================================================
fprintf('\n=== Part 2: 4×4 array, 9 focus points, two sources ===\n');

% ---- microphone array geometry: 4×4 grid ----
spacing  = 0.05;          % 5 cm between adjacent mics
[gx, gy] = meshgrid(0:3, 0:3);
mic_pos  = [gx(:)*spacing, gy(:)*spacing, zeros(16,1)];   % (16, 3)
fprintf('Mic array: %d microphones, %.0f cm spacing\n', ...
    size(mic_pos,1), spacing*100);

% ---- two sources ----
src_right = [0.0,  2.0,  1.5];   % right-centre (y>0)
src_left  = [0.0, -2.0,  1.5];   % left-centre  (y<0)

% ---- 9 focus points: 3×3 grid at 2 m range ----
angles_az = linspace(-60, 60, 3);   % azimuth  degrees
angles_el = linspace(-30, 30, 3);   % elevation degrees
[AZ, EL]  = meshgrid(angles_az, angles_el);
AZ = AZ(:); EL = EL(:);

% Convert spherical (az, el) to Cartesian at range r = 2 m, centred on array.
% Standard convention:  x = r*cos(el)*cos(az)
%                       y = r*cos(el)*sin(az)
%                       z = r*sin(el)
array_centre = mean(mic_pos, 1);
r = 2.0;
focus_points = [r*cosd(EL).*cosd(AZ), r*cosd(EL).*sind(AZ), r*sind(EL)];
focus_points = focus_points + repmat(array_centre, 9, 1);

fprintf('Focus points:\n');
for k = 1:9
    fprintf('  fp%d: [%.2f, %.2f, %.2f] m   (az=%.0f°, el=%.0f°)\n', ...
        k, focus_points(k,1), focus_points(k,2), focus_points(k,3), ...
        AZ(k), EL(k));
end

% ---- simulate signals: delay-and-sum model ----
% Use mixture2_target as the target signal, mixture2_noise as noise
N_samp = min(size(s_tgt,1), size(s_noise,1));
c_sound = 343;            % speed of sound [m/s]

fprintf('Simulating 16-channel mixture (N=%d samples)...\n', N_samp);
data16 = simulate_array(s_tgt(1:N_samp,1), s_noise(1:N_samp,1), ...
    mic_pos, src_right, src_left, fs, c_sound);

% ---- run beamformer for each of the 9 focus points ----
pwr_at_fp = zeros(9, 1);    % total output power per focus point

fprintf('\nBeamforming on 9 focus points:\n');
for k = 1:9
    fp = focus_points(k, :);
    fprintf('  Focus point %d/9: [%.2f, %.2f, %.2f] m  ...', k, fp(1),fp(2),fp(3));

    [s_target_k, pd_k] = cgmm_mvdr_beamform( ...
        data16, mic_pos, fp, fs, c_sound, ...
        'nperseg', 200, 'noverlap', 100, ...
        'ny_k', 10, 'ny_n', 10);

    pwr_at_fp(k) = mean(s_target_k.^2);
    fprintf(' output power = %.4e\n', pwr_at_fp(k));

    % Save the enhanced audio
    if max(abs(s_target_k)) > eps
        audiowrite(fullfile(audio_dir, sprintf('fp%d_enhanced.wav', k)), ...
            s_target_k / max(abs(s_target_k)+eps), fs);
    end
end

% ---- final power comparison across focus points ----
figure('Name','Part 2 — Power across 9 focus points','Position',[100 100 800 400]);
bar(1:9, 10*log10(pwr_at_fp + eps));
hold on;
xlabel('Focus point index');
ylabel('Output power [dBfs²]');
title('CGMM-MVDR output power at each focus point (4×4 array, 9 focus points)');
xticks(1:9);
xticklabels(arrayfun(@(k) sprintf('fp%d\naz%+.0f° el%+.0f°', k, AZ(k), EL(k)), 1:9, ...
    'UniformOutput', false));
grid on;
drawnow;

fprintf('\nFocus point with highest power: fp%d (az=%.0f°, el=%.0f°)\n', ...
    find(pwr_at_fp == max(pwr_at_fp)), ...
    AZ(pwr_at_fp == max(pwr_at_fp)), ...
    EL(pwr_at_fp == max(pwr_at_fp)));

fprintf('\nAll done.\n');


%% ================================================================
%% Helper functions
%% ================================================================

function [spec, freqs, t] = stft_multichannel(data, fs, window, noverlap, nfft)
%STFT_MULTICHANNEL  Compute STFT for all channels.
%   data : (N_samples, N_ch)
%   spec : (N_ch, F, T)
    [N_samp, N_ch] = size(data);
    [S1, freqs, t] = stft(data(:,1), fs, ...
        'Window', window, 'OverlapLength', noverlap, 'FFTLength', nfft, ...
        'FrequencyRange', 'onesided');
    [F, T] = size(S1);
    spec = zeros(N_ch, F, T, 'like', 1+1i);
    spec(1,:,:) = S1;
    for ch = 2:N_ch
        spec(ch,:,:) = stft(data(:,ch), fs, ...
            'Window', window, 'OverlapLength', noverlap, 'FFTLength', nfft, ...
            'FrequencyRange', 'onesided');
    end
end


function data_mc = simulate_array(s_target, s_noise, mic_pos, src_target, ...
                                   src_noise, fs, c)
%SIMULATE_ARRAY  Simulate multi-channel audio with two point sources.
%
%   Each source signal is delayed to each microphone using integer-sample
%   rounding (simple far-field approximation).
%
%   s_target, s_noise : (N_samples,1) single-channel source signals
%   mic_pos           : (N_mics, 3)
%   src_target/noise  : (1, 3) source positions [m]
%   Returns data_mc   : (N_samples, N_mics)

    N     = numel(s_target);
    N_mic = size(mic_pos, 1);
    data_mc = zeros(N, N_mic);

    for m = 1:N_mic
        % delay from target source
        d_t   = norm(mic_pos(m,:) - src_target) / c;
        delay_t = round(d_t * fs);

        % delay from noise source
        d_n   = norm(mic_pos(m,:) - src_noise) / c;
        delay_n = round(d_n * fs);

        % apply integer delays
        sig_t = apply_delay(s_target, delay_t, N);
        sig_n = apply_delay(s_noise,  delay_n,  N);

        data_mc(:, m) = sig_t + 0.3 * sig_n;    % SNR ~10 dB
    end
end


function out = apply_delay(sig, delay, N)
%APPLY_DELAY  Circular shift then trim/pad to length N.
    out = circshift(sig(:), delay);
    if numel(out) < N
        out = [out; zeros(N - numel(out), 1)];
    else
        out = out(1:N);
    end
end
