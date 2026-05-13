function [R_target, R_noise] = compute_steering_covariance(mic_pos, focus_point, freqs, c)
%COMPUTE_STEERING_COVARIANCE  Build prior SCMs from array geometry.
%
%   Given microphone positions and a target focus point, constructs:
%     - R_target : regularised rank-1 covariance derived from the steering
%                  vector toward focus_point (near-field model).
%     - R_noise  : scaled identity (isotropic / diffuse noise model).
%
%   The target SCM is deliberately regularised:
%     R_target(f) = alpha * d(f)*d(f)^H / ||d(f)||^2  +  (1-alpha) * I / N
%   A purely rank-1 matrix is singular; applying this regularisation keeps
%   R_target positive-definite and prevents Cholesky failures inside the
%   complex Gaussian PDF evaluator — which was the root cause of the mask
%   collapse (all masks ≈ 0.5, no spatial discrimination).
%
%   Relative delays (Δτ_i = (||mic_i−fp|| − ||mic_ref−fp||) / c) are used
%   instead of absolute delays; only phase differences affect beamforming.
%
%   Usage:
%     [R_target, R_noise] = compute_steering_covariance(mic_pos, focus_point, freqs, c)
%
%   Inputs:
%     mic_pos     - microphone positions, (N_mics, 3) [metres]
%     focus_point - target position,      (1, 3)      [metres]
%     freqs       - STFT frequency bins,  (F, 1)      [Hz]
%     c           - speed of sound                    [m/s]  (e.g. 343)
%
%   Outputs:
%     R_target - (F, N_mics, N_mics) complex  – target prior SCM
%     R_noise  - (F, N_mics, N_mics) complex  – noise  prior SCM

N_mics = size(mic_pos, 1);
F      = numel(freqs);

% Regularisation weight: 0 = flat (identity) prior, 1 = pure rank-1 (singular).
% 0.9 gives a strong spatial prior while keeping R_target positive-definite.
alpha  = 0.9;

% Propagation delay from focus_point to each microphone (near-field model).
% Use delays *relative to microphone 1* so that only phase differences
% between microphones enter the steering vector (standard beamforming
% convention; a global phase offset is irrelevant for MVDR).
focus_point = focus_point(:).';
dist_abs    = sqrt(sum((mic_pos - repmat(focus_point, N_mics, 1)).^2, 2));  % (N_mics,1)
delays      = (dist_abs - dist_abs(1)) / c;                                  % (N_mics,1) relative

R_target = zeros(F, N_mics, N_mics, 'like', 1+1i);
R_noise  = zeros(F, N_mics, N_mics, 'like', 1+1i);

for f = 1:F
    % Steering vector  a(f,i) = exp(-j*2*pi*freq*Δτ_i)
    a = exp(-1j * 2 * pi * freqs(f) * delays);          % (N_mics, 1)

    % Normalised rank-1 component (trace = 1, so R_target trace = N for
    % consistency with the identity noise prior whose trace = N_mics).
    ddn = (a * a') / real(a' * a);                      % trace(ddn) = 1

    % Regularised target SCM: combine rank-1 spatial prior with isotropic
    % background, scaled so trace(R_target) = N_mics for both terms.
    R_target(f, :, :) = alpha * ddn * N_mics ...
                      + (1 - alpha) * eye(N_mics);

    % Identity noise SCM (diffuse noise model, trace = N_mics)
    R_noise(f, :, :)  = eye(N_mics);
end
end
