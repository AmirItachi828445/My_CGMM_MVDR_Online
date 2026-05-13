function [R_target, R_noise] = compute_steering_covariance(mic_pos, focus_point, freqs, c)
%COMPUTE_STEERING_COVARIANCE  Build prior SCMs from array geometry.
%
%   Given microphone positions and a target focus point, constructs:
%     - R_target : rank-1 covariance from the steering vector toward focus_point
%     - R_noise  : scaled identity (isotropic / diffuse noise model)
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

% Propagation delay from focus_point to each microphone (near-field model)
% tau_i = ||mic_pos_i - focus_point|| / c
focus_point = focus_point(:)';                       % ensure row vector (1,3)
diff        = mic_pos - repmat(focus_point, N_mics, 1);   % (N_mics, 3)
delays      = sqrt(sum(diff.^2, 2)) / c;              % (N_mics, 1)

R_target = zeros(F, N_mics, N_mics, 'like', 1+1i);
R_noise  = zeros(F, N_mics, N_mics, 'like', 1+1i);

for f = 1:F
    % Steering vector  a(f,i) = exp(-j*2*pi*freq*tau_i)
    a  = exp(-1j * 2 * pi * freqs(f) * delays);      % (N_mics, 1)

    % Normalise so that trace(R_target) = N_mics  (matches identity noise trace)
    a  = a * sqrt(N_mics) / norm(a);

    % Rank-1 target SCM
    R_target(f, :, :) = a * a';

    % Identity noise SCM (diffuse noise)
    R_noise(f, :, :)  = eye(N_mics);
end
end
