function s_target = CGMM_MVDR_Zoom(data, mic_pos, focus_point, fs, c)
%CGMM_MVDR_ZOOM  Geometry-guided CGMM-MVDR spatial zoom beamformer.
%
%   s_target = CGMM_MVDR_ZOOM(data, mic_pos, focus_point, fs, c)
%
%   Focuses on sound originating from focus_point using a two-stage approach:
%     1. CGMM mask estimation – identifies which (f,t) TF bins are target-
%        dominated, using a geometry-informed spatial covariance prior seeded
%        from the steering vector toward focus_point.
%     2. Steering-vector MVDR – applies a distortionless-response filter that
%        preserves sound arriving from focus_point while suppressing other
%        directions, using the mask-weighted noise covariance.
%
%   Key algorithmic improvements over a naive CGMM-MVDR:
%     • Regularised rank-1 target prior: prevents the singular-matrix
%       failure that caused masks to collapse to 0.5 and produced no
%       spatial discrimination.
%     • Lambda_prev is saved *before* the mask accumulation step
%       (fixing the Python shared-reference alias bug that froze R at the
%       prior and stopped the CGMM from learning).
%     • Final MVDR weights are computed explicitly from the steering vector
%       and the mask-weighted noise covariance — guaranteeing unit gain
%       toward focus_point regardless of mask quality.
%     • Diagonal loading based on matrix trace for numerical stability.
%
%   Inputs:
%     data        (N_samples × N_mics)  multichannel time-domain audio
%     mic_pos     (N_mics    × 3)       microphone positions  [m]
%     focus_point (1         × 3)       target zoom point     [m]
%     fs          scalar                sampling frequency    [Hz]
%     c           scalar                speed of sound        [m/s]
%
%   Output:
%     s_target    (N_samples × 1)       spatially zoomed output signal
%
%   References:
%     [1] T. Higuchi et al., "Frame-by-Frame Closed-Form Update for
%         Mask-Based Adaptive MVDR Beamforming," ICASSP 2018.
%     [2] T. Higuchi et al., "Online MVDR Beamformer Based on CGMM With
%         Spatial Prior for Noise Robust ASR," IEEE/ACM TASLP 2017.

if nargin < 5 || isempty(c), c = 343; end

%% ── STFT parameters ────────────────────────────────────────────────────
frame_ms      = 32;                          % window length [ms]
overlap_ratio = 0.75;                        % overlap fraction
nperseg       = round(fs * frame_ms / 1e3);  % samples per frame
noverlap      = round(nperseg * overlap_ratio);
win           = hann(nperseg, 'periodic');   % analysis/synthesis window

[N_samples, N_mics] = size(data);

%% ── Multi-channel STFT  →  Y : (N_mics, F, T) ─────────────────────────
[Y, freqs, T_frames] = cgmm_stft_all(data, fs, win, nperseg, noverlap);
F = numel(freqs);

%% ── Steering vector (near-field, relative delays) ─────────────────────
%   Phase reference: microphone 1.
%   d(f,i) = exp(−j·2π·f·Δτ_i)   where Δτ_i = (||mic_i−fp|| − ||mic_1−fp||)/c
focus_point = focus_point(:).';
dist_abs    = sqrt(sum((mic_pos - focus_point).^2, 2));  % (N_mics,1) [m]
delays_rel  = (dist_abs - dist_abs(1)) / c;              % (N_mics,1) [s]

% D : (F, N_mics) — steering matrix
D = exp(-1j * 2 * pi * freqs(:) * delays_rel(:).');

%% ── Geometry-guided prior spatial covariance matrices ──────────────────
%   Target prior : regularised rank-1 formed from the steering vector.
%     R_target(f) = α · (d·d^H) / ||d||² + (1−α) · I/N_mics
%   A purely rank-1 matrix is singular; the (1−α)·I term keeps the matrix
%   positive-definite and prevents Cholesky failures in the Gaussian PDF.
%
%   Noise prior  : scaled identity  (isotropic / diffuse field model).

alpha_prior    = 0.9;     % strength of spatial prior  (0 = flat, 1 = rank-1)
R_target_prior = zeros(F, N_mics, N_mics, 'like', 1+1i);
R_noise_prior  = zeros(F, N_mics, N_mics, 'like', 1+1i);

for f = 1:F
    d_f  = D(f, :).';                              % (N_mics,1)
    ddn  = (d_f * d_f') / real(d_f' * d_f);       % normalised rank-1, trace = 1
    R_target_prior(f, :, :) = alpha_prior * ddn ...
        + (1 - alpha_prior) * eye(N_mics) / N_mics;
    R_noise_prior(f, :, :)  = eye(N_mics);
end

%% ── CGMM hyperparameters ────────────────────────────────────────────────
ny_k      = 10;
ny_n      = 10;
%   Force the first 5 % of frames as noise-only to give the model a clean
%   noise reference before speech / target signal begins.
beg_noise = max(1, floor(0.05 * T_frames));
end_noise = 0;

%% ── Online CGMM — mask estimation + covariance accumulation ────────────
%   We accumulate two mask-weighted spatial covariance matrices (SCMs):
%     Phi_s(f) = Σ_t λ_kn(f,t) · y(f,t)·y(f,t)^H   (target  SCM)
%     Phi_n(f) = Σ_t λ_n(f,t)  · y(f,t)·y(f,t)^H   (noise   SCM)
%
%   Phi_n is then used to form the steering-vector MVDR weights.

Phi_s = zeros(F, N_mics, N_mics, 'like', 1+1i);
Phi_n = zeros(F, N_mics, N_mics, 'like', 1+1i);

% Initialise CGMM state
R_kn       = R_target_prior;          % (F, N_mics, N_mics)
R_n        = R_noise_prior;
phi_kn     = [];                       % initialized on first frame
phi_n      = [];
Lambda_kn  = zeros(F, 1);
Lambda_n   = zeros(F, 1);
alpha_kn   = 0.5 * ones(F, 1);        % mixture weights (kept equal; priors
alpha_n    = 0.5 * ones(F, 1);        %   differentiate target vs noise)

for t = 1:T_frames
    y_t = squeeze(Y(:, :, t));         % (N_mics, F)

    % ── Initialise phi on the very first frame ──────────────────────────
    if isempty(phi_kn)
        phi_kn = cgmm_compute_phi(y_t, R_kn, N_mics, F);
        phi_n  = cgmm_compute_phi(y_t, R_n,  N_mics, F);
    end

    % ── CRITICAL FIX: save Lambda BEFORE accumulation ───────────────────
    %   The original Python code had a shared-reference alias bug where
    %   Lambda_prev pointed to the same array as Lambda, so after the
    %   in-place "+=" Lambda_prev == Lambda, giving nom==denom in update_R
    %   and zeroing the forgetting term (R never updates from the prior).
    %   Here we save a proper copy before calling the mask update.
    Lambda_kn_prev = Lambda_kn;
    Lambda_n_prev  = Lambda_n;

    % ── Mask update ─────────────────────────────────────────────────────
    is_noise = (t <= beg_noise) || (t > T_frames - end_noise);
    [lam_kn, lam_n] = cgmm_update_masks(y_t, R_kn, R_n, phi_kn, phi_n, ...
                                         alpha_kn, alpha_n, is_noise, N_mics, F);

    Lambda_kn = Lambda_kn + lam_kn;
    Lambda_n  = Lambda_n  + lam_n;

    % ── phi update ──────────────────────────────────────────────────────
    phi_kn = cgmm_compute_phi(y_t, R_kn, N_mics, F);
    phi_n  = cgmm_compute_phi(y_t, R_n,  N_mics, F);

    % ── R update  (Eq. 33 in [2]) ───────────────────────────────────────
    [R_kn, R_n] = cgmm_update_R(y_t, R_kn, R_n, lam_kn, lam_n, ...
                                 phi_kn, phi_n, ...
                                 Lambda_kn_prev, Lambda_n_prev, ...
                                 Lambda_kn, Lambda_n, ...
                                 ny_k, ny_n, N_mics, F);

    % ── Accumulate mask-weighted SCMs ───────────────────────────────────
    for f = 1:F
        yf  = y_t(:, f);
        yyH = yf * yf';
        Phi_s(f, :, :) = squeeze(Phi_s(f, :, :)) + lam_kn(f) * yyH;
        Phi_n(f, :, :) = squeeze(Phi_n(f, :, :)) + lam_n(f)  * yyH;
    end
end

%% ── MVDR weight computation (steering-vector constrained) ───────────────
%   w(f) = Φ_n^{−1}(f) · d(f)  /  [d(f)^H · Φ_n^{−1}(f) · d(f)]
%
%   This is the classical MVDR / Capon beamformer.  Using the steering
%   vector directly (rather than the principal eigenvector of Phi_s) gives
%   guaranteed unit gain toward focus_point regardless of mask quality, and
%   is the correct formulation for acoustic spatial zoom.
%
%   Diagonal loading: δ · tr(Φ_n) · I is added to Φ_n before inversion.
%   This is proportional to the matrix energy, providing scale-invariant
%   regularisation.  A value of 1e-3 is appropriate for speech-level inputs.

reg_diag = 1e-3;
W        = zeros(F, N_mics, 'like', 1+1i);

for f = 1:F
    Pn_f  = squeeze(Phi_n(f, :, :));

    % Diagonal loading
    tr_Pn = max(real(trace(Pn_f)), 1e-20);
    Pn_f  = Pn_f + reg_diag * tr_Pn * eye(N_mics);

    d_f   = D(f, :).';                          % (N_mics,1)

    try
        Pn_inv_d = Pn_f \ d_f;                  % (N_mics,1)
    catch
        Pn_inv_d = pinv(Pn_f) * d_f;
    end

    denom = real(d_f' * Pn_inv_d);              % real scalar for Hermitian Pn
    if denom < 1e-20
        % Degenerate fallback: delay-and-sum toward focus_point
        W(f, :) = d_f.' / N_mics;
    else
        W(f, :) = (Pn_inv_d / denom).';         % (1, N_mics)
    end
end

%% ── Apply beamformer to all STFT frames ─────────────────────────────────
Y_out = zeros(F, T_frames, 'like', 1+1i);

for t = 1:T_frames
    y_t = squeeze(Y(:, :, t));          % (N_mics, F)
    for f = 1:F
        % w^H · y  =  conj(w) · y  (dot product over microphones)
        Y_out(f, t) = conj(W(f, :)) * y_t(:, f);
    end
end

%% ── ISTFT → time-domain output ──────────────────────────────────────────
s_target = cgmm_istft(Y_out, N_samples, win, nperseg, noverlap);

end   % ── end of CGMM_MVDR_Zoom ─────────────────────────────────────────


% =========================================================================
%  LOCAL HELPER FUNCTIONS
%  These replicate and correct the OnlineCGMMMVDR / complex_gaussian logic
%  from the Matlab/ directory, all within one self-contained file.
% =========================================================================

function [Y, freqs, T_frames] = cgmm_stft_all(data, fs, win, nperseg, noverlap)
%CGMM_STFT_ALL  Short-time Fourier transform for every channel.
%   Y : (N_mics, F, T_frames)  one-sided complex spectrum.
    [~, N_ch] = size(data);
    hop    = nperseg - noverlap;
    pad    = floor(nperseg / 2);
    N_freq = floor(nperseg / 2) + 1;
    freqs  = (0:N_freq-1).' * fs / nperseg;   % frequency axis [Hz]
    % --- determine T_frames from padded length ---
    x0_pad = [zeros(pad,1); data(:,1); zeros(pad,1)];
    n_fr   = ceil((length(x0_pad) - nperseg) / hop) + 1;
    tgt    = (n_fr - 1) * hop + nperseg;
    if length(x0_pad) < tgt
        x0_pad = [x0_pad; zeros(tgt - length(x0_pad), 1)]; %#ok<NASGU>
    end
    T_frames = n_fr;
    Y = zeros(N_ch, N_freq, T_frames, 'like', 1+1i);
    for ch = 1:N_ch
        xp = [zeros(pad,1); data(:,ch); zeros(pad,1)];
        if length(xp) < tgt, xp = [xp; zeros(tgt - length(xp),1)]; end
        for m = 1:T_frames
            idx        = (m-1)*hop + (1:nperseg);
            frame      = xp(idx) .* win;
            X          = fft(frame, nperseg);
            Y(ch,:,m)  = X(1:N_freq);
        end
    end
end


function s = cgmm_istft(S, target_len, win, nperseg, noverlap)
%CGMM_ISTFT  Overlap-add inverse STFT.  S : (F, T_frames).
    [~, n_frames] = size(S);
    hop = nperseg - noverlap;
    if rem(nperseg, 2) == 0
        mirror = conj(S(end-1:-1:2, :));
    else
        mirror = conj(S(end:-1:2, :));
    end
    S_full  = [S; mirror];
    out_len = (n_frames - 1) * hop + nperseg;
    x_ola   = zeros(out_len, 1);
    wsum    = zeros(out_len, 1);
    for m = 1:n_frames
        idx        = (m-1)*hop + (1:nperseg);
        frame      = real(ifft(S_full(:,m), nperseg));
        frame      = frame(:) .* win;
        x_ola(idx) = x_ola(idx) + frame;
        wsum(idx)  = wsum(idx)  + win.^2;
    end
    nz       = wsum > 1e-12;
    x_ola(nz) = x_ola(nz) ./ wsum(nz);
    pad = floor(nperseg / 2);
    if length(x_ola) > 2*pad
        s = x_ola(pad+1 : end-pad);
    else
        s = x_ola;
    end
    if ~isempty(target_len)
        if length(s) > target_len
            s = s(1:target_len);
        elseif length(s) < target_len
            s = [s; zeros(target_len - length(s), 1)];
        end
    end
    s = real(s(:));
end


function phi = cgmm_compute_phi(y, R, C, F)
%CGMM_COMPUTE_PHI  Time-dependent variance, Eq.(20) in [2].
%   phi(f) = (1/C) · y_f^H · R_f^{-1} · y_f
    phi = zeros(F, 1);
    for f = 1:F
        yf   = y(:, f);
        Rf   = reshape(R(f,:,:), C, C);
        Rf   = Rf + 1e-10 * eye(C);    % minimal regularisation
        phi(f) = max(real(yf' * (Rf \ yf)) / C, 1e-12);
    end
end


function p = cgmm_complex_gaussian(y, R, phi, C, F)
%CGMM_COMPLEX_GAUSSIAN  Zero-mean complex Gaussian PDF, Eq.(15) in [2].
%   Computed in log-domain via Cholesky for numerical stability.
    p = zeros(F, 1);
    for f = 1:F
        Rf    = reshape(R(f,:,:), C, C);
        Sigma = phi(f) * Rf + 1e-10 * eye(C);
        yf    = y(:, f);
        try
            L        = chol(Sigma, 'lower');
            v        = L \ yf;
            log_quad = real(v' * v);
            log_det  = 2 * sum(log(abs(diag(L))));
            p(f)     = exp(-log_quad - C*log(pi) - log_det);
        catch
            % Fall back to LU if Cholesky fails (near-singular Sigma)
            try
                [~, U, ~] = lu(Sigma);
                log_quad  = real(yf' * (Sigma \ yf));
                log_det   = sum(log(abs(diag(U))));
                p(f)      = exp(-log_quad - C*log(pi) - log_det);
            catch
                p(f) = 1e-300;
            end
        end
    end
    p = max(real(p), 0);
end


function [lam_kn, lam_n] = cgmm_update_masks(y, R_kn, R_n, phi_kn, phi_n, ...
                                               alpha_kn, alpha_n, is_noise, C, F)
%CGMM_UPDATE_MASKS  Posterior mask, Eq.(19),(25) in [2].
%
%   lam_kn(f) = P(target | y_f)   —  target mask
%   lam_n(f)  = P(noise  | y_f)   —  noise  mask
    p_kn = cgmm_complex_gaussian(y, R_kn, phi_kn, C, F);
    p_n  = cgmm_complex_gaussian(y, R_n,  phi_n,  C, F);
    lk   = alpha_kn .* p_kn;
    ln   = alpha_n  .* p_n;
    tot  = lk + ln + 1e-15;
    lam_kn = lk ./ tot;
    lam_n  = ln ./ tot;
    if is_noise
        lam_kn = zeros(F, 1);
        lam_n  = ones(F, 1);
    end
end


function [R_kn_new, R_n_new] = cgmm_update_R(y, R_kn, R_n, ...
                                               lam_kn, lam_n, phi_kn, phi_n, ...
                                               Lkn_prev, Ln_prev, Lkn, Ln, ...
                                               ny_k, ny_n, C, F)
%CGMM_UPDATE_R  Normalized covariance update, Eq.(33) in [2].
%
%   R_new(f) = [nom/den] · R_old(f)
%            + [1/den]   · (λ(f)/φ(f)) · y_f · y_f^H
%
%   where  nom = Λ_prev(f) + (ν + C + 1)/2
%          den = Λ(f)      + (ν + C + 1)/2
    R_kn_new = zeros(F, C, C, 'like', 1+1i);
    R_n_new  = zeros(F, C, C, 'like', 1+1i);
    half_k   = (ny_k + C + 1) / 2;
    half_n   = (ny_n + C + 1) / 2;
    for f = 1:F
        yf  = y(:, f);
        yyH = yf * yf';
        % ── target ─────────────────────────────────────────────────────
        Rk_f   = reshape(R_kn(f,:,:), C, C);
        nom_k  = Lkn_prev(f) + half_k;
        den_k  = max(Lkn(f) + half_k, 1e-30);
        R_kn_new(f,:,:) = (nom_k / den_k) * Rk_f ...
                        + (lam_kn(f) / den_k / max(phi_kn(f), 1e-30)) * yyH;
        % ── noise ──────────────────────────────────────────────────────
        Rn_f   = reshape(R_n(f,:,:), C, C);
        nom_n  = Ln_prev(f) + half_n;
        den_n  = max(Ln(f) + half_n, 1e-30);
        R_n_new(f,:,:) = (nom_n / den_n) * Rn_f ...
                       + (lam_n(f) / den_n / max(phi_n(f), 1e-30)) * yyH;
    end
end
