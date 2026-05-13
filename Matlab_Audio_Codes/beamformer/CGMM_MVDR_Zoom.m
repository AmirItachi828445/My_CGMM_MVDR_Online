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
%   Key algorithmic and performance improvements:
%     • Prior trace correction: both R_target and R_noise have trace = N_mics,
%       preventing the mask-collapse that was caused by the original
%       R_target having trace ≈ 1 while R_noise had trace = N_mics.
%     • Precomputed R^{-1} and log|det(R)|: eliminates per-frame Cholesky
%       calls from the inner loop (major speedup for real-time use).
%     • Analytical log-likelihood: log p = -C·(1+log π) - C·log φ - log|det R|
%       (exact for the zero-mean complex Gaussian; reduces to simple
%       vector operations per frame).
%     • Sherman-Morrison rank-1 inverse update: O(C²) update per bin instead
%       of O(C³) matrix inversion after each R update.
%     • Lambda_prev saved before accumulation (fixes shared-reference bug).
%     • Final MVDR from steering vector + noise SCM: guaranteed unit gain.
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
%   d(f,i) = exp(−j·2π·f·Δτ_i)   where Δτ_i = (‖mic_i−fp‖ − ‖mic_1−fp‖)/c
focus_point = focus_point(:).';
dist_abs    = sqrt(sum((mic_pos - focus_point).^2, 2));   % (N_mics,1) [m]
delays_rel  = (dist_abs - dist_abs(1)) / c;               % (N_mics,1) [s]

D = exp(-1j * 2 * pi * freqs(:) * delays_rel(:).');       % (F, N_mics)

%% ── Geometry-guided prior spatial covariance matrices ──────────────────
%   Target prior: regularised rank-1 from the steering vector.
%     R_target(f) = α · N · (d·d^H/‖d‖²) + (1−α) · I
%   IMPORTANT: trace(R_target) = α·N + (1−α)·N = N = trace(R_noise).
%   Matching traces is critical: when trace(R_target) ≪ trace(R_noise)
%   the phi estimates become asymmetric and masks collapse to 0 (all noise).
%
%   Noise prior: identity (isotropic / diffuse field model), trace = N_mics.

alpha_prior    = 0.95;    % spatial prior strength  (0 = flat, 1 = rank-1)
R_target_prior = zeros(F, N_mics, N_mics, 'like', 1+1i);
R_noise_prior  = zeros(F, N_mics, N_mics, 'like', 1+1i);

for f = 1:F
    d_f = D(f, :).';                               % (N_mics,1)
    ddn = (d_f * d_f') / real(d_f' * d_f);        % rank-1, trace = 1
    % Correct scaling: trace = alpha_prior*N_mics + (1-alpha_prior)*N_mics = N_mics
    R_target_prior(f, :, :) = alpha_prior * ddn * N_mics ...
                             + (1 - alpha_prior) * eye(N_mics);
    R_noise_prior(f, :, :)  = eye(N_mics);
end

%% ── CGMM hyperparameters ────────────────────────────────────────────────
ny_k      = 10;
ny_n      = 10;
%   10 % of frames forced as noise-only (gives a clean noise reference
%   before the algorithm has learned the spatial distributions).
beg_noise = max(1, floor(0.10 * T_frames));  % 10% noise-only frames
end_noise = 0;

%% ── Initialise CGMM state ───────────────────────────────────────────────
R_kn      = R_target_prior;
R_n       = R_noise_prior;
Lambda_kn = zeros(F, 1);
Lambda_n  = zeros(F, 1);
alpha_kn  = 0.5 * ones(F, 1);
alpha_n   = 0.5 * ones(F, 1);

%   Precompute R^{-1} and log|det(R)| for both components.
%   These are updated analytically after every R update (Sherman-Morrison),
%   so no Cholesky factorisation is required inside the frame loop.
[R_kn_inv, ld_kn] = cgmm_precompute_inv(R_kn, N_mics, F);
[R_n_inv,  ld_n]  = cgmm_precompute_inv(R_n,  N_mics, F);

%% ── Accumulation matrices for MVDR ─────────────────────────────────────
Phi_n = zeros(F, N_mics, N_mics, 'like', 1+1i);

%% ── Online CGMM loop — vectorised over F ────────────────────────────────
%   Analytical log-likelihood for the zero-mean complex Gaussian:
%     log p(f) = −C·(1+log π) − C·log φ(f) − log|det R(f)|
%   where φ(f) = (1/C)·y_f^H·R^{-1}(f)·y_f  (ML estimate per frame).
%   Since Σ = φ·R, the quadratic y^H·Σ^{-1}·y = y^H·R^{-1}·y / φ = C
%   (by the ML definition of φ), so no Cholesky is needed per frame.
C_lnpi = N_mics * (1 + log(pi));   % constant part of log-likelihood

for t = 1:T_frames
    y_t = Y(:, :, t);              % (N_mics, F) — no squeeze needed

    % ── phi and R_inv·y for all F (vectorised) ──────────────────────────
    [phi_kn, u_kn] = cgmm_phi_and_u(y_t, R_kn_inv, N_mics, F);
    [phi_n,  u_n ] = cgmm_phi_and_u(y_t, R_n_inv,  N_mics, F);

    % ── analytical log-likelihoods ──────────────────────────────────────
    log_p_kn = -C_lnpi - N_mics * log(max(phi_kn, 1e-300)) - ld_kn;
    log_p_n  = -C_lnpi - N_mics * log(max(phi_n,  1e-300)) - ld_n;

    % ── masks (log-sum-exp stable) ──────────────────────────────────────
    log_lk = log(alpha_kn + 1e-300) + log_p_kn;
    log_ln = log(alpha_n  + 1e-300) + log_p_n;
    mx     = max(log_lk, log_ln);
    lk     = exp(log_lk - mx);
    ln     = exp(log_ln - mx);
    tot    = lk + ln + 1e-15;
    lam_kn = lk ./ tot;
    lam_n  = ln ./ tot;

    is_noise = (t <= beg_noise) || (t > T_frames - end_noise);
    if is_noise
        lam_kn = zeros(F, 1);
        lam_n  = ones(F, 1);
    end

    % ── save Lambda BEFORE accumulation ─────────────────────────────────
    Lambda_kn_prev = Lambda_kn;
    Lambda_n_prev  = Lambda_n;
    Lambda_kn      = Lambda_kn + lam_kn;
    Lambda_n       = Lambda_n  + lam_n;

    % ── R update + Sherman-Morrison inverse update ───────────────────────
    %   R_new(f) = a(f)·R(f) + b(f)·y_f·y_f^H
    %   R_new^{-1}  updated via matrix inversion lemma (O(C²) per bin).
    %   log|det(R_new)| = C·log a + log|det R| + log(1 + (b/a)·y^H·R^{-1}·y)
    [R_kn, R_kn_inv, ld_kn] = cgmm_R_sm_update( ...
        y_t, R_kn, R_kn_inv, ld_kn, u_kn, phi_kn, ...
        lam_kn, Lambda_kn_prev, Lambda_kn, ny_k, N_mics, F);

    [R_n, R_n_inv, ld_n] = cgmm_R_sm_update( ...
        y_t, R_n, R_n_inv, ld_n, u_n, phi_n, ...
        lam_n, Lambda_n_prev, Lambda_n, ny_n, N_mics, F);

    % ── accumulate noise SCM (vectorised outer products) ─────────────────
    for m1 = 1:N_mics
        for m2 = 1:N_mics
            Phi_n(:, m1, m2) = Phi_n(:, m1, m2) ...
                + lam_n .* conj(y_t(m1, :)).' .* y_t(m2, :).';
        end
    end
end

%% ── MVDR weight computation (steering-vector constrained) ───────────────
%   w(f) = Φ_n^{−1}(f)·d(f) / [d(f)^H·Φ_n^{−1}(f)·d(f)]
%   Diagonal loading δ·tr(Φ_n)·I is added to Φ_n for numerical stability.

reg_diag = 1e-3;
W        = zeros(F, N_mics, 'like', 1+1i);

for f = 1:F
    Pn_f  = Phi_n(f, :, :);
    Pn_f  = reshape(Pn_f, N_mics, N_mics);

    tr_Pn = max(real(trace(Pn_f)), 1e-20);
    Pn_f  = Pn_f + reg_diag * tr_Pn * eye(N_mics);

    d_f   = D(f, :).';

    try
        Pn_inv_d = Pn_f \ d_f;
    catch
        Pn_inv_d = pinv(Pn_f) * d_f;
    end

    denom = real(d_f' * Pn_inv_d);
    if denom < 1e-20
        W(f, :) = d_f.' / N_mics;
    else
        W(f, :) = (Pn_inv_d / denom).';
    end
end

%% ── Apply beamformer to all STFT frames ─────────────────────────────────
%   Vectorised: Y_out(f,t) = conj(W(f,:)) * Y(:,f,t)  for all f,t.
%   Reshape Y to (F, N_mics, T) for efficient contraction.
Y_perm = permute(Y, [2, 1, 3]);   % (F, N_mics, T_frames)
W_conj = conj(W);                  % (F, N_mics)
Y_out  = zeros(F, T_frames, 'like', 1+1i);
for f = 1:F
    Y_out(f, :) = W_conj(f, :) * reshape(Y_perm(f, :, :), N_mics, T_frames);
end

%% ── ISTFT → time-domain output ──────────────────────────────────────────
s_target = cgmm_istft(Y_out, N_samples, win, nperseg, noverlap);

end   % ── end of CGMM_MVDR_Zoom ─────────────────────────────────────────


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function [Y, freqs, T_frames] = cgmm_stft_all(data, fs, win, nperseg, noverlap)
%CGMM_STFT_ALL  Short-time Fourier transform for every channel.
%   Y : (N_mics, F, T_frames)  one-sided complex spectrum.
    [~, N_ch] = size(data);
    hop    = nperseg - noverlap;
    pad    = floor(nperseg / 2);
    N_freq = floor(nperseg / 2) + 1;
    freqs  = (0:N_freq-1).' * fs / nperseg;
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
            idx       = (m-1)*hop + (1:nperseg);
            frame     = xp(idx) .* win;
            X         = fft(frame, nperseg);
            Y(ch,:,m) = X(1:N_freq);
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
    nz = wsum > 1e-12;
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


function [R_inv, log_det] = cgmm_precompute_inv(R, C, F)
%CGMM_PRECOMPUTE_INV  Compute R^{-1} and log|det R| for all F bins via Cholesky.
%   Called once at initialisation and when a full re-inversion is needed.
    R_inv   = zeros(F, C, C, 'like', R);
    log_det = zeros(F, 1);
    I_reg   = 1e-8 * eye(C);
    for f = 1:F
        Rf = reshape(R(f,:,:), C, C) + I_reg;
        try
            L = chol(Rf, 'lower');
            Linv       = L \ eye(C);
            R_inv(f,:,:) = Linv' * Linv;
            log_det(f)   = 2 * sum(log(abs(diag(L))));
        catch
            Rf2 = Rf + 1e-6 * eye(C);
            R_inv(f,:,:) = inv(Rf2);
            log_det(f)   = log(abs(det(Rf2)));
        end
    end
end


function [phi, u_mat] = cgmm_phi_and_u(y, R_inv, C, F)
%CGMM_PHI_AND_U  Compute phi(f) = (1/C)·y_f^H·R^{-1}·y_f and u = R^{-1}·y_f.
%   Both outputs reuse the same matrix-vector product, avoiding redundant work.
%   u_mat : (C, F)  — R^{-1}·y for each frequency bin.
    phi   = zeros(F, 1);
    u_mat = zeros(C, F, 'like', R_inv);
    for f = 1:F
        yf = y(:, f);
        Ri = reshape(R_inv(f,:,:), C, C);
        u  = Ri * yf;
        phi(f)   = max(real(yf' * u) / C, 1e-12);
        u_mat(:,f) = u;
    end
end


function [R_new, R_inv_new, ld_new] = cgmm_R_sm_update( ...
    y, R, R_inv, ld, u_mat, phi, lam, Lprev, L, ny, C, F)
%CGMM_R_SM_UPDATE  Rank-1 R update (Eq. 33 [2]) + Sherman-Morrison inverse update.
%
%   R_new(f) = a(f)·R(f) + b(f)·y_f·y_f^H
%
%   Sherman-Morrison (matrix inversion lemma):
%     (a·R + b·y·y^H)^{-1}
%       = (1/a)·[ R^{-1} − (b/a)·(R^{-1}·y)·(y^H·R^{-1}) / (1+(b/a)·y^H·R^{-1}·y) ]
%
%   log|det R_new| update (matrix determinant lemma):
%     = C·log a + log|det R| + log(1 + (b/a)·y^H·R^{-1}·y)
%
%   Both updates reuse u = R^{-1}·y already computed for phi.
    half      = (ny + C + 1) / 2;
    R_new     = zeros(F, C, C, 'like', R);
    R_inv_new = zeros(F, C, C, 'like', R_inv);
    ld_new    = zeros(F, 1);

    for f = 1:F
        a  = max((Lprev(f) + half) / max(L(f) + half, 1e-30), 1e-30);
        b  = lam(f) / max(L(f) + half, 1e-30) / max(phi(f), 1e-30);

        yf  = y(:, f);
        Rf  = reshape(R(f,:,:), C, C);
        Ri  = reshape(R_inv(f,:,:), C, C);
        u   = u_mat(:, f);   % R^{-1}·y (already computed)

        yyH = yf * yf';
        R_new(f,:,:) = a * Rf + b * yyH;

        % Sherman-Morrison rank-1 inverse update
        ba       = b / a;
        s        = real(yf' * u);    % y^H·R^{-1}·y  (numerically = phi·C since phi = s/C)
        denom_sm = 1 + ba * s;

        if denom_sm > 1e-10
            R_inv_new(f,:,:) = (1/a) * (Ri - (ba / denom_sm) * (u * u'));
            ld_new(f)        = C * log(a) + ld(f) + log(denom_sm);
        else
            % Degenerate case: full re-inversion (rare)
            Rn_f = reshape(R_new(f,:,:), C, C) + 1e-8 * eye(C);
            try
                Ln = chol(Rn_f, 'lower');
                Linv = Ln \ eye(C);
                R_inv_new(f,:,:) = Linv' * Linv;
                ld_new(f)        = 2 * sum(log(abs(diag(Ln))));
            catch
                R_inv_new(f,:,:) = inv(Rn_f);
                ld_new(f)        = log(abs(det(Rn_f)));
            end
        end
    end
end
