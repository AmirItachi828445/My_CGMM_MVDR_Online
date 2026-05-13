classdef OnlineCGMMMVDR < handle
%ONLINECGMMMVDR Online CGMM-MVDR Beamformer (MATLAB port of cgmm_mvdr.py)
%
%   Line-by-line port of the Python OnlineCGMMMVDR class.
%   Notation: C = channels, F = frequency bins.
%
%   References:
%     [1] T. Higuchi et al., "Frame-by-Frame Closed-Form Update for
%         Mask-Based Adaptive MVDR Beamforming," ICASSP 2018.
%         doi:10.1109/ICASSP.2018.8461850
%     [2] T. Higuchi et al., "Online MVDR Beamformer Based on Complex
%         Gaussian Mixture Model With Spatial Prior for Noise Robust ASR,"
%         IEEE/ACM TASLP, vol. 25, no. 4, pp. 780-793, Apr. 2017.
%         doi:10.1109/TASLP.2017.2665341
%
%   Usage:
%     obj = OnlineCGMMMVDR(ny_k, ny_n, priorRk, priorRn, C, F)
%     obj = OnlineCGMMMVDR(ny_k, ny_n, priorRk, priorRn, C, F, beg_noise, end_noise)
%     [s_out, lambda_kn] = obj.step(y, l, T)

    % ------------------------------------------------------------------ %
    properties
        ny_k          % CGMM hyperparameter for target component
        ny_n          % CGMM hyperparameter for noise component
        C             % number of channels
        F             % number of frequency bins
        beg_noise     % initial frames that contain noise only
        end_noise     % final  frames that contain noise only

        % Notation: RR corresponds to \mathcal{R} in [2]
        RR_y_inv      % inverse observed spatial cov,  (F, C, C)
        RR_k          % target spatial covariance,     (F, C, C)

        Lambda_kn       % cumulated target mask, (F, 1)
        Lambda_n        % cumulated noise  mask, (F, 1)
        Lambda_kn_prev  % previous Lambda_kn
        Lambda_n_prev   % previous Lambda_n

        R_kn          % normalized cov for target, (F, C, C)
        R_n           % normalized cov for noise,  (F, C, C)
        phi_kn        % time-dependent variance, target, (F, 1)
        phi_n         % time-dependent variance, noise,  (F, 1)
        alpha_kn      % mixture weight, target, (F, 1)
        alpha_n       % mixture weight, noise,  (F, 1)

        lambda_kn     % current target mask, (F, 1)
        lambda_n      % current noise  mask, (F, 1)
    end

    % ------------------------------------------------------------------ %
    methods (Access = public)

        % ---------------------------------------------------------- %
        function obj = OnlineCGMMMVDR(ny_k, ny_n, priorRk, priorRn, ...
                                       channels, frequency_bins, ...
                                       beg_noise, end_noise)
            %ONLINECGMMMVDR  Constructor.
            %
            %   Inputs:
            %     ny_k          - CGMM hyperparameter for target (float)
            %     ny_n          - CGMM hyperparameter for noise  (float)
            %     priorRk       - prior normalized SCM, target, (F, C, C)
            %     priorRn       - prior normalized SCM, noise,  (F, C, C)
            %     channels      - number of microphones C
            %     frequency_bins- number of STFT frequency bins F
            %     beg_noise     - (optional) initial noise-only frames  [0]
            %     end_noise     - (optional) final   noise-only frames  [0]

            if nargin < 7, beg_noise = 0; end
            if nargin < 8, end_noise = 0; end

            obj.ny_k      = ny_k;
            obj.ny_n      = ny_n;
            obj.C         = channels;
            obj.F         = frequency_bins;
            obj.beg_noise = beg_noise;
            obj.end_noise = end_noise;

            % Python: np.tile(np.eye(C)[None], (F,1,1)) * 1e10
            I_big = eye(channels) * 1e10;
            obj.RR_y_inv = repmat(reshape(I_big, [1, channels, channels]), ...
                                  [frequency_bins, 1, 1]);

            obj.RR_k = zeros(frequency_bins, channels, channels, 'like', 1+1i);

            obj.Lambda_kn      = zeros(frequency_bins, 1);
            obj.Lambda_n       = zeros(frequency_bins, 1);
            obj.Lambda_kn_prev = zeros(frequency_bins, 1);
            obj.Lambda_n_prev  = zeros(frequency_bins, 1);

            obj.R_kn    = priorRk;
            obj.R_n     = priorRn;
            obj.phi_kn  = [];        % initialized on first step()
            obj.phi_n   = [];

            obj.alpha_kn = 0.5 * ones(frequency_bins, 1);
            obj.alpha_n  = 0.5 * ones(frequency_bins, 1);
        end

        % ---------------------------------------------------------- %
        function [s_out, lam_kn] = step(obj, y, l, T)
            %STEP  One frame of online CGMM-MVDR inference + beamforming.
            %
            %   Inputs:
            %     y - complex STFT frame, shape (C, F)
            %     l - 1-based frame index  (1 .. T)
            %     T - total number of frames
            %
            %   Outputs:
            %     s_out   - cell {s_k1, s_k2}, each (F, 1) complex
            %     lam_kn  - target mask for this frame, (F, 1) real

            % ---- phi initialisation (first call only) ----
            % Python: if self.phi_kn is None: ...
            if isempty(obj.phi_kn)
                obj.phi_kn = obj.compute_phi(y, obj.R_kn);
                obj.phi_n  = obj.compute_phi(y, obj.R_n);
            end

            % Python (0-based l_py = l-1):
            %   is_all_noise = (l_py < beg_noise) or (l_py > T - end_noise)
            l_py        = l - 1;
            is_noise    = (l_py < obj.beg_noise) || (l_py > T - obj.end_noise);

            % ---- NOTE on Lambda_kn_prev / Lambda_n_prev ----
            % In Python:  self.Lambda_kn_prev, self.Lambda_n_prev = self.Lambda_kn, self.Lambda_n
            % This is a shared-reference assignment, NOT a copy.
            % _update_masks then does:  self.Lambda_kn += self.lambda_kn  (in-place)
            % Because Lambda_kn_prev IS the same object as Lambda_kn, Lambda_kn_prev
            % is ALSO updated.  As a result, in _update_R: nom == denom => forgetting = 1.
            % We replicate this exactly: save prev AFTER masks are updated.

            obj.update_masks(y, is_noise);

            % Assign prev AFTER mask update → prev == current  (matches Python behaviour)
            obj.Lambda_kn_prev = obj.Lambda_kn;
            obj.Lambda_n_prev  = obj.Lambda_n;

            obj.update_phi(y);
            obj.update_R(y);
            obj.update_RRk(y);
            obj.update_RRyinv(y);

            % Python: if l > beg_noise - 1  (i.e. l_py >= beg_noise)
            if l_py >= obj.beg_noise
                [w_k1, w_k2] = obj.compute_beamformer();
                [s_k1, s_k2] = obj.beamform(y, w_k1, w_k2);
            else
                s_k1 = zeros(obj.F, 1);
                s_k2 = zeros(obj.F, 1);
            end

            s_out  = {s_k1, s_k2};
            lam_kn = obj.lambda_kn;
        end

    end   % public methods

    % ------------------------------------------------------------------ %
    methods (Access = private)

        % ---------------------------------------------------------- %
        function phi = compute_phi(obj, y, R)
            %COMPUTE_PHI  Eq.(20) in [2] helper used at initialisation.
            %   phi(f) = (1/C) * y_f^H * R_f^{-1} * y_f
            phi = zeros(obj.F, 1);
            for f = 1:obj.F
                yf  = y(:, f);
                Rf  = reshape(R(f, :, :), obj.C, obj.C);
                phi(f) = max(real(yf' * (Rf \ yf)) / obj.C, 1e-10);
            end
        end

        % ---------------------------------------------------------- %
        function update_masks(obj, y, is_all_noise)
            %UPDATE_MASKS  Eq.(19),(25) in [2]: posterior mask estimation.
            %
            %   Sets obj.lambda_kn, obj.lambda_n  (F,1)
            %   Accumulates obj.Lambda_kn, obj.Lambda_n

            p_y_kn = complex_gaussian(y, obj.R_kn, obj.phi_kn);
            p_y_n  = complex_gaussian(y, obj.R_n,  obj.phi_n);

            lam_kn = obj.alpha_kn .* p_y_kn;
            lam_n  = obj.alpha_n  .* p_y_n;
            total  = lam_kn + lam_n + 1e-6;

            obj.lambda_kn = lam_kn ./ total;
            obj.lambda_n  = lam_n  ./ total;

            if is_all_noise
                obj.lambda_kn = zeros(obj.F, 1);
                obj.lambda_n  = ones(obj.F, 1);
            end

            obj.Lambda_kn = obj.Lambda_kn + obj.lambda_kn;
            obj.Lambda_n  = obj.Lambda_n  + obj.lambda_n;
        end

        % ---------------------------------------------------------- %
        function update_phi(obj, y)
            %UPDATE_PHI  Eq.(20) in [2]: time-dependent variance update.
            obj.phi_kn = obj.compute_phi(y, obj.R_kn);
            obj.phi_n  = obj.compute_phi(y, obj.R_n);
        end

        % ---------------------------------------------------------- %
        function update_R(obj, y)
            %UPDATE_R  Eq.(33) in [2]: normalized covariance update.
            %
            %   R_new = (nom/denom)*R_old
            %         + (1/denom)*(lambda/phi)*y*y^H
            %
            %   where nom  = Lambda_prev + (ny + C + 1)/2
            %         denom = Lambda      + (ny + C + 1)/2

            R_kn_new = zeros(obj.F, obj.C, obj.C, 'like', 1+1i);
            R_n_new  = zeros(obj.F, obj.C, obj.C, 'like', 1+1i);

            for f = 1:obj.F
                yf     = y(:, f);
                yyH    = yf * yf';           % outer product (C, C)

                % --- target ---
                Rkn_f   = reshape(obj.R_kn(f, :, :), obj.C, obj.C);
                nom_k   = obj.Lambda_kn_prev(f) + (obj.ny_k + obj.C + 1) / 2;
                denom_k = obj.Lambda_kn(f)      + (obj.ny_k + obj.C + 1) / 2;
                R_kn_new(f, :, :) = (nom_k  / denom_k) * Rkn_f  ...
                                  + (1       / denom_k) * (obj.lambda_kn(f) / obj.phi_kn(f)) * yyH;

                % --- noise ---
                Rn_f    = reshape(obj.R_n(f,  :, :), obj.C, obj.C);
                nom_n   = obj.Lambda_n_prev(f) + (obj.ny_n + obj.C + 1) / 2;
                denom_n = obj.Lambda_n(f)      + (obj.ny_n + obj.C + 1) / 2;
                R_n_new(f, :, :)  = (nom_n  / denom_n) * Rn_f   ...
                                  + (1       / denom_n) * (obj.lambda_n(f)  / obj.phi_n(f))  * yyH;
            end

            obj.R_kn = R_kn_new;
            obj.R_n  = R_n_new;
        end

        % ---------------------------------------------------------- %
        function update_RRk(obj, y)
            %UPDATE_RRK  Eq.(8) in [1]: target SCM accumulation.
            %
            %   RR_k(f) += lambda_kn(f) * y_f * y_f^H

            for f = 1:obj.F
                yf    = y(:, f);
                RRk_f = reshape(obj.RR_k(f, :, :), obj.C, obj.C);
                obj.RR_k(f, :, :) = RRk_f + obj.lambda_kn(f) * (yf * yf');
            end
        end

        % ---------------------------------------------------------- %
        function update_RRyinv(obj, y)
            %UPDATE_RRYINV  Eq.(7) in [1]: rank-1 Sherman-Morrison update
            %   of inverse observed SCM.
            %
            %   RR_y_inv_new = RR_y_inv
            %     - (RR_y_inv * y * y^H * RR_y_inv) / (1 + y^H * RR_y_inv * y)

            for f = 1:obj.F
                M   = reshape(obj.RR_y_inv(f, :, :), obj.C, obj.C);
                yf  = y(:, f);
                My  = M * yf;                            % (C,1)
                % y^H * M * y is real for Hermitian M; real() removes
                % tiny floating-point imaginary artefacts that accumulate
                % over frames as M departs slightly from exact Hermitian symmetry.
                d   = 1 + real(yf' * My);                % scalar
                nom = My * (yf' * M);                    % (C,C) outer product
                obj.RR_y_inv(f, :, :) = M - nom / d;
            end
        end

        % ---------------------------------------------------------- %
        function [w_k1, w_k2] = compute_beamformer(obj, refs)
            %COMPUTE_BEAMFORMER  Eq.(9) in [1]: MVDR filter computation.
            %
            %   w_k = (RR_y_inv * RR_k)[:, ref] / trace(RR_y_inv * RR_k)
            %
            %   Returns two filters for two reference microphones.

            if nargin < 2, refs = [1, 2]; end
            ref1 = refs(1);
            ref2 = refs(2);

            w_k1  = zeros(obj.F, obj.C, 'like', 1+1i);
            w_k2  = zeros(obj.F, obj.C, 'like', 1+1i);
            denom = zeros(obj.F, 1);

            for f = 1:obj.F
                M_inv   = reshape(obj.RR_y_inv(f, :, :), obj.C, obj.C);
                M_k     = reshape(obj.RR_k(f,     :, :), obj.C, obj.C);
                product = M_inv * M_k;              % (C, C)
                denom(f)   = real(trace(product));
                w_k1(f, :) = product(:, ref1);
                w_k2(f, :) = product(:, ref2);
            end

            % Python: w_k1 / (denom[:,None] + 1e-10)
            w_k1 = w_k1 ./ (denom + 1e-10);
            w_k2 = w_k2 ./ (denom + 1e-10);
        end

        % ---------------------------------------------------------- %
        function [s_k1, s_k2] = beamform(obj, y, w_k1, w_k2)
            %BEAMFORM  Apply beamforming filters to one STFT frame.
            %
            %   Python: s_k1(f) = einsum('fc,cf->f', w_k1.conj(), y)
            %         = sum_c  conj(w_k1(f,c)) * y(c,f)
            %
            %   y: (C,F), w_k1/w_k2: (F,C) → s_k1/s_k2: (F,1)
            s_k1 = sum(conj(w_k1) .* y.', 2);   % (F,1)
            s_k2 = sum(conj(w_k2) .* y.', 2);
        end

    end   % private methods

end
