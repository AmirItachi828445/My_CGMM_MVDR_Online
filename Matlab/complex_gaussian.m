function p = complex_gaussian(y, R, phi)
%COMPLEX_GAUSSIAN Computes zero-mean complex Gaussian PDF (Eq.15 in [2]).
%
%   p = COMPLEX_GAUSSIAN(y, R, phi)
%
%   Inputs:
%     y   - complex observation matrix, shape (C, F)
%     R   - normalized covariance matrix, shape (F, C, C)
%     phi - time-dependent variance (scalar per freq bin), shape (F, 1)
%
%   Output:
%     p   - real probability density, shape (F, 1)
%
%   References:
%     [2] Higuchi et al., IEEE/ACM TASLP 2017, doi:10.1109/TASLP.2017.2665341

[C, F] = size(y);
p = zeros(F, 1);

for f = 1:F
    Rf    = reshape(R(f, :, :), C, C);
    Sigma = phi(f) * Rf;
    yf    = y(:, f);

    % Use Cholesky decomposition: Sigma = L*L^H (Hermitian PD)
    % This is ~2x faster and more numerically stable than LU for this case.
    % Log-domain computation for numerical stability:
    %   log p = -(y^H * Sigma^{-1} * y) - C*log(pi) - log|det(Sigma)|
    %         = -(||L^{-1} y||^2) - C*log(pi) - 2*sum(log|diag(L)|)
    try
        L         = chol(Sigma, 'lower');   % Sigma = L * L^H
        v         = L \ yf;                 % v = L^{-1} * y
        log_quad  = real(v' * v);           % y^H * Sigma^{-1} * y  (real)
        log_det   = 2 * sum(log(abs(diag(L))));
        log_p     = -log_quad - C * log(pi) - log_det;
        p(f)      = exp(log_p);
    catch
        % Fall back to LU if Cholesky fails (e.g. near-singular Sigma)
        try
            [~, U, ~] = lu(Sigma);
            log_quad  = real(yf' * (Sigma \ yf));
            log_det   = sum(log(abs(diag(U))));
            log_p     = -log_quad - C * log(pi) - log_det;
            p(f)      = exp(log_p);
        catch
            p(f) = 0;
        end
    end
end

p = real(p);
end
