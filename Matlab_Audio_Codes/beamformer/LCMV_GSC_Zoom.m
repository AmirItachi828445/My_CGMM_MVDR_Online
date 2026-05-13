% %% Main_Fucntion_Of_LCMV_GSC
function enhanced_signal = LCMV_GSC_Zoom(data, mic_pos, focus_point, fs, c)
    [N_samples, N_mics] = size(data);
    frame_len_ms    = 64;
    overlap_percent = 75;
    nperseg  = round(fs * frame_len_ms / 1000);
    noverlap = round(nperseg * overlap_percent / 100);
    win = hann(nperseg, 'periodic');
    for i = 1:N_mics
        [Zxx_mic, f, ~] = Stft_Custom(data(:, i), fs, win, nperseg, noverlap);
        if i == 1
            [N_freq, N_frames] = size(Zxx_mic);
            Y = complex(zeros(N_freq, N_frames, N_mics));
        end
        Y(:, :, i) = Zxx_mic;
    end
    
    % Find SPP
    SPP = ones(N_frames, 1);
    n_noise_frames = max(1, floor(0.2 * N_frames));
    SPP(1:n_noise_frames) = 0;

    % Steering vector D
    focus_point = focus_point(:).';
    dist   = sqrt(sum((mic_pos - focus_point).^2, 2));
    delays = dist / c;
    D = complex(zeros(N_freq, N_mics));
    for k = 1:N_freq
        freq = f(k);
        D(k, :) = exp(-1j * 2 * pi * freq * delays(:).');
    end

    % Blocking matrix B
    B = complex(zeros(N_freq, N_mics, N_mics - 1));
    for k = 1:N_freq
        d_k = D(k, :).';
        [~, ~, V] = svd(d_k');
        B_k = V(:, 2:end);
        B(k, :, :) = B_k;
    end

    % Noise covariance
    R_n = complex(zeros(N_freq, N_mics, N_mics));
    noise_frames_idx = (SPP == 0);
    num_noise_frames = sum(noise_frames_idx);
    for k = 1:N_freq
        if num_noise_frames > 0
            Y_k = squeeze(Y(k, :, :)); 
            Y_noise = Y_k(noise_frames_idx, :);
            R_n(k, :, :) = (Y_noise' * Y_noise) / num_noise_frames;
        else
            R_n(k, :, :) = eye(N_mics) * 1e-6;
        end
    end

    % LCMV / MVDR weights
    W_q = complex(zeros(N_freq, N_mics));
    for k = 1:N_freq
        d_k   = reshape(D(k, :), [], 1);
        R_n_k = squeeze(R_n(k, :, :));
        A = R_n_k + 1e-6 * eye(N_mics);
        numerator   = A \ d_k;
        denominator = d_k' * numerator;
        W_q(k, :) = (numerator / denominator).';
    end

    % Main beamformer output
    Y_q = complex(zeros(N_freq, N_frames));
    for k = 1:N_freq
        Y_f = squeeze(Y(k, :, :)).';   % [N_mics x N_frames]
        Y_q(k, :) = conj(W_q(k, :)) * Y_f;
    end
    enhanced_signal = IStft_Custom(Y_q, fs, win, nperseg, noverlap, N_samples);
    if length(enhanced_signal) > N_samples
        enhanced_signal = enhanced_signal(1:N_samples);
    elseif length(enhanced_signal) < N_samples
        enhanced_signal = [enhanced_signal; zeros(N_samples - length(enhanced_signal), 1)];
    end
    enhanced_signal = real(enhanced_signal);
end

%% Function of STFT
function [S, f, t] = Stft_Custom(x, fs, win, nperseg, noverlap)
    x = x(:);
    hop = nperseg - noverlap;
    pad = floor(nperseg / 2);
    x_pad = [zeros(pad,1); x; zeros(pad,1)];
    n_frames = ceil((length(x_pad) - nperseg) / hop) + 1;
    target_len = (n_frames - 1) * hop + nperseg;
    if length(x_pad) < target_len
        x_pad = [x_pad; zeros(target_len - length(x_pad), 1)];
    end
    N_freq = floor(nperseg / 2) + 1;
    S = complex(zeros(N_freq, n_frames));
    for m = 1:n_frames
        idx = (m-1)*hop + (1:nperseg);
        frame = x_pad(idx) .* win;
        X = fft(frame, nperseg);
        S(:, m) = X(1:N_freq);
    end
    f = (0:N_freq-1).' * fs / nperseg;
    t = ((0:n_frames-1) * hop) / fs;
end

%% Function of ISTFT
function x = IStft_Custom(S, ~, win, nperseg, noverlap, target_len)
    [~, n_frames] = size(S);
    hop = nperseg - noverlap;
    if rem(nperseg, 2) == 0
        mirror_part = conj(S(end-1:-1:2, :));
    else
        mirror_part = conj(S(end:-1:2, :));
    end
    S_full = [S; mirror_part];
    out_len = (n_frames - 1) * hop + nperseg;
    x_ola = zeros(out_len, 1);
    wsum  = zeros(out_len, 1);
    for m = 1:n_frames
        idx = (m-1)*hop + (1:nperseg);
        frame_t = real(ifft(S_full(:, m), nperseg));
        frame_t = frame_t(:) .* win;
        x_ola(idx) = x_ola(idx) + frame_t;
        wsum(idx)  = wsum(idx) + win.^2;
    end
    nz = wsum > 1e-12;
    x_ola(nz) = x_ola(nz) ./ wsum(nz);
    pad = floor(nperseg / 2);
    if length(x_ola) > 2 * pad
        x = x_ola(pad+1:end-pad);
    else
        x = x_ola;
    end
    if nargin >= 6 && ~isempty(target_len)
        if length(x) > target_len
            x = x(1:target_len);
        elseif length(x) < target_len
            x = [x; zeros(target_len - length(x), 1)];
        end
    end
end
