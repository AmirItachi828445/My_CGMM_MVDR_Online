function [enhanced_sig, best_shift] = MVDR_Zoom_Freq_Domain(data, mic_pos, target_point, fs, c)
    if nargin < 5
        c = 343.0; 
    end
    % STFT Parameters
    [N_samples, num_mics] = size(data);
    frame_len_ms    = 32;
    overlap_percent = 75;
    nperseg  = round(fs * frame_len_ms / 1000);
    noverlap = round(nperseg * overlap_percent / 100);
    win = hann(nperseg, 'periodic');
    
    % --- 1. Compute STFT for all channels ---
    [~, F, ~] = Stft_Custom(data(:, 1), fs, win, nperseg, noverlap);
    num_freqs = length(F);
    
    Y = [];
    for m = 1:num_mics
        [S, ~, T] = Stft_Custom(data(:, m), fs, win, nperseg, noverlap);
        if isempty(Y)
            num_frames = length(T);
            Y = zeros(num_freqs, num_frames, num_mics);
        end
        Y(:, :, m) = S;
    end
    
    % --- 2. Steering Vector & Delay Calculation ---
    dist = zeros(num_mics, 1);
    for m = 1:num_mics
        dist(m) = norm(mic_pos(m, :) - target_point);
    end
    
    % Calculate Time Difference of Arrival (TDOA) / Delays
    delays = dist / c; 
    
    % Output the shift in samples (useful for analysis or time-domain alignment)
    best_shift = round(delays * fs);
    
    % Construct the frequency-domain steering vector
    D = zeros(num_freqs, num_mics);
    for k = 1:num_freqs
        D(k, :) = exp(-1j * 2 * pi * F(k) * delays).';
    end
    
    % --- 3. Speech Presence Probability (SPP) Estimation ---
    % Calculate signal energy across all frequencies and microphones
    E = squeeze(sum(sum(abs(Y).^2, 1), 3)); 
    mean_E = mean(E);
    
    % Basic Voice Activity Detection (VAD): 1 if Speech, 0 if Noise
    SPP = E > (0.5 * mean_E); 
    noise_frames_idx = find(SPP == 0);
    num_noise_frames = length(noise_frames_idx);
    
    % --- 4. MVDR Processing (Frequency Domain) ---
    Y_out = zeros(num_freqs, num_frames);
    
    for k = 1:num_freqs
        Y_k = squeeze(Y(k, :, :)).'; % Extract freqs: [num_mics x num_frames]
        d_k = D(k, :).';             % Steering vector for current frequency
        
        % Estimate the Spatial Noise Covariance Matrix (R_n)
        R_n_k = zeros(num_mics, num_mics);
        if num_noise_frames > 0
            % Use only noise frames to estimate noise characteristics
            Y_noise = Y_k(:, noise_frames_idx);
            R_n_k = (Y_noise * Y_noise') / num_noise_frames;
        else
            % Fallback (MPDR): If no noise frames, use all frames (Signal+Noise)
            R_n_k = (Y_k * Y_k') / num_frames;
        end
        
        % Diagonal loading (Regularization) to prevent singular matrix issues
        R_n_k = R_n_k + 1e-6 * eye(num_mics); 
        
        % Compute MVDR Weights using the formula: W = (R_n^-1 * d) / (d^H * R_n^-1 * d)
        R_inv = inv(R_n_k);
        num = R_inv \ d_k;
        den = d_k' * num;
        W_mvdr = num / den;
        
        % Apply the weights to the multichannel signal to get the enhanced output
        Y_out(k, :) = W_mvdr' * Y_k; % [1 x num_frames]
    end
    
    % --- 5. Signal Reconstruction (ISTFT) ---
    enhanced_sig = IStft_Custom(Y_out, fs, win, nperseg, noverlap, N_samples);
    if length(enhanced_sig) > N_samples
        enhanced_sig = enhanced_sig(1:N_samples);
    elseif length(enhanced_sig) < N_samples
        enhanced_sig = [enhanced_sig; zeros(N_samples - length(enhanced_sig), 1)];
    end
    enhanced_sig = real(enhanced_sig);
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