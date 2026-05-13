% %% Main_Fucntion_Of_LCMV_GSC
function enhanced_signal = LCMV_Spatial_Time_Domain(data, mic_pos, focus_point, fs, c)
    [N_samples, ~] = size(data);
    % parameters like python
    frame_len_ms = 32;
    overlap_percent = 75;
    num_mics = size(data,2);
    % sep and overlap calculation
    nperseg = round(fs * frame_len_ms / 1000);
    noverlap = round(nperseg * overlap_percent / 100);
    win = hann(nperseg,"periodic");
    % null points
    points.Center       = [0, 0, 0.6];
    points.TopCenter    = [0, 0.2, 0.6];
    points.RightCenter  = [0.2, 0, 0.6];
    
    null_coords = [
        points.Center;
        points.TopCenter;
        points.RightCenter
    ];

    % STFT for first microphone
    [Zxx_all, f, ~] = Stft_Custom(data(:,1), fs, win,  nperseg,noverlap);
    [num_freq_bins, num_frames] = size(Zxx_all);
    % multi-chaneel Zxx calculation
    Zxx_multi = zeros(num_mics, num_freq_bins, num_frames);
    Zxx_multi(1,:,:) = Zxx_all;
    % And for other mics
    for i = 2:num_mics
        [Zxx_tmp, f, ~] = Stft_Custom(data(:,i), fs, win,  nperseg, noverlap);
        Zxx_multi(i,:,:) = Zxx_tmp;
    end
    % enhanced STFT output
    enhanced_stft = zeros(num_freq_bins, num_frames);
    for k = 1:num_freq_bins
        freq = f(k);
        if freq == 0
            enhanced_stft(k,:) = squeeze(Zxx_multi(1,k,:));
            continue
        end
        a_zoom = calculate_steering_vector(mic_pos, focus_point, fs, freq, c);
        a_zoom = a_zoom(:);
        num_nulls = size(null_coords,1);
        a_nulls = cell(num_nulls,1);
        for ni = 1:num_nulls
            nc = null_coords(ni,:); 
            a_tmp = calculate_steering_vector(mic_pos, nc, fs, freq, c);
            a_nulls{ni} = a_tmp(:);
        end
        % C Matric Calculation
        num_constraints = 1 + num_nulls;
        C = zeros(num_mics, num_constraints);
        C(:,1) = a_zoom;
        for ni = 1:num_nulls
            C(:,1+ni) = a_nulls{ni};
        end
        
        % d Matric Calculation
        d = zeros(num_constraints,1);
        d(1) = 1.0;
        Zxx_multi_k_frames = squeeze(Zxx_multi(:,k,:));
        Ryy_k = cov(Zxx_multi_k_frames.');
        Ryy_k = Ryy_k + 1e-3 * eye(num_mics);
        Ryy_inv_k = pinv(Ryy_k);
        % two term : term1 & term2
        term1 = Ryy_inv_k * C;
        term2 = C' * term1;
        term2_inv = pinv(term2);
        w_k = term1 * (term2_inv * d);   % [num_mics x 1]
        enhanced_stft(k,:) = (w_k') * Zxx_multi_k_frames;
    end
    enhanced_signal = IStft_Custom(enhanced_stft, fs, win, nperseg, noverlap, N_samples);
end

%% Steering Vector Calculation
function steering_vectors = calculate_steering_vector(mic_pos, source_coord, ~, frequencies, c)
    if nargin < 5
        c = 343;
    end
    [num_mics, ~] = size(mic_pos);
    source_coord = source_coord(:).'; 
    frequencies = frequencies(:); 
    num_freqs   = length(frequencies);
    steering_tmp = complex(zeros(num_freqs, num_mics));
    dists = sqrt(sum((mic_pos - source_coord).^2, 2));
    ref_dist = sqrt(sum((mic_pos(1,:) - source_coord).^2, 2));
    delays = (dists - ref_dist) / c;
    for f_idx = 1:num_freqs
        freq = frequencies(f_idx);
        if freq == 0
            steering_tmp(f_idx, :) = ones(1, num_mics);
            continue;
        end
        steering_tmp(f_idx, :) = exp(-1j * 2 * pi * freq * delays).';
    end
    steering_vectors = steering_tmp.';
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
