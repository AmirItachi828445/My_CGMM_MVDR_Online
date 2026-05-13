%% Frequency Domain Processing Function (delay and sum) of Base Beamforming
function out_time = Base_Beamforming_Freq_Domain(data, mic_pos, focus_point, fs, c)
    % This function mimics the mic_host/kernel.py logic
    % 1. Convert to Frequency Domain (FFT)
    % 2. Apply Phase Shift (Complex Multiply)
    % 3. Inverse FFT
    freq_range = [200, 8000];
    [N, M] = size(data); % N: samples, M: mics
    nfft = 2^nextpow2(N);
    
    % FFT of each channel
    X = fft(data, nfft);
    
    % Frequency vector
    f = (0:nfft-1)' * (fs / nfft);
    
    % Calculate distances and delays
    dist = sqrt(sum((mic_pos - focus_point).^2, 2)); % [M x 1]
    delays = dist / c; % [M x 1]
    
    % Prepare Output Spectrum
    Y = zeros(nfft, 1);
    
    % Frequency Mask (mic_host often filters specific bands)
    mask = (f >= freq_range(1) & f <= freq_range(2)) | ...
           (f >= (fs - freq_range(2)) & f <= (fs - freq_range(1)));
    
    % Apply Phase Shift for each frequency bin
    % Logic: Y(f) = sum_i( X_i(f) * exp(j * 2 * pi * f * delay_i) )
    for i = 1:M
        % Phase shift term: exp(j * 2 * pi * f * tau)
        phase_shift = exp(1j * 2 * pi * f * delays(i));
        
        % Sum the shifted spectra
        Y = Y + X(:, i) .* phase_shift;
    end
    
    % Average across microphones
    Y = Y / M;
    
    % Apply frequency mask (optional, but mimics mic_host's band focus)
    Y(~mask) = 0;
    
    % Convert back to Time Domain
    out_time = real(ifft(Y, nfft));
    
    % Trim to original length
    out_time = out_time(1:N);
    
    % Normalization
    if max(abs(out_time)) > 0
        out_time = out_time / max(abs(out_time));
    end
end