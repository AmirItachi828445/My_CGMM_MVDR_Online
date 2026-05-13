function energy = compute_energy(signal, window_size)
% COMPUTE_ENERGY  Mean energy in consecutive non-overlapping time windows.
%
% signal       Column vector [numSamples x 1]
% window_size  Window length in samples (integer)
%
% Returns energy as a column vector with one value per full window; trailing
% samples shorter than one window are ignored.

    num_windows = floor(length(signal) / window_size);
    energy = zeros(num_windows, 1);

    for i = 1:num_windows
        idx_start = (i - 1) * window_size + 1;
        idx_end = i * window_size;
        window = signal(idx_start:idx_end);
        energy(i) = sum(window .^ 2) / window_size;
    end
end
