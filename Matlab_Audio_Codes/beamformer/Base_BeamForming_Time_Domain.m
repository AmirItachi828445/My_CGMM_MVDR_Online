%% Core Processing Function of BaseBeamformer Zoom (Delay-and-Sum) on Time Domain
function out = Base_BeamForming_Time_Domain(data, mic_pos, focus_point, fs, c)
    % data: [Samples x Mics]
    % mic_pos: [Mics x 3]
    % focus_point: [1 x 3]
    
    [num_samples, num_mics] = size(data);
    % Calculate distances from focus point to each microphone
    % dist(i) = sqrt((x_mic - x_f)^2 + (y_mic - y_f)^2 + (z_mic - z_f)^2)
    dist = sqrt(sum((mic_pos - focus_point).^2, 2)); 
    
    % Calculate travel time (delays) in seconds
    delays = dist / c; 
    
    % Time vector for the original signal
    t = (0:num_samples-1)' / fs;
    out = zeros(num_samples, 1);
    
    % Beamforming: sum( mic_i(t + delay_i) )
    % We use linear interpolation to handle fractional delays accurately
    for i = 1:num_mics
        shifted_t = t + delays(i);
        % Interpolate to find signal value at the delayed time point
        % 'linear' matches the basic BeamformerTime behavior
        mic_signal = interp1(t, data(:, i), shifted_t, 'linear', 0);
        out = out + mic_signal;
    end
    
    % Normalize by number of microphones (Acoular BeamformerTime convention)
    out = out / num_mics;

    % Do NOT peak-normalize here. Acoular BeamformerTime keeps physical scaling so
    % mean(out.^2) compares across focus points (Python beamforming_comparison).
    % Export to WAV normalizes separately in save_zoomed_audio.m when needed.
end