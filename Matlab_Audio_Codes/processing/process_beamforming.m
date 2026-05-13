function results = process_beamforming(audio_data, mic_pos, cfg)
% PROCESS_BEAMFORMING  Run every configured beamformer on every focus point.
%
% audio_data  [numSamples x numMics] pressure time series
% mic_pos     [numMics x 3] Cartesian coordinates of each channel
% cfg         Configuration struct from zoom_config(), with cfg.fs set to sample rate
%
% Returns a struct array with one element per (beamformer, point) pair:
%   .beamformer - Name string from cfg.beamformers{:}.name
%   .point      - Name string from cfg.points(:).name
%   .signal     - Column vector of beamformed audio
%   .energy     - Short-time energy curve from compute_energy()
%
% Actual calls to /beamformer implementations go through beamformer_dispatch()
% so optional per-method parameters stay in cfg.beamformers{:}.params.

    num_beamformers = numel(cfg.beamformers);
    num_points = numel(cfg.points);

    fs = cfg.fs;
    window_size = round(cfg.energy_window_ms * fs / 1000);

    results = struct('beamformer', {}, 'point', {}, 'signal', {}, 'energy', {});
    idx = 1;

    fprintf('\n Processing beamforming...\n');

    for b = 1:num_beamformers
        bf_entry = cfg.beamformers{b};
        bf_name = bf_entry.name;

        fprintf('\n Beamformer: %s\n', bf_name);

        for p = 1:num_points
            point_name = cfg.points(p).name;
            point_pos = cfg.points(p).position;

            fprintf('  → Point: %s at [%.2f, %.2f, %.2f]\n', ...
                point_name, point_pos(1), point_pos(2), point_pos(3));

            signal = beamformer_dispatch(bf_entry, audio_data, mic_pos, point_pos, fs, cfg);
            energy = compute_energy(signal, window_size);

            results(idx).beamformer = bf_name;
            results(idx).point = point_name;
            results(idx).signal = signal;
            results(idx).energy = energy;
            idx = idx + 1;
        end
    end
end
