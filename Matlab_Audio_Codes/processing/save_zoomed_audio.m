function save_zoomed_audio(results, cfg)
% SAVE_ZOOMED_AUDIO  Export each beamformed trace as a normalized mono WAV file.
%
% results  Output of process_beamforming()
% cfg      Must contain .output_dir and .fs

    fprintf('\n Saving audio files...\n');

    for i = 1:numel(results)
        filename = sprintf('%s_%s.wav', results(i).beamformer, results(i).point);
        filepath = fullfile(cfg.output_dir, filename);

        sig = real(results(i).signal);
        sig = sig / max(abs(sig) + 1e-12);
        audiowrite(filepath, sig, cfg.fs);
        fprintf(' %s (method: %s, point: %s)\n', ...
            filename, results(i).beamformer, results(i).point);
    end
end
