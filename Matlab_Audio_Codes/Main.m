% MAIN  Command-line entry: load data, run all beamformers, export WAV clips.
%
% Add project folders to the MATLAB path, then execute this script from the
% repository root (or ensure relative paths in zoom_config() still resolve).

clear; clc; close all;

addpath('beamformer');
addpath('utils');
addpath('config');
addpath('processing');

%% 1. Load configuration (beamformer list, points, paths, physical constants)
cfg = zoom_config();

%% 2. Microphone geometry from XML
% mic_pos = load_mic_positions(cfg.xml_file);
% disp(mic_pos)
mic_pos = load_mic_positions(cfg.xml_file);
selected_idx = [9 11 13 15];
mic_pos = mic_pos(selected_idx, :);

%% 3. Multichannel audio (HDF5 or WAV; channels aligned to mic count)
audio_file = cfg.audio_file;
[audio_data, fs] = load_audio(audio_file, size(mic_pos, 1));
cfg.fs = fs;

fprintf('Audio loaded: %d samples | %d mics | fs=%d Hz\n', ...
    size(audio_data, 1), size(audio_data, 2), fs);

%% 4. Beamforming pass
results = process_beamforming(audio_data, mic_pos, cfg);

%% 5. Write one WAV per (method, point)
save_zoomed_audio(results, cfg);

%% 6. Generate spatial-response plots (output/plots/)
save_beamforming_plots(results, cfg);

fprintf('\n Finished. All outputs saved in /output\n');
