function [audio_data, fs] = load_audio(filepath, num_mics)
% LOAD_AUDIO  Load multichannel audio from WAV or HDF5 (.h5).
%
% filepath   Full path to a .wav or .h5 file.
% num_mics   Optional. When provided, the matrix is oriented to
%            [numSamples x numMics] if the channel dimension matches num_mics.
%
% HDF5 layout (same as Main.m):
%   Dataset: /time_data
%   Attribute on dataset: sample_freq (scalar)
%
% WAV: standard audioread() behavior.

    if nargin < 2
        num_mics = [];
    end

    if ~isfile(filepath)
        error('load_audio:FileNotFound', 'File not found: %s', filepath);
    end

    [~, ~, ext] = fileparts(filepath);
    ext = lower(ext);

    switch ext
        case '.wav'
            [audio_data, fs] = audioread(filepath);
        case '.h5'
            audio_data = h5read(filepath, '/time_data');
            fs = double(h5readatt(filepath, '/time_data', 'sample_freq'));
        otherwise
            error('load_audio:UnsupportedFormat', ...
                'Unsupported extension "%s". Use .wav or .h5.', ext);
    end

    if ~isempty(num_mics)
        if size(audio_data, 2) ~= num_mics && size(audio_data, 1) == num_mics
            audio_data = audio_data';
        end
    end
end
