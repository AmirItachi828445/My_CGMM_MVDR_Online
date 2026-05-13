function signal = beamformer_dispatch(bf_entry, audio_data, mic_pos, point_pos, fs, cfg)
% BEAMFORMER_DISPATCH  Route beamforming jobs to core implementations in /beamformer.
%
% This is the single place that maps configuration entries to function calls.
% Core beamformer files keep a fixed signature:
%     signal = CoreFunction(data, mic_pos, focus_point, fs, c)
% Optional algorithm knobs live in bf_entry.params and are extracted here,
% so you can add or remove options per method without changing the number of
% positional arguments the core functions take.
%
% To register a new beamformer:
%   1) Add a struct entry in config/zoom_config.m  (name, core, params).
%   2) Add a case below that calls the core function.
%   See BEAMFORMER_DEVELOPER_GUIDE.md for the full guide.
%
% bf_entry   Struct from cfg.beamformers{:}:
%            .name   - Display name for logs and UI
%            .core   - Function name under /beamformer (on MATLAB path)
%            .params - Struct of optional parameters (may be empty)
% audio_data [numSamples x numMics]
% mic_pos    [numMics x 3]
% point_pos  [1 x 3] focus coordinates
% fs         sample rate (Hz)
% cfg        full config from zoom_config(); uses at least cfg.c
%
% Returns column vector signal.

    if ~isfield(bf_entry, 'params') || isempty(bf_entry.params)
        params = struct();
    else
        params = bf_entry.params;
    end

    core = bf_entry.core;
    c    = cfg.c;

    switch core

        case 'CGMM_MVDR_Zoom'
            signal = CGMM_MVDR_Zoom(audio_data, mic_pos, point_pos, fs, c);

        case 'Base_BeamForming_Time_Domain'
            signal = Base_BeamForming_Time_Domain(audio_data, mic_pos, point_pos, fs, c);

        case 'Base_Beamforming_Freq_Domain'
            signal = Base_Beamforming_Freq_Domain(audio_data, mic_pos, point_pos, fs, c);

        case 'MVDR_Zoom_Time_Domain'
            signal = MVDR_Zoom_Time_Domain(audio_data, mic_pos, point_pos, fs, c);

        case 'MVDR_Zoom_Freq_Domain'
            signal = MVDR_Zoom_Freq_Domain(audio_data, mic_pos, point_pos, fs, c);

        case 'LCMV_Spatial_Time_Domain'
            signal = LCMV_Spatial_Time_Domain(audio_data, mic_pos, point_pos, fs, c);

        case 'LCMV_GSC_Zoom'
            signal = LCMV_GSC_Zoom(audio_data, mic_pos, point_pos, fs, c);

        otherwise
            error('beamformer_dispatch:UnknownCore', ...
                'Unknown beamformer core "%s". Register it in beamformer_dispatch.m.', core);
    end

    signal = signal(:);
end

% -----------------------------------------------------------------------
function v = local_get(s, field_name, default_value)
    if isfield(s, field_name)
        v = s.(field_name);
    else
        v = default_value;
    end
end
