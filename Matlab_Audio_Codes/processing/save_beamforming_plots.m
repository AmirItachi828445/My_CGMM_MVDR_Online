function save_beamforming_plots(results, cfg)
% SAVE_BEAMFORMING_PLOTS  Generate spatial-response bar-chart figures for
%   each beamformer and a combined comparison figure.
%
% For every (beamformer, focus-point) pair the mean short-time energy
% computed by process_beamforming() is converted to dB and used as the
% signal power metric.  SNR is derived relative to the minimum energy
% observed across all results (noise floor estimate).
%
% Output files are written to <cfg.output_dir>/plots/:
%   <BF_name>_spatial_response.png   – two-panel plot for one beamformer
%   all_beamformers_comparison.png   – combined figure (2 rows × N_bf cols)
%
% results  Output of process_beamforming()  (struct array)
% cfg      Configuration struct; uses cfg.output_dir

    plots_dir = fullfile(cfg.output_dir, 'plots');
    if ~exist(plots_dir, 'dir')
        mkdir(plots_dir);
    end

    % ---- Collect unique beamformer and point names (preserve order) ----
    bf_names    = unique_stable({results.beamformer});
    point_names = unique_stable({results.point});
    n_bf  = numel(bf_names);
    n_pts = numel(point_names);

    % ---- Build energy matrix  [n_bf x n_pts] ---------------------------
    energy_mat = zeros(n_bf, n_pts);
    for i = 1:numel(results)
        bf_idx = find(strcmp(bf_names,    results(i).beamformer), 1);
        pt_idx = find(strcmp(point_names, results(i).point),      1);
        if ~isempty(results(i).energy)
            energy_mat(bf_idx, pt_idx) = mean(results(i).energy);
        end
    end

    % ---- dB conversion -------------------------------------------------
    eps_val   = 1e-12;
    power_db  = 10 * log10(energy_mat + eps_val);

    % Spatial discrimination SNR:
    %   For each (beamformer, point) pair, SNR is defined as the ratio of
    %   the output energy at that focus point to the mean output energy
    %   across all other focus points.  This reflects how well the
    %   beamformer isolates the focused direction from the rest of the
    %   scene, which is exactly what a spatial zoom should achieve.
    snr_db = zeros(n_bf, n_pts);
    for b = 1:n_bf
        for p = 1:n_pts
            focal_energy = energy_mat(b, p);
            other_idx    = setdiff(1:n_pts, p);
            if isempty(other_idx)
                snr_db(b, p) = 0;
            else
                mean_other = mean(energy_mat(b, other_idx));
                snr_db(b, p) = 10 * log10((focal_energy + eps_val) / ...
                                           (mean_other  + eps_val));
            end
        end
    end

    % ---- Colour map (blue bars; red/orange for the focus centre) -------
    blue   = [0.22 0.55 0.80];
    red    = [0.85 0.25 0.10];
    bar_colors = repmat(blue, n_pts, 1);
    center_idx = find(strcmpi(point_names, 'Center'), 1);
    if ~isempty(center_idx)
        bar_colors(center_idx, :) = red;
    end

    fprintf('\n Generating beamforming plots...\n');

    % ================================================================== %
    %  Individual plot per beamformer                                      %
    % ================================================================== %
    for b = 1:n_bf
        fig = figure('Visible', 'off', 'Position', [0 0 1100 500]);

        % --- Power (dB) ---
        ax1 = subplot(1, 2, 1);
        bh1 = bar(ax1, 1:n_pts, power_db(b, :));
        bh1.FaceColor = 'flat';
        bh1.CData = bar_colors;
        title(ax1, sprintf('%s  –  Power (dB)', bf_names{b}), ...
            'Interpreter', 'none');
        set(ax1, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45);
        ylabel(ax1, 'Power (dB)');
        grid(ax1, 'on');

        % --- SNR (dB) ---
        ax2 = subplot(1, 2, 2);
        bh2 = bar(ax2, 1:n_pts, snr_db(b, :));
        bh2.FaceColor = 'flat';
        bh2.CData = bar_colors;
        title(ax2, sprintf('%s  –  SNR (dB)', bf_names{b}), ...
            'Interpreter', 'none');
        set(ax2, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45);
        ylabel(ax2, 'SNR (dB)');
        grid(ax2, 'on');

        sgtitle(sprintf('Beamformer: %s  –  Spatial Response', bf_names{b}), ...
            'Interpreter', 'none');

        outpath = fullfile(plots_dir, sprintf('%s_spatial_response.png', bf_names{b}));
        print(fig, outpath, '-dpng', '-r150');
        close(fig);
        fprintf('  %s\n', outpath);
    end

    % ================================================================== %
    %  Combined comparison figure (2 rows × n_bf columns)                 %
    % ================================================================== %
    fig_w = max(300 * n_bf, 900);
    fig = figure('Visible', 'off', 'Position', [0 0 fig_w 800]);

    for b = 1:n_bf
        % Top row – Power (dB)
        ax = subplot(2, n_bf, b);
        bh = bar(ax, 1:n_pts, power_db(b, :));
        bh.FaceColor = 'flat';
        bh.CData = bar_colors;
        title(ax, sprintf('%s\nPower (dB)', bf_names{b}), ...
            'Interpreter', 'none', 'FontSize', 8);
        set(ax, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45, 'FontSize', 7);
        if b == 1, ylabel(ax, 'Power (dB)'); end
        grid(ax, 'on');

        % Bottom row – SNR (dB)
        ax2 = subplot(2, n_bf, n_bf + b);
        bh2 = bar(ax2, 1:n_pts, snr_db(b, :));
        bh2.FaceColor = 'flat';
        bh2.CData = bar_colors;
        title(ax2, sprintf('%s\nSNR (dB)', bf_names{b}), ...
            'Interpreter', 'none', 'FontSize', 8);
        set(ax2, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45, 'FontSize', 7);
        if b == 1, ylabel(ax2, 'SNR (dB)'); end
        grid(ax2, 'on');
    end

    sgtitle('All Beamformers  –  Spatial Response Comparison');

    outpath = fullfile(plots_dir, 'all_beamformers_comparison.png');
    print(fig, outpath, '-dpng', '-r150');
    close(fig);
    fprintf('  %s\n', outpath);

    fprintf(' Plots saved to %s\n', plots_dir);
end

% -----------------------------------------------------------------------
function names = unique_stable(cell_arr)
% Return unique entries preserving first-occurrence order.
    n     = numel(cell_arr);
    seen  = cell(1, n);
    names = cell(1, n);
    count = 0;
    for i = 1:n
        v = cell_arr{i};
        if ~any(strcmp(seen(1:count), v))
            count          = count + 1;
            seen{count}    = v;
            names{count}   = v;
        end
    end
    names = names(1:count);
end
