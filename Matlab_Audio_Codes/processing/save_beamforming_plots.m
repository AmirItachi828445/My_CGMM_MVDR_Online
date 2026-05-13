function save_beamforming_plots(results, cfg)
% SAVE_BEAMFORMING_PLOTS  Generate spatial-response bar-chart figures for
%   each beamformer and a combined comparison figure.
%
% For every (beamformer, focus-point) pair the mean short-time energy
% computed by process_beamforming() is converted to dB and used as the
% signal power metric.  SNR is derived relative to the mean output energy
% across all OTHER focus points — reflecting spatial zoom quality.
%
% Bars for focus points listed in cfg.source_points are drawn in red/dark-red
% to mark actual source locations.  All other bars are blue.
%
% Power bars start at the lowest measured dB and grow upward toward the
% maximum.  SNR bars are shown as deviations from zero (positive = source
% direction stands out; negative = suppressed).
%
% Output files are written to <cfg.output_dir>/plots/:
%   <BF_name>_spatial_response.png   – two-panel plot for one beamformer
%   all_beamformers_comparison.png   – combined figure (2 rows × N_bf cols)
%
% results  Output of process_beamforming()  (struct array)
% cfg      Configuration struct; uses cfg.output_dir and cfg.source_points

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
    eps_val  = 1e-12;
    power_db = 10 * log10(energy_mat + eps_val);

    % ---- Spatial-discrimination SNR ------------------------------------
    %   SNR(b,p) = output energy at focus p  vs  mean over all other points.
    %   Positive = focus point stands above the background.
    %   Negative = focus point is suppressed relative to background.
    snr_db = zeros(n_bf, n_pts);
    for b = 1:n_bf
        for p = 1:n_pts
            other_idx  = setdiff(1:n_pts, p);
            if isempty(other_idx)
                snr_db(b, p) = 0;
            else
                mean_other = mean(energy_mat(b, other_idx));
                snr_db(b, p) = 10 * log10( ...
                    (energy_mat(b, p) + eps_val) / (mean_other + eps_val));
            end
        end
    end

    % ---- Colour map ----------------------------------------------------
    %   Red/dark-red for actual source locations; steel-blue for all others.
    blue = [0.20 0.51 0.78];
    red  = [0.80 0.15 0.10];

    % Determine which focus points are source locations
    if isfield(cfg, 'source_points') && ~isempty(cfg.source_points)
        src_pts = cfg.source_points;
    else
        % Legacy fallback: highlight 'Center'
        src_pts = {'Center'};
    end

    bar_colors = repmat(blue, n_pts, 1);
    for si = 1:numel(src_pts)
        idx = find(strcmpi(point_names, src_pts{si}), 1);
        if ~isempty(idx)
            bar_colors(idx, :) = red;
        end
    end

    fprintf('\n Generating beamforming plots...\n');

    % ================================================================== %
    %  Individual plot per beamformer                                      %
    % ================================================================== %
    for b = 1:n_bf
        fig = figure('Visible', 'off', 'Position', [0 0 1000 520]);

        % --- Power (dB) — bars grow upward (negative scale, top = 0 dB) ---
        ax1 = subplot(2, 1, 1);
        bh1 = bar(ax1, 1:n_pts, power_db(b, :));
        bh1.FaceColor = 'flat';
        bh1.CData = bar_colors;
        bh1.EdgeColor = 'none';
        title(ax1, sprintf('\\bf%s\\rm\\newlinePower (dB)', bf_names{b}), ...
            'Interpreter', 'tex');
        set(ax1, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45);
        ylabel(ax1, 'Power (dB)');
        grid(ax1, 'on');
        ax1.YLim(2) = 0;          % top of power axis always at 0 dB

        % --- SNR (dB) — zero line visible; positive = source stands out ---
        ax2 = subplot(2, 1, 2);
        bh2 = bar(ax2, 1:n_pts, snr_db(b, :));
        bh2.FaceColor = 'flat';
        bh2.CData = bar_colors;
        bh2.EdgeColor = 'none';
        yline(ax2, 0, 'k-', 'LineWidth', 0.8);
        title(ax2, sprintf('\\bf%s\\rm\\newlineSNR (dB)', bf_names{b}), ...
            'Interpreter', 'tex');
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
    fig_w = max(320 * n_bf, 900);
    fig = figure('Visible', 'off', 'Position', [0 0 fig_w 820]);

    for b = 1:n_bf
        % Top row — Power (dB)
        ax = subplot(2, n_bf, b);
        bh = bar(ax, 1:n_pts, power_db(b, :));
        bh.FaceColor = 'flat';
        bh.CData = bar_colors;
        bh.EdgeColor = 'none';
        title(ax, sprintf('\\bf%s\\rm\nPower (dB)', bf_names{b}), ...
            'Interpreter', 'tex', 'FontSize', 8);
        set(ax, 'XTick', 1:n_pts, 'XTickLabel', point_names, ...
            'XTickLabelRotation', 45, 'FontSize', 7);
        if b == 1, ylabel(ax, 'Power (dB)'); end
        ax.YLim(2) = 0;
        grid(ax, 'on');

        % Bottom row — SNR (dB)
        ax2 = subplot(2, n_bf, n_bf + b);
        bh2 = bar(ax2, 1:n_pts, snr_db(b, :));
        bh2.FaceColor = 'flat';
        bh2.CData = bar_colors;
        bh2.EdgeColor = 'none';
        yline(ax2, 0, 'k-', 'LineWidth', 0.8);
        title(ax2, sprintf('\\bf%s\\rm\nSNR (dB)', bf_names{b}), ...
            'Interpreter', 'tex', 'FontSize', 8);
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
