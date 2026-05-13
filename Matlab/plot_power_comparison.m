function plot_power_comparison(power_data, focus_point)
%PLOT_POWER_COMPARISON  Plot power spectra before and after beamforming.
%
%   Called automatically by cgmm_mvdr_beamform.m after each focus point.
%   Shows input vs enhanced power for both reference-microphone beamformers.
%
%   Inputs:
%     power_data  - struct returned by cgmm_mvdr_beamform
%     focus_point - (1,3) focus-point coordinates [m]

freqs = power_data.freqs(:);

% Smooth with short median filter for readability
win_med = 5;
p_in  = medfilt1(10*log10(power_data.input     + eps), win_med);
p_e1  = medfilt1(10*log10(power_data.enhanced1 + eps), win_med);
p_e2  = medfilt1(10*log10(power_data.enhanced2 + eps), win_med);

fig = figure('Name', sprintf('CGMM-MVDR Power — focus [%.2f %.2f %.2f] m', ...
    focus_point(1), focus_point(2), focus_point(3)), ...
    'NumberTitle', 'off', 'Position', [100 100 900 600]);

% --- Top: power spectra ---
subplot(2, 1, 1);
plot(freqs, p_in,  'k-',  'LineWidth', 1.5, 'DisplayName', 'Input  (ch 1)');
hold on;
plot(freqs, p_e1, 'b-',  'LineWidth', 1.5, 'DisplayName', 'Enhanced (ref-1)');
plot(freqs, p_e2, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Enhanced (ref-2)');
hold off;
grid on;
xlabel('Frequency [Hz]');
ylabel('Power [dBfs²]');
title(sprintf('Power spectrum comparison — focus point [%.2f, %.2f, %.2f] m', ...
    focus_point(1), focus_point(2), focus_point(3)));
legend('Location', 'best');
xlim([freqs(1), freqs(end)]);

% --- Bottom: estimated target mask ---
subplot(2, 1, 2);
imagesc(1:size(power_data.masks, 1), freqs, power_data.masks');
axis xy;
colorbar;
xlabel('Frame index');
ylabel('Frequency [Hz]');
title('Estimated target mask \lambda_{kn}(f,t)');
colormap(gca, 'hot');
clim([0, 1]);

drawnow;
end
