% part a
% Signal parameters
Ts = 0.0005; % 500 microseconds
t = 0:Ts:0.02; % Time range from 0 to 0.02 seconds

% Original signal
x = sin(1000*pi*t) + sin(2000*pi*t);

% Desired times for interpolation
delta = 0.00005; % 50 microseconds
t_recon = 0:delta:0.02;

% Use the interpolation function
y_recon = sinc_interpolation(x, t, t_recon);

% Plot the original and interpolated signals
figure;
plot(t, x, 'o', 'DisplayName', 'Sampled Signal'); % Sampled signal
hold on;
plot(t_recon, y_recon, '-', 'DisplayName', 'Reconstructed Signal'); % Interpolated signal
title('Signal Interpolation using sinc');
xlabel('Time (s)');
ylabel('Amplitude');
legend;
grid on;


%%
% part b
clc; clear; close all;

% Signal parameters
Ts = 0.0005; % 500 microseconds
t = 0:Ts:0.02; % Time range from 0 to 0.02 seconds

% Original signal
x = sin(1000*pi*t) + sin(2000*pi*t);

% Desired times for interpolation
delta = 0.00005; % 50 microseconds
t_recon = 0:delta:0.02;

% Use the interpolation function with limited sinc
y_recon = limited_sinc_interpolation(x, t, t_recon);

% Plot the original and interpolated signals
figure;
plot(t, x, 'o', 'DisplayName', 'Sampled Signal'); % Sampled signal
hold on;
plot(t_recon, y_recon, '-', 'DisplayName', 'Reconstructed Signal'); % Interpolated signal
title('Signal Interpolation using limited sinc');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
legend

%%
clc; clear; close all;



clc; clear; close all;

% Signal parameters
Fs_initial = [1000, 2000, 4000]; % Initial sampling frequencies (Hz)
Fs_final = [5000, 10000, 20000]; % Final sampling frequencies (Hz)
t_duration = 0.02; % Duration of signal (seconds)

% Original signal
original_signal = @(t) sin(1000*pi*t) + sin(2000*pi*t);

figure;

subplot_idx = 1;
for i = 1:length(Fs_initial)
    for j = 1:length(Fs_final)
        Ts_initial = 1 / Fs_initial(i); % Initial sampling period
        t_initial = 0:Ts_initial:t_duration; % Initial sample times
        x = original_signal(t_initial); % Sampled signal

        Ts_final = 1 / Fs_final(j); % Final sampling period
        t_final = 0:Ts_final:t_duration; % Desired sample times for interpolation

        % Use the sinc interpolation function (full and limited)
        y_recon_full = sinc_interpolation(x, t_initial, t_final);
        y_recon_limited = limited_sinc_interpolation(x, t_initial, t_final);

        % Plot the sampled and reconstructed signals in subplots
        subplot(length(Fs_initial), length(Fs_final), subplot_idx);
        plot(t_initial, x, 'o', 'DisplayName', 'Sampled Signal'); % Sampled signal
        hold on;
        plot(t_final, y_recon_full, '-', 'DisplayName', 'Reconstructed Signal (Full sinc)'); % Interpolated signal (full sinc)
        plot(t_final, y_recon_limited, '--', 'DisplayName', 'Reconstructed Signal (Limited sinc)'); % Interpolated signal (limited sinc)
        title(sprintf('Fs_{initial} = %d Hz, Fs_{final} = %d Hz', Fs_initial(i), Fs_final(j)));
        xlabel('Time (s)');
        ylabel('Amplitude');
        legend;
        grid on;

        subplot_idx = subplot_idx + 1;
    end
end



% Define the sinc_interpolation function
function y_reconstructed = sinc_interpolation(sampled_signal, sample_times, desired_times)
    num_samples = length(sampled_signal);
    y_reconstructed = zeros(1, length(desired_times));
    for n = 1:num_samples
        y_reconstructed = y_reconstructed + sampled_signal(n) * sinc((desired_times - sample_times(n)) / (sample_times(2) - sample_times(1)));
    end
end

% Define the limited_sinc_interpolation function
function y_reconstructed = limited_sinc_interpolation(sampled_signal, sample_times, desired_times)
    num_samples = length(sampled_signal);
    y_reconstructed = zeros(1, length(desired_times));
    sinc_lobe_limit = 4.5;
    for n = 1:num_samples
        sinc_values = (desired_times - sample_times(n)) / (sample_times(2) - sample_times(1));
        limited_sinc = sinc(sinc_values) .* (abs(sinc_values) <= sinc_lobe_limit);
        y_reconstructed = y_reconstructed + sampled_signal(n) * limited_sinc;
    end
end
