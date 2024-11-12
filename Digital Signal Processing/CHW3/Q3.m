clc, clear, close all

% Parameters
f1 = 4e3;      % Initial frequency in Hz
mu = 600e3;    % Sweep rate in Hz/s
phi = 0;       % Initial phase
fs = 8e3;      % Sampling frequency in Hz
t_duration = 0.05;  % Duration of the signal in seconds

% Time vector for continuous signal
t = linspace(0, t_duration, 1000);

% Continuous chirp signal
x_t = cos(pi * mu * t.^2 + 2 * pi * f1 * t + phi);

% Instantaneous frequency calculation
syms t_sym
phase = pi * mu * t_sym^2 + 2 * pi * f1 * t_sym + phi;  % Phase of the signal
instantaneous_freq = diff(phase, t_sym) / (2 * pi);     % Instantaneous frequency in Hz
instantaneous_freq_values = double(subs(instantaneous_freq, t_sym, t)); % Substitute t values for plotting

% Plot continuous signal
figure;
plot(t, x_t);
title('Continuous Chirp Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot instantaneous frequency
figure
plot(t, instantaneous_freq_values);
title('Instantaneous Frequency');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
grid on

% Sampling the signal
n = 0:1/fs:t_duration;
x_sampled = cos(pi * mu * n.^2 + 2 * pi * f1 * n + phi);

% Plot sampled signal
figure
stem(n, x_sampled);
title('Sampled Chirp Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Aliasing check
max_instantaneous_freq = max(instantaneous_freq_values);
if max_instantaneous_freq > fs / 2
    disp('Aliasing might occur since the maximum instantaneous frequency exceeds Nyquist rate.');
else
    disp('No aliasing issue detected.');
end
