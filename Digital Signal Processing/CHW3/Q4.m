clc; clear; close all;

% Define the range of n for time domain signals
time_n = -50:50;  % Adjust range as needed for accurate FFT

% Define signals sig1[n] and sig2[n]
sig1 = (sin(pi/10 * time_n).^2) ./ ((pi/10 * time_n).^2);
sig2 = sin(pi/10 * time_n) ./ (pi/10 * time_n);

% Handle the time_n = 0 case to avoid division by zero
sig1(time_n == 0) = 1; % lim sig1 as time_n -> 0
sig2(time_n == 0) = 1; % lim sig2 as time_n -> 0

% Plot sig1[n] and sig2[n] in the time domain
figure;
subplot(2,1,1);
stem(time_n, sig1, 'filled');
title('Time Domain Signal sig1[n]');
xlabel('time_n');
ylabel('sig1[n]');
grid on;

subplot(2,1,2);
stem(time_n, sig2, 'filled');
title('Time Domain Signal sig2[n]');
xlabel('time_n');
ylabel('sig2[n]');
grid on;

% Compute Fourier Transforms using fft and fftshift for centered frequency range
FT_sig1 = fftshift(fft(sig1, 1024));  % Use 1024-point FFT for better resolution
FT_sig2 = fftshift(fft(sig2, 1024));
freq_f = linspace(-pi, pi, 1024);  % Frequency vector in [-pi, pi]

% Plot Fourier Transforms of sig1[n] and sig2[n]
figure;
subplot(2,1,1);
plot(freq_f, abs(FT_sig1));
title('Magnitude of Fourier Transform |FT\_sig1(\omega)|');
xlabel('Frequency (\omega)');
ylabel('|FT\_sig1(\omega)|');
grid on;

subplot(2,1,2);
plot(freq_f, abs(FT_sig2));
title('Magnitude of Fourier Transform |FT\_sig2(\omega)|');
xlabel('Frequency (\omega)');
ylabel('|FT\_sig2(\omega)|');
grid on;

% Part (b): Define modified signals mod_sig1[n], mod_sig2[n], mod_sig3[n] based on sig2[n]
% mod_sig1[n] = sig2[2n]
mod_sig1 = sig2(1:2:end);  % Direct downsampling, resulting in half the length of sig2
mod_sig1 = mod_sig1(mod_sig1 ~= 0);

% mod_sig2[n] as described in the question
mod_sig2 = zeros(size(time_n));
mod_sig2(mod(time_n, 2) == 0) = sig2((time_n(mod(time_n, 2) == 0) / 2) + (length(time_n) + 1) / 2);

% mod_sig3[n] = sig2[n] * sin(2π * 0.3 * n)
mod_sig3 = sig2 .* sin(2 * pi * 0.3 * time_n);

% Plot sig2[n], mod_sig1[n], mod_sig2[n], mod_sig3[n] in the time domain
figure
subplot(4,1,1);
stem(time_n, sig2, 'filled');
title('Time Domain Signal sig2[n]');
xlabel('time_n');
ylabel('sig2[n]');
grid on;

subplot(4,1,2);
stem(-25:25, mod_sig1, 'filled');  % Plot mod_sig1 with correct indices
title('Time Domain Signal mod\_sig1[n] = sig2[2n]');
xlabel('time_n');
ylabel('mod\_sig1[n]');
grid on;

subplot(4,1,3);
stem(time_n, mod_sig2, 'filled');
title('Time Domain Signal mod\_sig2[n]');
xlabel('time_n');
ylabel('mod\_sig2[n]');
grid on;

subplot(4,1,4);
stem(time_n, mod_sig3, 'filled');
title('Time Domain Signal mod\_sig3[n] = sig2[n] \cdot \sin(2 \pi \cdot 0.3 \cdot time_n)');
xlabel('time_n');
ylabel('mod\_sig3[n]');
grid on;

% Compute Fourier Transforms of mod_sig1[n], mod_sig2[n], mod_sig3[n]
FT_mod_sig1 = fftshift(fft(mod_sig1, 1024));
FT_mod_sig2 = fftshift(fft(mod_sig2, 1024));
FT_mod_sig3 = fftshift(fft(mod_sig3, 1024));

% Plot Fourier Transforms of mod_sig1[n], mod_sig2[n], mod_sig3[n]
figure;
subplot(4,1,1);
plot(freq_f, abs(FT_sig2));
title('Magnitude of Fourier Transform |FT\_sig2(\omega)|');
xlabel('Frequency (\omega)');
ylabel('|FT\_sig2(\omega)|');
grid on;

subplot(4,1,2);
plot(freq_f, abs(FT_mod_sig1));
title('Magnitude of Fourier Transform |FT\_mod\_sig1(\omega)| for mod\_sig1[n] = sig2[2n]');
xlabel('Frequency (\omega)');
ylabel('|FT\_mod\_sig1(\omega)|');
grid on;

subplot(4,1,3);
plot(freq_f, abs(FT_mod_sig2));
title('Magnitude of Fourier Transform |FT\_mod\_sig2(\omega)| for mod\_sig2[n]');
xlabel('Frequency (\omega)');
ylabel('|FT\_mod\_sig2(\omega)|');
grid on;

subplot(4,1,4);
plot(freq_f, abs(FT_mod_sig3));
title('Magnitude of Fourier Transform |FT\_mod\_sig3(\omega)| for mod\_sig3[n] = sig2[n] \cdot \sin(2 \pi \cdot 0.3 \cdot time_n)');
xlabel('Frequency (\omega)');
ylabel('|FT\_mod\_sig3(\omega)|');
grid on;
