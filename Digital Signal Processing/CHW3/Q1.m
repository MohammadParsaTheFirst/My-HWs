close all; 
clc; 
clear;

% Define the system coefficients
coeff_b = [1, 2, 0, 1]; % Coefficients for x[n]
coeff_a = [1, -0.5, 0.25]; % Coefficients for y[n]

%% Part (a): Impulse response using recursive method
num_samples = 100; % Number of samples
input_x = zeros(1, num_samples + 1); % Initialize input
output_y = zeros(1, num_samples + 1); % Initialize impulse response
input_x(1) = 1; % Impulse at n=0

% Recursive calculation based on difference equation
for index_n = 1:num_samples
    if index_n == 1
        output_y(index_n) = input_x(index_n);
    elseif index_n == 2
        output_y(index_n) = 2*input_x(index_n-1) + 0.5*output_y(index_n-1);
    elseif index_n == 3
        output_y(index_n) = 0.5*output_y(index_n-1) - 0.25*output_y(index_n-2);
    elseif index_n == 4
        output_y(index_n) = input_x(index_n-3) + 0.5*output_y(index_n-1) - 0.25*output_y(index_n-2);
    else 
        output_y(index_n) = 0.5*output_y(index_n-1) - 0.25*output_y(index_n-2);
    end
end

% Plot impulse response
figure;
stem(0:num_samples, output_y, 'filled');
title('Impulse Response h[n]');
xlabel('n');
ylabel('h[n]');
grid on;

%% Part (b): Z-transform and stability check
% Transfer function H(z)
transfer_H = tf(coeff_b, coeff_a, -1); % Define transfer function with discrete time
disp('Transfer Function H(z):');
transfer_H

% Display the impulse response using impz
impulse_response_len = 100; % Length of impulse response
impulse_response_vals = impz(coeff_b , coeff_a, impulse_response_len);
% Plot the impulse response
index_n = 0:impulse_response_len-1;
figure;
stem(index_n, impulse_response_vals , 'filled');
xlabel('n');
ylabel('h[n]');
title('Impulse Response h[n] of the System');
grid on;

% Check poles for stability
poles_H = pole(transfer_H);
disp('Poles of the system:');
disp(poles_H);
if all(abs(poles_H) < 1)
    disp('The system is stable.');
else
    disp('The system is unstable.');
end

%% Part (c): Input response
index_n = 0:100; % Define the range for n
input_x = 5 + 3*cos(0.2*pi*index_n) + 4*sin(0.6*pi*index_n); % Define the input signal

% Compute the output using the filter function
output_y = filter(coeff_b, coeff_a, input_x); % Apply the transfer function to the input signal

% Plot the input
figure;
stem(index_n, input_x, 'filled');
title('Input x[n]');
xlabel('n');
ylabel('x[n]');
grid on;

% Plot the response to the specific input
figure;
stem(index_n, output_y, 'filled');
title('Response to Input y[n]');
xlabel('n');
ylabel('y[n]');
grid on;

%% Part (d): Amplitude of frequency response
% Frequency response plot
[response_H, freq_w] = freqz(coeff_b, coeff_a, 'whole', 1024);
freq_w = freq_w - pi;  % Shift to range [-pi, pi]
response_H = fftshift(response_H); % Center the zero-frequency component

% Plot magnitude
figure;
plot(freq_w, abs(response_H));
title('Magnitude Response');
xlabel('Frequency (rad/sample)');
ylabel('|H(e^{j\omega})|');
grid on

%%  Part (e): Phase of frequency response
% Plot phase (wrapped)
figure;
plot(freq_w, angle(response_H));
title('Phase Response (Wrapped)');
xlabel('Frequency (rad/sample)');
ylabel('Phase (radians)');
grid on

% Plot phase (unwrapped)
figure;
plot(freq_w, unwrap(angle(response_H)));
title('Phase Response (Unwrapped)');
xlabel('Frequency (rad/sample)');
ylabel('Phase (radians)');
grid on

%% Part (f): Pole-Zero plot
% Compute the zeros , poles , and gain using tf2zp
[zeros_H, poles_H, gain_H] = tf2zpk(coeff_b, coeff_a);

% Display the zeros , poles
disp('Zeros of the system:');
disp(zeros_H);
disp('Poles of the system:');
disp(poles_H);

% Plot the zeros and poles using pzplot
figure;
pzplot(transfer_H); % -1 indicates Z-domain
title('Pole -Zero Plot of the System');
grid on;
