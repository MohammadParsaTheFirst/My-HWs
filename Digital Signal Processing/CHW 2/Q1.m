clc, clear, close all

%%
% Example Transfer Function H(z)
num = [1 -0.4 9.64 -3.856 5.76 -2.304];  % Numerator coefficients of H(z)
den = [1 -6 -0.25 1.5 0 0];  % Denominator coefficients of H(z)

% Step 1: Separate H(z) into H_allpass and H_min
[num_Hmin, den_Hmin, num_Hallpass, den_Hallpass] = separateTransferFunction(num, den);

% Step 2: Plot Zeros and Poles
% Plot zeros and poles of H(z)
figure;
zplane(num, den);
title('Zeros and Poles of H(z)');

% Plot zeros and poles of H_min(z)
figure;
zplane(num_Hmin, den_Hmin);
title('Zeros and Poles of H_{min}(z)');

% Plot zeros and poles of H_allpass(z)
figure;
zplane(num_Hallpass, den_Hallpass);
title('Zeros and Poles of H_{allpass}(z)');

% Step 3: Plot Amplitude and Phase Responses
% Define frequency range (normalized from 0 to pi)
omega = linspace(0, pi, 512);  % Frequencies in radians/sample

% Compute frequency responses
[H, W] = freqz(num, den, omega);
[Hmin, Wmin] = freqz(num_Hmin, den_Hmin, omega);
[Hallpass, Whall] = freqz(num_Hallpass, den_Hallpass, omega);

% Plot Amplitude Responses
figure;
subplot(2, 1, 1);
plot(W, abs(H), 'b', 'LineWidth', 2); hold on;
plot(Wmin, abs(Hmin), 'g', 'LineWidth', 2);
plot(Whall, abs(Hallpass), 'r', 'LineWidth', 2);
xlabel('Frequency (radians/sample)');
ylabel('Magnitude');
legend('H(z)', 'H_{min}(z)', 'H_{allpass}(z)');
title('Magnitude Response');
grid on;

% Plot Phase Responses
subplot(2, 1, 2);
plot(W, angle(H), 'b', 'LineWidth', 2); hold on;
plot(Wmin, angle(Hmin), 'g', 'LineWidth', 2);
plot(Whall, angle(Hallpass), 'r', 'LineWidth', 2);
xlabel('Frequency (radians/sample)');
ylabel('Phase (radians)');
legend('H(z)', 'H_{min}(z)', 'H_{allpass}(z)');
title('Phase Response');
grid on;

%%
% Example Transfer Function H(z)
num = [1 -0.4 9.64 -3.856 5.76 -2.304];  % Numerator coefficients of H(z)
den = [1 0 -0.25 0 0 0];  % Denominator coefficients of H(z)

% Step 1: Separate H(z) into H_allpass and H_min
[num_Hmin, den_Hmin, num_Hallpass, den_Hallpass] = separateTransferFunction(num, den);

% Step 2: Plot Zeros and Poles
% Plot zeros and poles of H(z)
figure;
zplane(num, den);
title('Zeros and Poles of H(z)');

% Plot zeros and poles of H_min(z)
figure;
zplane(num_Hmin, den_Hmin);
title('Zeros and Poles of H_{min}(z)');

% Plot zeros and poles of H_allpass(z)
figure;
zplane(num_Hallpass, den_Hallpass);
title('Zeros and Poles of H_{allpass}(z)');

% Step 3: Plot Amplitude and Phase Responses
% Define frequency range (normalized from 0 to pi)
omega = linspace(0, pi, 512);  % Frequencies in radians/sample

% Compute frequency responses
[H, W] = freqz(num, den, omega);
[Hmin, Wmin] = freqz(num_Hmin, den_Hmin, omega);
[Hallpass, Whall] = freqz(num_Hallpass, den_Hallpass, omega);

% Plot Amplitude Responses
figure;
subplot(2, 1, 1);
plot(W, abs(H), 'b', 'LineWidth', 2); hold on;
plot(Wmin, abs(Hmin), 'g', 'LineWidth', 2);
plot(Whall, abs(Hallpass), 'r', 'LineWidth', 2);
xlabel('Frequency (radians/sample)');
ylabel('Magnitude');
legend('H(z)', 'H_{min}(z)', 'H_{allpass}(z)');
title('Magnitude Response');
grid on;

% Plot Phase Responses
subplot(2, 1, 2);
plot(W, angle(H), 'b', 'LineWidth', 2); hold on;
plot(Wmin, angle(Hmin), 'g', 'LineWidth', 2);
plot(Whall, angle(Hallpass), 'r', 'LineWidth', 2);
xlabel('Frequency (radians/sample)');
ylabel('Phase (radians)');
legend('H(z)', 'H_{min}(z)', 'H_{allpass}(z)');
title('Phase Response');
grid on;

%%

function [num_Hmin, den_Hmin, num_Hallpass, den_Hallpass] = separateTransferFunction(num, den)
    % num and den are the numerator and denominator of H(z)
    
    % Step 1: Get zeros and poles of H(z)
    [zeros_H, poles_H, gain_H] = tf2zp(num, den);
    
    % Step 2: Classify zeros and poles
    min_phase_zeros = [];
    min_phase_poles = [];
    allpass_zeros = [];
    allpass_poles = [];
    
    % Classify zeros
    for i = 1:length(zeros_H)
        if abs(zeros_H(i)) < 1  % Inside unit circle: minimum-phase
            min_phase_zeros = [min_phase_zeros, zeros_H(i)];
        else  % Outside unit circle: allpass
            allpass_zeros = [allpass_zeros, zeros_H(i)];
        end
    end
    
    % Classify poles
    for i = 1:length(poles_H)
        if abs(poles_H(i)) < 1  % Inside unit circle: minimum-phase
            min_phase_poles = [min_phase_poles, poles_H(i)];
        else  % Outside unit circle: allpass
            allpass_poles = [allpass_poles, poles_H(i)];
        end
    end
    
    % Step 3: Check zeros in H_allpass and add conjugate reciprocal poles if needed
    for i = 1:length(allpass_zeros)
        z = allpass_zeros(i);  % A zero in H_allpass
        
        % Check if the conjugate reciprocal is in H_allpass poles
        reciprocal_conjugate = 1 / conj(z);
        
        % If the conjugate reciprocal is not already in allpass poles, add it
        if ~any(abs(allpass_poles - reciprocal_conjugate) < 1e-6)
            allpass_poles = [allpass_poles, reciprocal_conjugate];  % Add to H_allpass poles
            min_phase_zeros = [min_phase_zeros, reciprocal_conjugate];  % Add to H_min zeros
        end
    end
    
    % Step 4: Check poles in H_allpass and add conjugate reciprocal zeros if needed
    for i = 1:length(allpass_poles)
        p = allpass_poles(i);  % A pole in H_allpass
        
        % Check if the conjugate reciprocal is in H_allpass zeros
        reciprocal_conjugate = 1 / conj(p);
        
        % If the conjugate reciprocal is not already in allpass zeros, add it
        if ~any(abs(allpass_zeros - reciprocal_conjugate) < 1e-6)
            allpass_zeros = [allpass_zeros, reciprocal_conjugate];  % Add to H_allpass zeros
            min_phase_poles = [min_phase_poles, reciprocal_conjugate];  % Add to H_min poles
        end
    end
    
    % Step 5: Reconstruct the transfer functions H_min and H_allpass
    % For H_min: Use the modified zeros and poles
    num_Hmin = poly(min_phase_zeros);  % Construct numerator from min_phase_zeros
    den_Hmin = poly(min_phase_poles);  % Construct denominator from min_phase_poles
    
    % For H_allpass: Use the modified allpass zeros and poles
    num_Hallpass = poly(allpass_zeros);  % Construct numerator from allpass_zeros
    den_Hallpass = poly(allpass_poles);  % Construct denominator from allpass_poles
    
end
