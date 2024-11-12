clc, clear, close all

% Define numerator and denominator coefficients
num = [1 0 -1];
den = [1 0.9 0.6 0.05];

% Use residue function to get the partial fraction expansion
[r, p, k] = residue(num, den);

% Define the number of samples for impulse response
n_samples = 20; % you can adjust this as needed
n = 0:n_samples-1;

% Initialize the impulse response array
h = zeros(1, n_samples);

% Sum the terms from each residue and pole
for i = 1:length(r)
    h = h + r(i) * (p(i) .^ n); % Accumulate each term in the impulse response
end

% Display the impulse response
stem(n, h, 'filled');
title('Impulse Response h[n]');
xlabel('n');
ylabel('h[n]');
grid on;
