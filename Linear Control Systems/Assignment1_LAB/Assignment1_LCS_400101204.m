%% LCS LAB Assignment1
% 400101204
% MohammadParsa Dini
disp('MohammadParsa Dini - std id: 400101204')
%% 2.1
%definitions
t = linspace(-1, 15, 1000); 
u1 = t >= 0;      % u(t)
u2 = t >= 1;      % u(t-1)

y1 = 2 * exp(-0.5 * t) .* u1;
y2 = (t.^2 .* sin(2 * pi * t) .* exp(-0.5 * t)) .* u2;

%plottings
figure('Name','2.1 Plot');
plot(t, y1, 'LineWidth', 0.95,'Color',[1 0 0]); 
hold on;                  
plot(t, y2, 'LineWidth', 1,'LineStyle','--','Color',[0 0 1]); 
title('2.1 Plot');
xlabel('Time (s)');
xlim([-1 15])
ylabel('y(t)');
legend('y_1(t)', 'y_2(t)');
grid minor;
hold off;

%% 2.2


% Part (2.2 - a)

% Define state-space matrices
A = [-4, 2, 1; 1, -4, 1; -1, 0, -3];
B = [1; 0; 1];
C = [1, 1, 1];
D = [0];

% Create state-space model
sys_ss = ss(A, B, C, D);

sys_tf = tf(sys_ss);
disp('Transfer Function:\n');
sys_tf

%% Part (2.2 - b)

poles = pole(sys_tf);
zeros = zero(sys_tf);
gain = dcgain(sys_tf); % DC gain of the system

disp('2.2 - b\n')
disp('Poles:\n');
disp(poles);
disp('Zeros:\n');
disp(zeros);
disp('Gain:\n');
disp(gain);

%% Part (2.2 - c)

% Step response
subplot(2,2,1);
step(sys_tf);
title('2.2 -c: Step Response');
%grid('True');
subplot(2,2,2);
rlocus(sys_tf);
title('2.2 -c: Root Locus');
%grid('True');
subplot(2,2,3);
bode(sys_tf);
title('2.2 -c: Bode Diagram');
%grid('True');
subplot(2,2,4);
nyquist(sys_tf);
title('2.2 -c: Nyquist Diagram');
%grid('True');

%% Part (2.2 - d)

% Calculate margins explicitly using the margin command
[gm, pm, Wcg, Wcp] = margin(sys_tf);

if isinf(gm)
    disp('Gain Margin: Infinite (system is stable for all positive gains)');
else
    disp(['Gain Margin (dB): ', num2str(20*log10(gm))]);
end

if isinf(pm)
    disp('Phase Margin: Infinite (system is stable for all positive gains)');
else
    disp(['Phase Margin (degrees): ', num2str(pm)]);
end

if isnan(Wcg)
    disp('Gain Crossover Frequency: Undefined (system does not cross 0 dB)');
else
    disp(['Gain Crossover Frequency (rad/s): ', num2str(Wcg)]);
end

if isnan(Wcp)
    disp('Phase Crossover Frequency: Undefined (system does not reach -180° phase)');
else
    disp(['Phase Crossover Frequency (rad/s): ', num2str(Wcp)]);
end

% Plot margin diagram
figure;
margin(sys_tf);
title('Gain and Phase Margins');

%% Part (2.2 - e)

% Step response information
info = stepinfo(sys_tf);

disp('Step Response Properties:');
disp(['Rise Time: ', num2str(info.RiseTime)]);
disp(['Settling Time: ', num2str(info.SettlingTime)]);
disp(['Overshoot: ', num2str(info.Overshoot)]);
disp(['Peak: ', num2str(info.Peak)]);
disp(['Peak Time: ', num2str(info.PeakTime)]);
%%



