%% Part 1

path_folder = 'C:\Users\USER\Downloads\Wav_Files\'; 
path1 = 'DialedSequence_NoNoise.wav';

full_path = [path_folder, path1];
[x_t, f_s] = audioread(full_path);
t = 0:1:length(x_t)-1;
plot(t, x_t);

%% Part 1
path_folder = 'C:\Users\USER\Downloads\Wav_Files\';
path1 = 'DialedSequence_NoNoise.wav';
full_path = [path_folder, path1];
keys_pressed = detect_dtmf(full_path);

path2 = 'DialedSequence_SNR00dB.wav';
full_path2 = [path_folder, path2];
%% Part 1
path3 = 'DialedSequence_SNR10dB.wav';
full_path3 = [path_folder, path3];
%% Part 1
path4 = 'DialedSequence_SNR20dB.wav';
full_path4 = [path_folder, path4];
%% Part 1
path5 = 'DialedSequence_SNR30dB.wav';
full_path5 = [path_folder, path5];
%% Part 2

detect_dtmf_real_time1(10);


%% functions needed
function keys_pressed = detect_dtmf(full_path)

    [x_t, f_s] = audioread(full_path);

    dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
    dtmf_keys = ['1', '2', '3', 'A'; 
                 '4', '5', '6', 'B'; 
                 '7', '8', '9', 'C'; 
                 '*', '0', '#', 'D'];

    N = length(x_t);
    t = (0:N-1)/f_s; 
    X = fft(x_t); 

    P2 = abs(X/N);
    P1 = P2(1:floor(N/2)+1);
    f = f_s*(0:floor(N/2))/N;

    threshold = 0.1; 
    detected_freqs = f(P1 < threshold);

    keys_pressed = '';
    for i = 1:length(detected_freqs)
        for j = 1:length(dtmf_freqs)
            if abs(detected_freqs(i) - dtmf_freqs(j)) < 0.1
                keys_pressed = [keys_pressed, dtmf_keys(floor((j-1)/4)+1, mod(j-1,4)+1)];
            end
        end
    end

    disp(['Detected numbers: ', keys_pressed]);
end

function detect_dtmf_real_time()
    % Define DTMF frequencies and keys
    dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
    dtmf_keys = ['1', '2', '3', 'A'; 
                 '4', '5', '6', 'B'; 
                 '7', '8', '9', 'C'; 
                 '*', '0', '#', 'D'];

    % Sampling rate (typical for audio signals)
    fs = 8192;

    % Create bandpass filters for each DTMF frequency
    filters = cell(1, length(dtmf_freqs));
    for i = 1:length(dtmf_freqs)
        filters{i} = designfilt('bandpassiir', 'FilterOrder', 8, ...
                                'HalfPowerFrequency1', dtmf_freqs(i) - 10, ...
                                'HalfPowerFrequency2', dtmf_freqs(i) + 10, ...
                                'SampleRate', fs);
    end

    % Create an audio recorder object
    recObj = audiorecorder(fs, 16, 1);

    % Start recording
    disp('Start speaking...')
    record(recObj);

    % Continuously process audio input
    while true
        % Get current audio data
        audio_data = getaudiodata(recObj);

        % Check for DTMF tones
        keys_pressed = '';
        for i = 1:length(dtmf_freqs)
            filtered_signal = filter(filters{i}, audio_data);
            if max(abs(filtered_signal)) > 0.1 % Adjust threshold as needed
                [row, col] = ind2sub(size(dtmf_keys), i);
                keys_pressed = [keys_pressed, dtmf_keys(row, col)];
            end
        end

        % Display detected keys
        if ~isempty(keys_pressed)
            disp(['Detected numbers: ', keys_pressed]);
        end
        
        % Pause for a short time to avoid overwhelming the CPU
        pause(0.1);
    end
end


function detect_dtmf_real_time1(duration)
    % Define DTMF frequencies and keys
    dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633];
    dtmf_keys = ['1', '2', '3', 'A'; 
                 '4', '5', '6', 'B'; 
                 '7', '8', '9', 'C'; 
                 '*', '0', '#', 'D'];

    % Sampling rate (typical for audio signals)
    fs = 8192;

    % Create bandpass filters for each DTMF frequency
    filters = cell(1, length(dtmf_freqs));
    for i = 1:length(dtmf_freqs)
        filters{i} = designfilt('bandpassiir', 'FilterOrder', 8, ...
                                'HalfPowerFrequency1', dtmf_freqs(i) - 10, ...
                                'HalfPowerFrequency2', dtmf_freqs(i) + 10, ...
                                'SampleRate', fs);
    end

    % Create an audio recorder object
    recObj = audiorecorder(fs, 16, 1);

    % Start recording
    disp('Start speaking...');
    record(recObj);

    % Initialize timer
    start_time = tic;
    keys_pressed = '';

    % Continuously process audio input until the specified duration
    while toc(start_time) < duration
        % Pause to allow some audio data to be recorded
        pause(0.5);
        
        % Get current audio data
        audio_data = getaudiodata(recObj, 'double');

        % Check for DTMF tones
        for i = 1:length(dtmf_freqs)
            filtered_signal = filter(filters{i}, audio_data);
            if max(abs(filtered_signal)) > 0.1 % Adjust threshold as needed
                [row, col] = ind2sub(size(dtmf_keys), i);
                keys_pressed = [keys_pressed, dtmf_keys(row, col)];
            end
        end

        % Display detected keys
        if ~isempty(keys_pressed)
            disp(['Detected numbers: ', keys_pressed]);
        end
    end

    % Stop recording
    stop(recObj);

    % Save the recorded audio to a file
    filename = 'Downloads\recorded_audio.wav';
    recorded_audio = getaudiodata(recObj, 'double');
    audiowrite(filename, recorded_audio, fs);

    % Display final detected keys and save confirmation
    disp(['Final detected numbers: ', keys_pressed]);
    disp(['Recorded audio saved to: ', filename]);
end
