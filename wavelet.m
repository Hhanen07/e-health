clear all
close all
clc

% Specify the path and filename of the EDF file
edfFile = 'aaaaaaju_s005_t000.edf'; 
fs = 250;
% Read the EDF file using sload function
data = edfread(edfFile);

% Select a channel for visualization
channelIndex = 1; 

% Extract the selected channel data
channelData = data(:, channelIndex);

% Reconstruct the EEG signal into a vector and save the segments in a matrix
EEGFP1_REF = data.EEGFP1_REF;
EEGFP1_REF_Matrix = zeros(fs, height(channelData));
EEGFP1_REF_vector = zeros(1, height(channelData));
for i = 1:height(channelData)
    instcell = cell2mat(EEGFP1_REF(i));
    center_cell = instcell-mean(instcell); % center the segments
    EEGFP1_REF_Matrix(:,i) = center_cell';
    EEGFP1_REF_vector = [EEGFP1_REF_vector instcell'];
end
clear i
clear instcell
clear EEGFP1_REF

% define the time vector for the signal
time = (0:height(channelData)/(length(EEGFP1_REF_vector)-1):height(channelData));



% Remove the mean
EEGFP1_REF_vector = EEGFP1_REF_vector-mean(EEGFP1_REF_vector);

%% plot the power spectra diagram
windowSize = 1024; % Window size in samples
overlap = windowSize / 2; % Overlap between consecutive windows
nfft = 1024; % Number of FFT points for frequency resolution

% Compute the power spectrum
[powerSpectrum, frequency] = pwelch(EEGFP1_REF_vector, windowSize, overlap, nfft, fs);

% Plot the power spectrum
figure;
plot(frequency, 10*log10(powerSpectrum));
xlabel('Frequency (Hz)');
ylabel('Power (dB)');
title('Power Spectrum of EEG Signal');

%% Define parameters for wavelet 
types = {'bior3.5', 'bior1.3', 'bior5.5'};
threshold = 380;

% iterate through the levels to chose the best level for the wavelet
% decomposition
for level = 1:6
    [C,L] = wavedec(EEGFP1_REF_vector,level,'bior5.5');
    % Plot the approximation coefficients
    subplot(1, level+1, 1);
    plot(C(1:L(1)));
    title('Approximation Coefficients');
    % Plot the detail coefficients
    for i = 1:level
        subplot(1, level+1, i+1);
        plot(C(L(i)+1:L(i+1)));
        title(sprintf('Detail Coefficients Level %d', i));
    end
    % extract the artifacts
    modifiedCoefficients = wthresh(C, 'h', threshold);
    % recontruction of the signal with artifacts
    artifacts = waverec(modifiedCoefficients, L, 'bior5.5');
    figure;
    subplot(2,1,1);
    plot(time, EEGFP1_REF_vector);
    title('Original EEG Signal', level);
    subplot(2,1,2);
    plot(time, artifacts);
    ylim([-5000, 5000]);
    title('Filtered artifacts');
end

level = 3; % chose the level 3 as the most suitable level
ydata = zeros(1,width(EEGFP1_REF_Matrix));
for i=1:width(EEGFP1_REF_Matrix)
    [C,L] = wavedec(EEGFP1_REF_Matrix(:,i)',level,'bior5.5');
    modifiedCoefficients = wthresh(C, 'h', threshold);
    %create dataset for model training    
    if not(all(modifiedCoefficients == 0))
       ydata(i) = 1;
    end
end


%% apply LDA
trainingRatio = 0.8; 
testingRatio = 0.2;

% mix the indexes to get random data for training and testing
indices = randperm(width(EEGFP1_REF_Matrix));
trainingIndices = indices(1:round(trainingRatio * length(indices)));
testingIndices = indices(round(trainingRatio * length(indices)) + 1:end);

trainingData = EEGFP1_REF_Matrix(:,trainingIndices);
trainingLabels = ydata(trainingIndices);
testingData = EEGFP1_REF_Matrix(:,testingIndices);
testingLabels = ydata(testingIndices);
ldaModel = fitcdiscr(trainingData', trainingLabels');
predictedLabels = predict(ldaModel, testingData');

% plot the segments used for testing the model
figure
t = (0:1/(fs-1):1);
subplot(2, 2, 1);
plot(t, testingData, 'LineWidth', 1)
grid on;
xlabel('Time [s]')
ylabel('EEG segments [ \mu V]')
title('EEG signal testing segments')

predictedArtifacts = [];
artifacts_not_predicted = [];
predicted_good_segments_index = [];
for i=1:length(predictedLabels)

    % Get the artifacts that are successfully predicted by the model
    if (predictedLabels(i) ==1 && testingLabels(i) == 1)
        predictedArtifacts = [predictedArtifacts i];
    end

     % Get the artifacts that are not predicted by the model
    if (testingLabels(i) == 1 && predictedLabels(i) == 0)
        artifacts_not_predicted = [artifacts_not_predicted i];
    end

    % Get the good segments that was successfully predicted by the model
    if (predictedLabels(i) == 0 && testingLabels(i) == 0)
        predicted_good_segments_index = [predicted_good_segments_index i];
    end
end

predictedArtifacts_segment = testingData(:,predictedArtifacts);
artifacts_not_predicted_segment = testingData(:,artifacts_not_predicted);
predicted_good_segments = testingData(:,predicted_good_segments_index);

t = (0:1/(fs-1):1);
% plot the artifacts that are successfully predicted by the model
if not(isempty(predictedArtifacts_segment))
    subplot(2, 2, 2);
    plot(t, predictedArtifacts_segment, 'LineWidth', 1)
    grid on;
    xlabel('Time [s]')
    ylabel('EEG segments [ \mu V]')
    title('EEG signal predicted artifacts')
end

% plot the artifacts that are not predicted by the model
if not(isempty(artifacts_not_predicted_segment))
    subplot(2, 2, 3);
    plot(t, artifacts_not_predicted_segment, 'LineWidth', 1)
    grid on;
    xlabel('Time [s]')
    ylabel('EEG segments [ \mu V]')
    title('EEG signal not predicted artifacts')
end 

% plot the good segments that was successfully predicted by the model
if not(isempty(predicted_good_segments))
    subplot(2, 2, 4);
    plot(t, predicted_good_segments, 'LineWidth', 1)
    grid on;
    xlabel('Time [s]')
    ylabel('EEG segments [ \mu V]')
    title('EEG signal predicted good segments')
end 

accuracy = 0;
for i=1:length(predictedLabels)
    if predictedLabels(i) == testingLabels(i)
        accuracy = accuracy + 1/length(predictedLabels);
    end
end

disp(['Accuracy: ' num2str(accuracy)]);
confusionMatrix = confusionmat(testingLabels, predictedLabels)


