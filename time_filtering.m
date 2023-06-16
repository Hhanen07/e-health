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

% center the signal
EEGFP1_REF_vector = EEGFP1_REF_vector - mean(EEGFP1_REF_vector);

% define the time vector for the segements
t = (0:1/(fs-1):1);
% define the time vector for the signal
time = (0:height(channelData)/(length(EEGFP1_REF_vector)-1):height(channelData));

% Plot the original signal
figure 
subplot(2, 2, 1);
plot(time,EEGFP1_REF_vector, 'LineWidth', 1);
hold on
grid on
xlabel('Time (s)');
ylabel('Amplitude');
title('Original signal');


% Open the CSV file
fileID = readtable('aaaaaaju_s005_t000.csv');

% read the inputs related to FP1 sensor
for i=1:height(fileID) 
    if ismember(cell2mat(fileID.channel(i)),'{FP1-F7}')
        lines(i,:) = fileID(i,:);
    end
end


%% Plot all the segments together
subplot(2, 2, 3);
plot(t, EEGFP1_REF_Matrix, 'LineWidth', 1)
grid on;
xlabel('Time [s]')
ylabel('EEG segments [ \mu V]')
title('EEG signal segments')

% Identify the segments with artifacts according the CSV inputs
artifacts = [];
for i=1:height(lines)
    start_time = lines.start_time(i);
    stop_time = lines.stop_time(i);
    artifacts = [artifacts (floor(start_time):ceil(stop_time))];
end

% adding the artifact observed when visualizing the segmenst
artifacts = [artifacts [1,11,12]];
segments = 1:width(EEGFP1_REF_Matrix);

% get the good segments
list_good_channels = setdiff(segments, artifacts);
good_segments = EEGFP1_REF_Matrix(:,list_good_channels);
% Plot all the good segments together
subplot(2, 2, 4);
plot(t, good_segments, 'LineWidth', 1)
grid on;
xlabel('Time [s]')
ylabel('EEG segments [ \mu V]')
ylim([-1000,1000])
title('EEG signal Good segments')


% Reconstruct the signal after removing the segments with artifacts
signal_with_no_artifacts = zeros(1,1);
for i = 1:width(good_segments)
    signal_with_no_artifacts = [signal_with_no_artifacts good_segments(:,i)'];
end

% Plot the signal with no artifacts
t = (0:(length(signal_with_no_artifacts)/fs)/(length(signal_with_no_artifacts)-1):length(signal_with_no_artifacts)/fs);
subplot(2, 2, 2);
plot(t,signal_with_no_artifacts', 'b');
grid on
xlabel('Time (s)');
ylabel('Amplitude');
ylim([-1000, 1000]);
title('Filtred signal');

% plot the segments with artifacts
figure
plot((0:1/(fs-1):1), EEGFP1_REF_Matrix(:,artifacts), 'LineWidth', 1)
grid on;
xlabel('Time [s]')
ylabel('EEG segments [ \mu V]')
title('EEG signal artifacted segments')

%create lables for model training 
ydata = zeros(1,width(EEGFP1_REF_Matrix));
for i=1:width(EEGFP1_REF_Matrix)
    if ismember(i, artifacts)
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


