%%
clc
close all
clearvars

% Loading the datasets:
DATA1 = load("train_FD001.txt");
DATA2 = load("train_FD002.txt");
DATA3 = load("train_FD003.txt");
DATA4 = load("train_FD004.txt");

Data(1).D = DATA1;
Data(2).D = DATA2;
Data(3).D = DATA3;
Data(4).D = DATA4;


for k = 1:length(Data)
    % Exploratory Analysis
    nobs = length(Data(k).D(:,1));
    nvar = length(Data(k).D(1,:));
    
    for i = 1:nvar
        Count(i)    = nnz(~isnan(Data(k).D(:,i)));
        Mean(i)     = mean(Data(k).D(:,i));
        SD(i)       = std(Data(k).D(:,i));
        Min(i)      = min(Data(k).D(:,i));
        P25(i)      = prctile(Data(k).D(:,i), 25);
        P50(i)      = prctile(Data(k).D(:,i), 50);
        P75(i)      = prctile(Data(k).D(:,i), 75);
        Max(i)      = max(Data(k).D(:,i));
    end
    
    % Centering importance
    treatments = ["Non-treated data", "Mean Centered", "Z-score (STD)", ...
        "Z-score (robust)", "Center and p-norm", "Center and MAD"];
    for i = 1:length(treatments)
        model(i).treatmentName = treatments(i);
    end
    
    % Scale Data
    model(1).X      = Data(k).D(:,3:end);
    model(2).X      = model(1).X - mean(model(1).X);
    model(3).X      = zscore(model(1).X);
    model(4).X      = normalize(model(1).X, 'zscore', 'robust');
    model(5).X      = normalize(model(2).X, "norm");
    model(6).X      = normalize(model(1).X, 'center', 'mean', 'scale', 'mad');
    
    figure
    for j = 1:length(treatments)
        subplot(2,3,j)
        boxplot(model(j).X)
        hold on
        title(strcat("Scores plot - ", model(j).treatmentName));
    end
    hold off
end

%% Visualizing the pretreated data

% the center and p-norm were chosen to be the normalization method.
% Visualizing all the datasets as normalized

figure
subplot(2,2,1)
boxplot(normalize(DATA1(:, 1:end)))
title("train\_FD001")

subplot(2,2,2)
boxplot(normalize(DATA2(:, 1:end)))
title("train\_FD002")

subplot(2,2,3)
boxplot(normalize(DATA3(:, 1:end)))
title("train\_FD003")

subplot(2,2,4)
boxplot(normalize(DATA4(:, 1:end)))
title("train\_FD004")
%%
ColNames = string([]);
for i = 1:21
    ColNames(i) = sprintf("Sensor %d", i);
end
ColNames(22) = "RUL";
VarLabels = ColNames;

DATA = cell(1,4);
model_DATA = cell(1, 4);
for i = 1:4
    DATA{i} = load(sprintf("train_FD00%d.txt", i));
    DATA{i} = DATA{i}(:, 6:end);
    ind = find(var(DATA{i}) > 10^(-10));
    model_DATA{i}.VarLabels = VarLabels(ind);
    temp = DATA{i}(:, ind);
    DATA{i} = normalize(temp);
end


for i=1:length(DATA)
    figure
    hold on
    for j = 1:size(DATA{i},2)
        subplot(4,6,j)
        histogram(DATA{i}(:,j));
    end
    sgtitle("Variable histograms for the model " + i)
end

