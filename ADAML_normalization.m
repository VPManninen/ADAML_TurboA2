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
    model(5).X      = normalize(model(2).X, 'norm');
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

















