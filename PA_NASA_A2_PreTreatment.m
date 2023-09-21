% BM20A6100 Advanced Data Analysis and Machine Learning
% Practical Assignment - NASA A2
clc; close all; clearvars

%% Settings:
TrainPercentage         = 70;
ValidationPercentage    = 10;
TestPercentage          = 20;


rng(0)   % Default permutation of the train, val and test sets.

%% Column names:
% this does not work now, some columns deleted because they were constants
ColNames = string([]);
for i = 1:21
    ColNames(i) = sprintf("Sensor %d", i);
end
ColNames(22) = "RUL";
VarLabels = ColNames;

%% Loading the datasets, compute RUL, cut the first 5 columns % and normalize (center and scale)
DATA = cell(1,4);
model_DATA = cell(1, 4);
for i = 1:4
    DATA{i} = RUL_fun(load(sprintf("train_FD00%d.txt", i)));
    DATA{i} = DATA{i}(:, 6:end);
    ind = find(var(DATA{i}) ~=0);
    model_DATA{i}.VarLabels = VarLabels(ind);
    temp = DATA{i}(:, ind);
    DATA{i} = normalize(temp);
end

%% Put the data into a single matrix: ? or dodo we dodo 4 model for different settings
% DATA = [DATA1; DATA2; DATA3; DATA4];

%% Box plots:
figure;
for i = 1:4
    subplot(2, 2, i)
    boxplot(DATA{i}, model_DATA{i}.VarLabels);
    title(sprintf("Box plot of normalized dataset %d", i))
    xtickangle(90)  
end

%% Split the data into three groups: calibration, validation, testing
for i = 1:4
    % Number of data in each dataset:
    N = size(DATA{i}, 1);

    % Permutation of the data:
    I = randperm(N);

    % Index cutoffs:
    Train = floor(N * TrainPercentage / 100);
    Test = floor(N - N * TestPercentage / 100);
    
    % Split the data:
    model_DATA{i}.train = DATA{i}(I(1:Train), :);
    model_DATA{i}.test = DATA{i}(I(Test:end), :);
    model_DATA{i}.validation = DATA{i}(I(Train+1:Test-1), :);
end

%% PCA:
for i = 1:4
    [model_DATA{i}.coeff, model_DATA{i}.score, model_DATA{i}.latent, ...
        model_DATA{i}.tsquared, model_DATA{i}.explained, model_DATA{i}.mu] ...
        = pca(model_DATA{i}.train);
end

%% Principal component explained %'s
figure;
for i = 1:4
    subplot(2, 2, i)
    hold on
    title("Dataset " + i)
    yyaxis left
    bar(model_DATA{i}.explained)
    ylabel("Individual componenet (%)")
    yyaxis right
    plot(cumsum(model_DATA{i}.explained), 'x--', 'linewidth', 2, 'markersize', 10)
    xlabel("Principal component")
    ylabel("Cumulative componenet (%)")
    hold off
    model_DATA{i}.ncomp = find(cumsum(model_DATA{i}.explained) >= 90, 1, 'first');
end
sgtitle("PCA: Explained variance")

%% T2 and SPEx

% T2 and SPEx
for i = 1:4
    model_DATA{i}.T2 = t2comp(model_DATA{i}.train, model_DATA{i}.coeff, model_DATA{i}.latent, model_DATA{i}.ncomp);
    model_DATA{i}.Q = qcomp(model_DATA{i}.train, model_DATA{i}.coeff, model_DATA{i}.ncomp);
    
    % T2 Control limits:
    model_DATA{i}.meanT2 = mean(model_DATA{i}.T2);          % Mean of T2
    model_DATA{i}.stdT2 = std(model_DATA{i}.T2);            % STD of T2
    model_DATA{i}.T2l = model_DATA{i}.meanT2 - 2*model_DATA{i}.stdT2; % 2 stds below
    model_DATA{i}.T2u = model_DATA{i}.meanT2 + 2*model_DATA{i}.stdT2; % 2 stds above
    
    % SPEx Control limits:
    model_DATA{i}.meanQ = mean(model_DATA{i}.Q);          % Mean of SPEx
    model_DATA{i}.stdQ = std(model_DATA{i}.Q);            % STD of SPEx
    model_DATA{i}.Ql = model_DATA{i}.meanQ - 2*model_DATA{i}.stdQ; % 2 stds below
    model_DATA{i}.Qu = model_DATA{i}.meanQ + 2*model_DATA{i}.stdQ; % 2 stds above
end
 
% Samples outside of the bounds:
for i = 1:4
    model_DATA{i}.T2_over = find((model_DATA{i}.T2 > model_DATA{i}.T2u));
    model_DATA{i}.Q_over = find((model_DATA{i}.Q > model_DATA{i}.Qu));
    model_DATA{i}.C_over = intersect(model_DATA{i}.T2_over, model_DATA{i}.Q_over);
end

% Classify the samples in such a way that the "worst" group is the defined
% label:
for i = 1:4
    model_DATA{i}.N = length(model_DATA{i}.T2);
    SampleC = string(ones(1, model_DATA{i}.N));
    SampleC(model_DATA{i}.T2_over) = "T2";
    SampleC(model_DATA{i}.Q_over) = "SPEx";
    SampleC(model_DATA{i}.C_over) = "Common";
    % Categorize the sample class vector
    model_DATA{i}.SampleC = categorical(SampleC);
end

%% Plot the T2 and SPEx charts:

% T2
figure;
for i = 1:4
    subplot(2, 2, i)
    samp = 1:model_DATA{i}.N;
    hold on
    title("Dataset " + i)
    ylabel("T2")
    xlabel("Sample")
    gscatter(samp', model_DATA{i}.T2, model_DATA{i}.SampleC', 'krbc', '.oxx')
    plot(samp, model_DATA{i}.T2u*ones(size(samp)), 'r--')
    plot(samp, model_DATA{i}.T2l*ones(size(samp)), 'r--')
    hold off
end
sgtitle("T2 chart")

% SPEx
figure;
for i = 1:4
    subplot(2, 2, i)
    samp = 1:model_DATA{i}.N;
    hold on
    title("Dataset " + i)
    ylabel("T2")
    xlabel("Sample")
    gscatter(samp', model_DATA{i}.Q, model_DATA{i}.SampleC', 'krbc', '.oxx')
    plot(samp, model_DATA{i}.Qu*ones(size(samp)), 'r--')
    plot(samp, model_DATA{i}.Ql*ones(size(samp)), 'r--')
    hold off
end
sgtitle("SPEx chart")

% Combined
figure;
for i = 1:4
    subplot(2, 2, i)
    hold on
    title("Dataset " + i)
    ylabel("SPEx")
    xlabel("T2")
    gscatter(model_DATA{i}.T2, model_DATA{i}.Q, model_DATA{i}.SampleC', 'krbc', '.oxx')
    plot([min(model_DATA{i}.T2), max(model_DATA{i}.T2)], [model_DATA{i}.Qu, model_DATA{i}.Qu], 'r--')
    plot([model_DATA{i}.T2u, model_DATA{i}.T2u], [min(model_DATA{i}.Q), max(model_DATA{i}.Q)], 'r--')
    hold off
end
sgtitle("Combined chart")
    
%% Functions:
function DATA = RUL_fun(DATA)
    col = size(DATA, 2) + 1;
    for i = 1:max(DATA(:, 1))
        ind = find(DATA(:, 1) == i);
        DATA(ind, col) = max(DATA(ind, 2));
    end
    DATA(:, col) = DATA(:, col) - DATA(:, 2);
end

function T2     = t2comp(data, loadings, latent, comp)
        score       = data * loadings(:,1:comp);
        standscores = bsxfun(@times, score(:,1:comp), 1./sqrt(latent(1:comp,:))');
        T2          = sum(standscores.^2,2);
end

% SPEx
function [lev, levlim]=leveragex(T)
% INPUT:
% 	T =  X scores
% OUTPUT:
% 	lev = leverage values (X space)
%   levlim = approximation for 95 % confidence limits

lev = zeros(length(T),1);
[m,~] = size(T);
for i=1:m
    lev(i,1)=T(i,:)*inv(T'*T)*T(i,:)'+1/length(T); %#ok<MINV>
end
plot(lev,'-o');hold on
levlim=2*length(T(1,:))/length(T(:,1));
plot([1,length(T)],[levlim levlim]','-');
hold off
end

function T2varcontr    = t2contr(data, loadings, latent, comp)
score           = data * loadings(:,1:comp);
standscores     = bsxfun(@times, score(:,1:comp), 1./sqrt(latent(1:comp,:))');
T2contr         = abs(standscores*loadings(:,1:comp)');
T2varcontr      = sum(T2contr,1);
end

function Qcontr   = qcontr(data, loadings, comp)
score         = data * loadings(:,1:comp);
reconstructed = score * loadings(:,1:comp)';
residuals     = bsxfun(@minus, data, reconstructed);
Qcontr        = sum(residuals.^2);
end

function Qfac   = qcomp(data, loadings, comp)
score       = data * loadings(:,1:comp);
reconstructed = score * loadings(:,1:comp)';
residuals   = bsxfun(@minus, data, reconstructed);
Qfac        = sum(residuals.^2,2);
end
