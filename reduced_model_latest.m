% BM20A6100 Advanced Data Analysis and Machine Learning
% Practical Assignment - NASA A2
clc; close all; clearvars

%% Settings:
TrainPercentage         = 80;   % Percentage of data used for k-fold and calibration
TestPercentage          = 20;   % Percentage of data used for testing

rng(0)   % Default permutation of the train, val and test sets.

%% Column names:
ColNames = string([]);
% Create a vector of column names for the variables
for i = 1:21
    ColNames(i) = sprintf("Sensor %d", i);
end
ColNames(22) = "RUL";
VarLabels = ColNames;

%% Loading the datasets, compute RUL, split the data, cut the first 5 columns and normalize (center and scale)

% Define the variables for the modelling storage
DATA = cell(1,4);
model_DATA = cell(1, 4);
% Load the dataset, compute the RUL and add it to the dataset
for i = 1:4
    DATA{i} = RUL_fun(load(sprintf("train_FD00%d.txt", i)));
end
%%
% define which variables are not meaningful to the model
deleted = cell(1,4);
deleted{1} = [ 6, 14,  9,  3,  2, 13, 17,  8, 20, 21, 15];
            %[6,  14,  8,  9,  3,  2, 17, 20, 13, 21, 15];
deleted{2} = [ 5,  6, 13,  9, 12, 19,  7,  2,  8,  3, 21, 17, 18];
            %[ 6, 19,  5,  7, 9,  8, 12, 13, 17,  2, 20, 3, 18];
deleted{3} = [15, 21, 20,  6,  7, 12, 10, 14,  9,  2,  3,  8];
            %[15, 21, 20,  6, 7, 12, 10, 14,  9,  2,  3, 8];
deleted{4} = [ 5,  6,  2, 12,  9,  7, 13, 21, 19, 17,  1, 20,  3];
            %[5,   6,  9, 13, 2,  7, 21, 16,  8, 17, 12, 20, 3];
            
% Pretreatment (without normalization):
for i = 1:4
    % Take the number of units in the dataset:
    n_i = max(DATA{i}(:, 1));

    % Randomize the units which are included in the test partition:
    motor_num_i = randperm(n_i, floor(TestPercentage/100 * n_i));

    % Take the units not included in the test partition into the train one:
    model_DATA{i}.train = DATA{i}( ~ismember(DATA{i}(:,1),motor_num_i),:);
    % Take the test partition unit into the test set:
    model_DATA{i}.test = DATA{i}(ismember(DATA{i}(:,1),motor_num_i),:);
    
    % Save the RUL value as the response (Y) for each of the partitions: 
    model_DATA{i}.Ytrain = model_DATA{i}.train(:,end);
    model_DATA{i}.Ytest = model_DATA{i}.test(:,end);
    
    % Save the 1st row to know which unit, each of the entries are for:
    model_DATA{i}.testUnit = model_DATA{i}.test(:, 1);
    model_DATA{i}.trainUnit = model_DATA{i}.train(:, 1);
    
    % Cut all the other measurements from the datasets excl. sensor
    % measurements:
    model_DATA{i}.train = model_DATA{i}.train(:, 6:end-1);
    model_DATA{i}.test = model_DATA{i}.test(:, 6:end-1);

    % delete the non-meaningful variables from the data
    model_DATA{i}.train(:, deleted{i}) = [];
    model_DATA{i}.test(:, deleted{i}) = [];
    
    % Find those sensors, which have 0 variance in the measurements and remove them:
    ind = find(var(model_DATA{i}.train) >= 1e-6); 

    % Cut the labels correspondingly:
    model_DATA{i}.VarLabels = VarLabels;
    model_DATA{i}.VarLabels(deleted{i}) = [];
    model_DATA{i}.VarLabels = model_DATA{i}.VarLabels(ind);
    
    % Cut the dataset correspondingly:
    model_DATA{i}.train = model_DATA{i}.train(:, ind);
    model_DATA{i}.test = model_DATA{i}.test(:, ind);
end


%% PLS

% Allocate the variable for the modelled values:
plsModel = cell(1,4);

LV_per_Data = [4,8,4,8];

% Loop over the 4 different datasets:
for dataset = 1:4
    % Find the units belonging to the train partition:
    units = unique(model_DATA{dataset}.trainUnit);
    % Find the number of units and cut it into 1/4th (k-fold):
    nVal = floor(length(units)/4);

    % Perform PLS on all the cross-validation combinations (4):
    for kfold = 1:4
        % Take the units for the validation partition:
        valUnits = units((kfold - 1) * nVal + 1 : (kfold - 1) * nVal + nVal);

        % Split the data into the calibration and validation partitions:
        plsModel{dataset}.calibration{kfold} = model_DATA{dataset}.train(~ismember(model_DATA{dataset}.trainUnit,valUnits),:);
        plsModel{dataset}.validation{kfold} = model_DATA{dataset}.train(ismember(model_DATA{dataset}.trainUnit,valUnits),:);
        
        % Similarly for the responses:
        plsModel{dataset}.calibrationY{kfold} = model_DATA{dataset}.Ytrain(~ismember(model_DATA{dataset}.trainUnit,valUnits),:);
        plsModel{dataset}.validationY{kfold} = model_DATA{dataset}.Ytrain(ismember(model_DATA{dataset}.trainUnit,valUnits),:);
        
        % Loop over the possible number of latent variables (number of columns):
        for LV = 1:LV_per_Data(dataset)
            % This section of the code is basically the same as in the
            % workshop 2:

            % Normalize the calibration partition (X)
            [plsModel{dataset}.calibrationN{kfold}, plsModel{dataset}.mu{kfold},...
                plsModel{dataset}.sig{kfold}] = ...
                normalize(plsModel{dataset}.calibration{kfold});
            X = plsModel{dataset}.calibrationN{kfold};
    
            % Normalize the validation partition with calb. mean and var (Xt)
            plsModel{dataset}.validationN{kfold} = normalize(plsModel{dataset}.validation{kfold}, "center", ...
                plsModel{dataset}.mu{kfold}, "scale", plsModel{dataset}.sig{kfold});
            Xt = plsModel{dataset}.validationN{kfold};
    
            % Center the responses (Y and Yt) with calibration sets mean:
            plsModel{dataset}.meanY{kfold} = mean(plsModel{dataset}.calibrationY{kfold});
            plsModel{dataset}.calibrationYN{kfold} = plsModel{dataset}.calibrationY{kfold} - plsModel{dataset}.meanY{kfold}; 
            Y = plsModel{dataset}.calibrationYN{kfold};
            plsModel{dataset}.validationYN{kfold} = plsModel{dataset}.validationY{kfold} - plsModel{dataset}.meanY{kfold};
            Yt = plsModel{dataset}.validationYN{kfold};
    
            % Performing PLS:
            [plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.P, ... 
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.T, ...
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.Q, ...
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.U, ...
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.B, ...
                ~, ...
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.MSE, ...
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.stats] = plsregress(X, Y, LV);
    
            % Calculate R2 value:
            % Modelled responses for calibration: 
            Yfit    = [ones(size(X,1),1), X] * plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.B;
            % Squared residuals from mean:
            TSSRes  = sum((Y - mean(Y)).^2);
            % Squared residuals from model values:
            RSSRes  = sum((Y - Yfit).^2);
            % R2:
            plsModel{dataset}.ncomp{LV}.R2(kfold) = 1 - RSSRes / TSSRes;

            % Calculate Q2
            
            % Modelled responses for validation: 
            YfitT   = [ones(size(Xt,1),1), Xt] * plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.B;
            % Squared residuals from model values:
            PRESS = sum((Yt - YfitT).^2);
            % Q2:
            plsModel{dataset}.ncomp{LV}.Q2(kfold) = 1 - PRESS / TSSRes;

            % Storing for later:
            plsModel{dataset}.ncomp{LV}.B(kfold,:) = plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.B;
            
            % Check if the R2 and Q2 values are calculated in a stable
            % manner:
            plsModel{dataset}.ncomp{LV}.Q2(isnan(plsModel{dataset}.ncomp{LV}.Q2)) = 0;
            plsModel{dataset}.ncomp{LV}.Q2(find(plsModel{dataset}.ncomp{LV}.Q2==-Inf)) = 0;
            plsModel{dataset}.ncomp{LV}.meanR2 = nanmean(plsModel{dataset}.ncomp{LV}.R2');
            plsModel{dataset}.ncomp{LV}.meanQ2 = nanmean(plsModel{dataset}.ncomp{LV}.Q2'); 
        end
    end
end

%% Box plots:
% figure;
% for i = 1:4
%     subplot(2, 2, i)
%     boxplot(model_DATA{i}.train, model_DATA{i}.VarLabels);
%     title(sprintf("Box plot of dataset %d", i))
%     xtickangle(90)  
% end

%%  VIP scores

for dataset = 1:4

    nLV = LV_per_Data(dataset);
    plsModel{dataset}.VIP_index = [];
    nVar = length(model_DATA{dataset}.VarLabels);

    count = 1;

    
    VIP_values_per_LV = [];
    for LV = 1:nLV
        nPlot = ceil(sqrt(nLV));
        
        %subplot(nPlot, nPlot, LV)
        VIP_values_per_kfold = [];
        for kfold = 1:4
            % Uses the normalized PLS weights
            plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.W0 = ... 
                plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.stats.W ./  ...
                sqrt(sum(plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.stats.W.^2,1));
            
            p              = size(plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.P, 1);
            
            sumSq          = sum(plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.T.^2,1).* ...
                                sum(plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.Q.^2,1);
            
            vipScore       = sqrt(p*sum(sumSq.* ...
                (plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.W0.^2),2) ./ sum(sumSq,2));
            
            indVIP         = find(vipScore >= 1);
            VIP_values_per_kfold = [VIP_values_per_kfold; vipScore'];

            
            count = count + 1;
        end
        VIP_values_per_LV = [VIP_values_per_LV ;mean(VIP_values_per_kfold)];
        
    end
    VIP_values_per_LV = mean(VIP_values_per_LV);
    indVIP         = find(VIP_values_per_LV >= 1);
    figure;
    hold on
    s = scatter(1:length(VIP_values_per_LV),VIP_values_per_LV,'bx', linewidth=3);
    s.SizeData = 50;
    s = scatter(indVIP, VIP_values_per_LV(indVIP),'rx', linewidth=3);
    s.SizeData = 50;
    plot([1 length(VIP_values_per_LV)],[1 1],'--k')
    axis tight
    title("Dataset " + dataset + " remaining variables")
    xlabel('Predictor Variables')
    ylabel('VIP Scores')
    xticks(1:nVar)
    xticklabels(model_DATA{dataset}.VarLabels);
    hold off
    [least_important_var, del_ind] = min(VIP_values_per_LV);
    disp("For dataset " + dataset + " remove " + model_DATA{dataset}.VarLabels(del_ind) + ", with VIP = " + least_important_var)
end

% calculate the mean over different kfolds and over different latent
% variables used and delete the one that is the least important for all of
% the latent variables



%% R2 - Q2
for dataset = 1:4
    R2 = [];
    Q2 = [];
    for kfold = 1:4
        for LV = 1:LV_per_Data(dataset)
            R2(kfold,LV) = plsModel{dataset}.ncomp{LV}.R2(kfold);
            Q2(kfold,LV) = plsModel{dataset}.ncomp{LV}.Q2(kfold);
        end
    end
    
    figure;
    
    subplot(2, 1, 1)
    % yvalues = {'1', '2', '3', '4'};
    % xvalues = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'};
    % xvalues = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'};
    heatmap(R2);
    ylabel("K-fold");
    xlabel("No. components in the model");
    title("R2 values")
    
    subplot(2, 1, 2)
    % yvalues = {'1', '2', '3', '4'};
    % xvalues = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'};
    % xvalues = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'};
    heatmap(Q2);
    ylabel("K-fold");
    xlabel("No. components in the model");
    title("Q2 values");

    sgtitle("Dataset " + dataset + ", R2 and Q2 values")

%     subplot(3,1,3)
%     plot(1:1:LV_per_Data(dataset),mean(R2,1), 'k')
%     hold on
%     plot(1:1:LV_per_Data(dataset),mean(Q2,1), 'm')
%     xlabel("No. components in the model");
%     title("Mean of the R^2 and Q^2 values over the k-folds")
%     legend("R^2", "Q^2")
%     ylim([0,1])
end


%% Coefficients
for dataset = 1:4
    coeffs = [];
    nLV = LV_per_Data(dataset);
    figure;
    for LV = 1:nLV
        nPlot = ceil(sqrt(nLV));
        subplot(nPlot, nPlot, LV)
        hold on
        coeffs(LV, :) = mean(plsModel{dataset}.ncomp{LV}.B);
        bar(coeffs(LV, :))
        title("LV " + LV)
        hold off
    end
    sgtitle("Dataset " + dataset)
end

%% Testitesting

% Loop over the 4 different datasets:
for dataset = 1:4
    % Perform PLS on all the cross-validation combinations (4):
    for kfold = 1:4
        LV = LV_per_Data(dataset);
        X = plsModel{dataset}.calibrationN{kfold};
        % Normalize the validation partition with calb. mean and var (Xt)
        plsModel{dataset}.testXN{kfold} = normalize(model_DATA{dataset}.test,...
             "center", plsModel{dataset}.mu{kfold}, "scale", plsModel{dataset}.sig{kfold});
        Xt = plsModel{dataset}.testXN{kfold};

        % Center the responses (Y and Yt) with calibration sets mean:
        Y = plsModel{dataset}.calibrationYN{kfold};
        plsModel{dataset}.testYN{kfold} = model_DATA{dataset}.Ytest - plsModel{dataset}.meanY{kfold};
        Yt = plsModel{dataset}.testYN{kfold};

        % Squared residuals from mean:
        TSSRes  = sum((Y - mean(Y)).^2);

        % Calculate Q2
        
        % Modelled responses for validation: 
        plsModel{dataset}.YfitT{kfold}   = [ones(size(Xt,1),1), Xt] * plsModel{dataset}.ncomp{LV}.KFOLD{kfold}.B;
        % Squared residuals from model values:
        PRESS = sum((Yt - plsModel{dataset}.YfitT{kfold}).^2);
        % Q2:
        plsModel{dataset}.Q2test(kfold) = 1 - PRESS / TSSRes;
    end
end

% Plotting true Y values and estimated Y values against each other
mark = ["mo", "co", "ro", "ko"];
for dataset = 1:4
  figure;
  for kfold = 1:4
    subplot(2, 2, kfold)
    hold on
    plot(plsModel{dataset}.testYN{kfold}, plsModel{dataset}.YfitT{kfold}, mark(kfold))
    plot([-400, 400], [-400,400], 'g--', LineWidth=3)
    xlabel("True values")
    ylabel("Predicted values")
    title("Kfold " + kfold)
    hold off
    grid on
  end
  sgtitle("Dataset " + dataset + ", true Y vs. predicted Y")
end

%% Functions:

% Data loading and RUL computation:
function DATA = RUL_fun(DATA)
    % Input:    Loaded dataset
    % Output:   RUL included dataset

    % Compute the RUL for each unit present in the dataset:
    col = size(DATA, 2) + 1;    % RUL value included into the last column
    for i = 1:max(DATA(:, 1)) % Go through the engine entries
        % Find the current engine entries:
        ind = find(DATA(:, 1) == i);
        % Set the last column as the maximum of that units operating
        % cycles:
        DATA(ind, col) = max(DATA(ind, 2));
    end
    % Compute the RUL via. N. of Operating cycles - current opertating
    % cycle
    DATA(:, col) = DATA(:, col) - DATA(:, 2);
end


% Functions from example codes that were used earlier:
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