% BM20A6100 Advanced Data Analysis and Machine Learning
% Practical Assignment - NASA A2

clc
close all
clearvars

% Loading the datasets:
DATA1 = load("train_FD001.txt");
DATA2 = load("train_FD002.txt");
DATA3 = load("train_FD003.txt");
DATA4 = load("train_FD004.txt");

% compute the pca to the datasets and plot two first PC's in biplot

lab = ["1","2","3","4","5","5","6","7","8","9","10","11","12","13","14",...
    "15","16","18","19","20","21","22","23","24","25","26"];

[coeff1,score1,latent1,tsquared1,explained1,mu1] = pca(DATA1);

figure
biplot(coeff1(:,1:2), 'Scores', score1(:,1:2), VarLabels=lab)
title("train\_FD001")

[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(DATA2);

figure
biplot(coeff2(:,1:2), 'Scores', score2(:,1:2), VarLabels=lab)
title("train\_FD002")

[coeff3,score3,latent3,tsquared3,explained3,mu3] = pca(DATA3);

figure
biplot(coeff3(:,1:2), 'Scores', score3(:,1:2), VarLabels=lab)
title("train\_FD003")

[coeff4,score4,latent4,tsquared4,explained4,mu4] = pca(DATA4);

figure
biplot(coeff4(:,1:2), 'Scores', score4(:,1:2), VarLabels=lab)
title("train\_FD004")

