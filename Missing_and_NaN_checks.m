% BM20A6100 Advanced Data Analysis and Machine Learning
% Practical Assignment - NASA A2
clc; close all; clearvars
% Loading the datasets:
DATA1 = load("train_FD001.txt");
DATA2 = load("train_FD002.txt");
DATA3 = load("train_FD003.txt");
DATA4 = load("train_FD004.txt");

% Plotting out the timeseries time steps:
figure(1)
plot(DATA1(:, 2), '.')

figure(2)
plot(DATA2(:, 2), '.')

figure(3)
plot(DATA3(:, 2), '.')

figure(4)
plot(DATA4(:, 2), '.')

% Check if there are missing entries (NaN values):
DATA1_nan_count = sum(isnan(DATA1), 'all');
DATA2_nan_count = sum(isnan(DATA2), 'all');
DATA3_nan_count = sum(isnan(DATA3), 'all');
DATA4_nan_count = sum(isnan(DATA4), 'all');

% Check if there are zeros in the data:
DATA1_zero_count = sum(DATA1 == 0, 'all');
DATA2_zero_count = sum(DATA2 == 0, 'all');
DATA3_zero_count = sum(DATA3 == 0, 'all');
DATA4_zero_count = sum(DATA4 == 0, 'all');

% Check if the zeros are within the options:
DATA1_zero_opt_count = sum(DATA1(:, 3:4) == 0, 'all');
DATA2_zero_opt_count = sum(DATA2(:, 3:4) == 0, 'all');
DATA3_zero_opt_count = sum(DATA3(:, 3:4) == 0, 'all');
DATA4_zero_opt_count = sum(DATA4(:, 3:4) == 0, 'all');

% Check if there are missing steps in the time series for each unit:
non_full_timeseries_count = 0;
for i = 1:max(DATA1(:, 1))
    R = find(DATA1(:, 1) == i); % Find the number of observations for the unit
    non_full_timeseries_count = non_full_timeseries_count + (length(R) ~= max(DATA1(R, 2))); % Check if the operating cycle count ~= to number of obs.
end

for i = 1:max(DATA2(:, 1))
    R = find(DATA2(:, 1) == i);
    non_full_timeseries_count = non_full_timeseries_count + (length(R) ~= max(DATA2(R, 2)));
end

for i = 1:max(DATA3(:, 1))
    R = find(DATA3(:, 1) == i);
    non_full_timeseries_count = non_full_timeseries_count + (length(R) ~= max(DATA3(R, 2)));
end

for i = 1:max(DATA4(:, 1))
    R = find(DATA4(:, 1) == i);
    non_full_timeseries_count = non_full_timeseries_count + (length(R) ~= max(DATA4(R, 2)));
end

% Visualize the data on the settings and measurements:
figure(5)
boxplot(normalize(DATA1(:, 1:end)))
title("train\_FD001")

figure(6)
boxplot(normalize(DATA2(:, 1:end)))
title("train\_FD002")

figure(7)
boxplot(normalize(DATA3(:, 1:end)))
title("train\_FD003")

figure(8)
boxplot(normalize(DATA4(:, 1:end)))
title("train\_FD004")

% Plot timeseries for units sensor measurements over time:
scalingF = @(x) x./max(x); % + min(x);
figure(9)
xlabel("Start of experiment to breakage (0, 1)")
ylabel("Sensor measurement")
title("train\_FD001, column 6")
hold on
for i = 1:max(DATA1(:, 1))
    ind = (DATA1(:, 1) == i);
    plot(scalingF(DATA1(ind, 2)), DATA1(ind, 6))
end
hold off

figure(10)
hold on
xlabel("Start of experiment to breakage (0, 1)")
ylabel("Sensor measurement")
title("train\_FD001, column 12")
for i = 1:max(DATA1(:, 1))
    ind = (DATA1(:, 1) == i);
    plot(scalingF(DATA1(ind, 2)), DATA1(ind, 12))
end
hold off

figure(11)
hold on
xlabel("Start of experiment to breakage (0, 1)")
ylabel("Sensor measurement")
title("train\_FD001, column 14")
for i = 1:max(DATA1(:, 1))
    ind = (DATA1(:, 1) == i);
    plot(scalingF(DATA1(ind, 2)), DATA1(ind, 14))
end
hold off

figure(12)
hold on
xlabel("Start of experiment to breakage (0, 1)")
ylabel("Sensor measurement")
title("train\_FD001, column 26")
for i = 1:max(DATA1(:, 1))
    ind = (DATA1(:, 1) == i);
    plot(scalingF(DATA1(ind, 2)), DATA1(ind, 26))
end
hold off
