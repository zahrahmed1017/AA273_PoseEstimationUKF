close all;

% Load SHIRT ROE1 GT
varNames = {'filename', 'xmin', 'ymin', 'xmax', 'ymax', ...
            'qw', 'qx', 'qy', 'qz', 'rx', 'ry', 'rz', ...
            'x0', 'y0', ...
            'kx1',  'ky1',  'kx2',  'ky2',  'kx3',  'ky3', ...
            'kx4',  'ky4',  'kx5',  'ky5',  'kx6',  'ky6', ...
            'kx7',  'ky7',  'kx8',  'ky8',  'kx9',  'ky9', ...
            'kx10', 'ky10', 'kx11', 'ky11'};
varTypes = {'char'};
[varTypes{2:36}] = deal('double');
opts1 = delimitedTextImportOptions('VariableNames', varNames,...
    'VariableTypes', varTypes,...
    'DataLines', 1);
labelFile = "roe1.csv";
partition = 'roe1/lightbox';
dataroot = fullfile('/Users/Zahra1/Documents/Stanford/Research/datasets/shirtv1', partition);
csvVal = readtable(fullfile(dataroot, "labels", labelFile), opts1);

% Load SPN results:
pred_syn = "/Users/Zahra1/Documents/Stanford/Classes/AA273/FinalProject/slab-spn/output/efficientnet_b0.ra_in1k/baseline_incAug_20251022/roe1/synthetic/predictions_pose.mat";
predSyn = load(pred_syn);

pred_lb  = "/Users/Zahra1/Documents/Stanford/Classes/AA273/FinalProject/slab-spn/output/efficientnet_b0.ra_in1k/baseline_incAug_20251022/roe1/lightbox/predictions_pose.mat";
predLb = load(pred_lb);


% Calculate the trajectory position and orientation errors as individual
% components.
N = size(predSyn.heat_t, 1);

E_x_syn     = NaN(1, N);
E_y_syn     = NaN(1, N);
E_z_syn     = NaN(1, N);
E_yaw_syn   = NaN(1, N);
E_pitch_syn = NaN(1, N);
E_roll_syn  = NaN(1, N);

E_x_lb      = NaN(1, N);
E_y_lb      = NaN(1, N);
E_z_lb      = NaN(1, N);
E_yaw_lb    = NaN(1, N);
E_pitch_lb  = NaN(1, N);
E_roll_lb   = NaN(1, N);

% For each entry, calculate the x, y, z translation error and the yaw (Z),
% pitch (Y), roll (X) rotation errors
for i = 1:N

    pos_gt   = table2array(csvVal(i, 10:12));
    quat_gt  = table2array(csvVal(i, 6:9));

    % --- Synthetic ---
    if predSyn.reject(i) == 0
        pos_pred_syn  = predSyn.heat_t(i,:);
        quat_pred_syn = predSyn.heat_q(i,:);

        E_x_syn(i) = pos_gt(1) - pos_pred_syn(1);
        E_y_syn(i) = pos_gt(2) - pos_pred_syn(2);
        E_z_syn(i) = pos_gt(3) - pos_pred_syn(3);

        R1             = quat2dcm(quat_gt);
        R2             = quat2dcm(quat_pred_syn);
        eul_ZYX        = rotm2eul(R2' * R1);
        E_yaw_syn(i)   = rad2deg(eul_ZYX(1));
        E_pitch_syn(i) = rad2deg(eul_ZYX(2));
        E_roll_syn(i)  = rad2deg(eul_ZYX(3));
    end

    % --- Lightbox ---
    if predLb.reject(i) == 0
        pos_pred_lb  = predLb.heat_t(i,:);
        quat_pred_lb = predLb.heat_q(i,:);

        E_x_lb(i) = pos_gt(1) - pos_pred_lb(1);
        E_y_lb(i) = pos_gt(2) - pos_pred_lb(2);
        E_z_lb(i) = pos_gt(3) - pos_pred_lb(3);

        R1            = quat2dcm(quat_gt);
        R2            = quat2dcm(quat_pred_lb);
        eul_ZYX       = rotm2eul(R2' * R1);
        E_yaw_lb(i)   = rad2deg(eul_ZYX(1));
        E_pitch_lb(i) = rad2deg(eul_ZYX(2));
        E_roll_lb(i)  = rad2deg(eul_ZYX(3));
    end

end

time = (1:N) / N * 2;

% Compute mean errors (ignoring NaN from rejected samples)
mu_x_syn     = nanmean(abs(E_x_syn));     mu_x_lb     = nanmean(abs(E_x_lb));
mu_y_syn     = nanmean(abs(E_y_syn));     mu_y_lb     = nanmean(abs(E_y_lb));
mu_z_syn     = nanmean(abs(E_z_syn));     mu_z_lb     = nanmean(abs(E_z_lb));
mu_yaw_syn   = nanmean(abs(E_yaw_syn));   mu_yaw_lb   = nanmean(abs(E_yaw_lb));
mu_pitch_syn = nanmean(abs(E_pitch_syn)); mu_pitch_lb = nanmean(abs(E_pitch_lb));
mu_roll_syn  = nanmean(abs(E_roll_syn));  mu_roll_lb  = nanmean(abs(E_roll_lb));

%% Plots

figure()
fontsize(gca, 20, "points")

subplot(2, 3, 1)
plot(time, E_x_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_x_lb, 'magenta', 'LineWidth', 1)
ylabel('X-axis [m]')
title(sprintf('\\mu_{syn} = %.2f m,  \\mu_{lb} = %.2f m', mu_x_syn, mu_x_lb))
fontsize(gca, 16, "points")
ylim([-1 1])
grid on;
legend('Synthetic', 'Lightbox')

subplot(2, 3, 2)
plot(time, E_y_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_y_lb, 'magenta', 'LineWidth', 1)
ylabel('Y-axis [m]')
title(sprintf('\\mu_{syn} = %.2f m,  \\mu_{lb} = %.2f m', mu_y_syn, mu_y_lb))
fontsize(gca, 16, "points")
ylim([-1 1])
grid on;

subplot(2, 3, 3)
plot(time, E_z_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_z_lb, 'magenta', 'LineWidth', 1)
ylabel('Z-axis [m]')
title(sprintf('\\mu_{syn} = %.2f m,  \\mu_{lb} = %.2f m', mu_z_syn, mu_z_lb))
fontsize(gca, 16, "points")
ylim([-10 10])
grid on;

subplot(2, 3, 4)
plot(time, E_yaw_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_yaw_lb, 'magenta', 'LineWidth', 1)
ylabel(['Yaw [' char(176) ']'])
xlabel('Time [orbits]')
title(sprintf('\\mu_{syn} = %.1f%s,  \\mu_{lb} = %.1f%s', mu_yaw_syn, char(176), mu_yaw_lb, char(176)))
fontsize(gca, 16, "points")
% ylim([-50 50])
grid on;

subplot(2, 3, 5)
plot(time, E_pitch_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_pitch_lb, 'magenta', 'LineWidth', 1)
ylabel(['Pitch [' char(176) ']'])
xlabel('Time [orbits]')
title(sprintf('\\mu_{syn} = %.1f%s,  \\mu_{lb} = %.1f%s', mu_pitch_syn, char(176), mu_pitch_lb, char(176)))
fontsize(gca, 16, "points")
% ylim([-50 50])
grid on;

subplot(2, 3, 6)
plot(time, E_roll_syn, 'LineWidth', 1.5)
hold on;
plot(time, E_roll_lb, 'magenta', 'LineWidth', 1)
ylabel(['Roll [' char(176) ']'])
xlabel('Time [orbits]')
title(sprintf('\\mu_{syn} = %.1f%s,  \\mu_{lb} = %.1f%s', mu_roll_syn, char(176), mu_roll_lb, char(176)))
fontsize(gca, 16, "points")
% ylim([-50 50])
grid on;

