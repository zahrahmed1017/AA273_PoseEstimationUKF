% plot_results.m
%
% Load run_filter_out.mat and generate two figures:
%   Figure 1 - 2x3: NS-ROE errors with ±3σ covariance bounds
%   Figure 2 - 3x3: Camera position error | Euler angle error | RAV error
%
% Run from the AA273_PoseEstimationCNN/ directory, or set MAT_FILE below.

clear; close all;

MAT_FILE = fullfile(fileparts(mfilename('fullpath')), 'results', 'run_filter_out.mat');
d = load(MAT_FILE);

%% Time axis
N       = size(d.filter_roe, 1);
GM      = 3.986004418e14;              % m^3/s^2
T_orb   = 2*pi / sqrt(GM / d.sma_m^3);
time    = (0:N-1)' * d.dt / T_orb;   % [orbits]

% =========================================================================
%  FIGURE 1 — NS-ROE errors with ±3σ bounds
% =========================================================================
roe_labels = {'a\delta{a}', 'a\delta\lambda', 'a\delta{e_x}', ...
               'a\delta{e_y}', 'a\delta{i_x}', 'a\delta{i_y}'};

roe_err  = d.filter_roe - d.gt_roe;          % [N x 6]  m
roe_3sig = 3 * sqrt(abs(d.P_roe_diag));       % [N x 6]  m

fig1 = figure('Color','w','Position',[50 50 1400 800]);
tl1  = tiledlayout(fig1, 2, 3, 'Padding','compact', 'TileSpacing','compact');
title(tl1, 'NS-ROE Filter Errors with 3\sigma Bounds', 'FontSize', 14);

gray = [0.6 0.6 0.6];
for k = 1:6
    ax = nexttile; hold on; grid on; box off;

    % ±3σ shaded band
    fill([time; flipud(time)], [roe_3sig(:,k); flipud(-roe_3sig(:,k))], ...
         gray, 'FaceAlpha', 0.25, 'LineStyle', 'none');

    % Error trace
    plot(time, roe_err(:,k), 'k', 'LineWidth', 1.5);

    % ±3σ dashed lines
    plot(time,  roe_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);
    plot(time, -roe_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);

    xlabel('Time [orbits]', 'FontSize', 11);
    ylabel([roe_labels{k} ' [m]'], 'FontSize', 11);
    set(ax, 'FontSize', 11);
end

% =========================================================================
%  FIGURE 2 — 3x3: position | attitude | angular velocity errors
% =========================================================================
fig2 = figure('Color','w','Position',[50 50 1400 900]);
tl2  = tiledlayout(fig2, 3, 3, 'Padding','compact', 'TileSpacing','compact');
title(tl2, 'Camera-Frame Pose and State Errors', 'FontSize', 14);

% --- Row 1: Camera-frame position error  (pose_t - gt_t)  [m] -----------
pos_err    = d.pose_t - d.gt_t;           % [N x 3]
pos_labels = {'x error [m]', 'y error [m]', 'z error [m]'};
colors_pos = {'b', [0 0.6 0], 'r'};

for k = 1:3
    ax = nexttile; hold on; grid on; box off;
    plot(time, pos_err(:,k), 'Color', colors_pos{k}, 'LineWidth', 1.5);
    yline(0, 'k--', 'LineWidth', 0.8);
    xlabel('Time [orbits]', 'FontSize', 11);
    ylabel(pos_labels{k}, 'FontSize', 11);
    set(ax, 'FontSize', 11);
end

% --- Row 2: Euler angle errors from q_spri2tpri  [deg] ------------------
% dq = conj(q_true) * q_est  →  ZYX Euler angles = [yaw, pitch, roll]
% Convention matches cnnukf computeStateErrorThreeSigma.m
eul_err = zeros(N, 3);   % [yaw, pitch, roll] in deg
for i = 1:N
    q_est  = d.filter_q(i,:);    % [qw qx qy qz]
    q_true = d.gt_q_spri(i,:);
    dq     = quatMulLocal(quatConjLocal(q_true), q_est);
    eul_err(i,:) = dq2eul321_deg(dq);
end

% 3-sigma for Euler angles (propagated from MRP covariance via Jacobian)
att_3sig = zeros(N, 3);
for i = 1:N
    Pmrp     = diag(d.P_mrp_diag(i,:));   % [3x3] MRP covariance
    J        = mrp2euler_jacobian(d.filter_q(i,:));   % [3x3] approx Jacobian
    Peuler   = J * Pmrp * J';
    att_3sig(i,:) = rad2deg(3 * sqrt(abs(diag(Peuler)')));
end

eul_labels = {'Yaw error [deg]', 'Pitch error [deg]', 'Roll error [deg]'};
colors_att = {'b', [0 0.6 0], 'r'};
for k = 1:3
    ax = nexttile; hold on; grid on; box off;
    fill([time; flipud(time)], ...
         [att_3sig(:,k); flipud(-att_3sig(:,k))], ...
         gray, 'FaceAlpha', 0.25, 'LineStyle', 'none');
    plot(time, eul_err(:,k), 'Color', colors_att{k}, 'LineWidth', 1.5);
    plot(time,  att_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);
    plot(time, -att_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);
    xlabel('Time [orbits]', 'FontSize', 11);
    ylabel(eul_labels{k}, 'FontSize', 11);
    set(ax, 'FontSize', 11);
end

% --- Row 3: Angular velocity error  [deg/s] ------------------------------
rav_err  = rad2deg(d.filter_rav - d.gt_rav);   % [N x 3]
rav_3sig = rad2deg(3 * sqrt(abs(d.P_rav_diag)));

rav_labels = {'\omega_x error [deg/s]', '\omega_y error [deg/s]', '\omega_z error [deg/s]'};
colors_rav = {'b', [0 0.6 0], 'r'};
for k = 1:3
    ax = nexttile; hold on; grid on; box off;
    fill([time; flipud(time)], ...
         [rav_3sig(:,k); flipud(-rav_3sig(:,k))], ...
         gray, 'FaceAlpha', 0.25, 'LineStyle', 'none');
    plot(time, rav_err(:,k), 'Color', colors_rav{k}, 'LineWidth', 1.5);
    plot(time,  rav_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);
    plot(time, -rav_3sig(:,k), '--', 'Color', gray, 'LineWidth', 0.8);
    xlabel('Time [orbits]', 'FontSize', 11);
    ylabel(rav_labels{k}, 'FontSize', 11);
    set(ax, 'FontSize', 11);
end

% =========================================================================
%  Local helper functions
% =========================================================================

function qc = quatConjLocal(q)
% Conjugate of [qw qx qy qz]
qc = [q(1), -q(2), -q(3), -q(4)];
end

function qr = quatMulLocal(p, q)
% Hamilton product of two [qw qx qy qz] quaternions
pw=p(1); px=p(2); py=p(3); pz=p(4);
qw=q(1); qx=q(2); qy=q(3); qz=q(4);
qr = [pw*qw - px*qx - py*qy - pz*qz, ...
      pw*qx + px*qw + py*qz - pz*qy, ...
      pw*qy - px*qz + py*qw + pz*qx, ...
      pw*qz + px*qy - py*qx + pz*qw];
end

function eul = dq2eul321_deg(dq)
% Convert error quaternion dq = [qw qx qy qz] to ZYX Euler angles [deg]
% Returns [yaw, pitch, roll] matching cnnukf computeStateErrorThreeSigma
dq0=dq(1); dqx=dq(2); dqy=dq(3); dqz=dq(4);

sinPhi   = 2*(dq0*dqx + dqy*dqz);
cosPhi   = 1 - 2*(dqx^2 + dqy^2);
sinTheta = 2*(dq0*dqy - dqz*dqx);
sinPsi   = 2*(dq0*dqz + dqx*dqy);
cosPsi   = 1 - 2*(dqy^2 + dqz^2);

roll  = atan2d(sinPhi,   cosPhi);
pitch = asind(clamp(sinTheta, -1, 1));
yaw   = atan2d(sinPsi,   cosPsi);

eul = [yaw, pitch, roll];
end

function v = clamp(v, lo, hi)
v = max(lo, min(hi, v));
end

function J = mrp2euler_jacobian(q)
% Approximate Jacobian d(Euler)/d(MRP) for small MRP perturbations.
% For MRP = 0 (reset after update), this linearises around identity.
% Returns [3x3]: rows = [yaw, pitch, roll], cols = [p1, p2, p3].
% At identity dq ≈ I, the Jacobian simplifies to 2*I (f=4 USQUE).
% For a quick 3-sigma estimate this is sufficient.
J = 2 * eye(3);   % [rad/rad] -- converts MRP sigma to Euler sigma
end
