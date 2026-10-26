function r = mean_tauw(logfile, tstart, varargin)
% MEAN_TAUW  Time-averaged wall shear stress and wall scalar gradients from a HiCHAN log file
%
%   r = mean_tauw('data/log.dat', 300)                % average from t = 300 to the end of the log
%   r = mean_tauw('data/log.dat', 300, 'tend', 1200)  % average over 300 <= t <= 1200
%   r = mean_tauw(..., 'plot', true)                  % plot tau_w(t) (and the scalar gradients)
%   r = mean_tauw(..., 'nbatch', 10)                  % number of batches for the error estimate
%   r = mean_tauw(..., 'outfile', 'run1/utau.dat')    % output file (default 'utau.dat'; '' = none)
%
% Columns of log.dat used: 2 time, 5 applied mean pressure gradient -dp/dx, 6 tau_w (= nu*(|dU/dy|_bottom
% + |dU/dy|_top)/2, averaged over both walls), 7 Re_tau (= sqrt(tau_w)/nu), and for every scalar n the
% wall-normal gradients dS_n/dy at the bottom wall (column 9+2n) and at the top wall (column 10+2n).
%
% * The averages are time weighted (trapezoidal rule), so a variable time step and the log interval
%   do not bias them.
% * Header/comment lines ('#') are skipped. After a restart the log is appended to; time levels that
%   occur more than once (restart from an earlier restart file) are taken from the latest run.
% * The statistical error of each mean is estimated by batch means (nbatch equal-time batches).
% * All results are written to the text file 'utau.dat' (one quantity per line: name, mean,
%   standard error), readable with  c = textscan(fopen('utau.dat'), '%s %f %f', 'CommentStyle', '#').
%
% Output structure r:
%   r.tauw      mean tau_w                    r.utau     sqrt(r.tauw)
%   r.retau     u_tau/nu (mean Re_tau)        r.nu       kinematic viscosity (1/re)
%   r.dpdx      mean -dp/dx (= tau_w for a fully developed channel, no buoyancy along x)
%   r.err_tauw  standard error of r.tauw      r.err_retau standard error of r.retau
%   r.err_dpdx  standard error of r.dpdx
%   r.nscal     number of scalars in the log
%   r.dsdy_bot(n), r.dsdy_top(n)     mean dS_n/dy at the bottom (y = -1) and top (y = +1) wall
%   r.dsdy_avg(n)                    (|dS_n/dy|_bottom + |dS_n/dy|_top)/2
%   r.err_dsdy_bot(n), r.err_dsdy_top(n), r.err_dsdy_avg(n)   standard errors
%   r.t0, r.t1  averaging interval            r.nsamples  number of log lines used
% The wall heat flux follows as q = -dS/dy/(re*pr) (into the fluid at the bottom wall: q_bottom =
% -dS/dy|_bottom/(re*pr), at the top wall: q_top = +dS/dy|_top/(re*pr)).
% Works in MATLAB and GNU Octave.

opt = struct('tend', inf, 'plot', false, 'nbatch', 10, 'outfile', 'utau.dat');
for k = 1:2:numel(varargin)
  opt.(lower(varargin{k})) = varargin{k+1};
end

d = read_log(logfile);
nscal = max(0, floor((size(d, 2) - 10)/2));

% keep the latest occurrence of every time level (restarts) and sort in time
t = d(:,2);
[~, ilast] = unique(flipud(t), 'stable');
keep = sort(numel(t) + 1 - ilast);
d = d(keep, :);
[~, is] = sort(d(:,2));
d = d(is, :);
t = d(:,2); dpdx = d(:,5); tau = d(:,6); ret = d(:,7);

sel = t >= tstart & t <= opt.tend;
if nnz(sel) < 2
  error('mean_tauw: fewer than two log entries between t = %g and t = %g (log covers %g ... %g)', ...
        tstart, min(opt.tend, t(end)), t(1), t(end));
end
ts = t(sel);

% kinematic viscosity from Re_tau = sqrt(tau_w)/nu (any line with tau_w > 0)
j = find(tau > 0 & ret > 0, 1, 'last');
nu = sqrt(tau(j))/ret(j);

% at least 3 log lines per batch (fewer batches for short averaging intervals)
nb = max(2, min(round(opt.nbatch), floor(nnz(sel)/3)));
[r.tauw, r.err_tauw, nbu] = tmean(ts, tau(sel), nb);
[r.dpdx, r.err_dpdx]      = tmean(ts, dpdx(sel), nb);
r.utau  = sqrt(r.tauw);
r.nu    = nu;
r.retau = r.utau/nu;
r.err_retau = r.retau*0.5*r.err_tauw/r.tauw;
r.t0 = ts(1); r.t1 = ts(end);
r.nsamples = numel(ts);

% wall-normal scalar gradients
r.nscal = nscal;
r.dsdy_bot = zeros(1, nscal); r.dsdy_top = zeros(1, nscal); r.dsdy_avg = zeros(1, nscal);
r.err_dsdy_bot = zeros(1, nscal); r.err_dsdy_top = zeros(1, nscal); r.err_dsdy_avg = zeros(1, nscal);
for n = 1:nscal
  gb = d(sel, 9 + 2*n); gt = d(sel, 10 + 2*n);
  [r.dsdy_bot(n), r.err_dsdy_bot(n)] = tmean(ts, gb, nb);
  [r.dsdy_top(n), r.err_dsdy_top(n)] = tmean(ts, gt, nb);
  [r.dsdy_avg(n), r.err_dsdy_avg(n)] = tmean(ts, 0.5*(abs(gb) + abs(gt)), nb);
end

T = r.t1 - r.t0;
fprintf('averaging interval     : t = %.4g ... %.4g  (%.4g time units, %d log lines)\n', ...
        r.t0, r.t1, T, r.nsamples);
fprintf('mean tau_w             : %.6e  +- %.1e (batch means, %d batches)\n', r.tauw, r.err_tauw, nbu);
fprintf('mean -dp/dx            : %.6e  +- %.1e\n', r.dpdx, r.err_dpdx);
fprintf('u_tau = sqrt(<tau_w>)  : %.6f\n', r.utau);
fprintf('Re_tau = u_tau/nu      : %.3f  +- %.2f   (nu = %.6e)\n', r.retau, r.err_retau, nu);
for n = 1:nscal
  fprintf('scalar %d: <dS/dy> bottom %.6e +- %.1e,  top %.6e +- %.1e,  (|bottom|+|top|)/2 %.6e +- %.1e\n', ...
          n, r.dsdy_bot(n), r.err_dsdy_bot(n), r.dsdy_top(n), r.err_dsdy_top(n), r.dsdy_avg(n), r.err_dsdy_avg(n));
end

if ~isempty(opt.outfile)
  write_stats(opt.outfile, logfile, r);
  fprintf('written to %s\n', opt.outfile);
end

if opt.plot
  figure('Color', 'w');
  if nscal > 0, subplot(2, 1, 1); end
  plot(t, tau, 'b-'); hold on
  plot([r.t0 r.t1], [r.tauw r.tauw], 'r-', 'LineWidth', 2);
  yl = ylim;
  plot([r.t0 r.t0], yl, 'k--'); plot([r.t1 r.t1], yl, 'k--');
  xlabel('t U/h'); ylabel('\tau_w');
  title(sprintf('<\\tau_w> = %.4e,  Re_\\tau = %.2f  (t = %.4g ... %.4g)', r.tauw, r.retau, r.t0, r.t1));
  grid on
  if nscal > 0
    subplot(2, 1, 2); hold on
    leg = {};
    for n = 1:nscal
      plot(t, d(:, 9 + 2*n), '-');  leg{end+1} = sprintf('dS_%d/dy bottom', n); %#ok<AGROW>
      plot(t, d(:, 10 + 2*n), '--'); leg{end+1} = sprintf('dS_%d/dy top', n);   %#ok<AGROW>
    end
    yl = ylim;
    plot([r.t0 r.t0], yl, 'k:'); plot([r.t1 r.t1], yl, 'k:');
    xlabel('t U/h'); ylabel('dS/dy at the walls'); legend(leg, 'Location', 'best'); grid on
  end
end
end

function [m, err, nbu] = tmean(t, f, nb)
% time-weighted mean of f(t) and its standard error from nb equal-time batches
m = trapz(t, f)/(t(end) - t(1));
edges = linspace(t(1), t(end), nb + 1);
bm = nan(nb, 1);
for b = 1:nb
  k = t >= edges(b) & t <= edges(b+1);
  if nnz(k) >= 2
    bm(b) = trapz(t(k), f(k))/(t(find(k, 1, 'last')) - t(find(k, 1)));
  end
end
bm = bm(~isnan(bm));
nbu = numel(bm);
if nbu >= 2
  err = std(bm)/sqrt(nbu);
else
  err = NaN;
end
end

function write_stats(fname, logfile, r)
% one quantity per line: name, mean, standard error (NaN if not applicable)
fid = fopen(fname, 'w');
if fid < 0
  error('mean_tauw: cannot write %s', fname);
end
fprintf(fid, '# HiCHAN mean statistics from %s (mean_tauw.m, %s)\n', logfile, datestr(now, 'yyyy-mm-dd HH:MM'));
fprintf(fid, '# time-weighted averages over t = %.6g ... %.6g; errors: batch means\n', r.t0, r.t1);
fprintf(fid, '# dSn_dy_*: wall-normal gradient of scalar n at the bottom (y=-1) and top (y=+1) wall;\n');
fprintf(fid, '#   dSn_dy_avg = (|bottom| + |top|)/2. Wall heat flux: q = -dS/dy/(re*pr)\n');
fprintf(fid, '# %-20s %22s %22s\n', 'quantity', 'mean', 'std_error');
w = @(name, m, e) fprintf(fid, '%-22s %22.14e %22.14e\n', name, m, e);
w('t_start', r.t0, NaN);
w('t_end', r.t1, NaN);
w('n_samples', r.nsamples, NaN);
w('nu', r.nu, NaN);
w('tau_w', r.tauw, r.err_tauw);
w('u_tau', r.utau, 0.5*r.err_tauw/r.utau);
w('Re_tau', r.retau, r.err_retau);
w('dpdx', r.dpdx, r.err_dpdx);
w('n_scalars', r.nscal, NaN);
for n = 1:r.nscal
  w(sprintf('dS%d_dy_bottom', n), r.dsdy_bot(n), r.err_dsdy_bot(n));
  w(sprintf('dS%d_dy_top', n), r.dsdy_top(n), r.err_dsdy_top(n));
  w(sprintf('dS%d_dy_avg', n), r.dsdy_avg(n), r.err_dsdy_avg(n));
end
fclose(fid);
end

function d = read_log(logfile)
% numeric lines of log.dat (comment/header lines and incomplete last lines are skipped)
fid = fopen(logfile, 'r');
if fid < 0
  error('mean_tauw: cannot open %s', logfile);
end
rows = {};
ncol = 0;
while true
  l = fgetl(fid);
  if ~ischar(l), break; end
  l = strtrim(l);
  if isempty(l) || l(1) == '#', continue; end
  v = sscanf(l, '%f')';
  if isempty(v), continue; end
  if ncol == 0, ncol = numel(v); end
  if numel(v) ~= ncol
    if numel(v) > ncol && ncol >= 7
      v = v(1:ncol);                    % more scalars in a later run: keep the common columns
    elseif numel(v) >= 7
      ncol = min(ncol, numel(v));       % fewer columns: truncate all to the common part
      rows = cellfun(@(x) x(1:ncol), rows, 'UniformOutput', false);
      v = v(1:ncol);
    else
      continue                          % incomplete line (e.g. a run that was killed)
    end
  end
  rows{end+1} = v; %#ok<AGROW>
end
fclose(fid);
if isempty(rows)
  error('mean_tauw: no data in %s', logfile);
end
d = vertcat(rows{:});
end
