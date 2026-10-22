function r = mean_tauw(logfile, tstart, varargin)
% MEAN_TAUW  Time-averaged wall shear stress from a HiCHAN log file (data/log.dat)
%
%   r = mean_tauw('data/log.dat', 300)                % average from t = 300 to the end of the log
%   r = mean_tauw('data/log.dat', 300, 'tend', 1200)  % average over 300 <= t <= 1200
%   r = mean_tauw(..., 'plot', true)                  % plot tau_w(t) and the averaging window
%   r = mean_tauw(..., 'nbatch', 10)                  % number of batches for the error estimate
%
% Columns of log.dat used: 2 time, 6 tau_w (= nu*(|dU/dy|_bottom + |dU/dy|_top)/2, averaged over both
% walls), 7 Re_tau (= sqrt(tau_w)/nu), 5 applied mean pressure gradient -dp/dx.
%
% * The average is time weighted (trapezoidal rule), so a variable time step and the log interval
%   do not bias it.
% * Header/comment lines ('#') are skipped. After a restart the log is appended to; time levels that
%   occur more than once (restart from an earlier restart file) are taken from the latest run.
% * The statistical error of the mean is estimated by batch means (nbatch equal-time batches).
%
% Output structure r:
%   r.tauw      mean tau_w                    r.utau     sqrt(r.tauw)
%   r.retau     u_tau/nu (mean Re_tau)        r.nu       kinematic viscosity (1/re)
%   r.dpdx      mean -dp/dx (= tau_w for a fully developed channel)
%   r.err_tauw  standard error of r.tauw      r.err_retau standard error of r.retau
%   r.t0, r.t1  averaging interval            r.nsamples  number of log lines used
% Works in MATLAB and GNU Octave.

opt = struct('tend', inf, 'plot', false, 'nbatch', 10);
for k = 1:2:numel(varargin)
  opt.(lower(varargin{k})) = varargin{k+1};
end

d = read_log(logfile);
t = d(:,2); dpdx = d(:,5); tau = d(:,6); ret = d(:,7);

% keep the latest occurrence of every time level (restarts) and sort in time
[~, ilast] = unique(flipud(t), 'stable');
keep = sort(numel(t) + 1 - ilast);
t = t(keep); tau = tau(keep); ret = ret(keep); dpdx = dpdx(keep);
[t, is] = sort(t); tau = tau(is); ret = ret(is); dpdx = dpdx(is);

sel = t >= tstart & t <= opt.tend;
if nnz(sel) < 2
  error('mean_tauw: fewer than two log entries between t = %g and t = %g (log covers %g ... %g)', ...
        tstart, min(opt.tend, t(end)), t(1), t(end));
end
ts = t(sel); taus = tau(sel);

% kinematic viscosity from Re_tau = sqrt(tau_w)/nu (any line with tau_w > 0)
j = find(tau > 0 & ret > 0, 1, 'last');
nu = sqrt(tau(j))/ret(j);

T = ts(end) - ts(1);
r.tauw  = trapz(ts, taus)/T;
r.dpdx  = trapz(ts, dpdx(sel))/T;
r.utau  = sqrt(r.tauw);
r.nu    = nu;
r.retau = r.utau/nu;
r.t0 = ts(1); r.t1 = ts(end);
r.nsamples = numel(ts);

% batch-means error estimate
nb = max(2, round(opt.nbatch));
edges = linspace(ts(1), ts(end), nb + 1);
bm = nan(nb, 1);
for b = 1:nb
  m = ts >= edges(b) & ts <= edges(b+1);
  if nnz(m) >= 2
    bm(b) = trapz(ts(m), taus(m))/(ts(find(m, 1, 'last')) - ts(find(m, 1)));
  end
end
bm = bm(~isnan(bm));
if numel(bm) >= 2
  r.err_tauw = std(bm)/sqrt(numel(bm));
else
  r.err_tauw = NaN;
end
r.err_retau = r.retau*0.5*r.err_tauw/r.tauw;

fprintf('averaging interval     : t = %.4g ... %.4g  (%.4g time units, %d log lines)\n', ...
        r.t0, r.t1, T, r.nsamples);
fprintf('mean tau_w             : %.6e  +- %.1e (batch means, %d batches)\n', r.tauw, r.err_tauw, numel(bm));
fprintf('mean -dp/dx            : %.6e\n', r.dpdx);
fprintf('u_tau = sqrt(<tau_w>)  : %.6f\n', r.utau);
fprintf('Re_tau = u_tau/nu      : %.3f  +- %.2f   (nu = %.6e)\n', r.retau, r.err_retau, nu);

if opt.plot
  figure('Color', 'w');
  plot(t, tau, 'b-'); hold on
  plot([r.t0 r.t1], [r.tauw r.tauw], 'r-', 'LineWidth', 2);
  yl = ylim;
  plot([r.t0 r.t0], yl, 'k--'); plot([r.t1 r.t1], yl, 'k--');
  xlabel('t U/h'); ylabel('\tau_w');
  title(sprintf('<\\tau_w> = %.4e,  Re_\\tau = %.2f  (t = %.4g ... %.4g)', r.tauw, r.retau, r.t0, r.t1));
  grid on
end
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
