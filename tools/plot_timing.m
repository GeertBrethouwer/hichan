function plot_timing(fname, nskip)
% PLOT_TIMING  plot the wall-clock time per time step written by HiCHAN with timing = .true.
%
%   plot_timing                          reads data/timing_steps.dat
%   plot_timing('run1/data/timing_steps.dat')
%   plot_timing(fname, nskip)            ignores the first nskip steps (start-up) in the statistics
%
% Top: time per step (slowest rank, fastest rank, mean); steps slower than 2 x the median are marked.
% Bottom: time per step spent in the FFTs (incl. transposes), the transposes alone, the velocity
% solve, the scalar solves and output (max over the ranks), to see which part causes a spike.
% Works in MATLAB and Octave.

if nargin < 1 || isempty(fname), fname = 'data/timing_steps.dat'; end
if nargin < 2, nskip = 0; end

d = load(fname);                      % '#' header lines are skipped
step = d(:,1); tmax = d(:,4); tmin = d(:,5); tmean = d(:,6);
parts = d(:,7:11);
names = {'FFT + transposes','transposes','velocity solve','scalar solves','output'};

m   = step > step(1) + nskip - 1;
med = median(tmax(m));
spk = m & tmax > 2*med;

fprintf('%s: %d steps\n', fname, numel(step));
fprintf('time per step (slowest rank): median %.4g s, mean %.4g s, max %.4g s (step %d)\n', ...
        med, mean(tmax(m)), max(tmax(m)), step(find(tmax == max(tmax(m)), 1)));
fprintf('load imbalance (max/min over ranks), mean: %.3f\n', mean(tmax(m)./tmin(m)));
fprintf('steps slower than 2 x median: %d\n', nnz(spk));
if any(spk)
  fprintf('   step     t_step   fft+tr  transp.   vsolve   ssolve   output\n');
  is = find(spk);
  [~, o] = sort(tmax(is), 'descend');
  for k = is(o(1:min(20, numel(o))))'
    fprintf('%7d %10.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n', step(k), tmax(k), parts(k,:));
  end
end

figure;
subplot(2,1,1);
plot(step, tmax, 'b-', step, tmin, 'g-', step, tmean, 'k-'); hold on;
plot(step(spk), tmax(spk), 'ro');
plot([step(1) step(end)], [med med], 'r--');
hold off; grid on;
xlabel('time step'); ylabel('wall-clock time per step [s]');
legend('slowest rank', 'fastest rank', 'mean', '> 2 x median', 'median', 'location', 'best');
title(strrep(fname, '_', '\_'));

subplot(2,1,2);
plot(step, parts); grid on;
xlabel('time step'); ylabel('time per step [s] (max over ranks)');
legend(names, 'location', 'best');
end
