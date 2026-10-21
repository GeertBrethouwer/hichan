function plot_slices(matfile, fields, varargin)
% PLOT_SLICES  2D visualisation of the slices written by tools/restart_slices.py
%
%   plot_slices('data/restart_slices.mat')                 % all fields in the file
%   plot_slices('data/restart_slices.mat', {'u','s1'})     % selected fields
%   plot_slices(file, fields, 'save', true)                % also write <file>_<field>.png
%   plot_slices(file, fields, 'clim', [-0.2 0.2])          % fixed colour range
%   plot_slices(file, fields, 'plane', 'xz')               % only 'xz', 'xy' or 'yz' planes
%
% One figure per field: the xz-planes (wall-parallel, one per stored y position) in the top
% row, the xy-plane (side view, at z = z_xy) and the yz-plane (cross-section, at x = x_yz)
% below. Coordinates are in units of the channel half-height h (walls at y = -1 and 1).
% Fields with both signs (v, w, fluctuations) use a symmetric blue-white-red colour scale.
%
% The .mat file contains: x, y, z (grids), time, step, lx, lz, y_xz, z_xy, x_yz, fields, and
% for every field f: xz_f (Nz x Nx x n_y), xy_f (ny x Nx), yz_f (ny x Nz).
% Works in MATLAB and GNU Octave.

if nargin < 2 || isempty(fields)
  fields = {};
end
opt = struct('save', false, 'clim', [], 'plane', 'all');
for k = 1:2:numel(varargin)
  opt.(lower(varargin{k})) = varargin{k+1};
end

S = load(matfile);
x = S.x(:)'; y = S.y(:)'; z = S.z(:)';
yxz = S.y_xz(:)';
if isempty(fields)
  fields = cellstr(S.fields);
elseif ischar(fields)
  fields = {fields};
end
fields = cellfun(@strtrim, fields, 'UniformOutput', false);
% periodic directions: append the first point (x = lx, z = lz) so the plots fill the box
xp = [x, S.lx]; zp = [z, S.lz];

for n = 1:numel(fields)
  f = fields{n};
  if ~isfield(S, ['xy_' f])
    warning('field %s not found in %s', f, matfile); continue
  end
  XZ = S.(['xz_' f]); XY = S.(['xy_' f]); YZ = S.(['yz_' f]);
  XZ = cat(2, XZ, XZ(:,1,:)); XZ = cat(1, XZ, XZ(1,:,:));
  XY = [XY, XY(:,1)]; YZ = [YZ, YZ(:,1)];

  % colour range common to all planes of this field
  if ~isempty(opt.clim)
    cl = opt.clim;
  else
    allv = [XZ(:); XY(:); YZ(:)];
    lo = min(allv); hi = max(allv);
    if lo < 0 && hi > 0                       % symmetric for signed quantities
      m = max(abs([lo hi])); cl = [-m m];
    else
      cl = [lo hi];
    end
    if cl(1) == cl(2), cl = cl + [-1 1]*max(1e-12, abs(cl(1))*1e-6); end
  end
  if cl(1) < 0 && cl(2) > 0, cmap = bwr(256); else, cmap = jet(256); end

  nxz = size(XZ, 3);
  doxz = any(strcmp(opt.plane, {'all','xz'}));
  doxy = any(strcmp(opt.plane, {'all','xy'}));
  doyz = any(strcmp(opt.plane, {'all','yz'}));
  nrow = doxz + (doxy || doyz);
  ncol = max([nxz*doxz, doxy + doyz, 1]);
  fig = figure('Name', sprintf('%s  (t = %.4g)', f, S.time), 'Color', 'w');
  set(fig, 'Position', [100 100 420*ncol 330*nrow]);
  colormap(cmap);
  row = 0;
  if doxz
    row = row + 1;
    for k = 1:nxz
      subplot(nrow, ncol, (row-1)*ncol + k);
      pcolor(xp, zp, XZ(:,:,k)); shading flat; axis image; caxis(cl);
      xlabel('x/h'); ylabel('z/h');
      title(sprintf('%s, xz-plane at y = %.4f (%.4f from wall)', lbl(f, S), yxz(k), 1-abs(yxz(k))));
      colorbar;
    end
  end
  if doxy || doyz
    row = row + 1; c = 0;
    if doxy
      c = c + 1;
      subplot(nrow, ncol, (row-1)*ncol + c);
      pcolor(xp, y, XY); shading flat; axis image; caxis(cl);
      xlabel('x/h'); ylabel('y/h');
      title(sprintf('%s, xy-plane at z = %.4f', lbl(f, S), S.z_xy));
      colorbar;
    end
    if doyz
      c = c + 1;
      subplot(nrow, ncol, (row-1)*ncol + c);
      pcolor(zp, y, YZ); shading flat; axis image; caxis(cl);
      xlabel('z/h'); ylabel('y/h');
      title(sprintf('%s, yz-plane at x = %.4f', lbl(f, S), S.x_yz));
      colorbar;
    end
  end
  if opt.save
    [p, b] = fileparts(matfile);
    png = fullfile(p, sprintf('%s_%s.png', b, f));
    print(fig, png, '-dpng', '-r150');
    fprintf('wrote %s\n', png);
  end
end
end

function s = lbl(f, S)
% field label, with a prime for fluctuations
if isfield(S, 'fluct') && S.fluct
  s = [f ''''];
else
  s = f;
end
end

function c = bwr(n)
% blue - white - red diverging colour map
h = floor(n/2);
up = linspace(0, 1, h)';
c = [[up up ones(h,1)]; [ones(n-h,1) flipud(linspace(0, 1, n-h)') flipud(linspace(0, 1, n-h)')]];
end
