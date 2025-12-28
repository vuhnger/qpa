clear; clc;

% repo-aware paths
here    = fileparts(mfilename('fullpath'));   % .../utils/env
repo    = fileparts(fileparts(here));         % repo root
addpath(genpath(repo));                       % utils/ etc.

% sizes (your Map A)
rows = 20; cols = 30; heights = 30;
pad  = [0 0 0];            % change to e.g. [5 5 0] for empty border
hole = 5;

grid_map = make_map_A(rows, cols, heights, pad, hole);

% save inside the repo
outDir = fullfile(repo,'utils','data','map');
if ~exist(outDir,'dir'), mkdir(outDir); end
outFile = fullfile(outDir, 'mapA_20x30x30.mat');
save(outFile,'grid_map');
fprintf('Saved: %s\n', outFile);

% quick preview (optional)
plot_grid_3d(grid_map);
title('Map A');
