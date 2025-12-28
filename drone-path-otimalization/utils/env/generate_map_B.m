% Generate and save Map B (diagonal holes) to your repo
clear; clc; close all;

here  = fileparts(mfilename('fullpath'));
repo  = fileparts(fileparts(here));     % repo root
addpath(genpath(repo));

rows = 20; cols = 30; heights = 30;
pad  = [0 0 0];          % e.g., [5 5 0] to add empty space around the box
hole = 5;
off  = 1;

grid_map = make_map_B(rows, cols, heights, pad, hole, off);

outDir  = fullfile(repo,'utils','data','map');
if ~exist(outDir,'dir'), mkdir(outDir); end
outFile = fullfile(outDir,'mapB_20x30x30.mat');
save(outFile,'grid_map');
fprintf('Saved Map B to: %s\n', outFile);

plot_grid_3d(grid_map); title('Map B (diagonal holes)');
