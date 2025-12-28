function plot_grid_3d(grid_map)
% Clean, bright, empty 3-D scene. Obstacles are 1×1×1 cubes (no sheets).
% Convention: X=rows, Y=cols, Z=heights. Values: 1=empty, 2=obstacle.

    [rows, cols, heights] = size(grid_map);

    % --- figure / axes ---
    f  = figure('Color','w','Renderer','opengl');
    ax = axes('Parent',f,'Color','w'); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    ax.GridColor = [0.95 0.95 0.95];
    ax.GridAlpha = 1;
    view(ax,40,25); camproj(ax,'perspective');

    % bounds & aspect
    xlim(ax,[0.5, rows+0.5]); ylim(ax,[0.5, cols+0.5]); zlim(ax,[0.5, heights+0.5]);
    daspect(ax,[1 1 1]); axis(ax,'vis3d');

    % light box + optional floor ticks (no surfaces)
    draw_bbox(ax, rows, cols, heights, [0.90 0.90 0.90]);
    draw_floor_grid(ax, rows, cols, heights, 5, [0.93 0.93 0.93]);

    % --- obstacles as cubes (voxels) ---
    obsMask = (grid_map == 2);
    if any(obsMask(:))
        [r,c,h] = ind2sub(size(grid_map), find(obsMask));
        draw_voxels(ax, [r c h], ...
            'FaceColor',[0.20 0.50 1.00], 'FaceAlpha',0.9, ...
            'EdgeColor',[0.15 0.25 0.60], 'EdgeAlpha',0.9, 'LineWidth',0.5);
    end

    % labels/ticks
    xlabel(ax,'X (rows)'); ylabel(ax,'Y (cols)'); zlabel(ax,'Z (heights)');
    set(ax,'XTick',0:5:rows,'YTick',0:5:cols,'ZTick',0:5:heights);
    title(ax, sprintf('%d×%d×%d Parking garage', rows, cols, heights));
    rotate3d(f,'on');
    hold(ax,'off');
end

% -------- helpers --------
function draw_voxels(ax, centers, varargin)
% centers: N×3 integer grid coordinates [r c h]
% Renders each as a unit cube centered on that cell.
    p = inputParser;
    addParameter(p,'FaceColor',[0.2 0.5 1]);
    addParameter(p,'EdgeColor',[0.1 0.1 0.1]);
    addParameter(p,'FaceAlpha',1.0);
    addParameter(p,'EdgeAlpha',1.0);
    addParameter(p,'LineWidth',0.5);
    parse(p,varargin{:});
    opt = p.Results;

    % Unit cube centered at (0,0,0), edge length 1
    V0 = [ -0.5 -0.5 -0.5
            0.5 -0.5 -0.5
            0.5  0.5 -0.5
           -0.5  0.5 -0.5
           -0.5 -0.5  0.5
            0.5 -0.5  0.5
            0.5  0.5  0.5
           -0.5  0.5  0.5 ];
    F0 = [1 2 3 4;   % bottom (z-)
          5 6 7 8;   % top (z+)
          1 2 6 5;   % y-
          2 3 7 6;   % x+
          3 4 8 7;   % y+
          4 1 5 8];  % x-

    n  = size(centers,1);
    V  = zeros(n*8,3);
    F  = zeros(n*6,4);
    for i = 1:n
        base = (i-1)*8;
        V(base+(1:8),:) = V0 + centers(i,:);      % shift cube to this cell center
        F((i-1)*6+(1:6),:) = F0 + base;           % faces reference this cube's verts
    end

    patch(ax, 'Vertices',V, 'Faces',F, ...
        'FaceColor',opt.FaceColor, 'FaceAlpha',opt.FaceAlpha, ...
        'EdgeColor',opt.EdgeColor, 'EdgeAlpha',opt.EdgeAlpha, ...
        'LineWidth',opt.LineWidth);
end

function draw_bbox(ax, X, Y, Z, col)
    xs = [0.5, X+0.5]; ys = [0.5, Y+0.5]; zs = [0.5, Z+0.5];
    for y = ys, for z = zs, line(ax, xs,[y y],[z z],'Color',col,'LineWidth',1); end, end
    for x = xs, for z = zs, line(ax,[x x],ys,[z z],'Color',col,'LineWidth',1); end, end
    for x = xs, for y = ys, line(ax,[x x],[y y],zs,'Color',col,'LineWidth',1); end, end
end

function draw_floor_grid(ax, X, Y, ~, step, col)
    z = 0.5;
    for y = 0.5:step:Y+0.5, line(ax,[0.5, X+0.5],[y y],[z z],'Color',col,'LineWidth',0.5); end
    for x = 0.5:step:X+0.5, line(ax,[x x],[0.5, Y+0.5],[z z],'Color',col,'LineWidth',0.5); end
end
