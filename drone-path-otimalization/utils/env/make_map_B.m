function grid_map = make_map_B(rows, cols, heights, pad, holeSize, offsetFromCorner)
%MAKE_MAP_B  Build Map B using obstacle indices + generate_grid (like your .mlx)
% - Diagonal holes: mid slab -> corners 1 & 4; upper slab -> corners 2 & 3
% - offsetFromCorner keeps the holes away from the corner columns
% - pad = [pr pc pz] adds empty space around the structure

if nargin < 4 || isempty(pad),              pad = [0 0 0];        end
if nargin < 5 || isempty(holeSize),         holeSize = 5;         end
if nargin < 6 || isempty(offsetFromCorner), offsetFromCorner = 1; end

pr = pad(1); pc = pad(2); pz = pad(3);
map_size = [rows+2*pr, cols+2*pc, heights+2*pz];

% offsets inside the padded map
r0 = pr; c0 = pc; z0 = pz;

% hole extents after offset
hR = max(0, min(holeSize, rows - 2*offsetFromCorner));
hC = max(0, min(holeSize, cols - 2*offsetFromCorner));

obstacle = [];

%% 1) Ground floor (full slab at z = 1)
[ Rf, Cf, Zf ] = ndgrid(r0+(1:rows), c0+(1:cols), z0+1);
obstacle = [obstacle; sub2ind(map_size, Rf(:), Cf(:), Zf(:))];

%% 2) Corner columns (all heights at the four corners)
rC = r0 + [1, rows];
cC = c0 + [1, cols];
zC = z0 + (1:heights);
[ RC, CC, ZC ] = ndgrid(rC, cC, zC);
obstacle = [obstacle; sub2ind(map_size, RC(:), CC(:), ZC(:))];

%% 3) Middle floor with holes in corners 1 & 4 (diagonal)
z_mid = z0 + round(heights/2);
[ Rm, Cm, Zm ] = ndgrid(r0+(1:rows), c0+(1:cols), z_mid);
slab_mid = sub2ind(map_size, Rm(:), Cm(:), Zm(:));

holes_mid = [
    corner_hole_idx_off(1, z_mid, rows, cols, hR, hC, map_size, r0, c0, offsetFromCorner);
    corner_hole_idx_off(4, z_mid, rows, cols, hR, hC, map_size, r0, c0, offsetFromCorner)
];
slab_mid = setdiff(slab_mid, holes_mid);
obstacle = [obstacle; slab_mid];

%% 3b) Upper floor with holes in corners 2 & 3 (other diagonal)
z_up = z0 + round(3*heights/4);
[ Ru, Cu, Zu ] = ndgrid(r0+(1:rows), c0+(1:cols), z_up);
slab_up = sub2ind(map_size, Ru(:), Cu(:), Zu(:));

holes_up = [
    corner_hole_idx_off(2, z_up, rows, cols, hR, hC, map_size, r0, c0, offsetFromCorner);
    corner_hole_idx_off(3, z_up, rows, cols, hR, hC, map_size, r0, c0, offsetFromCorner)
];
slab_up = setdiff(slab_up, holes_up);
obstacle = [obstacle; slab_up];

%% 4) Roof (full slab at top z)
[ Rr, Cr, Zr ] = ndgrid(r0+(1:rows), c0+(1:cols), z0+heights);
obstacle = [obstacle; sub2ind(map_size, Rr(:), Cr(:), Zr(:))];

%% finalize (exactly like your map_creator)
obstacle = unique(obstacle);
grid_map = generate_grid(map_size, obstacle);
end

% ---- helper (with offset + padding offsets) ----
function idx = corner_hole_idx_off(corner, z, rows, cols, hR, hC, map_size, r0, c0, off)
    switch corner
        case 1  % top-left
            r = r0 + (1+off : off+hR);
            c = c0 + (1+off : off+hC);
        case 2  % top-right
            r = r0 + (1+off : off+hR);
            c = c0 + (cols-hC-off+1 : cols-off);
        case 3  % bottom-left
            r = r0 + (rows-hR-off+1 : rows-off);
            c = c0 + (1+off : off+hC);
        case 4  % bottom-right
            r = r0 + (rows-hR-off+1 : rows-off);
            c = c0 + (cols-hC-off+1 : cols-off);
        otherwise
            error('corner must be 1..4');
    end
    % clamp to bounds
    r = max(r0+1, min(r0+rows, r));
    c = max(c0+1, min(c0+cols, c));
    [R,C,Z] = ndgrid(r, c, z);
    idx = sub2ind(map_size, R(:), C(:), Z(:));
end
