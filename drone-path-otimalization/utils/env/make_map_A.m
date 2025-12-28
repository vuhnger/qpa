function grid_map = make_map_A(rows, cols, heights, pad, holeSize)
%MAKE_MAP_A  Recreates your Map A. Adds optional empty padding around it.
% grid_map(r,c,z) = true for occupied voxels.

if nargin < 5 || isempty(holeSize), holeSize = 5; end
if nargin < 4 || isempty(pad),      pad      = [0 0 0]; end  % [pr pc pz]

pr = pad(1); pc = pad(2); pz = pad(3);
map_size = [rows+2*pr, cols+2*pc, heights+2*pz];
grid_map = false(map_size);

% offsets where the box sits inside the padded map
r0 = pr; c0 = pc; z0 = pz;

hR = min(holeSize, rows);
hC = min(holeSize, cols);

obstacle = [];

% 1) Ground floor (full slab at z = 1+z0)
[Rf,Cf,Zf] = ndgrid(r0+(1:rows), c0+(1:cols), z0+1);
obstacle = [obstacle; sub2ind(map_size, Rf(:), Cf(:), Zf(:))];

% 2) Corner columns (all heights at four corners)
rC = r0 + [1, rows];
cC = c0 + [1, cols];
zC = z0 + (1:heights);
[RC, CC, ZC] = ndgrid(rC, cC, zC);
obstacle = [obstacle; sub2ind(map_size, RC(:), CC(:), ZC(:))];

% 3) Middle floor (holes at corners 1 & 3)
z_mid = z0 + round(heights/2);
[Rm,Cm,Zm] = ndgrid(r0+(1:rows), c0+(1:cols), z_mid);
slab_mid = sub2ind(map_size, Rm(:), Cm(:), Zm(:));
holes_mid = [
    corner_hole_idx(1, z_mid, rows, cols, hR, hC, map_size, r0, c0);
    corner_hole_idx(3, z_mid, rows, cols, hR, hC, map_size, r0, c0)
];
slab_mid = setdiff(slab_mid, holes_mid);
obstacle = [obstacle; slab_mid];

% 3b) Upper floor (holes at corners 2 & 4)
z_upper = z0 + round(heights*3/4);
[Ru,Cu,Zu] = ndgrid(r0+(1:rows), c0+(1:cols), z_upper);
slab_up = sub2ind(map_size, Ru(:), Cu(:), Zu(:));
holes_up = [
    corner_hole_idx(2, z_upper, rows, cols, hR, hC, map_size, r0, c0);
    corner_hole_idx(4, z_upper, rows, cols, hR, hC, map_size, r0, c0)
];
slab_up = setdiff(slab_up, holes_up);
obstacle = [obstacle; slab_up];

% 4) Roof (full slab at top z)
[Rr,Cr,Zr] = ndgrid(r0+(1:rows), c0+(1:cols), z0+heights);
obstacle = [obstacle; sub2ind(map_size, Rr(:), Cr(:), Zr(:))];

% finalize
obstacle = unique(obstacle);
grid_map = generate_grid(map_size, obstacle);
end

% --- helper (same as your mlx, but with offsets) ---
function idx = corner_hole_idx(corner, z, rows, cols, hR, hC, map_size, r0, c0)
    switch corner
        case 1, r = r0+(1:hR);                 c = c0+(1:hC);
        case 2, r = r0+(1:hR);                 c = c0+(cols-hC+1:cols);
        case 3, r = r0+(rows-hR+1:rows);       c = c0+(1:hC);
        case 4, r = r0+(rows-hR+1:rows);       c = c0+(cols-hC+1:cols);
        otherwise, error('corner must be 1..4');
    end
    [R,C,Z] = ndgrid(r, c, z);
    idx = sub2ind(map_size, R(:), C(:), Z(:));
end
