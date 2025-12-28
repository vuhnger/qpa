function grid_map = generate_grid(map_size, obstacle)
%% generate_grid
% map_size: [rows cols heights]
% obstacle:
%   - [] or omitted -> no obstacles
%   - vector of linear indices into grid_map
%   - OR an N-by-3 list of [row col height] (1-based subscripts)
%
% Encoding:
%   1 = empty
%   2 = obstacle

    arguments
        map_size (1,3) {mustBeInteger, mustBePositive}
        obstacle = []
    end

    % Initialize all empty
    grid_map = ones(map_size);  % 1 = empty

    if isempty(obstacle)
        return;
    end

    % Accept either linear indices or [row col height] subscripts
    if size(obstacle,2) == 3
        linIdx = sub2ind(map_size, obstacle(:,1), obstacle(:,2), obstacle(:,3));
    else
        linIdx = obstacle(:);
    end

    % Place obstacles
    grid_map(linIdx) = 2;  % 2 = obstacle
end