function [path, goal_reached, cost, EXPAND] = a_star(map, start, goal)
% @file: a_star.m
% @brief: 3D A* motion planning

% initialize
OPEN = [];
CLOSED = [];
EXPAND = [];

cost = 0;
goal_reached = false;

% 3D motion: 26 neighbors (including diagonals)
motion = [];
for dx = -1:1
    for dy = -1:1
        for dz = -1:1
            if dx == 0 && dy == 0 && dz == 0
                continue
            end
            motion = [motion; dx, dy, dz, norm([dx, dy, dz])];
        end
    end
end
motion_num = size(motion, 1);

node_s = [start, 0, h(start, goal), start];
OPEN = [OPEN; node_s];

while ~isempty(OPEN)
    % pop
    f = OPEN(:, 4) + OPEN(:, 5);
    [~, index] = min(f);
    cur_node = OPEN(index, :);
    OPEN(index, :) = [];

    % exists in CLOSED set
    if loc_list(cur_node, CLOSED, [1, 2, 3])
        continue
    end

    % update expand zone
    if ~loc_list(cur_node, EXPAND, [1, 2, 3])
        EXPAND = [EXPAND; cur_node(1:3)];
    end

    % goal found
    if all(cur_node(1:3) == goal)
        CLOSED = [cur_node; CLOSED];
        goal_reached = true;
        cost = cur_node(4);
        break
    end

    % explore neighbors
    for i = 1:motion_num
        node_n = [
            cur_node(1) + motion(i, 1), ...
            cur_node(2) + motion(i, 2), ...
            cur_node(3) + motion(i, 3), ...
            cur_node(4) + motion(i, 4), ...
            0, ...
            cur_node(1), cur_node(2), cur_node(3)];
        node_n(5) = h(node_n(1:3), goal);

        % bounds check
        if node_n(1) < 1 || node_n(2) < 1 || node_n(3) < 1 || ...
           node_n(1) > size(map,1) || node_n(2) > size(map,2) || node_n(3) > size(map,3)
            continue
        end

        % exists in CLOSED set
        if loc_list(node_n, CLOSED, [1, 2, 3])
            continue
        end

        % obstacle
        if map(node_n(1), node_n(2), node_n(3)) == 2
            continue
        end

        % update OPEN set
        OPEN = [OPEN; node_n];
    end
    CLOSED = [cur_node; CLOSED];
end

% extract path
path = extract_path(CLOSED, start);
end

%%
function h_val = h(node, goal)
% @brief: 3D Manhattan distance
h_val = abs(node(1) - goal(1)) + abs(node(2) - goal(2)) + abs(node(3) - goal(3));
end

function index = loc_list(node, list, range)
% @brief: locate the node in given list
num = size(list);
index = 0;

if ~num(1)
    return
else
    for i = 1:num(1)
        if isequal(node(range), list(i, range))
            index = i;
            return
        end
    end
end
end

function path = extract_path(close, start)
% @brief: Extract the path based on the CLOSED set.
path = [];
closeNum = size(close, 1);
index = 1;

while 1
    path = [path; close(index, 1:3)];

    if isequal(close(index, 1:3), start)
        break
    end

    for i = 1:closeNum
        if isequal(close(i, 1:3), close(index, 6:8))
            index = i;
            break
        end
    end
end
end

% Example usage:
% map = zeros(10,10,10); % 3D map
% map(5,5,5) = 2; % obstacle
% start = [1,1,1];
% goal = [10,10,10];
% [path, goal_reached, cost, EXPAND] = a_star(map, start, goal);