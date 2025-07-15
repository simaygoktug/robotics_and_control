#A*: Heuristic + Cost (h + g)

#Dijkstra: Saf Cost (g)

#RRT: Sampling tabanlı planlama

#######################################################

import heapq

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, []))
    visited = set()

    while open_list:
        _, cost, current, path = heapq.heappop(open_list)
        if current == goal:
            return path + [goal]
        if current in visited:
            continue
        visited.add(current)
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[0]][neighbor[1]] == 0:
                    heapq.heappush(open_list, (cost+1+heuristic(neighbor, goal), cost+1, neighbor, path+[current]))
    return []

# Basit harita
grid = [[0]*10 for _ in range(10)]
grid[5][2:8] = [1]*6
path = astar(grid, (0,0), (9,9))

for x, y in path:
    grid[x][y] = '*'
for row in grid:
    print(''.join(str(i) for i in row))
