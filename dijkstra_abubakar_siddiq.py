import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

# Define the map and its dimensions
map_width, map_height = 200, 100
grid = np.zeros((map_height, map_width))

# Define the actions and their costs
actions = [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1),
           (-1, -1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (1, 1, 1.4)]

def create_map():
    clearance = 5  # 5 mm clearance

    # Define the obstacle space on the grid by marking them as -1
    # These are hypothetical obstacle coordinates, replace them with your actual obstacle coordinates
    # Example: A rectangle obstacle
    grid[20:40, 45:55] = -1
    
    grid[50:90, 45:55] = -1
    # Example: A circular obstacle
    # for x in range(map_width):
    #     for y in range(map_height):
    #         if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
    #             grid[y, x] = -1
    
    # Inflate the obstacles by the clearance
    inflated_grid = grid.copy()
    for x in range(map_width):
        for y in range(map_height):
            if grid[y, x] == -1:
                for dx in range(-clearance, clearance + 1):
                    for dy in range(-clearance, clearance + 1):
                        if 0 <= x + dx < map_width and 0 <= y + dy < map_height:
                            inflated_grid[y + dy, x + dx] = -1
    return inflated_grid

def dijkstra(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    exploration = [start]  # List to keep track of exploration order

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break

        for dx, dy, action_cost in actions:
            next_node = (current_node[0] + dx, current_node[1] + dy)
            if 0 <= next_node[0] < map_width and 0 <= next_node[1] < map_height and grid[next_node[1], next_node[0]] == 0:
                new_cost = current_cost + action_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost
                    heapq.heappush(open_set, (priority, next_node))
                    came_from[next_node] = current_node
                    exploration.append(next_node)  # Add node to exploration list

    return came_from, cost_so_far, exploration

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def animate(i):
    # Plotting the explored nodes
    if i < len(exploration):
        node = exploration[i]
        plt.plot(node[0], node[1], 'yo', markersize=2)
    # Plotting the path
    elif i == len(exploration):
        for node in path:
            plt.plot(node[0], node[1], 'ro', markersize=2)

# Create the map with obstacles and clearance
grid = create_map()

# Define start and goal positions
start = (5, 5)
goal = (100, 60)

# Run Dijkstra's algorithm
came_from, cost_so_far, exploration = dijkstra(grid, start, goal)

# Reconstruct the path from start to goal
path = reconstruct_path(came_from, start, goal)

# Create the base plot
fig, ax = plt.subplots()
ax.imshow(grid, cmap=colors.ListedColormap(['white', 'black']), interpolation='none', origin='lower')
ax.plot(start[0], start[1], 'go', markersize=10)  # Start in green
ax.plot(goal[0], goal[1], 'bo', markersize=10)    # Goal in blue

# Run the animation
ani = FuncAnimation(fig, animate, frames=len(exploration) + 1, interval=50)
plt.show()
