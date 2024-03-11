import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors

# Map size
map_width, map_height = 200, 100

# Initialize the grid, 0 is free space, -1 is obstacle
grid = np.zeros((map_height, map_width))

# Define the actions with their cost
actions = {
    (0, -1): 1,  # UP
    (0, 1): 1,   # DOWN
    (-1, 0): 1,  # LEFT
    (1, 0): 1,   # RIGHT
    (-1, -1): 1.4,  # UP-LEFT
    (1, -1): 1.4,   # UP-RIGHT
    (-1, 1): 1.4,   # DOWN-LEFT
    (1, 1): 1.4    # DOWN-RIGHT
}

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

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break

        for action, cost in actions.items():
            next_node = (current_node[0] + action[0], current_node[1] + action[1])
            if 0 <= next_node[0] < map_width and 0 <= next_node[1] < map_height and grid[next_node[1], next_node[0]] == 0:
                new_cost = current_cost + cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost
                    heapq.heappush(open_set, (priority, next_node))
                    came_from[next_node] = current_node

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def visualize_path(grid, path):
    # Create a color map where free space is white and obstacles are black
    cmap = colors.ListedColormap(['white', 'black'])
    # Create a matrix to represent the grid for visualization
    grid_vis = np.zeros_like(grid)
    grid_vis[grid == -1] = 1  # Mark obstacles as black
    # Plot the grid
    plt.imshow(grid_vis, cmap=cmap, origin='lower')
    # Extract x and y coordinates from the path
    x_coords, y_coords = zip(*path)
    # Plot the path
    plt.plot(x_coords, y_coords, color='red')
    # Show the plot
    plt.show()

# Use the create_map function to initialize the grid with obstacles and clearance
grid = create_map()

# Define the start and goal positions
start = (5, 5)  # Start should be a tuple (x, y)
goal = (190, 60)  # Goal should be a tuple (x, y)

# Run Dijkstra's algorithm
came_from, cost_so_far = dijkstra(grid, start, goal)

# Reconstruct the path
path = reconstruct_path(came_from, start, goal)

# Visualize the path
visualize_path(grid, path)
