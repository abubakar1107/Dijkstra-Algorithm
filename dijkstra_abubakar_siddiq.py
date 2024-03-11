import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

# Defining the map and its dimensions
map_width, map_height = 1200, 500
grid = np.zeros((map_height, map_width))

# Define the actions and their costs
actions = [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1),
           (-1, -1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (1, 1, 1.4)]

def create_map(map_width, map_height, clearance):

    #white map background
    grid = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255
    print(grid.shape)

    #rectangular obstacles
    cv2.rectangle(grid, (100, 100), (175, 500), (255, 150, 0), -1)
    cv2.rectangle(grid, (275, 0), (350, 400), (255, 0, 0), -1)
    cv2.rectangle(grid, (900, 50), (1100, 125), (255, 0, 0), -1)
    cv2.rectangle(grid, (900, 375), (1100, 450), (255, 0, 0), -1)
    cv2.rectangle(grid, (1020, 125), (1100, 375), (255, 0, 0), -1)

    #hexagonal obstacle
    hexagon = np.array([[650, 120], [537, 185], [537, 315], [650, 380], [763, 315], [763, 185]], np.int32)
    hexagon = hexagon.reshape((-1, 1, 2))
    cv2.polylines(grid, [hexagon], True, (0, 0, 0), thickness=2)
    cv2.fillPoly(grid, [hexagon], (0, 0, 0))

    #Inflate obstacles by the clearance using dilation
    kernel = np.ones((2*clearance+1, 2*clearance+1), np.uint8)
    grid = cv2.dilate(grid, kernel, iterations=1)

    # Convert to grayscale and threshold to make sure it is binary
    grid = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
    _, grid = cv2.threshold(grid, 127, 255, cv2.THRESH_BINARY_INV)
  
    return grid
    
  
def validate_goal(grid, goal):
    if grid[goal[1], goal[0]] == -1:  # Assuming grid[y, x] == -1 indicates an obstacle

        print("------------------------------- \n Goal Position Not reachable\n ---------------------------------")
        raise ValueError("Goal position Not reachable: It is within an obstacle space.")


def dijkstra(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    exploration = [start]  

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
                    exploration.append(next_node) 

    return came_from, cost_so_far, exploration

def reconstruct_path(came_from, start, goal):

    current = goal
    path = [current]

    while current != start:

        current = came_from[current]
        path.append(current)

    path.reverse()
    return path

exploration_step_size = 1000  #number to show more nodes per frame and speed up the animation

# Function to animate the exploration and the path
def animate(i):
    for node in exploration[i * exploration_step_size:(i + 1) * exploration_step_size]:
        plt.plot(node[0], node[1], 'yo', markersize=2)
    if (i + 1) * exploration_step_size >= len(exploration):
        for node in path:
            plt.plot(node[0], node[1], 'ro', markersize=1)
        ani.event_source.stop()  # Stop the animation once the path is fully drawn

# Create the map with obstacles and clearance
clearance = 5 
grid = create_map(map_width, map_height, clearance)

# Define start and goal positions
start = (5, 5)
goal = (500, 60)

#validates the possibility of the goal
validate_goal(grid, goal)

# Run Dijkstra's algorithm
came_from, cost_so_far, exploration = dijkstra(grid, start, goal)
total_frames = len(exploration) // exploration_step_size + 1


# Reconstruct the path from start to goal
path = reconstruct_path(came_from, start, goal)
print("Solution path: ",path)

# Create the base plot
fig, ax = plt.subplots()
ax.imshow(grid, cmap=colors.ListedColormap(['white', 'black']), interpolation='none', origin='lower')
ax.plot(start[0], start[1], 'go', markersize=10)  # Start in green
ax.plot(goal[0], goal[1], 'bo', markersize=10)    # Goal in blue

# Run the animation
ani = FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)

plt.show()
