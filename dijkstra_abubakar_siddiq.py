import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

#map with dimensions
map_width, map_height = 1200, 500

#actions and their costs
actions = [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1),
           (-1, -1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (1, 1, 1.4)]


#Function to create the map 
def create_map(map_width, map_height, clearance):
    # Create a binary grid for pathfinding
    grid_binary = np.zeros((map_height, map_width), dtype=np.uint8)

    # Create a visual map with colors
    grid_visual = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

    #defining border clearance 
    cv2.rectangle(grid_binary, (0, 0), (map_width, map_height), 1, clearance)

    # clear inner part of the border to get free space
    cv2.rectangle(grid_binary, (clearance, clearance), (map_width-clearance, map_height-clearance), 0, -1)

    # Visual representation of borders with clearance
    cv2.rectangle(grid_visual, (0, 0), (map_width, map_height), (255, 0, 0), clearance)

    #rectangular obstacles into the binary grid and visual grid
    cv2.rectangle(grid_binary, (100, 100), (175, 500), 1, -1)
    cv2.rectangle(grid_visual, (100, 100), (175, 500), (255, 150, 0), -1)

    cv2.rectangle(grid_binary, (275, 0), (350, 400), 1, -1)
    cv2.rectangle(grid_visual, (275, 0), (350, 400), (150, 255, 0), -1)

    cv2.rectangle(grid_binary, (900, 50), (1100, 125), 1, -1)
    cv2.rectangle(grid_visual, (900, 50), (1100, 125), (150, 80, 0), -1)

    cv2.rectangle(grid_binary, (900, 375), (1100, 450), 1, -1)
    cv2.rectangle(grid_visual, (900, 375), (1100, 450), (150, 80, 0), -1)

    cv2.rectangle(grid_binary, (1020, 125), (1100, 375), 1, -1)
    cv2.rectangle(grid_visual, (1020, 125), (1100, 375), (150, 80, 0), -1)

    #hexagonal obstacle into the binary grid and visual grid 
    hexagon = np.array([[650, 120], [537, 185], [537, 315], [650, 380], [763, 315], [763, 185]], np.int32)
    cv2.fillPoly(grid_binary, [hexagon], 1)
    cv2.fillPoly(grid_visual, [hexagon], (0, 0, 200))


    kernel = np.ones((2*clearance+1, 2*clearance+1), np.uint8)
    grid_binary = cv2.dilate(grid_binary, kernel, iterations=1)
    
    return grid_binary, grid_visual

# Validate goal position
def validate_goal(grid_binary, goal):
    if grid_binary[goal[1], goal[0]] == 1:
        raise ValueError("Goal position not reachable: It is within an obstacle space.")

def dijkstra(grid_binary, start, goal):
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
            if 0 <= next_node[0] < map_width and 0 <= next_node[1] < map_height and grid_binary[next_node[1], next_node[0]] == 0:
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

# Function to draw the start and goal on the visual grid
def draw_start_goal_on_visual_grid(grid_visual, start, goal):

    cv2.circle(grid_visual, start, 5, (0, 255, 0), -1)  # Green for start
    cv2.circle(grid_visual, goal, 5, (0, 0, 255), -1)  # Red for goal
    return grid_visual

# Animation function
def animate(i):
    explored_nodes = exploration[:i*exploration_step_size]
    if i*exploration_step_size < len(exploration):
        ax.clear()
        ax.imshow(grid_visual, origin='lower')
        ax.scatter([node[0] for node in explored_nodes], [node[1] for node in explored_nodes], color='yellow', s=1)
    else:
        ax.clear()
        ax.imshow(grid_visual, origin='lower')
        ax.plot([node[0] for node in path], [node[1] for node in path], color="red", linewidth=2)
        ani.event_source.stop()

sx = int(input("Enter X cordinate of start position: "))
sy = int(input("Enter X cordinate of start position: "))
gx = int(input("Enter X cordinate of start position: "))
gy = int(input("Enter X cordinate of start position: "))

#(50, 10)  # Example start position
#(200, 400)  # Example goal position

start = (sx,sy)
goal = (gx,gy)

clearance = 5
grid_binary, grid_visual = create_map(map_width, map_height, clearance)
grid_visual = draw_start_goal_on_visual_grid(grid_visual, start, goal)

try:
    validate_goal(grid_binary, goal)
except ValueError as e:
    print(e)
    exit()  

came_from, cost_so_far, exploration = dijkstra(grid_binary, start, goal)
path = reconstruct_path(came_from, start, goal)

print(path)

exploration_step_size = 1000  # Number of nodes to show per frame in the animation

fig, ax = plt.subplots()

ani = FuncAnimation(fig, animate, frames=(len(exploration)//exploration_step_size)+len(path), interval=100, repeat=False)

plt.show()
