import matplotlib.pyplot as plt
import random
import heapq
import numpy as np
import matplotlib.animation as animation
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
class WorldMap:
    def __init__(self, rows, cols, num_dirt_blocks, num_obs):
        self.rows = rows
        self.cols = cols
        self.num_dirt_blocks = num_dirt_blocks
        self.num_obs = num_obs
        self.world_map = [['clean' for _ in range(cols)] for _ in range(rows)]

        self.agent_positions = {}  # Dictionary to store agent positions

        # Place dirt blocks randomly on the map
        for _ in range(num_dirt_blocks):
            row = random.randint(0, rows - 1)
            col = random.randint(0, cols - 1)
            while self.world_map[row][col] == 'dirt' or self.world_map[row][col] == 'agent':
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
            self.world_map[row][col] = 'dirt'

        # Place obstacles randomly on the map (excluding corners)
        for _ in range(num_obs):
            row = random.randint(1, rows - 2)  # Avoid corners
            col = random.randint(1, cols - 2)  # Avoid corners
            while self.world_map[row][col] == 'dirt' or self.world_map[row][col] == 'agent' or self.world_map[row][col] == 'obs':
                row = random.randint(1, rows - 2)  # Avoid corners
                col = random.randint(1, cols - 2)  # Avoid corners
            self.world_map[row][col] = 'obs'

    def add_agent(self, agent_id):
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.cols - 1)
            if self.world_map[row][col] == 'clean':
                self.world_map[row][col] = 'agent'
                self.agent_positions[agent_id] = (row, col)
                break


    def getAgentPos(self, agent_id):
        if agent_id in self.agent_positions:
            return self.agent_positions[agent_id]
        else:
            return None  # Agent not found
        
    def move_agent(self, agent_id, new_position):
        if agent_id in self.agent_positions:
            current_position = self.agent_positions[agent_id]
            if self.world_map[current_position[0]][current_position[1]] == 'agent':
                self.world_map[current_position[0]][current_position[1]] = 'clean'  # Clear the current cell
            self.world_map[new_position[0]][new_position[1]] = 'agent'  # Place the agent in the new cell
            self.agent_positions[agent_id] = new_position  # Update the agent's position

        
    def display_map(self, ax):
        ax.clear()  # Clear the current plot
        for row in range(self.rows):
            for col in range(self.cols):
                if self.world_map[row][col] == 'dirt':
                    ax.plot(col + 0.5, self.rows - row - 0.5, 'ro', markersize=10)  # Display dirt as red dots
                elif self.world_map[row][col] == 'agent':
                    ax.plot(col + 0.5, self.rows - row - 0.5, 'bo', markersize=10)  # Display agents as blue dots
                elif self.world_map[row][col] == 'obs':
                    ax.plot(col + 0.5, self.rows - row - 0.5, 'ko', markersize=10)  # Display obstacles as black dots

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.invert_yaxis()
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.grid()

    def is_valid_position(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols


class VacuumAgent(Agent):
    def __init__(self, unique_id, model, world_map):
        super().__init__(unique_id, model)
        self.move_speed = 1  # How many cells the agent moves per step
        self.world_map = world_map
        self.beliefs = {'position': None}
        self.desires = {'clean': None}
        self.intentions = {'clean': None}
        self.orientation = random.choice(['N', 'S', 'E', 'W'])  # Initial
        self.roaming=False
        self.edge_path = self.generate_edge_path() # Initialize the class variable
    
    def perceive(self):
        self.beliefs['position'] = self.world_map.getAgentPos(self.unique_id)        
        visible_cells = self.get_visible_cells()
        self.desires['clean'] = self._find_dirt_in_visible_cells(visible_cells)

    def get_visible_cells(self):
        visible_cells = []
        row, col = self.beliefs['position']

        visible_range = 2

        for dr in range(-visible_range, visible_range + 1):
            for dc in range(-visible_range, visible_range + 1):
                new_row, new_col = row + dr, col + dc
                if self.world_map.is_valid_position(new_row, new_col):
                    visible_cells.append((new_row, new_col))
        return visible_cells
    
    def _find_dirt_in_visible_cells(self, visible_cells):
        dirt_positions = [(row, col) for row, col in visible_cells if self.world_map.world_map[row][col] == 'dirt']
        if dirt_positions:
            closest_dirt = min(dirt_positions, key=lambda pos: abs(pos[0] - self.beliefs['position'][0]) + abs(pos[1] - self.beliefs['position'][1]))
            return closest_dirt
        else:
            return None
            
    def choose_intention(self):
        if not self.intentions['clean'] or (self.desires['clean'] and self.roaming):
            self.intentions['clean'] = self.find_path(self.beliefs['position'],self.desires['clean'])
            self.roaming=False
            
    def find_path(self, current, goal):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def is_valid_move(row, col):
            return 0 <= row < self.world_map.rows and 0 <= col < self.world_map.cols and self.world_map.world_map[row][col] != 'obs'

        def a_star_search(start, target):
            if target:
                open_list = []
                closed_list = set()

                heapq.heappush(open_list, (0, start))
                g_costs = {start: 0}
                f_scores = {start: heuristic(start, target)}

                while open_list:
                    current = heapq.heappop(open_list)[1]

                    if current == target:
                        path = []
                        while current != start:
                            path.append(current)
                            current = came_from[current]
                        path.reverse()
                        return path

                    closed_list.add(current)

                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        neighbor = (current[0] + dr, current[1] + dc)

                        if not is_valid_move(neighbor[0], neighbor[1]):
                            continue

                        tentative_g = g_costs[current] + 1

                        if neighbor in closed_list and tentative_g >= g_costs.get(neighbor, 0):
                            continue

                        if tentative_g < g_costs.get(neighbor, 0) or neighbor not in [item[1] for item in open_list]:
                            came_from[neighbor] = current
                            g_costs[neighbor] = tentative_g
                            f_scores[neighbor] = tentative_g + heuristic(neighbor, target)
                            heapq.heappush(open_list, (f_scores[neighbor], neighbor))

                return []

        came_from = {}
        path = a_star_search(current, goal)
        return path
    
    def act(self):
        if self.intentions['clean']:
            target_row, target_col = self.intentions['clean'][0]
            current_row, current_col = self.beliefs['position']
            self.world_map.move_agent(self.unique_id, self.intentions['clean'][0])
            self.intentions['clean'].pop(0)
        else:
            # Roam randomly if there are no desires
            self.roam()

    def roam(self):
        # Find the closest point on the edge path to the current position
        current_position = self.beliefs['position']
        closest_edge_point = min(self.edge_path, key=lambda pos: self.distance(pos, current_position))

        # Calculate the obstacle-free path to the closest edge point
        obstacle_free_path = self.find_path(self.beliefs['position'], closest_edge_point)

        # Find the index of the closest edge point in the edge path
        closest_edge_index = self.edge_path.index(closest_edge_point)

        # Set the agent's intentions to the obstacle-free path
        self.intentions['clean'] = obstacle_free_path

        # Check if the agent is not at the last square of the edge path
        if closest_edge_index < len(self.edge_path) - 1:
            # Splice the remaining part of the edge path starting from the closest edge point
            remaining_edge_path = self.edge_path[closest_edge_index + 1:]

            # Append the remaining edge path to the agent's intentions
            self.intentions['clean'] += remaining_edge_path

            # Set the roaming status to True
            self.roaming = True


    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def generate_edge_path(self):
        new_edge_path = [(0, 0)]
        for i in reversed(range(self.world_map.rows)):
            for j in range(self.world_map.cols):
                if self.world_map.world_map[j][i-1]!='dirt':
                    add = self.find_path(new_edge_path[-1], (j, i - 1))
                    new_edge_path.extend(add)
        new_edge_path.extend(self.find_path(new_edge_path[-1],(0,0)))
        return new_edge_path
    
    def _find_dirt(self):
        dirt_positions = [(row, col) for row in range(self.world_map.rows) for col in range(self.world_map.cols) if self.world_map.world_map[row][col] == 'dirt']
        if dirt_positions:
            closest_dirt = min(dirt_positions, key=lambda pos: abs(pos[0] - self.beliefs['position'][0]) + abs(pos[1] - self.beliefs['position'][1]))
            return closest_dirt
        else:
            return None
    
    def step(self):
        if self.model.schedule.time < self.model.max_steps:  # Verifica el límite de tiempo
            self.perceive()
            self.choose_intention()
            self.act()
            
                       

class VacuumModel(Model):
    def __init__(self, width, height, num_agents, num_dirt_blocks, max_steps, num_obs):
        self.num_agents = num_agents
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.max_steps = max_steps
        self.performance_data = []

        self.world_map = WorldMap(width, height, num_dirt_blocks, num_obs)

        for i in range(num_agents):
            self.world_map.add_agent(i)  # Provide the agent_id
            agent = VacuumAgent(i, self, self.world_map)
            self.schedule.add(agent)
    def step(self):
        if self.schedule.time < self.max_steps:
            self.performance_data.append(self.calculate_performance())
            self.schedule.step()
        
        
    def calculate_performance(self):
        cleaned_cells =self.world_map.num_dirt_blocks - sum([1 for row in self.world_map.world_map for cell in row if cell == 'dirt'])
        total_cells = self.world_map.num_dirt_blocks
        performance = cleaned_cells / total_cells
        return performance

    def check_postconditions(self):
        for row in self.world_map.world_map:
            for cell in row:
                if cell=='dirt':
                    return False
        return True 

def animate(frame, model):
    if not model.check_postconditions():
        model.step()  # Step the model
        ax.clear()  # Clear the current plot
        model.world_map.display_map(ax)  # Display the updated map

n = 10
num_agents = 10
num_dirt_blocks = 20  # Número de espacios sucios iniciales
max_steps = 80  # Máximo de pasos en la simulación
num_obs=5
t=1
# Simulación con una configuración específica
if t==1:
    model = VacuumModel(width=n, height=n, num_agents=num_agents, num_dirt_blocks=num_dirt_blocks, max_steps=max_steps,num_obs=num_obs)
    fig, ax = plt.subplots(figsize=(7, 7))
    ani = animation.FuncAnimation(fig, animate, frames=max_steps, repeat=False, fargs=(model,))
    plt.show()

    # Genera un gráfico de rendimiento en función del tiempo
    performance_values = model.performance_data
    time_steps = list(range(len(performance_values)))
    plt.plot(time_steps, performance_values)

    plt.xlabel('Time Steps')
    plt.ylabel('Performance')
    plt.title('Performance over Time')
    plt.show()
    # Verificación de postcondiciones
    if model.check_postconditions():
        print("Postconditions met")
    else:
        print("Postconditions not met")
else:
    num_attempts = 100
    performance_results = []
    satisfied_postconditions = []
    time_result = []

    for attempt in range(num_attempts):
        model = VacuumModel(width=n, height=n, num_agents=num_agents, num_dirt_blocks=num_dirt_blocks, max_steps=max_steps,num_obs=num_obs)
        while not model.check_postconditions():
            if model.schedule.time >= max_steps:
                break
            model.step()
        time_result.append(model.schedule.time)
        performance_results.append(model.performance_data)
        satisfied_postconditions.append(model.check_postconditions())

    # Create a new figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot performance data for each attempt
    for attempt, performance in enumerate(performance_results):
        if len(performance) < max_steps:
            performance.extend([performance[-1]] * (max_steps - len(performance)))
        ax1.plot(range(len(performance)), performance, label=f'Attempt {attempt + 1}')

    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Performance')
    ax1.set_title('Performance over Time (Multiple Attempts)')

    # Create a pie chart for the number of satisfied postconditions
    num_satisfied = sum(satisfied_postconditions)
    num_unsatisfied = num_attempts - num_satisfied
    ax2.pie([num_satisfied, num_unsatisfied], labels=['Satisfied', 'Unsatisfied'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Percentage of Satisfied Postconditions')

    # Calculate average performance of agents and average number of steps to finish
    average_performance = np.mean(performance_results, axis=1)
    avg_per=np.mean(performance_results)
    # Create histograms for average performance and average number of steps to finish
    num_bins = 100  # You can adjust the number of bins as needed
    ax3.hist(average_performance, bins=num_bins, edgecolor='black')
    ax3.set_xlabel('Average Performance')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Average Performance')
    time_performance = np.mean(time_result)

    ax4.hist(time_result, bins=num_bins, edgecolor='black')
    ax4.set_xlabel('Average Steps to Finish')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Average Steps to Finish')

    print(f'Statistics: \nAttempts: {num_attempts}\nSolved: {num_satisfied}\nAvrg performance {avg_per}\nTime performance: {time_performance}')
    
    plt.show()