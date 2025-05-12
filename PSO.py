import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output
import time
from collections import deque






class PSOPathFinder:
    def __init__(self, maze, start_pos, end_pos, 
                 n_particles=50, n_iterations=200,
                 w=0.8, c1=2, c2=2):
    

    
    
        self.maze = maze
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.height, self.width = maze.shape
        self.global_best_position = start_pos
        self.global_best_score = -float('inf')
        self.history = []
        self.full_path_history = []
        
       
        self.positions = [self.get_valid_position() for _ in range(n_particles)]
        self.velocities = [(0, 0) for _ in range(n_particles)]
        self.best_positions = self.positions.copy()
        self.best_scores = [-float('inf')] * n_particles
        
        
        self.path_map = np.zeros((self.height, self.width))
       
    def get_valid_position(self):
        
        if random.random() < 0.7:  
            i = random.randint(0, min(5, self.height-1))
            j = random.randint(0, min(5, self.width-1))
            if self.is_valid_move((i, j)):
                return (i, j)
        return (random.randint(0, self.height-1), random.randint(0, self.width-1))
    
    def is_valid_move(self, pos):
        
        i, j = pos
        return (0 <= i < self.height and 0 <= j < self.width 
                and self.maze[i, j] != 1)
   
    def get_neighbors(self, pos):
       
        i, j = pos
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if self.is_valid_move((ni, nj)):
                    neighbors.append((ni, nj))
        return neighbors

    def calculate_fitness(self, pos):
    
        path = self.find_path(pos)
        if path and path[-1] == self.end_pos:
            length = len(path)
            straightness = sum(1 for i in range(1, len(path)) 
                          if path[i][0] == path[i-1][0] or path[i][1] == path[i-1][1])
            return 1 / (length + 0.3*(length - straightness) + 1)
        return 1 / (self.manhattan_distance(pos, self.end_pos) + 10)  
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])
    

    
    
    
    
    def find_path(self, start):
        
        path = []
        current = start
        visited = set([current])
        
        for _ in range(self.height * self.width * 2):  
            if current == self.end_pos:
                break
                
            neighbors = self.get_neighbors(current)
            neighbors = [n for n in neighbors if n not in visited]
            
            if not neighbors:
                if len(path) > 0:
                    current = path.pop()  
                    continue
                else:
                    break
            

            neighbors.sort(key=lambda p: (
                -self.path_map[p[0], p[1]], 
                self.manhattan_distance(p, self.end_pos) 
            ))
            next_pos = neighbors[0]
            
            path.append(next_pos)
            visited.add(next_pos)
            current = next_pos
        
        if current == self.end_pos:
            return [start] + path
        return None
    



    
    
    def update_particles(self):
        
        for i in range(self.n_particles):
            pos = self.positions[i]
            vel = self.velocities[i]
            
    
            r1, r2 = random.random(), random.random()
            new_vel_i = self.w * vel[0] + \
                        self.c1 * r1 * (self.best_positions[i][0] - pos[0]) + \
                        self.c2 * r2 * (self.global_best_position[0] - pos[0])
            new_vel_j = self.w * vel[1] + \
                        self.c1 * r1 * (self.best_positions[i][1] - pos[1]) + \
                        self.c2 * r2 * (self.global_best_position[1] - pos[1])
            
            
            norm = (new_vel_i**2 + new_vel_j**2)**0.5
            if norm > 0:
                new_vel_i, new_vel_j = new_vel_i/norm, new_vel_j/norm
            
            
            new_i = round(pos[0] + new_vel_i)
            new_j = round(pos[1] + new_vel_j)
            new_pos = (new_i, new_j)
            
        
            if not self.is_valid_move(new_pos):
                path = self.find_path(pos)
                if path and len(path) > 1:
                    new_pos = path[1]  
                    new_vel_i, new_vel_j = new_pos[0]-pos[0], new_pos[1]-pos[1]
                else:
                    
                    neighbors = self.get_neighbors(pos)
                    if neighbors:
                        new_pos = random.choice(neighbors)
                        new_vel_i, new_vel_j = new_pos[0]-pos[0], new_pos[1]-pos[1]
                    else:
                        new_pos = self.get_valid_position()
                        new_vel_i, new_vel_j = 0, 0
            
            
            self.positions[i] = new_pos
            self.velocities[i] = (new_vel_i, new_vel_j)
            
            
            current_fitness = self.calculate_fitness(new_pos)
            if current_fitness > self.best_scores[i]:
                self.best_positions[i] = new_pos
                self.best_scores[i] = current_fitness
                
                if current_fitness > self.global_best_score:
                    self.global_best_position = new_pos
                    self.global_best_score = current_fitness
    



    def update_path_map(self):
       
        for pos in self.positions:
            path = self.find_path(pos)
            if path:
                for p in path:
                    self.path_map[p[0], p[1]] += 1



    def run(self):
        
        for iteration in range(self.n_iterations):
            self.update_particles()
            self.update_path_map()
            
            
            current_path = self.find_path(self.start_pos)
            current_length = len(current_path) if current_path else float('inf')
            self.history.append(current_length)
            self.full_path_history.append(current_path.copy() if current_path else None)
            
           
            if iteration % 10 == 0:
                clear_output(wait=True)
                self.visualize_progress(iteration, current_path)
                time.sleep(0.1)
        
        
        best_path = self.find_path(self.start_pos)
        best_length = len(best_path) if best_path else float('inf')
        
        clear_output(wait=True)
        self.visualize_progress(self.n_iterations, best_path, final=True)
        
        return best_path, best_length, self.history
    
    def visualize_progress(self, iteration, current_path=None, final=False):
        
        plt.figure(figsize=(18, 6))
        
       
        plt.subplot(1, 3, 1)
        plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
        
       
        if current_path:
            path_x = [p[1] for p in current_path]
            path_y = [p[0] for p in current_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
            plt.scatter(path_x, path_y, c='blue', s=30, alpha=0.5)
        
        
        particles_x = [p[1] for p in self.positions]
        particles_y = [p[0] for p in self.positions]
        plt.scatter(particles_x, particles_y, c='magenta', s=50, alpha=0.6)
        
        plt.title(f"  develop PSO - rebetion  {iteration+1}/{self.n_iterations}")
        plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=300, marker='o', edgecolor='black')
        plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=300, marker='o', edgecolor='black')
        
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
        
        best_path = self.find_path(self.start_pos)
        if best_path:
            path_x = [p[1] for p in best_path]
            path_y = [p[0] for p in best_path]
            plt.plot(path_x, path_y, 'm-', linewidth=3)
            plt.scatter(path_x, path_y, c='magenta', s=50)
            
        
            for i in range(1, len(best_path)):
                plt.plot([best_path[i-1][1], best_path[i][1]], 
                         [best_path[i-1][0], best_path[i][0]], 
                         'm-', linewidth=3, alpha=0.7)
            
            status = f"path : {len(best_path)})" if best_path[-1] == self.end_pos else "path not compelete  "
            plt.title(f"  best path  \n{status}")
        else:
            plt.title("   no path   ")
        
        plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=300, marker='o', edgecolor='black')
        plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=300, marker='o', edgecolor='black')
        
      
        plt.subplot(1, 3, 3)
        plt.plot(self.history, 'm-', linewidth=2)
        plt.xlabel('rebetion  ', fontsize=12)
        plt.ylabel(' path length ', fontsize=12)
        plt.title('path develop    ', fontsize=14)
        plt.grid(True)
        
        if final:
            plt.annotate(f'best path  : {min(self.history)}', 
                         xy=(self.history.index(min(self.history)), min(self.history)),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'),
                         bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.show()

