import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output
import time



class ACOPathFinder:
    def __init__(self, maze, start_pos, end_pos, 
                 n_ants=30, n_iterations=100, 
                 evaporation_rate=0.5, 
                 alpha=1, beta=2, q=100):
        
        self.maze = maze
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q = q
        
        self.height, self.width = maze.shape
        self.pheromone = np.ones((self.height, self.width)) * 0.1  
        
    def is_valid_move(self, pos):
    
        i, j = pos
        return (0 <= i < self.height and 0 <= j < self.width and 
                self.maze[i, j] != 1)
    
    def get_neighbors(self, pos):
    
        i, j = pos
        neighbors = []
        for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:  
            new_pos = (i+di, j+dj)
            if self.is_valid_move(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def calculate_distance(self, pos1, pos2):
       
        return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    
    def run(self):
        
        best_path = None
        best_path_length = float('inf')
        history = []  
        
        for iteration in range(self.n_iterations):
            all_paths = []
            
            
            for ant in range(self.n_ants):
                current_pos = self.start_pos
                path = [current_pos]
                visited = set([current_pos])
                
                
                while current_pos != self.end_pos and len(path) < self.height * self.width * 2:
                    neighbors = [n for n in self.get_neighbors(current_pos) if n not in visited]
                    
                    if not neighbors:
                        break  
                    
                   
                    probabilities = []
                    total = 0
                    
                    for neighbor in neighbors:
                       
                        pheromone = self.pheromone[neighbor[0], neighbor[1]]
                        distance = self.calculate_distance(neighbor, self.end_pos)
                        if distance == 0:
                            distance = 0.1  
                        
                      
                        weight = (pheromone ** self.alpha) * ((1/distance) ** self.beta)
                        probabilities.append(weight)
                        total += weight
                    
                    if total == 0:
                       
                        next_pos = random.choice(neighbors)
                    else:
                        
                        probabilities = [p/total for p in probabilities]
                        next_pos = neighbors[np.random.choice(len(neighbors), p=probabilities)]
                    
                    path.append(next_pos)
                    visited.add(next_pos)
                    current_pos = next_pos
                
                if current_pos == self.end_pos:
                    all_paths.append((path, len(path)))
                    if len(path) < best_path_length:
                        best_path = path
                        best_path_length = len(path)
            
            
            self.update_pheromones(all_paths)
            
          
            history.append(best_path_length)
            
           
            if iteration % 10 == 0:
                self.visualize_progress(iteration, best_path, history)
        
        return best_path, best_path_length, history
    
    def update_pheromones(self, all_paths):
       
        self.pheromone *= (1 - self.evaporation_rate)
        
        
        for path, length in all_paths:
            for pos in path:
                self.pheromone[pos[0], pos[1]] += self.q / length
    
    def visualize_progress(self, iteration, best_path=None, history=None):
       
        clear_output(wait=True)
        plt.figure(figsize=(30, 10))
        
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
        plt.imshow(self.pheromone, cmap='hot', alpha=0.5)
        plt.colorbar(label=' Pheromone level')
        if best_path:
            path_x = [p[1] for p in best_path]
            path_y = [p[0] for p in best_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2)
        plt.title(f"ACO - Repetition {iteration+1}/{self.n_iterations}")
        plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=200, marker='o')
        plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=200, marker='o')
        
       
        if best_path:
            plt.subplot(1, 3, 2)
            plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
            path_x = [p[1] for p in best_path]
            path_y = [p[0] for p in best_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2)
            plt.scatter(path_x, path_y, c='blue', s=20)
            plt.title(f"Best path : {len(best_path)})")
            plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=200, marker='o')
            plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=200, marker='o')
        
       
        if history:
            plt.subplot(1, 3, 3)
            plt.plot(history, 'b-')
            plt.xlabel('Repetition')
            plt.ylabel('Length of the path')
            plt.title('  The development of the path length ')
        
        plt.tight_layout()
        plt.show()
