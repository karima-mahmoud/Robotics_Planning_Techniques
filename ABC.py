import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from IPython.display import display, clear_output
import time



class ABCPathFinder:
    
    
    
    
    
    def __init__(self, maze, start_pos, end_pos, 
                 n_bees=50, n_iterations=200, 
                 limit=30, scout_bee_ratio=0.3):
       
        self.maze = maze
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.n_bees = n_bees
        self.n_iterations = n_iterations
        self.limit = limit
        self.scout_bee_ratio = scout_bee_ratio
        
        self.height, self.width = maze.shape
        self.n_scout_bees = max(1, int(n_bees * scout_bee_ratio))
        self.n_worker_bees = n_bees - self.n_scout_bees
        
      
        self.food_sources = []
        self.trials = []  
        self.fitness = [] 
    
   
    def is_valid_move(self, pos):
       
        i, j = pos
        return (0 <= i < self.height and 0 <= j < self.width and self.maze[i, j] != 1)
    
    
    
    
    def get_neighbors(self, pos):
        
        i, j = pos
        neighbors = []
        for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:  
            new_pos = (i+di, j+dj)
            if self.is_valid_move(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    
   
    def generate_random_path(self):
        
        current_pos = self.start_pos
        path = [current_pos]
        visited = set(path)
        
        while current_pos != self.end_pos and len(path) < self.height * self.width:
            neighbors = [n for n in self.get_neighbors(current_pos) if n not in visited]
            
            if not neighbors:
                break  
            
        
            weights = []
            for n in neighbors:
                dist = (n[0]-self.end_pos[0])**2 + (n[1]-self.end_pos[1])**2
                weights.append(1/(dist+0.1))  
            
            next_pos = random.choices(neighbors, weights=weights, k=1)[0]
            path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
        
        return path
    

    def calculate_fitness(self, path):
       
        if not path or path[-1] != self.end_pos:
            return 0  
        
        length_penalty = len(path)
        straightness = 0
        for i in range(1, len(path)):
            prev_dir = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
            if i > 1:
                curr_dir = (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
                if prev_dir != curr_dir:
                    straightness += 1
        
        return 1 / (length_penalty + 0.5*straightness + 1)
    

    def initialize_food_sources(self):
        
        self.food_sources = []
        self.trials = []
        self.fitness = []
        
        for _ in range(self.n_bees):
            path = self.generate_random_path()
            self.food_sources.append(path)
            self.trials.append(0)
            self.fitness.append(self.calculate_fitness(path))
    
    
  

    def send_worker_bees(self):
      
        for i in range(self.n_worker_bees):
            if i >= len(self.food_sources):
                continue
                
            valid_partners = [j for j in range(len(self.food_sources)) if j != i]
            if not valid_partners:
                continue
                
            partner_idx = random.choice(valid_partners)
            partner_path = self.food_sources[partner_idx]
            
            new_path = self.mutate_path(partner_path)
            new_fitness = self.calculate_fitness(new_path)
            
            if new_fitness > self.fitness[i]:
                self.food_sources[i] = new_path
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
   

    def send_scout_bees(self):
      
        for i in range(len(self.food_sources)):
            if self.trials[i] > self.limit:
                self.food_sources[i] = self.generate_random_path()
                self.fitness[i] = self.calculate_fitness(self.food_sources[i])
                self.trials[i] = 0
    
    def mutate_path(self, path):
       
        if len(path) < 3 or path[-1] != self.end_pos:
            return self.generate_random_path()
        
        mutate_point = random.randint(1, min(len(path)-2, 20))
        new_path = path[:mutate_point]
        current_pos = new_path[-1]
        visited = set(new_path)
        
        while current_pos != self.end_pos and len(new_path) < self.height * self.width:
            neighbors = [n for n in self.get_neighbors(current_pos) if n not in visited]
            
            if not neighbors:
                break
                
            neighbors.sort(key=lambda pos: (pos[0]-self.end_pos[0])**2 + (pos[1]-self.end_pos[1])**2)
            next_pos = neighbors[0]
            
            new_path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
        
        return new_path



    def run(self):
       
        self.initialize_food_sources()
        best_path = None
        best_fitness = 0
        history = []
        
        for iteration in range(self.n_iterations):
            self.send_worker_bees()
            self.send_scout_bees()
            
            current_best_idx = np.argmax(self.fitness)
            current_best_path = self.food_sources[current_best_idx]
            current_best_fitness = self.fitness[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_path = current_best_path
                best_fitness = current_best_fitness
            
            if best_path is not None and best_path[-1] == self.end_pos:
                history.append(len(best_path))
            else:
                history.append(self.height * self.width)
            
            if iteration % 10 == 0:
                clear_output(wait=True)
                self.visualize_progress(iteration, best_path, history)
                time.sleep(0.1)
        
        clear_output(wait=True)
        self.visualize_progress(self.n_iterations, best_path, history)
        
        best_length = len(best_path) if (best_path is not None and best_path[-1] == self.end_pos) else float('inf')
        return best_path, best_length, history
    
    def visualize_progress(self, iteration, best_path=None, history=None):
       
        plt.figure(figsize=(15, 5))
        
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
        if best_path is not None and len(best_path) > 1:
            path_x = [p[1] for p in best_path]
            path_y = [p[0] for p in best_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2)
        plt.title(f"ABC - Rebetion  {iteration+1}/{self.n_iterations}")
        plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=200, marker='o')
        plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=200, marker='o')
        
        
        if best_path is not None and len(best_path) > 1:
            plt.subplot(1, 3, 2)
            plt.imshow(self.maze, cmap=ListedColormap(['white', 'black', 'green', 'red']))
            path_x = [p[1] for p in best_path]
            path_y = [p[0] for p in best_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2)
            plt.scatter(path_x, path_y, c='blue', s=20)
            title = f"Best path : {len(best_path)})" if best_path[-1] == self.end_pos else " path not  complete  "
            plt.title(title)
            plt.scatter(self.start_pos[1], self.start_pos[0], c='green', s=200, marker='o')
            plt.scatter(self.end_pos[1], self.end_pos[0], c='red', s=200, marker='o')
        
       
        if history:
            plt.subplot(1, 3, 3)
            plt.plot(history, 'b-')
            plt.xlabel('rebetion ')
            plt.ylabel('path length ')
            plt.title(' develop path length    ')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()