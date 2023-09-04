import networkx as nx
import matplotlib.pyplot as plt
from networkx.generators.random_graphs import erdos_renyi_graph
import numpy as np
class environment:

    def __init__(self, num_objects:int, prob_edge_relation:float, seed:int = 1):
        # Generate relationship graph between objects
        self.rel_graph = erdos_renyi_graph(num_objects, prob_edge_relation, seed, directed = True)
        self.rng = np.random.default_rng(seed+1)
        # add self connections with the same probability as other edges since the erdoes_reny_graph 
        # does not allow for self-directed edges
        for i in range(0, num_objects):
            rand_num = self.rng.random()
            if rand_num < prob_edge_relation:
                self.rel_graph.add_edge(i, i)
            # add 
            
            self.rel_graph.add_node(i, attribute_1_affinity = self.rng.random() - 0.5,\
                attribute_2_affinity = self.rng.random() -0.5, \
                attribute_3_affinity = self.rng.random() -0.5)
        for e in self.rel_graph.edges:
            rand_num = self.rng.random()
            # generate whether the two objects positively or negatively affect one another
            if rand_num < 0.5:
                pos_relationship = True
            else:
                pos_relationship = False
            self.rel_graph.add_edge(e[0], e[1], pos_relationship = pos_relationship)
        # save positions of nodes for consistency in graph generation
        self.pos = nx.spring_layout(self.rel_graph, seed=seed+2)
    def evaluate_node(self, node:int, neighbor_list:list, node_attr_values:np.array):
        attr_1_value = self.rel_graph.nodes[node]['attribute_1_affinity']*node_attr_values[0]
        attr_2_value = self.rel_graph.nodes[node]['attribute_2_affinity']*node_attr_values[1]
        attr_3_value = self.rel_graph.nodes[node]['attribute_3_affinity']*node_attr_values[2]
        full_value = attr_1_value + attr_2_value + attr_3_value
        for neighbor in neighbor_list:
            if (node, neighbor) in self.rel_graph.edges:
                if self.rel_graph.edges[(node, neighbor)]['pos_relationship']:
                    full_value *= 2
                else:
                    full_value /= 2
        return full_value
            
    def draw(self):
        pos_edges = [e for e in self.rel_graph.edges if self.rel_graph.edges[e]['pos_relationship']]
        
        neg_edges = [e for e in self.rel_graph.edges if not self.rel_graph.edges[e]['pos_relationship']]
        nx.draw_networkx_nodes(self.rel_graph, self.pos)
        nx.draw_networkx_labels(self.rel_graph, self.pos)
        nx.draw_networkx_edges(self.rel_graph, self.pos, edgelist=pos_edges,\
                                edge_color = 'green', width = 2, alpha = 0.5)
        nx.draw_networkx_edges(self.rel_graph, self.pos, edgelist=neg_edges,\
                                edge_color = 'red', style = 'dashed')
        plt.show()
class terrain:
    num_attributes = 3
    def __init__(self, size:int, rand_walk_eps:float = 0.1, seed = 1):
        self.grid = np.zeros((size, size, self.num_attributes))
        self.rng = np.random.default_rng(seed)
        for attr in range(0, self.num_attributes):
            for x in range(0, size):
                for y in range(0, size):
                    rand_num = self.rng.normal()
                    if x == 0 and y ==0:
                        self.grid[x,y,attr] = rand_num*rand_walk_eps
                    elif x==0 and y>0:
                        self.grid[x,y,attr] = self.grid[x, y -1, attr] + rand_num*rand_walk_eps
                    elif x > 0 and y ==0:
                        self.grid[x,y,attr] = self.grid[x -1, y, attr] + rand_num*rand_walk_eps
                    else:
                        self.grid[x,y,attr] = (self.grid[x, y -1, attr] + self.grid[x-1, y, attr])/2 + \
                            rand_num*rand_walk_eps 
    def draw(self):
        fig, ax = plt.subplots(1, self.num_attributes)
        for i in range(0, self.num_attributes):
            ax[i].imshow(self.grid[:,:,i])
            ax[i].set_title(f'Attribute {i + 1}')
        plt.show()

def evaluate(terr:terrain, env:environment, object_placements):
    # object placements should be a  list of int, (int, int)
    grid = np.zeros(terr.grid.shape[0:2], int) - 1
    score_table = np.zeros(terr.grid.shape[0:2])
    score = 0
    for placement in object_placements:
        node = placement[0]
        loc_x, loc_y = placement[1]
        if grid[loc_x, loc_y] != -1:
            raise KeyError("One or more objects have been given the same coordinates. " + \
                           "Objects should be placed in different locations")
        grid[loc_x, loc_y] = node
    for x in range(0, grid.shape[0]):
        for y in range(0, grid.shape[1]):
            if grid[x,y] == -1:
                continue
            neighbor_list = [grid[x,y]]
            if x > 0:
                neighbor_node = grid[x - 1,y]
                if neighbor_node != -1:
                    neighbor_list += [neighbor_node]
            if y > 0:
                neighbor_node = grid[x,y - 1]
                if neighbor_node != -1:
                    neighbor_list += [neighbor_node]
            if x < grid.shape[0] - 1:
                neighbor_node = grid[x + 1,y]
                if neighbor_node != -1:
                    neighbor_list += [neighbor_node]
            if y < grid.shape[1] - 1:
                neighbor_node = grid[x,y + 1]
                if neighbor_node != -1:
                    neighbor_list += [neighbor_node]
            attr_values = terr.grid[x,y,:]
            score_table[x,y] = env.evaluate_node(grid[x,y], neighbor_list, attr_values)
    score = np.sum(score_table)
    return score, score_table
def gen_greedy_answer(object_list:list, env:environment, terr:terrain):
    object_list = object_list.copy()
    full_placements =[]
    taken_coords = []
    if len(object_list) > np.prod(terr.grid.shape[0:2]):
        raise ValueError("# of objects is greater than slots on terrain")
    while len(object_list) > 0:
        temp_placements = full_placements.copy()
        max_score = -np.inf
        for i in range(0, len(object_list)):
            for x in range(0, terr.grid.shape[0]):
                for y in range(0, terr.grid.shape[1]):
                    if (x,y) not in taken_coords:
                        addition = (object_list[i], (x,y))
                        temp_placements += [addition]
                        score, _ = evaluate(terr, env, temp_placements)
                        if score > max_score:
                            max_score = score
                            best_addition = addition
                            index_to_pop = i
                        temp_placements.pop(len(temp_placements)-1)
        full_placements += [best_addition]
        object_list.pop(index_to_pop)
        taken_coords += [best_addition[1]]
        # print(taken_coords)
    return full_placements
def gen_random_answer(object_list:list, terr, seed = 1):
    object_list = object_list.copy() # don't mutate original
    rng = np.random.default_rng(seed)
    rng.shuffle(object_list)
    if len(object_list) > np.prod(terr.grid.shape[0:2]):
        raise ValueError("# of objects is greater than slots on terrain")
    all_coords = [(x,y) for x in range(0, terr.grid.shape[0]) for y in range(0,terr.grid.shape[1])]
    rng.shuffle(all_coords)
    full_placements = [(object_list[i], all_coords[i]) for i in range(0, len(object_list))]
    return full_placements

# class GA:
#     min_weight = 0.1
#     def __init__(self, parents, seed = 1):
#         # parents should be a list of tuples of the form (placements, overall_score, score_by_square_on_grid)
#         self.parent_placements = [x[0] for x in parents]
#         self.parent_scores = np.array([x[1] for x in parents])
#         self.parent_grids = [x[2] for x in parents]
#         self.num_parents = len(parents)
#         self.rng = np.random.default_rng(seed)
#         self.norm_scores()
#     def norm_scores(self):
#         if np.min(self.parent_scores) == np.max(self.parent_scores):
#             self.parent_scores_norm = np.ones(len(self.parent_scores))
#         else:
#             self.parent_scores_norm = (self.parent_scores - np.min(self.parent_scores))/(np.max(self.parent_scores) - np.min(self.parent_scores))*(1-self.min_weight) + self.min_weight
#         self.parent_scores_norm = self.parent_scores_norm/np.sum(self.parent_scores_norm)
#         return None

#     def cross_pollenate(self):
#         parent_indices = self.rng.choice(range(0, self.num_parents), replace = False, p = self.parent_scores_norm)
#         parent_1_placements = self.parent_placements[parent_indices[0]]
#         parent_2_placements = self.parent_placements[parent_indices[1]]
#         parent_1_obj_list = [x[0] for x in parent_1_placements]
#         def remove_():

#         raise RuntimeError("child object list is different from parents'")


env = environment(10, 0.2)
terr = terrain(5, seed = 1)
rng = np.random.default_rng(seed = 1)
object_list  = [int(np.floor(rng.random()*10)) for _ in range(0, 20)]
# Greedy algorithm that starts with an empty grid 
greedy_placements = gen_greedy_answer(object_list, env, terr)
greedy_score, greedy_score_table = evaluate(terr, env, greedy_placements)
# generate random assignments for comparison
rand_placements = gen_random_answer(object_list, terr)
rand_score = evaluate(terr, env, rand_placements)
multiple_rand_placements = [gen_random_answer(object_list, terr, seed = i) for i in range(0, 10000)]
rand_scores = [evaluate(terr, env, rand_place) for rand_place in multiple_rand_placements]
rand_score_arr = np.array([x for x,_ in rand_scores])
# get best random assignments for genetic algorithm 
score_cutoff = np.quantile(rand_score_arr, (len(rand_scores) - 99)/len(rand_scores))
best_rand_placements = [(place, score, score_table) for place, (score, score_table) in zip(multiple_rand_placements, rand_scores) if score > score_cutoff]
places_and_scores = [(greedy_placements, greedy_score, greedy_score_table)] + best_rand_placements
# run genetic algorithm
print(f"Greedy Algorithm score is {round(greedy_score,2)} compared to best random score of {round(np.max(rand_score_arr),2)}")
