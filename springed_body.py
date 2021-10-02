import numpy as np


class Organism:
    def __init__(self):
        self.score = 0.0            # For GA

        self.nodes = np.array([], dtype=int)
        self.node_friction = np.array([])
        self.node_mass = np.array([])

        self.contract_time = np.random.ranf()
        self.springs_damping = np.array([])

        self.connected_nodes = dict()
        self.node_positions = list()
        self.node_id = 0
        self.spring_id = 0
        self.rng = np.random.default_rng(np.random.randint(0,1e7))

    def add_node(self):
        side_len = 30
        x = np.random.uniform(0, side_len)
        y = np.random.uniform(0, side_len)

        other_node = -1
        if len(self.nodes) > 0:
            other_node = np.random.choice(self.nodes)
        self.nodes = np.append(self.nodes, self.node_id)

        self.node_positions.append((x, y))
        self.node_friction = np.append(self.node_friction, np.random.ranf())
        self.node_mass = np.append(self.node_mass, np.random.uniform(0, 10))

        if other_node != -1:
            self.connected_nodes[(other_node, self.node_id)] = self.spring_id
            self.springs_damping = np.append(self.springs_damping, np.random.uniform(0,10))
            self.spring_id += 1

        self.node_id += 1

    def make_random_connection(self):
        nodes = np.sort(self.rng.choice(self.nodes, 2, replace=False))
        node1 = nodes[0]
        node2 = nodes[1]
        if (node1, node2) not in self.connected_nodes:
            self.connected_nodes[(node1, node2)] = self.spring_id
            self.springs_damping = np.append(self.springs_damping, np.random.uniform(0,10))
            self.spring_id += 1

    def mutate_friction(self):
        node = np.random.choice(self.nodes)
        self.node_friction[node] = np.random.ranf()

    def mutate_mass(self):
        node = np.random.choice(self.nodes)
        self.node_mass[node] = np.random.uniform(0, 10)

    def mutate_spring_damping(self):
        spring = np.random.randint(0,len(self.springs_damping))
        self.springs_damping[spring] = np.random.uniform(0, 10)




