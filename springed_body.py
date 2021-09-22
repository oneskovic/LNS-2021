import numpy as np


class SpringedBody:
    def __init__(self, rng):
        self.score = 0.0            # For GA

        self.nodes = np.array([], dtype=int)
        self.node_friction = np.array([])
        self.node_mass = np.array([])

        self.contract_time = rng.uniform()
        self.springs_damping = np.array([])

        self.connected_nodes = dict()
        self.node_positions = list()
        self.node_id = 0
        self.spring_id = 0

    def add_node(self, rng):
        side_len = 30
        x = rng.uniform(0, side_len)
        y = rng.uniform(0, side_len)

        other_node = -1
        if len(self.nodes) > 0:
            other_node = rng.choice(self.nodes)
        self.nodes = np.append(self.nodes, self.node_id)

        self.node_positions.append((x, y))
        self.node_friction = np.append(self.node_friction, rng.uniform())
        self.node_mass = np.append(self.node_mass, rng.uniform(0, 10))

        if other_node != -1:
            self.connected_nodes[(other_node, self.node_id)] = self.spring_id
            self.springs_damping = np.append(self.springs_damping, rng.uniform(0,10))
            self.spring_id += 1

        self.node_id += 1

    def make_random_connection(self, rng):
        nodes = np.sort(rng.choice(self.nodes, 2, replace=False))
        node1 = nodes[0]
        node2 = nodes[1]
        if (node1, node2) not in self.connected_nodes:
            self.connected_nodes[(node1, node2)] = self.spring_id
            self.springs_damping = np.append(self.springs_damping, rng.uniform(0,10))
            self.spring_id += 1

    def mutate_friction(self, rng):
        node = rng.choice(self.nodes)
        self.node_friction[node] = rng.uniform()

    def mutate_mass(self, rng):
        node = rng.choice(self.nodes)
        self.node_mass[node] = rng.uniform(0, 10)

    def mutate_spring_damping(self, rng):
        spring = rng.integers(0,len(self.springs_damping))
        self.springs_damping[spring] = rng.uniform(0, 10)
