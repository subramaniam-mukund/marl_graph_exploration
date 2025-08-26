import textwrap 
import numpy as np 
from collections import defaultdict 

from env.environment import EnvironmentVariant, NetworkEnv 
from gym.spaces import Discrete 

from env.network import Network 
from util import one_hot_list 


class Data: 
    """ 
    A data packet. 
    """ 

    def __init__(self, id): 
        self.id = id 
        self.now = None 
        self.target = None 
        self.size = None 
        self.start = None 
        self.time = 0 
        self.edge = -1 
        self.neigh = None 
        self.ttl = None 
        self.shortest_path_weight = None 
        self.visited_nodes = None 

    def reset(self, start, target, size, ttl, shortest_path_weight): 
        self.now = start 
        self.target = target 
        self.size = size 
        self.start = start 
        self.time = 0 
        self.edge = -1 
        self.neigh = [self.id] 
        self.ttl = ttl 
        self.shortest_path_weight = shortest_path_weight 
        self.visited_nodes = set([start]) 


class Routing(NetworkEnv): 
    """ " 
    Routing environment based on the environment by 
    Jiang et al. https://github.com/PKU-RL/DGN/blob/master/Routing/routers.py 
    used for their DGN paper https://arxiv.org/abs/1810.09202.

    The task is to route packets from random source to random destination nodes in a 
    given network. Each agent controls a single packet. When a packet reaches its 
    destination, a new packet is instantly created at a random location with a new 
    random target. 
    """

    def __init__( 
        self, 
        network: Network, 
        n_data, 
        env_var: EnvironmentVariant, 
        k=3, 
        enable_congestion=True, 
        enable_action_mask=False, 
        ttl=0, 
        enable_link_failures=False, # <-- ADDED: Parameter to enable failures
        link_failure_rate=0.0, # <-- ADDED: Probability of a link failing per step
        num_actions=4,
    ):
        """ 
        Initialize the environment.

        :param network: a network 
        :param n_data: the number of data packets 
        :param env_var: the environment variant 
        :param k: include k neighbors in local observation (only for environment variant WITH_K_NEIGHBORS), defaults to 3 
        :param enable_congestion: whether to respect link capacities, defaults to True 
        :param enable_action_mask: whether to generate an action mask for agents that does not allow visiting nodes twice, defaults to False 
        :param ttl: time to live before packets are discarded, defaults to 0 
        :param enable_link_failures: whether to enable stochastic link failures, defaults to False 
        :param link_failure_rate: probability of a non-critical link failing at each step, defaults to 0.0 
        """
        super(Routing, self).__init__()

        self.network = network 
        assert isinstance(self.network, Network)

        self.n_data = n_data 
        self.data = [] 

        # make sure env_var is casted 
        self.env_var = EnvironmentVariant(env_var) 

        # optionally include k neighbors in local observation 
        self.k = k
        self.num_actions = num_actions

        # log information 
        self.agent_steps = np.zeros(self.n_data) 

        # whether to use random targets or target == 0 for all packets 
        self.num_random_targets = self.network.n_nodes 
        assert self.num_random_targets >= 0 

        # map from shortest path to actual agent steps 
        self.distance_map = defaultdict(list) 
        self.enable_ttl = ttl > 0 
        self.enable_congestion = enable_congestion 
        self.ttl = ttl 
        self.sum_packets_per_node = None 
        self.sum_packets_per_edge = None 

        # Link failure parameters
        self.enable_link_failures = enable_link_failures 
        self.link_failure_rate = link_failure_rate 

        self.enable_action_mask = enable_action_mask 
        self.action_mask = np.zeros((n_data, num_actions), dtype=bool) 

        self.action_space = Discrete(num_actions, start=0) # {0, 1, 2, 3} using gym action space 
        self.eval_info_enabled = False 

    def set_eval_info(self, val): 
        """ 
        Whether the step function should return additional info for evaluation. 

        :param val: the step function returns additional info if true 
        """ 
        self.eval_info_enabled = val 

    def reset_packet(self, packet: Data): 
        """ 
        Resets the given data packet using the settings of this environment. 

        :param packet: a data packet that will be reset *in-place* """ 
        # free resources on used edge 
        if packet.edge != -1: 
            self.network.edges[packet.edge].load -= packet.size 

        # reset packet in place 
        start = np.random.randint(self.network.n_nodes) 
        target = np.random.randint(self.num_random_targets) 

        # Ensure start and target nodes are still connected after potential failures
        while target not in self.network.shortest_paths[start]: 
            start = np.random.randint(self.network.n_nodes) 
            target = np.random.randint(self.num_random_targets) 

        packet.reset( 
            start=start, 
            target=target, 
            size=np.random.random(), 
            ttl=self.ttl, 
            shortest_path_weight=self.network.shortest_paths_weights[start][target], 
        ) 

        if self.enable_action_mask: 
            # all links are allowed 
            self.action_mask[packet.id] = 0 
            # idling is allowed if a packet spawns at the destination 
            self.action_mask[packet.id, 0] = packet.now != packet.target 

    def __str__(self) -> str: 
        return textwrap.dedent( 
            f"""\ 
            Routing environment with parameters 
            > Network: {self.network.n_nodes} nodes 
            > Number of packets: {self.n_data} 
            > Environment variant: {self.env_var.name} 
            > Number of considered neighbors (k): {self.k if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS else "disabled"} 
            > Congestion: {self.enable_congestion} 
            > Action mask: {self.enable_action_mask} 
            > TTL: {self.ttl if self.enable_ttl else "disabled"} 
            > Link Failures: {"Enabled with rate " + str(self.link_failure_rate) if self.enable_link_failures else "Disabled"}\ 
            """ 
        )

    def reset(self): 
        self.agent_steps = np.zeros(self.n_data) 
        self.network.reset() 
        for edge in self.network.edges: 
            # add new load attribute to edges 
            edge.load = 0

        if self.eval_info_enabled: 
            self.sum_packets_per_node = np.zeros(self.network.n_nodes) 
            self.sum_packets_per_edge = np.zeros(len(self.network.edges)) 

        # generate random data packets 
        self.data = [] 
        for i in range(self.n_data): 
            new_data = Data(i) 
            self.reset_packet(new_data) 
            self.data.append(new_data) 

        return self._get_observation(), self._get_data_adjacency() 

    def render(self): 
        # TODO: also render packets 
        self.network.render() 

    def get_nodes_adjacency(self): 
        return self.network.adj_matrix

    def get_node_observation(self): 
        """ 
        Get the node observation for each node in the network. 

        :return: node observations of shape (num_nodes, node_observation_size) 
        """ 
        obs = []
        for j in range(self.network.n_nodes): 
            ob = []

            # router info 
            # ob.append(j) 
            ob += one_hot_list(j, self.network.n_nodes) 
            num_packets = 0 
            total_load = 0 
            for i in range(self.n_data): 
                if self.data[i].now == j and self.data[i].edge == -1: 
                    num_packets += 1 
                    total_load += self.data[i].size 

            ob.append(num_packets) 
            ob.append(total_load) 

            # edge info 
            for k in self.network.nodes[j].edges: 
                other_node = self.network.edges[k].get_other_node(j) 
                ob += one_hot_list(other_node, self.network.n_nodes) 
                ob.append(self.network.edges[k].length) 
                ob.append(self.network.edges[k].load) 

            obs.append(ob) 
        return np.array(obs, dtype=np.float32) 

    def get_node_agent_matrix(self): 
        """ 
        Gets a matrix that indicates where agents are located, 
        matrix[n, a] = 1 iff agent a is on node n and 0 otherwise. 

        :return: the node agent matrix of shape (n_nodes, n_agents) 
        """ 
        node_agent = np.zeros((self.network.n_nodes, self.n_data), dtype=np.int8) 
        for a in range(self.n_data): 
            node_agent[self.data[a].now, a] = 1 

        return node_agent 

    def _get_observation(self):
        obs = []
        if self.env_var == EnvironmentVariant.GLOBAL:
            # for the global observation
            nodes_adjacency = self.get_nodes_adjacency().flatten()
            node_observation = self.get_node_observation().flatten()
            global_obs = np.concatenate((nodes_adjacency, node_observation))

        for i in range(self.n_data):
            ob = []
            # packet information
            ob += one_hot_list(self.data[i].now, self.network.n_nodes)
            ob += one_hot_list(self.data[i].target, self.network.n_nodes)

            # packets should know where they are coming from when traveling on an edge
            ob.append(int(self.data[i].edge != -1))
            if self.data[i].edge != -1:
                other_node = self.network.edges[self.data[i].edge].get_other_node(
                    self.data[i].now
                )
            else:
                other_node = -1
            ob += one_hot_list(other_node, self.network.n_nodes)

            ob.append(self.data[i].time)
            ob.append(self.data[i].size)
            ob.append(self.data[i].id)

            # --- MODIFIED: Padded Edge Information ---
            max_degree = self.num_actions  # Maximum possible neighbors for any node

            # Add info for actual, existing edges
            actual_edges = self.network.nodes[self.data[i].now].edges
            for edge_idx in actual_edges:
                other_node = self.network.edges[edge_idx].get_other_node(self.data[i].now)
                ob += one_hot_list(other_node, self.network.n_nodes)
                ob.append(self.network.edges[edge_idx].length)
                ob.append(self.network.edges[edge_idx].load)

            # Add padding for missing edges to ensure fixed size
            num_missing_edges = max_degree - len(actual_edges)
            for _ in range(num_missing_edges):
                # Use a zero vector for one-hot and -1 for scalar values
                ob += [0] * self.network.n_nodes
                ob.append(-1)  # Placeholder for length
                ob.append(-1)  # Placeholder for load

            # other data
            count = 0
            self.data[i].neigh = []
            self.data[i].neigh.append(i)
            for j in range(self.n_data):
                if j == i:
                    continue
                if (
                    self.data[j].now in self.network.nodes[self.data[i].now].neighbors
                ) | (self.data[j].now == self.data[i].now):
                    self.data[i].neigh.append(j)

                    # with neighbor information in observation (until k neighbors)
                    if (
                        self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS
                        and count < self.k
                    ):
                        count += 1
                        ob.append(self.data[j].now)
                        ob.append(self.data[j].target)
                        ob.append(self.data[j].edge)
                        ob.append(self.data[j].size)
                        ob.append(self.data[i].id)

            if self.env_var == EnvironmentVariant.WITH_K_NEIGHBORS:
                for j in range(self.k - count):
                    for _ in range(5):
                        ob.append(-1)  # invalid placeholder

            ob_numpy = np.array(ob)

            # add global information
            if self.env_var == EnvironmentVariant.GLOBAL:
                ob_numpy = np.concatenate((ob_numpy, global_obs))

            obs.append(ob_numpy)

        return np.array(obs, dtype=np.float32)

    def step(self, act): 
        reward = np.zeros(self.n_data, dtype=np.float32) 
        looped = np.zeros(self.n_data, dtype=np.float32) 
        done = np.zeros(self.n_data, dtype=bool) 
        drop_packet = np.zeros(self.n_data, dtype=bool) 
        success = np.zeros(self.n_data, dtype=bool) 
        blocked = 0 

        delays = [] 
        delays_arrived = [] 
        spr = [] 
        self.agent_steps += 1 

        # --- LINK FAILURE LOGIC ---
        failed_edge_idx = None 
        if self.enable_link_failures and np.random.random() < self.link_failure_rate: 
            failed_edge_idx = self.network.fail_random_edge() 

        if failed_edge_idx is not None: 
            # Check if any packets were on the failed link and drop them 
            for i in range(self.n_data): 
                if self.data[i].edge == failed_edge_idx: 
                    drop_packet[i] = True 
                    # Free up its load from the (now failed) edge 
                    self.network.edges[failed_edge_idx].load -= self.data[i].size 
                    self.data[i].edge = -1 # No longer on a valid edge
        # --- END LINK FAILURE LOGIC --- 

        # handle actions 
        for i in range(self.n_data): 
            # agent i controls data packet i 
            packet = self.data[i] 

            if self.eval_info_enabled: 
                if packet.edge == -1: 
                    self.sum_packets_per_node[packet.now] += 1 

            # select outgoing edge (act == 0 is idle) 
            if packet.edge == -1 and act[i] != 0: 
                # ADDED: Check if the chosen action is valid for the node's current number of edges
                if act[i] - 1 >= len(self.network.nodes[packet.now].edges): 
                    reward[i] -= 0.1 # Penalize for trying an invalid action
                    print("***** INVALID ACTION CHOSEN *****")
                    continue # Treat as an idle action and move to the next packet

                t = self.network.nodes[packet.now].edges[act[i] - 1] 

                if ( 
                    self.enable_congestion 
                    and self.network.edges[t].load + packet.size > 1 
                ): 
                    # not possible to take this edge => collision 
                    reward[i] -= 0.2 
                    blocked += 1 
                else: 
                    # take this edge 
                    packet.edge = t 
                    packet.time = self.network.edges[t].length 
                    # assign load to the selected edge 
                    self.network.edges[t].load += packet.size 

                    # already set the next position 
                    packet.now = self.network.edges[t].get_other_node(packet.now) 
                    if packet.now in packet.visited_nodes: 
                        looped[i] = 1 
                    else: 
                        packet.visited_nodes.add(packet.now) 

        if self.eval_info_enabled: 
            total_edge_load = 0 
            occupied_edges = 0 
            packets_on_edges = 0 
            total_packet_size = 0 
            packet_sizes = [] 

            for edge in self.network.edges: 
                total_edge_load += edge.load 
                if edge.load > 0: 
                    occupied_edges += 1 

            for i in range(self.n_data): 
                packet = self.data[i] 
                if packet.edge != -1: 
                    self.sum_packets_per_edge[packet.edge] += 1 

                total_packet_size += packet.size 
                packet_sizes.append(self.data[i].size) 
                if packet.edge != -1: 
                    packets_on_edges += 1 

            packet_distances = list( 
                map( 
                    lambda p: self.network.shortest_paths_weights[p.now].get(p.target, -1),
                    self.data, 
                ) 
            ) 

        # then simulate in-flight packets (=> effect of actions) 
        for i in range(self.n_data): 
            packet = self.data[i] 
            packet.ttl -= 1 

            if packet.edge != -1: 
                packet.time -= 1 
                # the packet arrived at the destination, reduce load from edge 
                if packet.time <= 0: 
                    self.network.edges[packet.edge].load -= packet.size 
                    packet.edge = -1 

            drop_packet[i] = drop_packet[i] or (self.enable_ttl and packet.ttl <= 0) 
            if self.enable_action_mask: 
                if packet.edge != -1: 
                    self.action_mask[i] = 0 
                else: 
                    self.action_mask[i, 0] = 1 
                    for edge_i, e in enumerate(self.network.nodes[packet.now].edges): 
                        self.action_mask[i, 1 + edge_i] = ( 
                            self.network.edges[e].get_other_node(packet.now) 
                            in packet.visited_nodes 
                        ) 

                    # packets that can't do anything are dropped 
                    if self.action_mask[i].sum() == self.num_actions: 
                        drop_packet[i] = True 

            # the packet has reached the target 
            has_reached_target = packet.edge == -1 and packet.now == packet.target 
            if has_reached_target or drop_packet[i]: 
                reward[i] += 10 if has_reached_target else -10 
                done[i] = True 
                success[i] = has_reached_target 

                # we need at least 1 step (idle) if we spawn at the target 
                opt_distance = max(packet.shortest_path_weight, 1) 

                # insert delays before resetting packets 
                if success[i]: 
                    delays_arrived.append(self.agent_steps[i]) 
                    spr.append(self.agent_steps[i] / opt_distance) 
                    if self.eval_info_enabled: 
                        self.distance_map[opt_distance].append(self.agent_steps[i]) 

                delays.append(self.agent_steps[i]) 

                self.agent_steps[i] = 0 
                self.reset_packet(packet) 

        obs = self._get_observation() 
        adj = self._get_data_adjacency() 
        info = { 
            "delays": delays, 
            "delays_arrived": delays_arrived, 
            # shortest path ratio in [1, inf) where 1 is optimal 
            "spr": spr, 
            "looped": looped.sum(), 
            "throughput": success.sum(), 
            "dropped": (done & ~success).sum(), 
            "blocked": blocked, 
        } 
        if self.eval_info_enabled: 
            info.update( 
                { 
                    "total_edge_load": total_edge_load, 
                    "occupied_edges": occupied_edges, 
                    "packets_on_edges": packets_on_edges, 
                    "total_packet_size": total_packet_size, 
                    "packet_sizes": packet_sizes, 
                    "packet_distances": packet_distances, 
                } 
            ) 
        return obs, adj, reward, done, info 

    def _get_data_adjacency(self): 
        """ 
        Get an adjacency matrix for data packets (agents) of shape (n_agents, n_agents) 
        where the second dimension contains the neighbors of the agents in the first 
        dimension, i.e. the matrix is of form (agent, neighbors). 

        :param data: current data list 
        :param n_data: number of data packets 
        :return: adjacency matrix 
        """ 
        # eye because self is also part of the neighborhood 
        adj = np.eye(self.n_data, self.n_data, dtype=np.int8) 
        for i in range(self.n_data): 
            for n in self.data[i].neigh: 
                if n != -1: 
                    # n is (currently) a neighbor of i 
                    adj[i, n] = 1 
        return adj 

    def get_final_info(self, info: dict): 
        agent_steps = self.agent_steps 
        for agent_step in agent_steps: 
            if agent_step != 0: 
                info["delays"].append(agent_step) 
        return info 

    def get_num_agents(self): 
        return self.n_data 

    def get_num_nodes(self): 
        return self.network.n_nodes