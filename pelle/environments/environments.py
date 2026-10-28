from __future__ import annotations
import networkx as nx
import matplotlib.pyplot as plt
from random import Random
import numpy as np


from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import gymnasium as gym
import matplotlib as mpl
from matplotlib.patches import Rectangle

def visualize_network(
    G,
    node_values=None,
    edge_values=None,
    *,
    index=None,
    show_grid=True,
    cmap='coolwarm',
    vmin=None,
    vmax=None,
    node_size=120,
    width=1.5,
    figsize=(8, 8),
    outpath=None,
    title="Maze Graph",
):
    """
    Visualize a maze graph with states as coordinates.

    Parameters
    ----------
    G : dict or networkx.Graph
        Maze graph, either as:
            {state: {action: next_state, ...}}
        or as a NetworkX graph whose nodes are state tuples (x, y).

    node_values : dict, list, tuple, np.ndarray, or None
        If dict: {state: value}
        If array-like: values[i] corresponds to index[state]
        If None: use zeros.

    edge_values : dict, np.ndarray, or None
        If dict: {(u, v): value}
        If array-like: edge matrix indexed by rows/cols of graph index.
        If None: no edge colouring.

    index : dict or None
        Mapping {state: i}. Required if node_values / edge_values are array-like.
    """

    # ------------------------------
    # Build NetworkX graph
    # ------------------------------
    if isinstance(G, dict):
        G_nx = nx.Graph()
        for state, neighbors in G.items():
            G_nx.add_node(state)
            for _, nxt in neighbors.items():
                G_nx.add_edge(state, nxt)
    else:
        G_nx = G

    # Need state coordinates
    states = []
    for n in G_nx.nodes():
        if isinstance(n, tuple) and len(n) == 2:
            states.append(n)

    if not states:
        raise ValueError("Graph nodes must be 2D state tuples like (x, y).")

    if index is None:
        index = {s: i for i, s in enumerate(sorted(states))}

    # Position is just the state coordinates
    pos = {s: (float(s[0]), float(s[1])) for s in states}

    # ------------------------------
    # Normalize node values
    # ------------------------------
    if node_values is None:
        node_values_dict = {s: 0.0 for s in states}
    elif isinstance(node_values, dict):
        node_values_dict = node_values
    else:
        node_values_dict = {s: node_values[index[s]] for s in states}

    # ------------------------------
    # Normalize edge values
    # ------------------------------
    if edge_values is None:
        edge_values_dict = {}
    elif isinstance(edge_values, dict):
        edge_values_dict = edge_values
    else:
        # assume edge_values is an array-like matrix shaped like (n, n)
        edge_values_dict = {}
        inv = {i: s for s, i in index.items()}
        for i in range(edge_values.shape[0]):
            for j in range(edge_values.shape[1]):
                if i == j:
                    continue
                val = edge_values[i, j]
                if val != 0:
                    edge_values_dict[(inv[i], inv[j])] = val

    # ------------------------------
    # Shared colormap range
    # ------------------------------
    vals = list(node_values_dict.values()) + [v for v in edge_values_dict.values()]
    if vals:
        vabs = max(abs(min(vals)),abs(max(vals)))
        vmin = -vabs if vmin is None else vmin
        vmax = vabs if vmax is None else vmax
    else:
        vmin, vmax = 0.0, 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # ------------------------------
    # Draw
    # ------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # background grid
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if show_grid:
        for x in range(int(np.floor(x_min)), int(np.ceil(x_max)) + 1):
            for y in range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1):
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        edgecolor="lightgray",
                        linewidth=0.5,
                    )
                )

    # draw edges
    edge_colors = []
    edge_list = []
    for u, v in G_nx.edges():
        edge_list.append((u, v))
        if (u, v) in edge_values_dict:
            val = edge_values_dict[(u, v)]
        elif (v, u) in edge_values_dict:
            val = edge_values_dict[(v, u)]
        else:
            val = np.nan
        if np.isnan(val):
            edge_colors.append((0.7, 0.7, 0.7, 0.8))
        else:
            edge_colors.append(cmap_obj(norm(float(val))))

    if edge_list:
        nx.draw_networkx_edges(
            G_nx,
            pos,
            edgelist=edge_list,
            edge_color=edge_colors,
            width=width,
            ax=ax,
            arrows=False,
        )

    # draw nodes
    node_colors = [cmap_obj(norm(float(node_values_dict.get(n, vmin)))) for n in G_nx.nodes()]
    nx.draw_networkx_nodes(
        G_nx,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors="black",
        linewidths=0.6,
        ax=ax,
    )

    ax.set_aspect("equal")
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1))
    ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 1))
    ax.set_title(title)
    ax.grid(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Curvature")


    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {outpath}")
    else:
        plt.show()

    plt.close(fig)

def minigrid_maze_to_graph(env):
    """
    Convert a MiniGrid maze env into a graph:
        graph[state] = {action: next_state, ...}
    where state is (x, y) and action is one of ("L", "R", "U", "D").
    """
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    grid = base_env.grid
    width, height = grid.width, grid.height

    graph = {}

    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)

            if cell is not None and getattr(cell, "type", None) == "wall":
                continue

            state = (x, y)
            graph[state] = {}

            for action, (dx, dy) in {
                "L": (0, -1),
                "R": (0, 1),
                "U": (-1, 0),
                "D": (1, 0),
            }.items():
                nx, ny = x + dx, y + dy

                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                next_cell = grid.get(nx, ny)
                if next_cell is not None and getattr(next_cell, "type", None) == "wall":
                    continue

                graph[state][action] = (nx, ny)

    return graph

def graph_to_adjacency_matrix(graph, directed=False):
    """
    Convert a graph dict to a NumPy adjacency matrix.

    Returns:
        A: np.ndarray of shape (n, n)
        index: dict mapping state -> row/column index
    """
    nodes = sorted(graph.keys())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    A = np.zeros((n, n), dtype=int)

    for src, edges in graph.items():
        i = index[src]
        for dst in edges.values():
            if dst not in index:
                continue
            j = index[dst]
            A[i, j] = 1
            if not directed:
                A[j, i] = 1

    return A, index

def graph_to_geomem_dict(graph):
    states = tuple(sorted(graph.keys()))
    actions = ("L", "R", "U", "D")

    transition = {}
    valid = set()

    for state in states:
        for action in actions:
            if action in graph.get(state, {}):
                nxt = graph[state][action]
                transition[(state, action)] = nxt
                valid.add((state, action))
            else:
                transition[(state, action)] = state

    return {
        "states": states,
        "actions": actions,
        "transition": transition,
        "valid": frozenset(valid)
    }

# functions for ease of execution
def make_env(EnvString, edge_length=10,render_mode=None):
    while True:
        env = gym.make(EnvString, size=edge_length, render_mode=render_mode)
        obs, info = env.reset()

        graph = minigrid_maze_to_graph(env)

        # build a NetworkX graph from the maze graph dict
        G = nx.Graph()
        for state, neighbors in graph.items():
            G.add_node(state)
            for nxt in neighbors.values():
                G.add_edge(state, nxt)

        # keep generating until the graph is connected
        if nx.number_connected_components(G) == 1:
            break
    return env



def get_measure_env(env,EnvString,visualise=False):
    graph = minigrid_maze_to_graph(env)

    A, index = graph_to_adjacency_matrix(graph, directed=False)

    geomem_dict = graph_to_geomem_dict(graph)
    env_adapt = DictEnvAdapter(geomem_dict)

    raw_commutativity = effective_commutativity(env_adapt)
    valid_commutativity = effective_commutativity(env_adapt, valid_only=True)

    eff_res = effective_resistance_adj(A)
    link_cur = link_curvature(env_adapt, eff_res)
    node_cur = node_curvature(env_adapt, eff_res)

    mean_link_cur = np.nanmean(link_cur)

    if visualise:
        G_vis = nx.DiGraph()
        for state, neighbors in graph.items():
            G_vis.add_node(state)
            for action, nxt in neighbors.items():
                G_vis.add_edge(state, nxt, action=action)

        index_state = {state: i for i, state in enumerate(sorted(graph.keys()))}
        visualize_network(G_vis, node_cur, link_cur, index=index_state,title=EnvString.split('-')[2])

    return raw_commutativity, valid_commutativity, mean_link_cur


# Measures
# Measures Commutativty
def action_strings(actions, length, rng, n):
    """Return n random tuples of actions of given length."""
    return [tuple(rng.choice(actions) for _ in range(length)) for _ in range(n)]

def swapped_pairs(seq):
    """Return list of (original, swapped) tuples by swapping two distinct indices."""
    pairs = []
    L = len(seq)
    for i in range(L):
        for j in range(i + 1, L):
            if seq[i] == seq[j]:
                # swapping identical actions yields same sequence; skip as repo does
                continue
            swapped = list(seq)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            pairs.append((tuple(seq), tuple(swapped)))
    return pairs

def effective_commutativity(env, *, length=4, n_sequences=200, seed=1, valid_only=False, equivalent=None):
    """
    Estimate fraction of sampled action-order swaps that preserve final state.

    - env: Environment instance (must implement rollout(start, actions) -> (state, valid_flag)
      and provide env.states and env.actions).
    - length: length of sampled action strings
    - n_sequences: number of sampled sequences
    - seed: RNG seed
    - valid_only: if True, count only swaps where both orders avoid invalid transitions
    - equivalent: callable(state_a, state_b) -> bool (defaults to equality)

    Returns a float in [0,1] or float('nan') if no pairs were considered.
    """
    rng = Random(seed)
    if equivalent is None:
        equivalent = lambda a, b: a == b

    total = 0
    preserved = 0

    starts = list(env.states)
    # mirror repo behavior: if valid_only and env.max_depth present, filter starts so length won't exceed max depth
    try:
        if valid_only and getattr(env, "max_depth", None) is not None:
            starts = [state for state in starts if len(state) + length <= env.max_depth]
            if not starts:
                return float("nan")
    except Exception:
        # if len(state) isn't meaningful for this env, ignore filtering
        pass

    for seq in action_strings(env.actions, length, rng, n_sequences):
        start = rng.choice(starts)
        for original, swapped in swapped_pairs(seq):
            a, valid_a = env.rollout(start, original)
            b, valid_b = env.rollout(start, swapped)
            if valid_only and not (valid_a and valid_b):
                continue
            total += 1
            preserved += 1 if equivalent(a, b) else 0

    return preserved / total if total else float("nan")

    # Helper class

class DictEnvAdapter:
    def __init__(self, env_dict):
        self.states = tuple(env_dict["states"])
        self.actions = tuple(env_dict["actions"])
        self.transition = env_dict["transition"]
        self.valid = set(env_dict.get("valid", set()))

    def is_valid(self, state, action):
        return (state, action) in self.valid

    def valid_actions(self, state):
        return tuple(a for a in self.actions if self.is_valid(state, a))

    def step(self, state, action):
        key = (state, action)
        if key not in self.valid:
            return state
        if key not in self.transition:
            return state
        return self.transition[key]

    def rollout(self, start, actions):
        state = start
        valid = True
        for action in actions:
            if (state, action) not in self.valid:
                valid = False
                break
            state = self.transition[(state, action)]
        return state, valid

def pairwise_commutator_matrix(env, *, valid_only=False):
    """
    Compute Pr[AB(s) == BA(s)] for every (A,B) pair, estimated exactly over all env.states.
    Returns a dict keyed by (action1, action2) -> fraction (float).
    """
    matrix = {}
    for a in env.actions:
        for b in env.actions:
            total = 0
            preserved = 0
            for s in env.states:
                ab, valid_ab = env.rollout(s, (a, b))
                ba, valid_ba = env.rollout(s, (b, a))
                if valid_only and not (valid_ab and valid_ba):
                    continue
                total += 1
                preserved += 1 if ab == ba else 0
            matrix[(a, b)] = preserved / total if total else float("nan")
    return matrix

# Measures effective resistance
def effective_resistance_adj(A):
    D = np.diag(A.sum(axis=1)) # diagonal matrix
    Q = D - A # Laplacian matrix

    Q_pseudo_inv = np.linalg.pinv(Q)
    q_diag = np.diag(Q_pseudo_inv)
    Omega = q_diag[:, None] + q_diag[None,:] - 2 * Q_pseudo_inv

    np.fill_diagonal(Omega,0.0)
    return Omega

def get_goal_state(env):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    # Try common names
    for attr in ["goal_pos", "goal_position", "goal_coords", "goal", "target_pos", "target"]:
        if hasattr(base_env, attr):
            value = getattr(base_env, attr)
            if hasattr(value, "x") and hasattr(value, "y"):
                return (value.x, value.y)
            if isinstance(value, (tuple, list)):
                return tuple(value)
            if isinstance(value, dict):
                if "x" in value and "y" in value:
                    return (value["x"], value["y"])

    # Fallback: scan grid for a goal object
    grid = base_env.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None:
                # Works for many Minigrid-like cells
                if getattr(cell, "type", None) == "goal" or "goal" in type(cell).__name__.lower():
                    return (x, y)

    return None

def node_curvature(env,Omega): # want to input relative resitance, in unweighted graph = effective resistance
    starts = list(env.states)
    state_lookup = {state:index for index,state in enumerate(starts)}
    p = np.zeros(Omega.shape[0])
    for i in range(Omega.shape[0]):
        s = env.states[i]
        states = [env.step(s,a) for a in env.actions]
        states_idx = [state_lookup[state] for state in states]
        for j in set(states_idx):
            p[i] += 0.5*Omega[i,j]
    p = 1 - p
    return p

def link_curvature(env,Omega):

    p = node_curvature(env,Omega)
    k = np.empty(Omega.shape)
    k[:] = np.nan
    starts = list(env.states)
    state_lookup = {state:index for index,state in enumerate(starts)}

    for i in range(Omega.shape[0]):
            s = env.states[i]
            states = [env.step(s,a) for a in env.actions]
            states_idx = [state_lookup[state] for state in states]
            for j in set(states_idx):
                  if i ==j: continue
                  k[i,j] = 2*(p[i] + p[j])/Omega[i,j]
    return k

# functions for ease of execution
def setup_env(EnvString, edge_length, visualise=False):
    while True:
        env = gym.make(EnvString, size=edge_length)
        obs, info = env.reset()

        graph = minigrid_maze_to_graph(env)

        # build a NetworkX graph from the maze graph dict
        G = nx.Graph()
        for state, neighbors in graph.items():
            G.add_node(state)
            for nxt in neighbors.values():
                G.add_edge(state, nxt)

        # keep generating until the graph is connected
        if nx.number_connected_components(G) == 1:
            break

    A, index = graph_to_adjacency_matrix(graph, directed=False)

    geomem_dict = graph_to_geomem_dict(graph)
    env_adapt = DictEnvAdapter(geomem_dict)

    raw_commutativity = effective_commutativity(env_adapt)
    valid_commutativity = effective_commutativity(env_adapt, valid_only=True)

    eff_res = effective_resistance_adj(A)
    link_cur = link_curvature(env_adapt, eff_res)
    node_cur = node_curvature(env_adapt, eff_res)

    mean_link_cur = np.nanmean(link_cur)

    if visualise:
        G_vis = nx.DiGraph()
        for state, neighbors in graph.items():
            G_vis.add_node(state)
            for action, nxt in neighbors.items():
                G_vis.add_edge(state, nxt, action=action)

        index_state = {state: i for i, state in enumerate(sorted(graph.keys()))}
        visualize_network(G_vis, node_cur, link_cur, index=index_state,title=EnvString.split('-')[2])

    return raw_commutativity, valid_commutativity, mean_link_cur