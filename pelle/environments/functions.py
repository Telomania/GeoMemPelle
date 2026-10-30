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

DIRECTION_VECTORS = {
    0: (1, 0),    # right
    1: (0, 1),    # down
    2: (-1, 0),   # left
    3: (0, -1),   # up
}

ORIENTATION_OFFSETS = {
    0: (0.44 * np.cos(np.pi/6), 0.44 * np.sin(np.pi/6)),   # right
    1: (0.44 * np.sin(np.pi/6), -0.44 * np.cos(np.pi/6)),   # down
    2: (-0.44 * np.cos(np.pi/6), -0.44 * np.sin(np.pi/6)),  # left
    3: (-0.44 * np.sin(np.pi/6), 0.44 * np.cos(np.pi/6)),  # up
}

ACTION_ORDER = {"L": 0, "R": 1, "F": 2, "B": 3}


def _tile_state(state):
    if isinstance(state, tuple) and len(state) >= 2:
        return (state[0], state[1])
    return tuple(state)


def _project_orientation_graph(graph):
    """Collapse the orientation-expanded graph to a tile-only undirected graph."""
    tile_graph = {}
    for state, neighbors in graph.items():
        tile = _tile_state(state)
        tile_graph.setdefault(tile, set())
        for nxt in neighbors.values():
            nxt_tile = _tile_state(nxt)
            if nxt_tile == tile:
                continue
            tile_graph[tile].add(nxt_tile)
            tile_graph.setdefault(nxt_tile, set()).add(tile)
    return {tile: {nbr: nbr for nbr in sorted(neighbors)} for tile, neighbors in tile_graph.items()}


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
    node_size=30,
    width=1.5,
    figsize=(8, 8),
    outpath=None,
    title="Maze Graph",
):
    """
    Visualize a maze graph whose states are (x, y, heading). All states sharing the same
    tile coordinate are positioned as a small cluster around that tile.
    """
    if isinstance(G, dict):
        G_nx = nx.Graph()
        for state, neighbors in G.items():
            G_nx.add_node(state)
            for _, nxt in neighbors.items():
                G_nx.add_edge(state, nxt)
    else:
        G_nx = G

    states = list(G_nx.nodes())
    if not states:
        raise ValueError("Graph is empty.")

    if all(isinstance(s, tuple) and len(s) == 2 for s in states):
        pos = {s: (float(s[0]), float(s[1])) for s in states}
    elif all(isinstance(s, tuple) and len(s) == 3 for s in states):
        pos = {}
        for s in states:
            x, y, heading = s
            dx, dy = ORIENTATION_OFFSETS.get(heading % 4, (0.0, 0.0))
            pos[s] = (float(x) + dx, float(y) + dy)
    else:
        raise ValueError("Graph nodes must be 2D or 3D tuples like (x, y) or (x, y, heading).")

    if index is None:
        index = {s: i for i, s in enumerate(sorted(states))}

    if node_values is None:
        node_values_dict = {s: 0.0 for s in states}
    elif isinstance(node_values, dict):
        node_values_dict = node_values
    else:
        node_values_dict = {s: node_values[index[s]] for s in states}

    if edge_values is None:
        edge_values_dict = {}
    elif isinstance(edge_values, dict):
        edge_values_dict = edge_values
    else:
        edge_values_dict = {}
        inv = {i: s for s, i in index.items()}
        for i in range(edge_values.shape[0]):
            for j in range(edge_values.shape[1]):
                if i == j:
                    continue
                val = edge_values[i, j]
                if val != 0:
                    edge_values_dict[(inv[i], inv[j])] = val

    vals = list(node_values_dict.values()) + [v for v in edge_values_dict.values()]
    if vals:
        vabs = max(abs(min(vals)), abs(max(vals)))
        vmin = -vabs if vmin is None else vmin
        vmax = vabs if vmax is None else vmax
    else:
        vmin, vmax = 0.0, 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_title(f'{title}: {np.nanmean(edge_values):.3f} curvature')
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


def minigrid_maze_to_graph(env, *, include_orientation=True, undirected=True):
    """
    Convert a MiniGrid maze env into a graph of coordinate+heading states.

    The state is (x, y, heading), where heading indicates the facing direction.
    We include four actions:
        - "L": turn left
        - "R": turn right
        - "F": move forward
        - "B": move backward

    This makes the graph undirected in the sense that every state can be reached again,
    but still preserves the distinction between facing-right / facing-left / facing-up / facing-down.
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

            if include_orientation:
                for heading in range(4):
                    state = (x, y, heading)
                    graph[state] = {}

                    left_heading = (heading + 3) % 4
                    right_heading = (heading + 1) % 4
                    graph[state]["L"] = (x, y, left_heading)
                    graph[state]["R"] = (x, y, right_heading)

                    dx, dy = DIRECTION_VECTORS[heading]

                    fx, fy = x + dx, y + dy
                    if 0 <= fx < width and 0 <= fy < height:
                        next_cell = grid.get(fx, fy)
                        if next_cell is None or getattr(next_cell, "type", None) != "wall":
                            graph[state]["F"] = (fx, fy, heading)

                    bx, by = x - dx, y - dy
                    if 0 <= bx < width and 0 <= by < height:
                        back_cell = grid.get(bx, by)
                        if back_cell is None or getattr(back_cell, "type", None) != "wall":
                            graph[state]["B"] = (bx, by, heading)
            else:
                state = (x, y)
                graph[state] = {}
                for dx, dy in DIRECTION_VECTORS.values():
                    nx_coord, ny_coord = x + dx, y + dy
                    if not (0 <= nx_coord < width and 0 <= ny_coord < height):
                        continue
                    next_cell = grid.get(nx_coord, ny_coord)
                    if next_cell is not None and getattr(next_cell, "type", None) == "wall":
                        continue
                    graph[state][(nx_coord, ny_coord)] = (nx_coord, ny_coord)

    if undirected:
        undirected_graph = {state: {} for state in graph}
        for state, neighbors in graph.items():
            for nxt in neighbors.values():
                undirected_graph.setdefault(state, {})[nxt] = nxt
                undirected_graph.setdefault(nxt, {})[state] = state
        graph = undirected_graph

    return graph


def graph_to_adjacency_matrix(graph, directed=False):
    """Convert a graph dict to a NumPy adjacency matrix."""
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
    actions = tuple(sorted({action for edges in graph.values() for action in edges.keys()}, key=lambda a: ACTION_ORDER.get(a, 99))) or ("L", "R", "F", "B")

    transition = {}
    valid = set()

    for state in states:
        for action in actions:
            nxt = graph.get(state, {}).get(action, state)
            transition[(state, action)] = nxt
            if action in graph.get(state, {}):
                valid.add((state, action))

    return {
        "states": states,
        "actions": actions,
        "transition": transition,
        "valid": frozenset(valid),
    }


# Measures Commutativity

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
                continue
            swapped = list(seq)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            pairs.append((tuple(seq), tuple(swapped)))
    return pairs


def effective_commutativity(env, *, length=4, n_sequences=200, seed=1, valid_only=False, equivalent=None):
    """
    Estimate fraction of sampled action-order swaps that preserve final state.
    """
    rng = Random(seed)
    if equivalent is None:
        equivalent = lambda a, b: a == b

    total = 0
    preserved = 0

    starts = list(env.states)
    try:
        if valid_only and getattr(env, "max_depth", None) is not None:
            starts = [state for state in starts if len(state) + length <= env.max_depth]
            if not starts:
                return float("nan")
    except Exception:
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
    """Compute Pr[AB(s) == BA(s)] for every (A,B) pair."""
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


def effective_resistance_adj(A):
    D = np.diag(A.sum(axis=1))
    Q = D - A
    Q_pseudo_inv = np.linalg.pinv(Q)
    q_diag = np.diag(Q_pseudo_inv)
    Omega = q_diag[:, None] + q_diag[None, :] - 2 * Q_pseudo_inv
    np.fill_diagonal(Omega, 0.0)
    return Omega


def get_goal_state(env):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

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

    grid = base_env.grid
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and (getattr(cell, "type", None) == "goal" or "goal" in type(cell).__name__.lower()):
                return (x, y)

    return None


def node_curvature(env, Omega):
    starts = list(env.states)
    state_lookup = {state: index for index, state in enumerate(starts)}
    p = np.zeros(Omega.shape[0])
    for i in range(Omega.shape[0]):
        s = env.states[i]
        states = [env.step(s, a) for a in env.actions]
        states_idx = [state_lookup[state] for state in states]
        for j in set(states_idx):
            p[i] += 0.5 * Omega[i, j]
    p = 1 - p
    return p


def link_curvature(env, Omega):
    p = node_curvature(env, Omega)
    k = np.empty(Omega.shape)
    k[:] = np.nan
    starts = list(env.states)
    state_lookup = {state: index for index, state in enumerate(starts)}

    for i in range(Omega.shape[0]):
        s = env.states[i]
        states = [env.step(s, a) for a in env.actions]
        states_idx = [state_lookup[state] for state in states]
        for j in set(states_idx):
            if i == j:
                continue
            k[i, j] = 2 * (p[i] + p[j]) / Omega[i, j]
    return k


def setup_env(EnvString, edge_length=None, visualise=False):
    counter = 0
    while True:
        env = gym.make(EnvString)#, size=edge_length)
        _, _ = env.reset()

        graph = minigrid_maze_to_graph(env, undirected=True)
        G = nx.Graph()
        for state, neighbors in graph.items():
            G.add_node(state)
            for nxt in neighbors.values():
                G.add_edge(state, nxt)

        if nx.number_connected_components(G) == 1:
            break

        if counter == 10:
            print(f"{EnvString.split('-')[2]} cannot be built with a single connected component")
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
        G_vis = nx.Graph()
        for state, neighbors in graph.items():
            G_vis.add_node(state)
            for _, nxt in neighbors.items():
                G_vis.add_edge(state, nxt)

        index_state = {state: i for i, state in enumerate(sorted(graph.keys()))}
        visualize_network(G_vis, node_cur, link_cur, index=index_state, title=EnvString.split('-')[1])

    return raw_commutativity, valid_commutativity, mean_link_cur


def prep_env(EnvString, edge_length):
    counter = 0
    while True:
        env = gym.make(EnvString, size=edge_length)
        _, _ = env.reset()
        graph = minigrid_maze_to_graph(env, undirected=True)
        G = nx.Graph()
        for state, neighbors in graph.items():
            G.add_node(state)
            for nxt in neighbors.values():
                G.add_edge(state, nxt)

        if nx.number_connected_components(G) == 1:
            break
        if counter == 10:
            print(f"{EnvString.split('-')} cannot be built with a single connected component")
            break
    return env


def measures_from_env(env, EnvString, visualise=False):
    graph = minigrid_maze_to_graph(env, undirected=True)
    geomem_dict = graph_to_geomem_dict(graph)
    env_adapt = DictEnvAdapter(geomem_dict)

    raw_commutativity = effective_commutativity(env_adapt)
    valid_commutativity = effective_commutativity(env_adapt, valid_only=True)

    A, index = graph_to_adjacency_matrix(graph, directed=False)
    eff_res = effective_resistance_adj(A)
    link_cur = link_curvature(env_adapt, eff_res)
    node_cur = node_curvature(env_adapt, eff_res)

    mean_link_cur = np.nanmean(link_cur)

    if visualise:
        G_vis = nx.Graph()
        for state, neighbors in graph.items():
            G_vis.add_node(state)
            for _, nxt in neighbors.items():
                G_vis.add_edge(state, nxt)

        index_state = {state: i for i, state in enumerate(sorted(graph.keys()))}
        visualize_network(G_vis, node_cur, link_cur, index=index_state, title=EnvString.split('-')[1])

    return raw_commutativity, valid_commutativity, mean_link_cur


def make_env(EnvString, edge_length=10, render_mode=None):
    while True:
        env = gym.make(EnvString, size=edge_length, render_mode=render_mode)
        _, _ = env.reset()
        graph = minigrid_maze_to_graph(env, undirected=True)
        G = nx.Graph()
        for state, neighbors in graph.items():
            G.add_node(state)
            for nxt in neighbors.values():
                G.add_edge(state, nxt)

        if nx.number_connected_components(G) == 1:
            break
    return env


def get_measure_env(env, EnvString, visualise=False):
    return measures_from_env(env, EnvString, visualise=visualise)


def tile_graph_from_orientation_graph(graph):
    return _project_orientation_graph(graph)


__all__ = [
    "visualize_network",
    "minigrid_maze_to_graph",
    "graph_to_adjacency_matrix",
    "graph_to_geomem_dict",
    "tile_graph_from_orientation_graph",
    "DictEnvAdapter",
    "effective_commutativity",
    "pairwise_commutator_matrix",
    "effective_resistance_adj",
    "node_curvature",
    "link_curvature",
    "setup_env",
    "prep_env",
    "measures_from_env",
    "make_env",
    "get_measure_env",
]
