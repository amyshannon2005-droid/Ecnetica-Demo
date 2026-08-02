import math
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider


# =====================================================================
# 1. BKT NODE CLASS
# =====================================================================
class BKTNode:
    def __init__(self, p_init, p_t, p_s, p_g, node_id=None):
        params = {"p_init": p_init, "p_t": p_t, "p_s": p_s, "p_g": p_g}
        for name, val in params.items():
            if not isinstance(val, (int, float)) or not (0 <= val <= 1):
                raise ValueError(
                    f'BKTNode: parameter "{name}" must be a number in [0, 1], got {val}'
                )

        self.id = node_id
        self.p_init = p_init
        self.p_t = p_t
        self.p_s = p_s
        self.p_g = p_g
        self.m = p_init
        self.history = [{"event": "init", "m": self.m}]

    def observe(self, correct: bool) -> float:
        m = self.m
        p_s, p_g, p_t = self.p_s, self.p_g, self.p_t
        if correct:
            numerator = m * (1 - p_s)
            denominator = numerator + (1 - m) * p_g
        else:
            numerator = m * p_s
            denominator = numerator + (1 - m) * (1 - p_g)

        m_obs = numerator / denominator
        m_next = m_obs + (1 - m_obs) * p_t
        self.m = m_next
        self.history.append(
            {"event": "correct" if correct else "wrong", "m_obs": m_obs, "m": m_next}
        )
        return m_next

    def reset(self):
        self.m = self.p_init
        self.history = [{"event": "reset", "m": self.m}]


# =====================================================================
# 2. CONCEPT GRAPH CLASS (With Debugged Kahn's & Transitivity DFS)
# =====================================================================
class ConceptGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, payload=None):
        self.nodes[node_id] = payload

    def add_edge(self, u, v, weight=1.0):
        self.edges[(u, v)] = weight

    def children(self, u):
        return [v for (uu, v) in self.edges if uu == u]

    def parents(self, v):
        return [u for (u, vv) in self.edges if vv == v]

    # -----------------------------------------------------------------
    # FIXED & OPTIMIZED Kahn's Topological Order Algorithm — O(|V| + |E|)
    # -----------------------------------------------------------------
    def kahn_topological_order(self):
        # Build in-degrees and adjacency list
        in_degree = {node_id: 0 for node_id in self.nodes}
        adj_list = {node_id: [] for node_id in self.nodes}

        for (u, v) in self.edges:
            in_degree[v] += 1
            adj_list[u].append(v)

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node_id = queue.popleft()
            order.append(node_id)

            for child in adj_list[node_id]:  # O(1) Adjacency Lookup
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)  # Enqueue when zero in-degree reached

        return order  # Returned after complete queue processing

    def is_valid_dag(self):
        order = self.kahn_topological_order()
        return len(order) == len(self.nodes)

    # -----------------------------------------------------------------
    # FIXED Transitive Reduction & Reachability DFS (_has_path)
    # -----------------------------------------------------------------
    def _has_path(self, start, target, exclude_edge=None):
        if start == target:
            return True

        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == target:
                return True

            if node in visited:
                continue

            visited.add(node)

            for (u, v) in self.edges:
                if exclude_edge is not None and (u, v) == exclude_edge:
                    continue

                if u == node and v not in visited:
                    stack.append(v)  # Continue pushing unvisited DFS nodes

        return False  # Target unreachable without excluded edge

    def transitive_reduction(self):
        redundant = []
        for (u, v) in self.edges:
            if self._has_path(u, v, exclude_edge=(u, v)):
                redundant.append((u, v))

        for edge in redundant:
            del self.edges[edge]

        return redundant


# =====================================================================
# 3. BKT PROPAGATION LOGIC
# =====================================================================
def clamp(m, eps=1e-3):
    return min(1 - eps, max(eps, m))

def logit(m):
    return math.log(clamp(m) / (1 - clamp(m)))

def sigmoid(ell):
    return 1 / (1 + math.exp(-ell))

def path_attenuation(g, source_id, base_beta):
    parent = {source_id: None}
    order = [source_id]
    queue = deque([source_id])

    while queue:
        node_id = queue.popleft()
        neighbours = g.children(node_id) + g.parents(node_id)
        for nb in neighbours:
            if nb not in parent:
                parent[nb] = node_id
                order.append(nb)
                queue.append(nb)

    atten = {source_id: 1.0}
    for node_id in order[1:]:
        p = parent[node_id]
        atten[node_id] = atten[p] * base_beta

    return atten

def observe_and_propagate(g, node_id, correct, base_beta, max_delta=3.0):
    node = g.nodes[node_id]
    m_before = node.m
    node.observe(correct)
    m_obs = node.history[-1]["m_obs"]
    delta_ell = logit(m_obs) - logit(m_before)

    atten = path_attenuation(g, node_id, base_beta)
    for nid, factor in atten.items():
        if nid == node_id:
            continue
        injection = max(-max_delta, min(max_delta, factor * delta_ell))
        g.nodes[nid].m = sigmoid(logit(g.nodes[nid].m) + injection)


# =====================================================================
# 4. INTERACTIVE MATPLOTLIB DEMO
# =====================================================================
NODE_POSITIONS = {
    "algebra_basics": (0.22, 0.88),
    "quadratics": (0.22, 0.58),
    "limits": (0.78, 0.88),
    "differentiation": (0.78, 0.58),
}

def build_demo_graph():
    g = ConceptGraph()
    nodes = {
        "algebra_basics": BKTNode(0.50, 0.20, 0.10, 0.20, "algebra_basics"),
        "quadratics": BKTNode(0.35, 0.18, 0.10, 0.20, "quadratics"),
        "limits": BKTNode(0.25, 0.15, 0.10, 0.20, "limits"),
        "differentiation": BKTNode(0.20, 0.15, 0.10, 0.15, "differentiation"),
    }
    for k, v in nodes.items():
        g.add_node(k, v)

    g.add_edge("algebra_basics", "quadratics")
    g.add_edge("quadratics", "limits")
    g.add_edge("limits", "differentiation")

    return g

def launch_interactive(g, threshold=0.90):
    fig = plt.figure(figsize=(12, 7))
    ax_graph = fig.add_axes([0.03, 0.15, 0.55, 0.80])
    ax_graph.set_xlim(0, 1)
    ax_graph.set_ylim(0, 1)
    ax_graph.axis("off")

    # RESTORED Base Beta Attenuation Slider
    ax_beta = fig.add_axes([0.08, 0.05, 0.45, 0.03])
    slider_beta = Slider(ax_beta, "Base Beta", 0.1, 0.9, valinit=0.60, valstep=0.05)

    patches, texts = {}, {}
    for node_id, (x, y) in NODE_POSITIONS.items():
        box = mpatches.FancyBboxPatch(
            (x - 0.10, y - 0.045),
            0.20,
            0.09,
            boxstyle="round,pad=0.015",
            fc="#d9dcdf",
            ec="black",
            zorder=2,
        )
        ax_graph.add_patch(box)