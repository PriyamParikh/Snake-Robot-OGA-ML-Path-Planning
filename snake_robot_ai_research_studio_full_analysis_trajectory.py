
import csv, heapq, math, random, time, tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle, Rectangle

try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor
    import joblib
    ML_AVAILABLE = True
    ML_IMPORT_ERROR = ""
except Exception as _ml_exc:
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(_ml_exc)

DOF_MIN, DOF_MAX = 6, 16
DEFAULT_DOF = 16
DEFAULT_LINK = 35.0


# ----------------------------- geometry -----------------------------

def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))

def inside_bounds(p, bounds):
    x, y = p
    xmin, xmax, ymin, ymax = bounds
    return xmin <= x <= xmax and ymin <= y <= ymax

def point_collision(p, obstacles, inflation=0.0):
    x, y = map(float, p)
    for o in obstacles:
        if o["type"] == "circle":
            if math.hypot(x-o["x"], y-o["y"]) <= o["r"] + inflation:
                return True
        else:
            if abs(x-o["x"]) <= o["w"]/2 + inflation and abs(y-o["y"]) <= o["h"]/2 + inflation:
                return True
    return False

def segment_collision(a, b, obstacles, inflation=0.0, step=4.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b-a)
    n = max(int(L/max(step, 1e-9)), 1)
    for i in range(n+1):
        p = a + (i/n)*(b-a)
        if point_collision(p, obstacles, inflation):
            return True
    return False

def route_length(path):
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

def route_smoothness(path):
    if path is None or len(path) < 3:
        return 0.0
    p = np.asarray(path, float)
    d = np.diff(p, axis=0)
    h = np.unwrap(np.arctan2(d[:,1], d[:,0]))
    return float(np.degrees(np.sum(np.abs(np.diff(h)))))

def route_clearance(path, obstacles):
    if path is None or len(path) == 0 or not obstacles:
        return float("inf")
    best = float("inf")
    for x, y in np.asarray(path, float):
        for o in obstacles:
            if o["type"] == "circle":
                c = math.hypot(x-o["x"], y-o["y"]) - o["r"]
            else:
                dx = max(abs(x-o["x"]) - o["w"]/2, 0)
                dy = max(abs(y-o["y"]) - o["h"]/2, 0)
                if abs(x-o["x"]) <= o["w"]/2 and abs(y-o["y"]) <= o["h"]/2:
                    c = -min(o["w"]/2-abs(x-o["x"]), o["h"]/2-abs(y-o["y"]))
                else:
                    c = math.hypot(dx, dy)
            best = min(best, c)
    return float(best)

def simplify(path, obstacles, inflation, step=4.0):
    if path is None or len(path) <= 2:
        return path
    pts = [np.asarray(p, float) for p in path]
    out, i = [pts[0]], 0
    while i < len(pts)-1:
        j = len(pts)-1
        while j > i+1 and segment_collision(pts[i], pts[j], obstacles, inflation, step):
            j -= 1
        out.append(pts[j])
        i = j
    return np.asarray(out)

def resample(path, spacing=8.0, max_points=1200):
    p = np.asarray(path, float)
    if len(p) < 2:
        return p.copy()
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    total = cum[-1]
    if total <= 1e-9:
        return p[:1]
    n = min(max(int(total/max(spacing,1e-9))+1, 2), max_points)
    ss = np.linspace(0, total, n)
    out, j = [], 0
    for s in ss:
        while j < len(seg)-1 and cum[j+1] < s:
            j += 1
        den = cum[j+1]-cum[j]
        u = 0 if den <= 1e-12 else (s-cum[j])/den
        out.append(p[j] + u*(p[j+1]-p[j]))
    return np.asarray(out)



def angle_diff_deg(a, b):
    """Smallest signed angular difference a-b in degrees."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def vector_heading_deg(v):
    v = np.asarray(v, dtype=float)
    if np.linalg.norm(v) < 1e-12:
        return None
    return math.degrees(math.atan2(v[1], v[0]))


def history_point_blocked(p, history_points, corridor_radius, current_start, release_radius):
    """
    Treat the previously traversed head path as a temporary no-retrace corridor.
    A release region around the current destination/start is excluded so the
    snake can leave that point without becoming trapped.
    """
    if history_points is None or len(history_points) == 0 or corridor_radius <= 0:
        return False
    p = np.asarray(p, dtype=float)
    current_start = np.asarray(current_start, dtype=float)

    if dist(p, current_start) <= release_radius:
        return False

    H = np.asarray(history_points, dtype=float)
    if len(H) == 0:
        return False

    # Ignore history points close to the current start / destination.
    mask = np.linalg.norm(H - current_start[None, :], axis=1) > release_radius
    H = H[mask]
    if len(H) == 0:
        return False

    return float(np.min(np.linalg.norm(H - p[None, :], axis=1))) < corridor_radius


def history_segment_blocked(a, b, history_points, corridor_radius, current_start, release_radius, sample_step=6.0):
    if history_points is None or len(history_points) == 0 or corridor_radius <= 0:
        return False
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    L = np.linalg.norm(b - a)
    n = max(int(L / max(sample_step, 1e-6)), 1)
    for i in range(n + 1):
        p = a + (i / n) * (b - a)
        if history_point_blocked(
            p, history_points, corridor_radius, current_start, release_radius
        ):
            return True
    return False


def body_self_intersection(points):
    """Detect intersections between non-adjacent snake-body segments."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        return False

    def orient(a, b, c):
        return np.cross(b - a, c - a)

    def intersects(a, b, c, d):
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return (o1 * o2 < 0) and (o3 * o4 < 0)

    for i in range(len(pts) - 1):
        for j in range(i + 2, len(pts) - 1):
            # Adjacent segments share a joint and are allowed.
            if j == i + 1:
                continue
            if intersects(pts[i], pts[i+1], pts[j], pts[j+1]):
                return True
    return False


# ----------------------------- grid search -----------------------------

def grid_search(
    start, goal, obstacles, bounds, resolution, inflation, astar=True,
    start_heading_deg=None, max_turn_deg=100.0, turn_penalty=0.35,
    history_points=None, retrace_radius=0.0, release_radius=0.0
):
    """
    Direction-aware Dijkstra/A*.

    State = (grid_x, grid_y, incoming_direction).
    Near-180 degree direction reversal is prohibited by max_turn_deg.
    A turn penalty makes smooth continuation cheaper than zig-zagging.
    Previously traversed route can be treated as a temporary no-retrace corridor.
    """
    xmin, xmax, ymin, ymax = bounds
    res = max(float(resolution), 3.0)
    nx = int((xmax-xmin)/res) + 1
    ny = int((ymax-ymin)/res) + 1

    def idx(p):
        return (int(round((p[0]-xmin)/res)), int(round((p[1]-ymin)/res)))
    def xy(i):
        return np.array([xmin+i[0]*res, ymin+i[1]*res], float)
    def valid(i):
        return (
            0 <= i[0] < nx and 0 <= i[1] < ny
            and not point_collision(xy(i), obstacles, inflation)
            and not history_point_blocked(
                xy(i), history_points, retrace_radius,
                np.asarray(start, float), release_radius
            )
        )

    s, g = idx(start), idx(goal)

    # The current start must remain usable even though it lies at the end of history.
    if not (0 <= s[0] < nx and 0 <= s[1] < ny):
        return None, 0
    if point_collision(xy(s), obstacles, inflation):
        return None, 0
    if not valid(g):
        return None, 0

    moves = [
        (-1,0,1.0), (1,0,1.0), (0,-1,1.0), (0,1,1.0),
        (-1,-1,math.sqrt(2)), (-1,1,math.sqrt(2)),
        (1,-1,math.sqrt(2)), (1,1,math.sqrt(2))
    ]
    move_headings = [
        math.degrees(math.atan2(dy, dx)) for dx, dy, _ in moves
    ]

    # Direction index -1 represents the initial state.
    start_state = (s[0], s[1], -1)
    pq = [(0.0, 0.0, start_state)]
    gs = {start_state: 0.0}
    came = {}
    closed = set()
    explored = 0
    goal_state = None

    while pq:
        _, gc, state = heapq.heappop(pq)
        if state in closed:
            continue
        closed.add(state)
        explored += 1

        cur = (state[0], state[1])
        prev_dir = state[2]

        if cur == g:
            goal_state = state
            break

        for move_i, (dx,dy,step_cost) in enumerate(moves):
            nxt = (cur[0]+dx, cur[1]+dy)
            if not valid(nxt):
                continue

            a = xy(cur)
            b = xy(nxt)

            if segment_collision(a, b, obstacles, inflation, res/4):
                continue
            if history_segment_blocked(
                a, b, history_points, retrace_radius,
                np.asarray(start,float), release_radius,
                sample_step=max(res/4, 2.5)
            ):
                continue

            new_heading = move_headings[move_i]

            if prev_dir >= 0:
                prev_heading = move_headings[prev_dir]
            else:
                prev_heading = start_heading_deg

            turn_deg = 0.0
            if prev_heading is not None:
                turn_deg = abs(angle_diff_deg(new_heading, prev_heading))
                if turn_deg > max_turn_deg:
                    continue

            turn_cost = turn_penalty * res * (turn_deg / max(max_turn_deg, 1.0))
            cand = gc + step_cost*res + turn_cost

            next_state = (nxt[0], nxt[1], move_i)
            if cand < gs.get(next_state, float("inf")):
                gs[next_state] = cand
                came[next_state] = state
                h = dist(xy(nxt), goal) if astar else 0.0
                heapq.heappush(pq, (cand+h, cand, next_state))

    if goal_state is None:
        return None, explored

    states = [goal_state]
    cur = goal_state
    while cur in came:
        cur = came[cur]
        states.append(cur)
    states.reverse()

    path = np.asarray([xy((st[0],st[1])) for st in states], dtype=float)
    path[0] = np.asarray(start,float)
    path[-1] = np.asarray(goal,float)
    return path, explored


# ----------------------------- RRT -----------------------------

@dataclass
class Node:
    p: np.ndarray
    parent: int
    cost: float

def rrt(
    start, goal, obstacles, bounds, step_size, max_iter, goal_radius,
    inflation, star=False, seed=42,
    start_heading_deg=None, max_turn_deg=100.0,
    history_points=None, retrace_radius=0.0, release_radius=0.0
):
    """
    Direction-aware RRT / RRT*.

    Extensions that require a local heading change greater than max_turn_deg
    are rejected. Previously traversed path can be treated as a no-retrace
    corridor.
    """
    rng = random.Random(seed)
    start, goal = np.asarray(start,float), np.asarray(goal,float)

    if point_collision(start, obstacles, inflation) or point_collision(goal, obstacles, inflation):
        return None, 0

    if history_point_blocked(goal, history_points, retrace_radius, start, release_radius):
        return None, 0

    nodes = [Node(start.copy(), -1, 0.0)]
    goal_idx = None
    radius = max(step_size*3, 25)

    def incoming_heading(node_index):
        node = nodes[node_index]
        if node.parent < 0:
            return start_heading_deg
        v = node.p - nodes[node.parent].p
        return vector_heading_deg(v)

    def allowed_edge(parent_idx, newp):
        parent_point = nodes[parent_idx].p
        new_heading = vector_heading_deg(newp - parent_point)
        prev_heading = incoming_heading(parent_idx)

        if prev_heading is not None and new_heading is not None:
            if abs(angle_diff_deg(new_heading, prev_heading)) > max_turn_deg:
                return False

        if history_segment_blocked(
            parent_point, newp, history_points, retrace_radius,
            start, release_radius, sample_step=max(step_size/6, 3.0)
        ):
            return False

        return True

    for it in range(max_iter):
        if rng.random() < 0.12:
            sample = goal.copy()
        else:
            xmin,xmax,ymin,ymax = bounds
            sample = np.array([rng.uniform(xmin,xmax), rng.uniform(ymin,ymax)], float)

        if history_point_blocked(sample, history_points, retrace_radius, start, release_radius):
            continue

        pts = np.asarray([n.p for n in nodes])
        ni = int(np.argmin(np.linalg.norm(pts-sample, axis=1)))
        v = sample - nodes[ni].p
        L = np.linalg.norm(v)
        if L <= 1e-12:
            continue

        newp = nodes[ni].p + v/L*min(step_size,L)

        if not inside_bounds(newp,bounds):
            continue
        if point_collision(newp,obstacles,inflation):
            continue
        if history_point_blocked(newp, history_points, retrace_radius, start, release_radius):
            continue
        if segment_collision(nodes[ni].p,newp,obstacles,inflation,step_size/6):
            continue
        if not allowed_edge(ni, newp):
            continue

        parent = ni
        newcost = nodes[ni].cost + dist(nodes[ni].p,newp)
        near = []

        if star:
            d = np.linalg.norm(pts-newp,axis=1)
            near = np.where(d <= radius)[0].tolist()

            for j in near:
                if not allowed_edge(j, newp):
                    continue
                cand = nodes[j].cost + dist(nodes[j].p,newp)
                if (
                    cand < newcost
                    and not segment_collision(nodes[j].p,newp,obstacles,inflation,step_size/6)
                ):
                    parent, newcost = j, cand

        nodes.append(Node(newp.copy(), parent, newcost))
        newi = len(nodes)-1

        if star:
            for j in near:
                if j == parent:
                    continue
                if not allowed_edge(newi, nodes[j].p):
                    continue
                cand = newcost + dist(newp,nodes[j].p)
                if (
                    cand < nodes[j].cost
                    and not segment_collision(newp,nodes[j].p,obstacles,inflation,step_size/6)
                ):
                    nodes[j].parent = newi
                    nodes[j].cost = cand

        if dist(newp,goal) <= goal_radius:
            if (
                not segment_collision(newp,goal,obstacles,inflation,step_size/6)
                and not history_segment_blocked(
                    newp,goal,history_points,retrace_radius,
                    start,release_radius,max(step_size/6,3.0)
                )
                and allowed_edge(newi, goal)
            ):
                nodes.append(Node(goal.copy(), newi, newcost+dist(newp,goal)))
                goal_idx = len(nodes)-1
                if not star:
                    break

        if star and goal_idx is not None and it > int(max_iter*0.6):
            break

    if goal_idx is None:
        return None, len(nodes)

    path, seen, i = [], set(), goal_idx
    while i >= 0 and i not in seen:
        seen.add(i)
        path.append(nodes[i].p.copy())
        i = nodes[i].parent
    path.reverse()
    return np.asarray(path), len(nodes)


# ----------------------------- FK snake -----------------------------

class Snake:
    def __init__(self, dof=16, link=35.0):
        self.configure(dof,link)
    def configure(self,dof,link):
        self.dof = int(dof)
        self.link = float(link)
    def points(self,q_deg,base,heading_deg):
        q = np.deg2rad(np.asarray(q_deg,float))
        base = np.asarray(base,float)
        pts = [base.copy()]
        x,y = base
        th = math.radians(float(heading_deg))
        for qi in q:
            th += qi
            x += self.link*math.cos(th)
            y += self.link*math.sin(th)
            pts.append(np.array([x,y]))
        return np.asarray(pts)


# ----------------------------- GUI -----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake Robot AI Research Studio — Full Mission Analysis + Dynamic Replanning")
        self.geometry("1600x930")
        self.minsize(1300,800)

        self.dof = DEFAULT_DOF
        self.robot = Snake(self.dof, DEFAULT_LINK)
        self.start = np.array([0.0,0.0])
        self.destinations = []
        self.obstacles = []

        self.current_path = None
        self.current_segments = []
        self.current_algorithm = None
        self.current_result = None
        self.compare_results = {}

        self.animating = False
        self.anim_job = None

        # Online / dynamic-obstacle navigation state
        self.active_seq = None
        self.active_index = 0
        self.active_head_trail = []
        self.active_body = None
        self.dynamic_event_count = 0
        self.dynamic_replan_count = 0
        self.mission_original_start = self.start.copy()
        self.mission_original_destinations = []
        self.mission_reference_path = None
        self.runtime_mission_started = False

        # Research analysis / logging
        self.runtime_body_log = []
        self.runtime_replan_flags = []
        self.analysis_cache = None
        self.error_cache = None
        self.workspace_cache = None
        self.workspace_singularity_cbar = None

        # ML motion-command state
        self.ml_X = None
        self.ml_y = None
        self.ml_models = {}
        self.ml_metrics = {}
        self.ml_predictions = {}
        self.ml_model_choice_name = ""
        self.ml_reference_path = None
        self.ml_last_route = None
        self.ml_confusions = {}
        self.ml_dataset_signature = None
        self.ml_training_signature = None
        self.ml_dataset_source_text = "No fresh ML dataset"
        self.ml_last_invalidation_reason = ""
        self.ml_training_scope = "environment"

        # Selected evaluation pair: ML command is evaluated FIRST, then the
        # selected path planner runs with that ML-derived initial heading.
        self.eval_ml_var = tk.StringVar(value="Neural Network")
        self.eval_planner_var = tk.StringVar(value="A*")
        self.active_pair_ml = ""
        self.active_pair_planner = ""
        self.pair_results = {}

        self._style()
        self._ui()
        self.redraw()

    def _style(self):
        st=ttk.Style(self)
        try: st.theme_use("clam")
        except: pass
        st.configure("Title.TLabel",font=("Segoe UI",15,"bold"))
        st.configure("Mono.TLabel",font=("Consolas",10))

    def _ui(self):
        top=ttk.Frame(self,padding=(10,7))
        top.pack(fill="x")
        ttk.Label(
            top,
            text="Snake Robot AI Research Studio — Kinematics | ML | Path Planning | Dynamic Replanning",
            style="Title.TLabel"
        ).pack(side="left")
        self.status=tk.StringVar(value="Ready")
        ttk.Label(top,textvariable=self.status).pack(side="right")

        self.nb=ttk.Notebook(self)
        self.nb.pack(fill="both",expand=True,padx=8,pady=(0,8))

        self.plan_tab=ttk.Frame(self.nb)
        self.ml_tab=ttk.Frame(self.nb)
        self.motion_tab=ttk.Frame(self.nb)
        self.error_tab=ttk.Frame(self.nb)
        self.singularity_tab=ttk.Frame(self.nb)
        self.workspace_tab=ttk.Frame(self.nb)
        self.comp_tab=ttk.Frame(self.nb)
        self.export_tab=ttk.Frame(self.nb)
        self.help_tab=ttk.Frame(self.nb)

        self.nb.add(self.plan_tab,text="Path Planning & Simulation")
        self.nb.add(self.ml_tab,text="ML Training & Commands")
        self.nb.add(self.motion_tab,text="Motion Analysis")
        self.nb.add(self.error_tab,text="Head Error & Safety")
        self.nb.add(self.singularity_tab,text="Singularity Analysis")
        self.nb.add(self.workspace_tab,text="Workspace Analysis")
        self.nb.add(self.comp_tab,text="Planner Comparison")
        self.nb.add(self.export_tab,text="Export Center")
        self.nb.add(self.help_tab,text="Course Workflow")

        self._planner_ui()
        self._ml_ui()
        self._motion_ui()
        self._error_ui()
        self._singularity_ui()
        self._workspace_ui()
        self._compare_ui()
        self._export_ui()
        self._help_ui()

    def _planner_ui(self):
        t=self.plan_tab
        t.columnconfigure(0,weight=0)
        t.columnconfigure(1,weight=1)
        t.rowconfigure(0,weight=1)

        # -------------------------------------------------------------
        # Scrollable control panel on the left
        # -------------------------------------------------------------
        left_outer=ttk.Frame(t)
        left_outer.grid(row=0,column=0,sticky="ns")
        left_outer.rowconfigure(0,weight=1)
        left_outer.columnconfigure(0,weight=1)

        self.control_canvas=tk.Canvas(
            left_outer,
            width=455,
            highlightthickness=0,
            borderwidth=0
        )
        self.control_scroll=ttk.Scrollbar(
            left_outer,
            orient="vertical",
            command=self.control_canvas.yview
        )
        self.control_canvas.configure(yscrollcommand=self.control_scroll.set)

        self.control_canvas.grid(row=0,column=0,sticky="ns")
        self.control_scroll.grid(row=0,column=1,sticky="ns")

        left=ttk.Frame(self.control_canvas,padding=8)
        self.control_window=self.control_canvas.create_window(
            (0,0),
            window=left,
            anchor="nw"
        )

        def update_scrollregion(event=None):
            self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )

        def resize_control_width(event):
            self.control_canvas.itemconfigure(
                self.control_window,
                width=event.width
            )

        def mousewheel(event):
            if event.delta:
                self.control_canvas.yview_scroll(
                    int(-event.delta/120),
                    "units"
                )
            return "break"

        def bind_mousewheel(event=None):
            self.control_canvas.bind_all("<MouseWheel>",mousewheel)

        def unbind_mousewheel(event=None):
            self.control_canvas.unbind_all("<MouseWheel>")

        left.bind("<Configure>",update_scrollregion)
        self.control_canvas.bind("<Configure>",resize_control_width)
        self.control_canvas.bind("<Enter>",bind_mousewheel)
        self.control_canvas.bind("<Leave>",unbind_mousewheel)
        left.bind("<Enter>",bind_mousewheel)
        left.bind("<Leave>",unbind_mousewheel)

        # Right-side map remains fixed and fully expandable.
        right=ttk.Frame(t,padding=6)
        right.grid(row=0,column=1,sticky="nsew")
        right.rowconfigure(0,weight=1)
        right.columnconfigure(0,weight=1)

        # Robot
        f=ttk.LabelFrame(left,text="Robot / Animation",padding=7); f.pack(fill="x",pady=(0,6))
        self.dof_var=tk.IntVar(value=self.dof); self.link_var=tk.DoubleVar(value=DEFAULT_LINK)
        self.amp_var=tk.DoubleVar(value=22.0); self.phase_var=tk.DoubleVar(value=40.0)
        self.wave_var=tk.DoubleVar(value=120.0); self.body_radius=tk.DoubleVar(value=5.0)
        self.speed_var=tk.DoubleVar(value=1.0)
        self.show_axes_var=tk.BooleanVar(value=True)
        self.axis_scale_var=tk.DoubleVar(value=0.60)
        rows=[("DOF:",self.dof_var),("Link length:",self.link_var),("Gait amplitude:",self.amp_var),
              ("Phase / joint:",self.phase_var),("Wavelength:",self.wave_var),
              ("Body radius:",self.body_radius)]
        for r,(lab,var) in enumerate(rows):
            ttk.Label(f,text=lab).grid(row=r,column=0,sticky="w",pady=2)
            if r==0:
                ttk.Combobox(f,textvariable=var,values=list(range(DOF_MIN,DOF_MAX+1)),state="readonly",width=8).grid(row=r,column=1,padx=3)
            else:
                ttk.Entry(f,textvariable=var,width=9).grid(row=r,column=1,padx=3)
        ttk.Button(f,text="Apply Robot",command=self.apply_robot).grid(row=0,column=2,padx=3)
        ttk.Label(f,text="Anim speed:").grid(row=6,column=0,sticky="w",pady=(4,0))
        ttk.Scale(f,from_=0.2,to=5.0,variable=self.speed_var,orient="horizontal",length=150).grid(
            row=6,column=1,columnspan=2
        )

        ttk.Checkbutton(
            f,
            text="Show Global + Local XYZ Frames",
            variable=self.show_axes_var,
            command=self.redraw
        ).grid(row=7,column=0,columnspan=3,sticky="w",pady=(6,0))

        ttk.Label(f,text="Local axis length / link:").grid(
            row=8,column=0,sticky="w",pady=(3,0)
        )
        ttk.Entry(f,textvariable=self.axis_scale_var,width=8).grid(
            row=8,column=1,sticky="w",padx=3,pady=(3,0)
        )
        ttk.Button(f,text="Refresh Axes",command=self.redraw).grid(
            row=8,column=2,padx=3,pady=(3,0)
        )

        # Bounds
        f=ttk.LabelFrame(left,text="Workspace",padding=7); f.pack(fill="x",pady=6)
        self.xmin=tk.DoubleVar(value=-200); self.xmax=tk.DoubleVar(value=900)
        self.ymin=tk.DoubleVar(value=-450); self.ymax=tk.DoubleVar(value=450)
        vals=[("X min",self.xmin),("X max",self.xmax),("Y min",self.ymin),("Y max",self.ymax)]
        for i,(lab,var) in enumerate(vals):
            ttk.Label(f,text=lab).grid(row=i//2,column=(i%2)*2,sticky="w",padx=2,pady=2)
            ttk.Entry(f,textvariable=var,width=8).grid(row=i//2,column=(i%2)*2+1,padx=2,pady=2)
        ttk.Button(f,text="Apply Bounds",command=self.redraw).grid(row=2,column=0,columnspan=4,sticky="ew",pady=(5,0))

        # Editing
        f=ttk.LabelFrame(left,text="Map Editing",padding=7); f.pack(fill="x",pady=6)
        self.mode=tk.StringVar(value="Destination")
        ttk.Label(f,text="Click mode:").grid(row=0,column=0,sticky="w")
        ttk.Combobox(f,textvariable=self.mode,values=["Start","Destination","Circle Obstacle","Rectangle Obstacle"],
                     state="readonly",width=20).grid(row=0,column=1,columnspan=2,padx=3)
        self.circle_r=tk.DoubleVar(value=45)
        self.rect_w=tk.DoubleVar(value=100)
        self.rect_h=tk.DoubleVar(value=100)
        self.manual_obs_x=tk.DoubleVar(value=250.0)
        self.manual_obs_y=tk.DoubleVar(value=0.0)

        ttk.Label(f,text="Circle radius:").grid(row=1,column=0,sticky="w",pady=(4,0))
        ttk.Entry(f,textvariable=self.circle_r,width=8).grid(row=1,column=1,pady=(4,0))

        ttk.Label(f,text="Rect W/H:").grid(row=2,column=0,sticky="w",pady=(4,0))
        ttk.Entry(f,textvariable=self.rect_w,width=8).grid(row=2,column=1,pady=(4,0))
        ttk.Entry(f,textvariable=self.rect_h,width=8).grid(row=2,column=2,pady=(4,0))

        ttk.Separator(f,orient="horizontal").grid(
            row=3,column=0,columnspan=3,sticky="ew",pady=6
        )

        ttk.Label(f,text="Manual obstacle X:").grid(row=4,column=0,sticky="w")
        ttk.Entry(f,textvariable=self.manual_obs_x,width=10).grid(row=4,column=1,padx=3)

        ttk.Label(f,text="Manual obstacle Y:").grid(row=5,column=0,sticky="w",pady=(3,0))
        ttk.Entry(f,textvariable=self.manual_obs_y,width=10).grid(row=5,column=1,padx=3,pady=(3,0))

        ttk.Button(
            f,
            text="Add Circle at X,Y",
            command=lambda:self.add_manual_static_obstacle("circle")
        ).grid(row=6,column=0,columnspan=2,sticky="ew",padx=2,pady=(6,0))

        ttk.Button(
            f,
            text="Add Rectangle at X,Y",
            command=lambda:self.add_manual_static_obstacle("rect")
        ).grid(row=6,column=2,sticky="ew",padx=2,pady=(6,0))

        ttk.Label(
            f,
            text="Either enter X,Y manually or select a click mode and click on the map.",
            justify="left"
        ).grid(row=7,column=0,columnspan=3,sticky="w",pady=(5,0))

        # Dynamic obstacle / online replanning
        f=ttk.LabelFrame(left,text="Dynamic Obstacle / Online Replanning",padding=7); f.pack(fill="x",pady=6)
        self.dynamic_mode=tk.BooleanVar(value=False)
        self.dynamic_auto_replan=tk.BooleanVar(value=True)
        self.dynamic_shape=tk.StringVar(value="Circle")
        self.dynamic_state=tk.StringVar(value="IDLE")

        ttk.Checkbutton(
            f,
            text="Enable sudden-obstacle insertion during animation",
            variable=self.dynamic_mode
        ).grid(row=0,column=0,columnspan=3,sticky="w")

        ttk.Label(f,text="New obstacle type:").grid(row=1,column=0,sticky="w",pady=(4,0))
        ttk.Combobox(
            f,
            textvariable=self.dynamic_shape,
            values=["Circle","Rectangle"],
            state="readonly",
            width=13
        ).grid(row=1,column=1,padx=3,pady=(4,0))

        ttk.Checkbutton(
            f,
            text="Automatically replan if future route is blocked",
            variable=self.dynamic_auto_replan
        ).grid(row=2,column=0,columnspan=3,sticky="w",pady=(4,0))

        ttk.Label(
            f,
            text="While the snake is moving: enable this mode and click on the map.\n"
                 "GREEN = continue | YELLOW = pause/replan | RED = emergency stop",
            justify="left"
        ).grid(row=3,column=0,columnspan=3,sticky="w",pady=(5,0))

        self.dynamic_x=tk.DoubleVar(value=350.0)
        self.dynamic_y=tk.DoubleVar(value=0.0)

        ttk.Separator(f,orient="horizontal").grid(
            row=4,column=0,columnspan=3,sticky="ew",pady=6
        )

        ttk.Label(f,text="Manual dynamic X:").grid(row=5,column=0,sticky="w")
        ttk.Entry(f,textvariable=self.dynamic_x,width=10).grid(row=5,column=1,padx=3)

        ttk.Label(f,text="Manual dynamic Y:").grid(row=6,column=0,sticky="w",pady=(3,0))
        ttk.Entry(f,textvariable=self.dynamic_y,width=10).grid(row=6,column=1,padx=3,pady=(3,0))

        ttk.Button(
            f,
            text="INSERT DYNAMIC OBSTACLE AT X,Y",
            command=self.add_manual_dynamic_obstacle
        ).grid(row=7,column=0,columnspan=3,sticky="ew",pady=(6,0))

        ttk.Label(f,text="Navigation state:").grid(row=8,column=0,sticky="w",pady=(5,0))
        ttk.Label(f,textvariable=self.dynamic_state,style="Mono.TLabel").grid(
            row=8,column=1,columnspan=2,sticky="w",pady=(5,0)
        )

        # destinations
        f=ttk.LabelFrame(left,text="Ordered Destinations",padding=7); f.pack(fill="both",expand=True,pady=6)

        self.manual_dest_x=tk.DoubleVar(value=400.0)
        self.manual_dest_y=tk.DoubleVar(value=0.0)

        md=ttk.Frame(f)
        md.pack(fill="x",pady=(0,6))
        ttk.Label(md,text="Manual D X:").grid(row=0,column=0,sticky="w")
        ttk.Entry(md,textvariable=self.manual_dest_x,width=9).grid(row=0,column=1,padx=3)
        ttk.Label(md,text="Y:").grid(row=0,column=2,sticky="w")
        ttk.Entry(md,textvariable=self.manual_dest_y,width=9).grid(row=0,column=3,padx=3)
        ttk.Button(
            md,
            text="ADD DESTINATION",
            command=self.add_manual_destination
        ).grid(row=0,column=4,padx=(6,0))

        ttk.Label(
            md,
            text="Destinations can be added by X,Y or by mouse click.",
            justify="left"
        ).grid(row=1,column=0,columnspan=5,sticky="w",pady=(4,0))

        self.dest_list=tk.Listbox(f,height=8,width=38,font=("Consolas",10)); self.dest_list.pack(fill="both",expand=True)
        r=ttk.Frame(f); r.pack(fill="x",pady=(4,0))
        ttk.Button(r,text="▲",width=4,command=lambda:self.move_dest(-1)).pack(side="left",padx=2)
        ttk.Button(r,text="▼",width=4,command=lambda:self.move_dest(1)).pack(side="left",padx=2)
        ttk.Button(r,text="Remove",command=self.remove_dest).pack(side="left",padx=2)
        ttk.Button(r,text="Clear Dests",command=self.clear_dests).pack(side="left",padx=2)
        ttk.Button(f,text="Clear Obstacles",command=self.clear_obs).pack(fill="x",pady=(4,0))
        ttk.Button(f,text="Clear Entire Map",command=self.clear_all).pack(fill="x",pady=(4,0))

        # Planner
        f=ttk.LabelFrame(left,text="Path Planner",padding=7); f.pack(fill="x",pady=6)
        self.alg=tk.StringVar(value="A*"); self.clearance=tk.DoubleVar(value=20)
        self.grid_res=tk.DoubleVar(value=18); self.rrt_step=tk.DoubleVar(value=38); self.rrt_iter=tk.IntVar(value=3500)
        ttk.Label(f,text="Algorithm:").grid(row=0,column=0,sticky="w")
        ttk.Combobox(f,textvariable=self.alg,values=["Dijkstra","A*","RRT","RRT*"],state="readonly",width=12).grid(row=0,column=1,padx=3)
        for r,(lab,var) in enumerate([("Safety clearance:",self.clearance),("Grid resolution:",self.grid_res),
                                      ("RRT step:",self.rrt_step),("RRT max iter:",self.rrt_iter)],start=1):
            ttk.Label(f,text=lab).grid(row=r,column=0,sticky="w",pady=2); ttk.Entry(f,textvariable=var,width=9).grid(row=r,column=1,padx=3)
        self.smooth=tk.BooleanVar(value=True)
        ttk.Checkbutton(f,text="Line-of-sight smoothing",variable=self.smooth).grid(row=5,column=0,columnspan=2,sticky="w",pady=3)

        self.directional_mode=tk.BooleanVar(value=True)
        ttk.Checkbutton(
            f,
            text="Directional snake mode / prevent backtracking",
            variable=self.directional_mode
        ).grid(row=6,column=0,columnspan=2,sticky="w",pady=3)

        self.max_turn_var=tk.DoubleVar(value=100.0)
        self.retrace_radius_var=tk.DoubleVar(value=32.0)
        self.turn_penalty_var=tk.DoubleVar(value=0.45)

        ttk.Label(f,text="Max local turn (deg):").grid(row=7,column=0,sticky="w",pady=2)
        ttk.Entry(f,textvariable=self.max_turn_var,width=9).grid(row=7,column=1,padx=3)

        ttk.Label(f,text="No-retrace corridor:").grid(row=8,column=0,sticky="w",pady=2)
        ttk.Entry(f,textvariable=self.retrace_radius_var,width=9).grid(row=8,column=1,padx=3)

        ttk.Label(f,text="Turn penalty:").grid(row=9,column=0,sticky="w",pady=2)
        ttk.Entry(f,textvariable=self.turn_penalty_var,width=9).grid(row=9,column=1,padx=3)

        ttk.Label(
            f,
            text="Full-body inflation = safety + body radius + 0.60 × link\n"
                 "Directional mode carries heading from D(i) into D(i+1)\n"
                 "and avoids the already-traversed head corridor.",
            justify="left"
        ).grid(row=10,column=0,columnspan=2,sticky="w",pady=(3,3))

        ttk.Button(f,text="REFERENCE / BASELINE PLAN",command=self.plan_selected).grid(
            row=11,column=0,columnspan=2,sticky="ew",pady=(5,0)
        )
        ttk.Button(f,text="COMPARE ALL 4",command=self.compare_all).grid(
            row=12,column=0,columnspan=2,sticky="ew",pady=(4,0)
        )
        ttk.Label(
            f,
            text="Hybrid operational workflow is in ML tab:\n"
                 "ML command FIRST → selected planner SECOND.",
            justify="left"
        ).grid(row=13,column=0,columnspan=2,sticky="w",pady=(5,0))

        # Export / animation
        f=ttk.LabelFrame(left,text="Animation / Export",padding=7); f.pack(fill="x",pady=(6,0))
        self.anim_spacing=tk.DoubleVar(value=8)
        ttk.Label(f,text="Path step:").grid(row=0,column=0,sticky="w"); ttk.Entry(f,textvariable=self.anim_spacing,width=8).grid(row=0,column=1,padx=3)
        ttk.Button(f,text="▶ Animate",command=self.animate_route).grid(row=1,column=0,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="■ Stop",command=self.stop).grid(row=1,column=1,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="Save GIF",command=lambda:self.save_anim("gif")).grid(row=2,column=0,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="Save MP4",command=lambda:self.save_anim("mp4")).grid(row=2,column=1,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="Path CSV",command=self.save_path_csv).grid(row=3,column=0,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="Figure",command=self.save_map_fig).grid(row=3,column=1,sticky="ew",padx=2,pady=3)
        ttk.Button(f,text="Comparison CSV",command=self.save_compare_csv).grid(row=4,column=0,columnspan=2,sticky="ew",padx=2,pady=3)

        # map
        self.fig=Figure(figsize=(10.5,8.2),dpi=100); self.ax=self.fig.add_subplot(111)
        self.canvas=FigureCanvasTkAgg(self.fig,master=right); self.canvas.get_tk_widget().grid(row=0,column=0,sticky="nsew")
        tb=NavigationToolbar2Tk(self.canvas,right,pack_toolbar=False); tb.update(); tb.grid(row=1,column=0,sticky="ew")
        self.info=tk.StringVar(value="Add Start/Destinations/Obstacles by clicking the map.")
        ttk.Label(right,textvariable=self.info,style="Mono.TLabel").grid(row=2,column=0,sticky="ew",pady=(4,0))
        self.canvas.mpl_connect("button_press_event",self.map_click)

    def _ml_ui(self):
        t=self.ml_tab
        t.columnconfigure(0,weight=1)
        t.rowconfigure(1,weight=1)

        top=ttk.Frame(t,padding=8)
        top.grid(row=0,column=0,sticky="ew")

        ttk.Label(top,text="Machine-Learning Motion Commands",style="Title.TLabel").grid(
            row=0,column=0,columnspan=12,sticky="w",pady=(0,6)
        )

        self.ml_test_pct=tk.DoubleVar(value=20.0)
        self.ml_step_var=tk.DoubleVar(value=14.0)
        self.ml_max_steps=tk.IntVar(value=1200)
        self.ml_lookahead_steps=tk.IntVar(value=5)
        self.ml_safety_shield=tk.BooleanVar(value=True)
        self.ml_include_compare=tk.BooleanVar(value=False)
        self.ml_training_source=tk.StringVar(value="ML First — Synthetic Commands")
        self.ml_synthetic_samples=tk.IntVar(value=3500)

        ttk.Label(top,text="Test data (%):").grid(row=1,column=0,sticky="w")
        ttk.Entry(top,textvariable=self.ml_test_pct,width=8).grid(row=1,column=1,padx=3)

        ttk.Label(top,text="ML step length:").grid(row=1,column=2,sticky="w",padx=(10,0))
        ttk.Entry(top,textvariable=self.ml_step_var,width=8).grid(row=1,column=3,padx=3)

        ttk.Label(top,text="Max ML steps / destination:").grid(row=1,column=4,sticky="w",padx=(10,0))
        ttk.Entry(top,textvariable=self.ml_max_steps,width=8).grid(row=1,column=5,padx=3)

        ttk.Label(top,text="Reference look-ahead:").grid(row=1,column=6,sticky="w",padx=(10,0))
        ttk.Entry(top,textvariable=self.ml_lookahead_steps,width=7).grid(row=1,column=7,padx=3)

        ttk.Checkbutton(
            top,text="Collision safety shield",
            variable=self.ml_safety_shield
        ).grid(row=1,column=8,columnspan=2,sticky="w",padx=(10,0))

        ttk.Checkbutton(
            top,
            text="Augment with current Compare-All routes",
            variable=self.ml_include_compare
        ).grid(row=1,column=10,columnspan=2,sticky="w",padx=(10,0))

        ttk.Label(top,text="Training source:").grid(row=2,column=4,sticky="w",padx=(10,0),pady=(6,0))
        ttk.Combobox(
            top,
            textvariable=self.ml_training_source,
            values=["ML First — Synthetic Commands","Reference Route — Imitation"],
            state="readonly",
            width=28
        ).grid(row=2,column=5,columnspan=3,sticky="w",padx=3,pady=(6,0))

        ttk.Label(top,text="Synthetic samples:").grid(row=2,column=8,sticky="e",pady=(6,0))
        ttk.Entry(top,textvariable=self.ml_synthetic_samples,width=8).grid(
            row=2,column=9,sticky="w",padx=3,pady=(6,0)
        )

        self.ml_algo_vars={}
        for j,name in enumerate(["KNN","Random Forest","Extra Trees","Neural Network"]):
            v=tk.BooleanVar(value=True)
            self.ml_algo_vars[name]=v
            ttk.Checkbutton(top,text=name,variable=v).grid(
                row=2,column=j,sticky="w",padx=3,pady=(6,0)
            )

        ttk.Button(top,text="1. Generate Training Data",command=self.ml_generate_dataset).grid(
            row=3,column=0,columnspan=2,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="2. Train Selected Models",command=self.ml_train_models).grid(
            row=3,column=2,columnspan=2,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="Predict Current Command",command=self.ml_predict_current_command).grid(
            row=3,column=4,columnspan=2,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="3. RUN ML COMMAND → PATH PLANNER",command=self.run_selected_pair).grid(
            row=3,column=6,columnspan=3,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="Save Model",command=self.ml_save_model).grid(
            row=3,column=9,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="Load Model",command=self.ml_load_model).grid(
            row=3,column=10,sticky="ew",padx=2,pady=(7,0)
        )
        ttk.Button(top,text="Export ML CSV",command=self.ml_export_csv).grid(
            row=3,column=11,sticky="ew",padx=2,pady=(7,0)
        )

        ttk.Label(top,text="Selected model:").grid(row=4,column=0,sticky="w",pady=(7,0))
        self.ml_model_var=tk.StringVar()
        self.ml_model_combo=ttk.Combobox(
            top,textvariable=self.ml_model_var,state="readonly",width=20
        )
        self.ml_model_combo.grid(row=4,column=1,columnspan=2,sticky="w",pady=(7,0))

        self.ml_command_text=tk.StringVar(value="ML command: not trained")
        ttk.Label(top,textvariable=self.ml_command_text,style="Mono.TLabel").grid(
            row=4,column=3,columnspan=9,sticky="w",padx=(8,0),pady=(7,0)
        )

        pair=ttk.LabelFrame(top,text="Graph / Hybrid Evaluation Pair",padding=5)
        pair.grid(row=5,column=0,columnspan=12,sticky="ew",pady=(8,0))

        ttk.Label(pair,text="ML Algorithm:").pack(side="left")
        self.eval_ml_combo=ttk.Combobox(
            pair,textvariable=self.eval_ml_var,state="readonly",width=18
        )
        self.eval_ml_combo.pack(side="left",padx=4)

        ttk.Label(pair,text="Path Planning Algorithm:").pack(side="left",padx=(12,0))
        ttk.Combobox(
            pair,textvariable=self.eval_planner_var,
            values=["Dijkstra","A*","RRT","RRT*"],
            state="readonly",width=12
        ).pack(side="left",padx=4)

        ttk.Button(pair,text="RUN SELECTED PAIR",command=self.run_selected_pair).pack(side="left",padx=8)
        ttk.Button(pair,text="LOAD PAIR GRAPHS",command=self.load_selected_pair_graphs).pack(side="left",padx=3)
        ttk.Button(pair,text="Save ML Figure",command=lambda:self.save_figure_generic(
            self.ml_fig,"ml_pair_results.png"
        )).pack(side="right",padx=3)

        self.ml_fig=Figure(figsize=(14.0,7.8),dpi=100)
        self.ml_axes=[
            self.ml_fig.add_subplot(241), self.ml_fig.add_subplot(242),
            self.ml_fig.add_subplot(243), self.ml_fig.add_subplot(244),
            self.ml_fig.add_subplot(245), self.ml_fig.add_subplot(246),
            self.ml_fig.add_subplot(247), self.ml_fig.add_subplot(248)
        ]
        self.ml_fig.tight_layout(pad=2)

        self.ml_canvas=FigureCanvasTkAgg(self.ml_fig,master=t)
        self.ml_canvas.get_tk_widget().grid(row=1,column=0,sticky="nsew",padx=8,pady=8)
        tb=NavigationToolbar2Tk(self.ml_canvas,t,pack_toolbar=False)
        tb.update()
        tb.grid(row=2,column=0,sticky="ew",padx=8)

        self.ml_status=tk.StringVar(
            value="Training target: steering command Δheading. Train KNN/RF/Extra Trees/MLP from successful routes."
        )
        ttk.Label(t,textvariable=self.ml_status,style="Mono.TLabel").grid(
            row=3,column=0,sticky="ew",padx=10,pady=(0,8)
        )
        self.ml_draw_empty()

    def _motion_ui(self):
        t=self.motion_tab
        top=ttk.Frame(t,padding=8)
        top.pack(fill="x")

        ttk.Label(top,text="Velocity, Acceleration & Displacement Analysis",style="Title.TLabel").pack(side="left")
        ttk.Label(top,text="ML:").pack(side="left",padx=(14,2))
        self.motion_ml_combo=ttk.Combobox(top,textvariable=self.eval_ml_var,state="readonly",width=16)
        self.motion_ml_combo.pack(side="left")
        ttk.Label(top,text="Planner:").pack(side="left",padx=(8,2))
        ttk.Combobox(
            top,textvariable=self.eval_planner_var,
            values=["Dijkstra","A*","RRT","RRT*"],state="readonly",width=10
        ).pack(side="left")
        ttk.Button(top,text="Load Pair",command=self.load_selected_pair_graphs).pack(side="left",padx=6)
        self.analysis_dt=tk.DoubleVar(value=0.05)
        ttk.Label(top,text="Analysis Δt (s):").pack(side="left",padx=(18,3))
        ttk.Entry(top,textvariable=self.analysis_dt,width=8).pack(side="left")
        ttk.Button(top,text="Refresh Analysis",command=self.refresh_motion_analysis).pack(side="left",padx=8)
        ttk.Button(top,text="Export Analysis CSV",command=self.export_motion_csv).pack(side="left",padx=3)
        ttk.Button(top,text="Save Figure",command=lambda:self.save_figure_generic(
            self.motion_fig,"snake_motion_analysis.png"
        )).pack(side="left",padx=3)

        self.motion_fig=Figure(figsize=(12.5,8.2),dpi=100)
        self.motion_axes=[
            self.motion_fig.add_subplot(231), self.motion_fig.add_subplot(232),
            self.motion_fig.add_subplot(233), self.motion_fig.add_subplot(234),
            self.motion_fig.add_subplot(235), self.motion_fig.add_subplot(236)
        ]
        self.motion_fig.tight_layout(pad=2)

        self.motion_canvas=FigureCanvasTkAgg(self.motion_fig,master=t)
        self.motion_canvas.get_tk_widget().pack(fill="both",expand=True,padx=8,pady=8)
        NavigationToolbar2Tk(self.motion_canvas,t).update()

        self.motion_summary=tk.StringVar(value="Run a plan/animation, then refresh.")
        ttk.Label(t,textvariable=self.motion_summary,style="Mono.TLabel").pack(
            fill="x",padx=10,pady=(0,8)
        )

    def _error_ui(self):
        t=self.error_tab
        top=ttk.Frame(t,padding=8)
        top.pack(fill="x")

        ttk.Label(top,text="Head Error, Singularity & Safety Analysis",style="Title.TLabel").pack(side="left")
        ttk.Label(top,text="ML:").pack(side="left",padx=(14,2))
        self.error_ml_combo=ttk.Combobox(top,textvariable=self.eval_ml_var,state="readonly",width=16)
        self.error_ml_combo.pack(side="left")
        ttk.Label(top,text="Planner:").pack(side="left",padx=(8,2))
        ttk.Combobox(
            top,textvariable=self.eval_planner_var,
            values=["Dijkstra","A*","RRT","RRT*"],state="readonly",width=10
        ).pack(side="left")
        ttk.Button(top,text="Load Pair",command=self.load_selected_pair_graphs).pack(side="left",padx=6)
        ttk.Button(top,text="Refresh Error Analysis",command=self.refresh_error_analysis).pack(side="left",padx=8)
        ttk.Button(top,text="Export Error CSV",command=self.export_error_csv).pack(side="left",padx=3)
        ttk.Button(top,text="Save Figure",command=lambda:self.save_figure_generic(
            self.error_fig,"snake_error_singularity_analysis.png"
        )).pack(side="left",padx=3)

        self.error_fig=Figure(figsize=(12.5,8.2),dpi=100)
        self.error_axes=[
            self.error_fig.add_subplot(231), self.error_fig.add_subplot(232),
            self.error_fig.add_subplot(233), self.error_fig.add_subplot(234),
            self.error_fig.add_subplot(235), self.error_fig.add_subplot(236)
        ]
        self.error_fig.tight_layout(pad=2)

        self.error_canvas=FigureCanvasTkAgg(self.error_fig,master=t)
        self.error_canvas.get_tk_widget().pack(fill="both",expand=True,padx=8,pady=8)
        NavigationToolbar2Tk(self.error_canvas,t).update()

        self.error_summary=tk.StringVar(value="Head error and singularity data not calculated yet.")
        ttk.Label(t,textvariable=self.error_summary,style="Mono.TLabel").pack(
            fill="x",padx=10,pady=(0,8)
        )

    def _singularity_ui(self):
        t=self.singularity_tab

        top=ttk.Frame(t,padding=8)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Dedicated Singularity Analysis",
            style="Title.TLabel"
        ).pack(side="left")

        ttk.Label(top,text="ML:").pack(side="left",padx=(14,2))
        self.singularity_ml_combo=ttk.Combobox(
            top,
            textvariable=self.eval_ml_var,
            state="readonly",
            width=15
        )
        self.singularity_ml_combo.pack(side="left")

        ttk.Label(top,text="Planner:").pack(side="left",padx=(8,2))
        ttk.Combobox(
            top,
            textvariable=self.eval_planner_var,
            values=["Dijkstra","A*","RRT","RRT*"],
            state="readonly",
            width=9
        ).pack(side="left")

        ttk.Label(top,text="Near-singularity ratio:").pack(side="left",padx=(10,2))
        self.singularity_ratio_threshold=tk.DoubleVar(value=0.05)
        ttk.Entry(
            top,
            textvariable=self.singularity_ratio_threshold,
            width=7
        ).pack(side="left")

        ttk.Button(
            top,
            text="Load Pair",
            command=self.load_selected_pair_graphs
        ).pack(side="left",padx=6)

        ttk.Button(
            top,
            text="Refresh",
            command=self.refresh_singularity_analysis
        ).pack(side="left",padx=3)

        ttk.Button(
            top,
            text="Export CSV",
            command=self.export_singularity_csv
        ).pack(side="left",padx=3)

        ttk.Button(
            top,
            text="Save Figure",
            command=lambda:self.save_figure_generic(
                self.singularity_fig,
                "snake_singularity_analysis.png"
            )
        ).pack(side="left",padx=3)

        self.singularity_fig=Figure(figsize=(12.8,8.3),dpi=100)
        self.singularity_axes=[
            self.singularity_fig.add_subplot(231),
            self.singularity_fig.add_subplot(232),
            self.singularity_fig.add_subplot(233),
            self.singularity_fig.add_subplot(234),
            self.singularity_fig.add_subplot(235),
            self.singularity_fig.add_subplot(236)
        ]
        self.singularity_fig.subplots_adjust(
            left=0.07,right=0.98,bottom=0.08,top=0.88,
            hspace=0.48,wspace=0.30
        )

        self.singularity_canvas=FigureCanvasTkAgg(
            self.singularity_fig,
            master=t
        )
        self.singularity_canvas.get_tk_widget().pack(
            fill="both",expand=True,padx=8,pady=8
        )
        NavigationToolbar2Tk(
            self.singularity_canvas,
            t
        ).update()

        self.singularity_summary=tk.StringVar(
            value="Run/plan a trajectory, then refresh singularity analysis."
        )
        ttk.Label(
            t,
            textvariable=self.singularity_summary,
            style="Mono.TLabel"
        ).pack(fill="x",padx=10,pady=(0,8))

    def _workspace_ui(self):
        t=self.workspace_tab

        top=ttk.Frame(t,padding=8)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="Kinematic Workspace Analysis",
            style="Title.TLabel"
        ).pack(side="left")

        ttk.Label(top,text="ML:").pack(side="left",padx=(14,2))
        self.workspace_ml_combo=ttk.Combobox(
            top,
            textvariable=self.eval_ml_var,
            state="readonly",
            width=15
        )
        self.workspace_ml_combo.pack(side="left")

        ttk.Label(top,text="Planner:").pack(side="left",padx=(8,2))
        ttk.Combobox(
            top,
            textvariable=self.eval_planner_var,
            values=["Dijkstra","A*","RRT","RRT*"],
            state="readonly",
            width=9
        ).pack(side="left")

        self.workspace_samples=tk.IntVar(value=6000)
        self.workspace_joint_limit=tk.DoubleVar(value=90.0)
        self.workspace_grid_res=tk.DoubleVar(value=15.0)
        self.workspace_seed=tk.IntVar(value=42)

        ttk.Label(top,text="Samples:").pack(side="left",padx=(12,2))
        ttk.Entry(
            top,
            textvariable=self.workspace_samples,
            width=7
        ).pack(side="left")

        ttk.Label(top,text="Joint limit ±°:").pack(side="left",padx=(8,2))
        ttk.Entry(
            top,
            textvariable=self.workspace_joint_limit,
            width=7
        ).pack(side="left")

        ttk.Label(top,text="Area grid:").pack(side="left",padx=(8,2))
        ttk.Entry(
            top,
            textvariable=self.workspace_grid_res,
            width=7
        ).pack(side="left")

        ttk.Button(
            top,
            text="GENERATE WORKSPACE",
            command=self.refresh_workspace_analysis
        ).pack(side="left",padx=8)

        ttk.Button(
            top,
            text="Export CSV",
            command=self.export_workspace_csv
        ).pack(side="left",padx=3)

        ttk.Button(
            top,
            text="Save Figure",
            command=lambda:self.save_figure_generic(
                self.workspace_fig,
                "snake_workspace_analysis.png"
            )
        ).pack(side="left",padx=3)

        self.workspace_fig=Figure(figsize=(12.9,8.4),dpi=100)

        self.workspace_axes=[
            self.workspace_fig.add_subplot(231),
            self.workspace_fig.add_subplot(232),
            self.workspace_fig.add_subplot(233),
            self.workspace_fig.add_subplot(234),
            self.workspace_fig.add_subplot(235),
            self.workspace_fig.add_subplot(236,projection="polar")
        ]

        self.workspace_fig.subplots_adjust(
            left=0.07,
            right=0.98,
            bottom=0.08,
            top=0.87,
            hspace=0.48,
            wspace=0.32
        )

        self.workspace_canvas=FigureCanvasTkAgg(
            self.workspace_fig,
            master=t
        )
        self.workspace_canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=8,
            pady=8
        )

        NavigationToolbar2Tk(
            self.workspace_canvas,
            t
        ).update()

        self.workspace_summary=tk.StringVar(
            value=(
                "Intrinsic fixed-base FK workspace. "
                "Generate after selecting DOF, link length and sampling limits."
            )
        )
        ttk.Label(
            t,
            textvariable=self.workspace_summary,
            style="Mono.TLabel"
        ).pack(
            fill="x",
            padx=10,
            pady=(0,8)
        )

    def _export_ui(self):
        t=self.export_tab

        top=ttk.Frame(t,padding=12)
        top.pack(fill="x")
        ttk.Label(top,text="Research Export Center",style="Title.TLabel").pack(side="left")

        body=ttk.Frame(t,padding=18)
        body.pack(fill="both",expand=True)

        f=ttk.LabelFrame(body,text="Figures",padding=12)
        f.grid(row=0,column=0,sticky="nsew",padx=8,pady=8)
        ttk.Button(f,text="Save Planner Figure",command=self.save_map_fig).pack(fill="x",pady=3)
        ttk.Button(f,text="Save ML Figure",command=lambda:self.save_figure_generic(
            self.ml_fig,"ml_training_results.png"
        )).pack(fill="x",pady=3)
        ttk.Button(f,text="Save Motion Analysis",command=lambda:self.save_figure_generic(
            self.motion_fig,"motion_analysis.png"
        )).pack(fill="x",pady=3)
        ttk.Button(f,text="Save Head Error/Safety",command=lambda:self.save_figure_generic(
            self.error_fig,"head_error_safety.png"
        )).pack(fill="x",pady=3)
        ttk.Button(f,text="Save Singularity Analysis",command=lambda:self.save_figure_generic(
            self.singularity_fig,"singularity_analysis.png"
        )).pack(fill="x",pady=3)
        ttk.Button(f,text="Save Workspace Analysis",command=lambda:self.save_figure_generic(
            self.workspace_fig,"workspace_analysis.png"
        )).pack(fill="x",pady=3)
        ttk.Button(f,text="Save Planner Comparison",command=self.save_compare_fig).pack(fill="x",pady=3)
        ttk.Button(f,text="SAVE ALL FIGURES",command=self.save_all_research_outputs).pack(fill="x",pady=(10,3))

        f=ttk.LabelFrame(body,text="Data",padding=12)
        f.grid(row=0,column=1,sticky="nsew",padx=8,pady=8)
        ttk.Button(f,text="Path CSV",command=self.save_path_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="Planner Comparison CSV",command=self.save_compare_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="ML Dataset / Metrics CSV",command=self.ml_export_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="Motion Analysis CSV",command=self.export_motion_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="Head Error / Safety CSV",command=self.export_error_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="Singularity CSV",command=self.export_singularity_csv).pack(fill="x",pady=3)
        ttk.Button(f,text="Workspace Samples CSV",command=self.export_workspace_csv).pack(fill="x",pady=3)

        f=ttk.LabelFrame(body,text="Animation",padding=12)
        f.grid(row=0,column=2,sticky="nsew",padx=8,pady=8)
        ttk.Button(f,text="Animate Current Route",command=self.animate_route).pack(fill="x",pady=3)
        ttk.Button(f,text="Save GIF",command=lambda:self.save_anim("gif")).pack(fill="x",pady=3)
        ttk.Button(f,text="Save MP4",command=lambda:self.save_anim("mp4")).pack(fill="x",pady=3)

        for c in range(3):
            body.columnconfigure(c,weight=1)
        body.rowconfigure(0,weight=1)

        self.export_status=tk.StringVar(
            value="Figures: 300 DPI. Data: planner, ML, motion, error, singularity, workspace. Animation: GIF/MP4."
        )
        ttk.Label(t,textvariable=self.export_status,style="Mono.TLabel").pack(
            fill="x",padx=16,pady=(0,12)
        )

    def _compare_ui(self):
        top=ttk.Frame(self.comp_tab,padding=8); top.pack(fill="x")
        ttk.Label(top,text="Algorithm Comparison",style="Title.TLabel").pack(side="left")
        ttk.Label(top,text="ML Algorithm:").pack(side="left",padx=(14,3))
        self.compare_ml_combo=ttk.Combobox(
            top,textvariable=self.eval_ml_var,state="readonly",width=16
        )
        self.compare_ml_combo.pack(side="left")
        ttk.Button(top,text="Run / Refresh",command=self.compare_all).pack(side="left",padx=10)
        ttk.Button(top,text="Save Figure",command=self.save_compare_fig).pack(side="left",padx=3)
        ttk.Button(top,text="Save CSV",command=self.save_compare_csv).pack(side="left",padx=3)

        self.cfig=Figure(figsize=(12,8),dpi=100)
        self.caxes=[self.cfig.add_subplot(221),self.cfig.add_subplot(222),self.cfig.add_subplot(223),self.cfig.add_subplot(224)]
        self.cfig.tight_layout(pad=2)
        self.ccanvas=FigureCanvasTkAgg(self.cfig,master=self.comp_tab); self.ccanvas.get_tk_widget().pack(fill="both",expand=True,padx=8,pady=8)
        NavigationToolbar2Tk(self.ccanvas,self.comp_tab).update()
        self.comp_text=tk.StringVar(value="No comparison yet.")
        ttk.Label(self.comp_tab,textvariable=self.comp_text,style="Mono.TLabel").pack(fill="x",padx=10,pady=(0,8))

    def _help_ui(self):
        tx=tk.Text(self.help_tab,wrap="word",font=("Segoe UI",11),padx=20,pady=20)
        tx.pack(fill="both",expand=True)
        tx.insert("1.0","""SNAKE ROBOT AI RESEARCH STUDIO — COURSE WORKFLOW

1. KINEMATICS / SIMULATION
• Select 6–16 DOF.
• The initial snake appears immediately before planning.
• Conventional robotics frame colours are used everywhere:
  X = Red, Y = Green, Z = Blue.
• Global XYZ frame is shown at the world origin.
• Every joint has a local XYZ frame.
• Because motion is planar, +Z is perpendicular to the screen and is shown
  using a blue circled-dot symbol.
• Set link length and travelling-wave parameters.
• Add Start, D1, D2, ... and obstacles.
• Animate the head-led full snake body.

2. PATH PLANNING
• Dijkstra
• A*
• RRT
• RRT*
• Directional continuity / anti-backtracking
• Full-body clearance
• Dynamic-obstacle online replanning

3. MACHINE LEARNING
The ML subsystem learns:
    robot state -> steering Δheading

Features:
• Goal direction and distance
• Current heading
• Nearest obstacle clearance
• Relative obstacle bearing

Models:
• KNN
• Random Forest
• Extra Trees
• Neural Network (MLP)

Recommended operational sequence:
1. Train/select ML command model FIRST.
2. Select ML Algorithm + Path Planning Algorithm as an evaluation pair.
3. Run "ML COMMAND → PATH PLANNER".
4. The ML model generates the first steering command.
5. The selected global planner then computes the feasible route.

The ML tab also reports a five-class command confusion matrix:
Hard Right / Right / Straight / Left / Hard Left.

4. MOTION ANALYSIS
• Head displacement
• Head velocity
• Head acceleration
• Joint angular displacement
• Joint angular velocity
• Overall X-Y trajectory

5. HEAD ERROR / SAFETY
• Head-to-destination error
• Heading error
• Full-body obstacle clearance

6. DEDICATED SINGULARITY ANALYSIS
• Smallest singular value σmin
• Jacobian condition number κ(J)
• Manipulability
• Planar Jacobian rank
• Normalized singularity ratio σmin/σmax
• Near-singularity threshold and event count

7. WORKSPACE ANALYSIS
• Monte-Carlo FK sampling for the fixed-base head workspace
• Adjustable number of samples and joint-angle limit
• Reachable X-Y workspace scatter
• Workspace occupancy density
• Radial reach distribution
• Singularity ratio mapped across workspace
• Manipulability versus radial reach
• Angular maximum/minimum reach envelope
• Empirical workspace area and theoretical maximum reach
• Workspace CSV and 300-DPI figure export

Manual destination entry is available using explicit X,Y coordinates.


6. EXPORT
• 300-DPI graphs
• CSV data
• Saved ML model
• GIF / MP4 animation
""")
        tx.configure(state="disabled")

    # ---------- state / editing ----------
    def bounds(self):
        try:
            b=(float(self.xmin.get()),float(self.xmax.get()),float(self.ymin.get()),float(self.ymax.get()))
            if b[1] <= b[0] or b[3] <= b[2]: raise ValueError
            return b
        except:
            messagebox.showerror("Bounds","Invalid workspace bounds.")
            return (-200,900,-450,450)

    def apply_robot(self):
        try:
            d=int(self.dof_var.get()); L=float(self.link_var.get())
            if not DOF_MIN <= d <= DOF_MAX or L <= 0: raise ValueError
        except:
            messagebox.showerror("Robot","DOF must be 6–16 and link length positive."); return
        self.dof=d
        self.robot.configure(d,L)
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.workspace_cache=None
        self.invalidate_ml_training("robot DOF/link geometry changed")
        self.redraw()
        self.status.set(f"Robot: {d} DOF — route and ML training cleared")

    def add_manual_destination(self):
        """Add an ordered head destination using typed world X/Y coordinates."""
        try:
            x=float(self.manual_dest_x.get())
            y=float(self.manual_dest_y.get())
        except Exception:
            messagebox.showerror(
                "Manual Destination",
                "Enter valid numeric X and Y destination coordinates."
            )
            return

        p=np.array([x,y],dtype=float)

        if not inside_bounds(p,self.bounds()):
            messagebox.showerror(
                "Manual Destination",
                "The entered destination lies outside the current workspace."
            )
            return

        self.destinations.append(p)

        # A destination change invalidates all route-dependent state.
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.active_seq=None
        self.active_index=0
        self.active_head_trail=[]
        self.active_body=None

        self.invalidate_ml_training(
            "manual destination added"
        )
        self.refresh_dest()
        self.redraw()

        self.status.set(
            f"Destination D{len(self.destinations)} added manually: "
            f"X={x:.2f}, Y={y:.2f}"
        )

    def add_manual_static_obstacle(self,shape):
        """Add a fixed obstacle using explicitly entered world X/Y coordinates."""
        try:
            x=float(self.manual_obs_x.get())
            y=float(self.manual_obs_y.get())
        except Exception:
            messagebox.showerror(
                "Manual Obstacle",
                "Enter valid numeric X and Y coordinates."
            )
            return

        p=np.array([x,y],dtype=float)

        if not inside_bounds(p,self.bounds()):
            messagebox.showerror(
                "Manual Obstacle",
                "The entered obstacle center lies outside the current workspace."
            )
            return

        try:
            if shape=="circle":
                r=max(float(self.circle_r.get()),1.0)
                obstacle={
                    "type":"circle",
                    "x":x,"y":y,"r":r
                }
            else:
                w=max(float(self.rect_w.get()),1.0)
                h=max(float(self.rect_h.get()),1.0)
                obstacle={
                    "type":"rect",
                    "x":x,"y":y,
                    "w":w,"h":h
                }
        except Exception:
            messagebox.showerror(
                "Manual Obstacle",
                "Check the obstacle radius / width / height values."
            )
            return

        self.obstacles.append(obstacle)

        # Environment changed: clear path-derived state and stale ML.
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.active_seq=None
        self.active_index=0
        self.active_head_trail=[]
        self.active_body=None

        self.invalidate_ml_training(
            "manual static obstacle added"
        )
        self.redraw()

        self.status.set(
            f"Static {shape} obstacle added manually at X={x:.2f}, Y={y:.2f}"
        )

    def add_manual_dynamic_obstacle(self):
        """
        Insert a sudden obstacle using typed X/Y coordinates.

        During animation this uses the same GREEN/YELLOW/RED online-replanning
        logic as mouse insertion.  If the robot is not moving, the obstacle is
        added to the environment but no online replan is triggered.
        """
        try:
            x=float(self.dynamic_x.get())
            y=float(self.dynamic_y.get())
        except Exception:
            messagebox.showerror(
                "Dynamic Obstacle",
                "Enter valid numeric dynamic-obstacle X and Y coordinates."
            )
            return

        p=np.array([x,y],dtype=float)

        if not inside_bounds(p,self.bounds()):
            messagebox.showerror(
                "Dynamic Obstacle",
                "The entered dynamic-obstacle center lies outside the workspace."
            )
            return

        if self.animating:
            self.handle_dynamic_obstacle_click(p)
            return

        # Allow the user to pre-position a 'dynamic' obstacle even while idle.
        # It then behaves as a normal fixed obstacle until motion begins.
        obstacle=self.create_dynamic_obstacle(p)
        obstacle["appear_frame"]=0
        self.obstacles.append(obstacle)

        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.invalidate_ml_training(
            "manual dynamic obstacle inserted while idle"
        )
        self.redraw()

        self.dynamic_state.set(
            "IDLE — obstacle added; start/re-plan mission"
        )
        self.status.set(
            f"Dynamic {self.dynamic_shape.get()} obstacle entered manually "
            f"at X={x:.2f}, Y={y:.2f}"
        )

    def map_click(self,e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return

        p=np.array([e.xdata,e.ydata],float)
        if not inside_bounds(p,self.bounds()):
            return

        # During live motion, dynamic-obstacle mode takes priority over the
        # ordinary map-editing mode.
        if (
            hasattr(self,"dynamic_mode")
            and self.dynamic_mode.get()
            and self.animating
        ):
            self.handle_dynamic_obstacle_click(p)
            return

        m=self.mode.get()
        if m=="Start":
            self.start=p
            self.mission_original_start=p.copy()
        elif m=="Destination":
            self.destinations.append(p)
        elif m=="Circle Obstacle":
            r=max(float(self.circle_r.get()),1.0)
            self.obstacles.append({"type":"circle","x":p[0],"y":p[1],"r":r})
        else:
            w=max(float(self.rect_w.get()),1.0)
            h=max(float(self.rect_h.get()),1.0)
            self.obstacles.append({"type":"rect","x":p[0],"y":p[1],"w":w,"h":h})

        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.active_seq=None
        self.active_index=0
        self.active_head_trail=[]
        self.active_body=None
        self.invalidate_ml_training("map/start/destination/obstacle changed")
        self.refresh_dest()
        self.redraw()

    def refresh_dest(self):
        self.dest_list.delete(0,"end")
        for i,p in enumerate(self.destinations,1):
            self.dest_list.insert("end",f"D{i:02d}: X={p[0]:8.2f}  Y={p[1]:8.2f}")

    def move_dest(self,d):
        sel=self.dest_list.curselection()
        if not sel:return
        i=int(sel[0]); j=i+d
        if 0<=j<len(self.destinations):
            self.destinations[i],self.destinations[j]=self.destinations[j],self.destinations[i]
            self.refresh_dest()
            self.dest_list.selection_set(j)
            self.current_path=None
            self.current_segments=[]
            self.invalidate_ml_training("destination order changed")
            self.redraw()

    def remove_dest(self):
        sel=self.dest_list.curselection()
        if sel:
            del self.destinations[int(sel[0])]
            self.refresh_dest()
            self.current_path=None
            self.current_segments=[]
            self.invalidate_ml_training("destination removed")
            self.redraw()

    def clear_dests(self):
        self.destinations.clear()
        self.refresh_dest()
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.invalidate_ml_training("destinations cleared")
        self.redraw()
    def clear_obs(self):
        self.obstacles.clear()
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.invalidate_ml_training("obstacles cleared")
        self.redraw()
    def clear_all(self):
        self.start=np.array([0.,0.]); self.destinations.clear(); self.obstacles.clear()
        self.current_path=None
        self.current_segments=[]
        self.current_result=None
        self.compare_results={}
        self.invalidate_ml_training("entire map cleared")
        self.refresh_dest()
        self.redraw()
        self.draw_compare()

    # ---------- AI / research analysis helpers ----------
    def ml_environment_signature(self):
        """
        Signature used for ML-FIRST training.
        It depends on the robot and current environment/destinations, but not
        on a path that has not yet been planned.
        """
        def rp(v):
            return round(float(v),4)

        obs=[]
        for o in self.obstacles:
            if o["type"]=="circle":
                obs.append(("c",rp(o["x"]),rp(o["y"]),rp(o["r"])))
            else:
                obs.append(("r",rp(o["x"]),rp(o["y"]),rp(o["w"]),rp(o["h"])))

        return (
            int(self.dof),
            rp(self.robot.link),
            (rp(self.start[0]),rp(self.start[1])),
            tuple((rp(p[0]),rp(p[1])) for p in self.destinations),
            tuple(obs)
        )

    def ml_signature_matches_current(self):
        if self.ml_training_signature is None:
            return False
        if self.ml_training_scope=="environment":
            return self.ml_training_signature == self.ml_environment_signature()
        return self.ml_training_signature == self.ml_current_signature()

    def graph_pair_text(self,pair=None):
        """Compact pair text used in multi-panel figures."""
        p=pair or self.pair_label()
        return str(p).replace("Neural Network","NN").replace("Random Forest","RF").replace("Extra Trees","ET")

    def panel_title(self,metric,pair=None):
        """
        Two-line title prevents neighbouring subplot headings from merging.
        Line 1 = compact evaluation pair; Line 2 = metric.
        """
        return f"{self.graph_pair_text(pair)}\n{metric}"

    def set_panel_title(self,ax,metric,pair=None,fontsize=8.2):
        ax.set_title(
            self.panel_title(metric,pair),
            fontsize=fontsize,
            pad=7,
            fontweight="semibold",
            wrap=True
        )

    def set_figure_pair_title(self,fig,label="Evaluation Pair",pair=None,fontsize=10.5):
        fig.suptitle(
            f"{label}: {self.graph_pair_text(pair)}",
            fontsize=fontsize,
            fontweight="semibold",
            y=0.985
        )

    def pair_label(self, planner=None, ml=None):
        planner = planner or self.active_pair_planner or self.eval_planner_var.get() or self.alg.get()
        ml = ml or self.active_pair_ml or self.eval_ml_var.get() or self.ml_model_var.get() or "Untrained ML"
        return f"{planner} + {ml}"

    @staticmethod
    def command_class_index(delta_deg):
        a=float(delta_deg)
        if a <= -35: return 0
        if a < -8: return 1
        if a <= 8: return 2
        if a < 35: return 3
        return 4

    @staticmethod
    def command_class_labels():
        return ["Hard Right","Right","Straight","Left","Hard Left"]

    def selected_pair_key(self):
        return (self.eval_planner_var.get(), self.eval_ml_var.get())

    def ml_current_signature(self):
        """
        Signature of the CURRENT planning/training scenario.
        Any change in robot geometry, environment, ordered destinations,
        or planned reference path creates a different signature.
        """
        def rp(v):
            return round(float(v),4)

        obs=[]
        for o in self.obstacles:
            if o["type"]=="circle":
                obs.append(("c",rp(o["x"]),rp(o["y"]),rp(o["r"])))
            else:
                obs.append(("r",rp(o["x"]),rp(o["y"]),rp(o["w"]),rp(o["h"])))

        path_sig=()
        if self.current_path is not None:
            arr=np.asarray(self.current_path,float)
            # Enough precision to detect a changed path without generating a
            # huge status string.
            path_sig=tuple(
                (rp(p[0]),rp(p[1]))
                for p in arr
            )

        return (
            int(self.dof),
            rp(self.robot.link),
            (rp(self.start[0]),rp(self.start[1])),
            tuple((rp(p[0]),rp(p[1])) for p in self.destinations),
            tuple(obs),
            str(self.current_algorithm),
            path_sig
        )

    def ml_signature_id(self,signature=None):
        if signature is None:
            signature=self.ml_current_signature()
        return f"S{abs(hash(signature)) % 100000000:08d}"

    def invalidate_ml_training(self,reason="",clear_models=True):
        """
        Clear stale learned data whenever the planning scenario changes.
        This prevents a new path from being trained/executed with samples
        retained from an older map or destination sequence.
        """
        self.ml_X=None
        self.ml_y=None
        self.ml_dataset_signature=None
        self.ml_training_signature=None
        self.ml_dataset_source_text="No fresh ML dataset"
        self.ml_last_invalidation_reason=str(reason)

        if clear_models:
            self.ml_models.clear()
            self.ml_metrics.clear()
            self.ml_predictions.clear()
            self.ml_confusions.clear()
            self.ml_model_choice_name=""
            if hasattr(self,"ml_model_combo"):
                self.ml_model_combo["values"]=[]
            if hasattr(self,"ml_model_var"):
                self.ml_model_var.set("")

        if hasattr(self,"ml_command_text"):
            self.ml_command_text.set("ML command: training invalidated — generate/train current route")
        if hasattr(self,"ml_status"):
            self.ml_status.set(
                "ML data cleared because the planning scenario changed."
                + (f" Reason: {reason}" if reason else "")
            )
        if hasattr(self,"ml_axes"):
            try:self.ml_draw_empty()
            except Exception:pass

    def ml_require(self):
        if ML_AVAILABLE:
            return True
        messagebox.showerror(
            "Machine Learning",
            "Install ML packages:\n\npip install scikit-learn joblib\n\n"
            f"Import error: {ML_IMPORT_ERROR}"
        )
        return False

    @staticmethod
    def command_label(delta_deg):
        a=float(delta_deg)
        if a <= -35: return "HARD RIGHT"
        if a < -8: return "RIGHT"
        if a >= 35: return "HARD LEFT"
        if a > 8: return "LEFT"
        return "FORWARD / STRAIGHT"

    def nearest_obstacle_features(self,p,heading_deg):
        p=np.asarray(p,float)
        if not self.obstacles:
            span=max(self.bounds()[1]-self.bounds()[0],self.bounds()[3]-self.bounds()[2],1.0)
            return span,0.0

        best=float("inf"); best_center=None
        for o in self.obstacles:
            center=np.array([o["x"],o["y"]],float)
            if o["type"]=="circle":
                surface=max(dist(p,center)-o["r"],0.0)
            else:
                dx=max(abs(p[0]-o["x"])-o["w"]/2,0.0)
                dy=max(abs(p[1]-o["y"])-o["h"]/2,0.0)
                surface=math.hypot(dx,dy)
            if surface<best:
                best=surface; best_center=center

        bearing=0.0
        if best_center is not None:
            bh=vector_heading_deg(best_center-p)
            if bh is not None:
                bearing=angle_diff_deg(bh,heading_deg)
        return float(best),float(bearing)

    def ml_feature(self,p,heading_deg,target):
        p=np.asarray(p,float); target=np.asarray(target,float)
        dx,dy=(target-p); d=math.hypot(dx,dy)
        clearance,bearing=self.nearest_obstacle_features(p,heading_deg)
        span=max(self.bounds()[1]-self.bounds()[0],self.bounds()[3]-self.bounds()[2],1.0)
        h=math.radians(float(heading_deg)); b=math.radians(float(bearing))
        return np.array([
            dx/span,dy/span,d/span,
            math.sin(h),math.cos(h),
            min(clearance/span,2.0),
            math.sin(b),math.cos(b)
        ],dtype=float)

    def ml_collect_routes(self):
        """
        CURRENT-SCENARIO training source.

        Default:
            only the currently planned route/segments.

        Optional:
            include successful routes from Compare All, but only from the
            current map/destination scenario.
        """
        routes=[]

        # Primary source = current selected planner route.
        if self.current_segments:
            for seg in self.current_segments:
                arr=np.asarray(seg,float)
                if len(arr)>=2:
                    routes.append(arr)

        # Fallback when segment bookkeeping is absent.
        if not routes and self.current_path is not None:
            arr=np.asarray(self.current_path,float)
            if len(arr)>=2:
                routes.append(arr)

        # Optional augmentation with current comparison routes.
        if hasattr(self,"ml_include_compare") and self.ml_include_compare.get():
            for r in self.compare_results.values():
                if not r or not r.get("success"):
                    continue
                segs=r.get("segments",[])
                if segs:
                    for seg in segs:
                        arr=np.asarray(seg,float)
                        if len(arr)>=2:
                            routes.append(arr)
                elif r.get("path") is not None:
                    arr=np.asarray(r["path"],float)
                    if len(arr)>=2:
                        routes.append(arr)

        return routes

    def ml_generate_dataset(self):
        if not self.ml_require():
            return

        source=self.ml_training_source.get()

        # --------------------------------------------------------
        # ML-FIRST mode: no planned path is required.
        # Ideal steering labels are generated from goal geometry plus
        # a simple obstacle-repulsion term.
        # --------------------------------------------------------
        if source.startswith("ML First"):
            if not self.destinations:
                messagebox.showinfo(
                    "ML Training",
                    "Add at least one destination first. No path planning is "
                    "required for ML-FIRST training."
                )
                return

            try:
                n=max(int(self.ml_synthetic_samples.get()),500)
            except Exception:
                n=3500

            rng=np.random.default_rng(42)
            xmin,xmax,ymin,ymax=self.bounds()
            effective=self.effective_planning_clearance()

            X=[]
            y=[]
            attempts=0
            max_attempts=n*12

            while len(X)<n and attempts<max_attempts:
                attempts+=1
                p=np.array([
                    rng.uniform(xmin,xmax),
                    rng.uniform(ymin,ymax)
                ],float)

                if point_collision(p,self.obstacles,effective):
                    continue

                target=np.asarray(
                    self.destinations[
                        int(rng.integers(0,len(self.destinations)))
                    ],
                    float
                )

                if dist(p,target)<max(float(self.robot.link),10.0):
                    continue

                heading=float(rng.uniform(-180.0,180.0))
                goal_heading=vector_heading_deg(target-p)
                if goal_heading is None:
                    continue

                goal_turn=angle_diff_deg(goal_heading,heading)
                clearance,bearing=self.nearest_obstacle_features(p,heading)

                # Obstacle avoidance component: if an obstacle is near and
                # approximately in front, turn away from it.
                safe_range=max(
                    3.0*float(self.robot.link),
                    2.5*effective
                )
                avoidance=0.0
                if clearance<safe_range and abs(bearing)<85.0:
                    proximity=1.0-clearance/max(safe_range,1e-9)
                    away_sign=-1.0 if bearing>=0 else 1.0
                    avoidance=away_sign*(25.0+50.0*proximity)

                ideal=float(np.clip(
                    0.88*goal_turn+avoidance,
                    -100.0,100.0
                ))

                X.append(self.ml_feature(p,heading,target))
                y.append(ideal)

            if len(X)<500:
                messagebox.showwarning(
                    "ML Training",
                    f"Only {len(X)} safe synthetic samples could be generated."
                )
                return

            self.ml_X=np.asarray(X,float)
            self.ml_y=np.asarray(y,float)
            self.ml_training_scope="environment"
            self.ml_dataset_signature=self.ml_environment_signature()
            self.ml_dataset_source_text=(
                f"ML-FIRST synthetic current environment | "
                f"scenario={self.ml_signature_id(self.ml_dataset_signature)} | "
                f"samples={len(self.ml_X):,}"
            )

        # --------------------------------------------------------
        # Optional imitation mode: learn from an already planned route.
        # --------------------------------------------------------
        else:
            if self.current_path is None:
                messagebox.showinfo(
                    "ML Training",
                    "Reference-route imitation requires a current planned route. "
                    "Use ML-FIRST Synthetic Commands if you want ML before planning."
                )
                return

            routes=self.ml_collect_routes()
            if not routes:
                messagebox.showinfo(
                    "ML Training",
                    "No usable current planner route was found."
                )
                return

            try:
                lookahead=max(int(self.ml_lookahead_steps.get()),1)
            except Exception:
                lookahead=5

            rng=np.random.default_rng(42)
            X=[]
            y=[]
            accepted_routes=0

            for route in routes:
                route=np.asarray(route,float)
                if len(route)<2:
                    continue

                route_len=route_length(route)
                if route_len<=1e-9:
                    continue

                spacing=max(float(self.anim_spacing.get()),6.0)
                desired_points=max(
                    int(route_len/max(spacing,1.0))+1,
                    10
                )
                dense=resample(
                    route,
                    max(route_len/max(desired_points-1,1),1.0),
                    1000
                )

                if len(dense)<5:
                    u=np.linspace(0.0,1.0,10)
                    dense=np.asarray([
                        route[0]+s*(route[-1]-route[0])
                        for s in u
                    ],float)

                accepted_routes+=1

                for i in range(1,len(dense)-1):
                    prev,cur,nxt=dense[i-1],dense[i],dense[i+1]
                    heading=vector_heading_deg(cur-prev)
                    next_heading=vector_heading_deg(nxt-cur)
                    if heading is None or next_heading is None:
                        continue

                    ref_idx=min(i+lookahead,len(dense)-1)
                    local_target=dense[ref_idx]
                    steer=float(np.clip(
                        angle_diff_deg(next_heading,heading),
                        -100.0,100.0
                    ))
                    feat=self.ml_feature(cur,heading,local_target)

                    X.append(feat)
                    y.append(steer)

                    for _ in range(4):
                        noisy=feat.copy()
                        noisy[0:3]+=rng.normal(0,0.010,3)
                        noisy[5]+=rng.normal(0,0.006)
                        X.append(noisy)
                        y.append(float(np.clip(
                            steer+rng.normal(0,1.25),
                            -100.0,100.0
                        )))

            if len(X)<20:
                messagebox.showwarning(
                    "ML Training",
                    f"Only {len(X)} imitation samples were produced."
                )
                return

            self.ml_X=np.asarray(X,float)
            self.ml_y=np.asarray(y,float)
            self.ml_training_scope="route"
            self.ml_dataset_signature=self.ml_current_signature()
            self.ml_dataset_source_text=(
                f"Reference-route imitation | "
                f"scenario={self.ml_signature_id(self.ml_dataset_signature)} | "
                f"routes={accepted_routes} | samples={len(self.ml_X):,}"
            )

        self.ml_models.clear()
        self.ml_metrics.clear()
        self.ml_predictions.clear()
        self.ml_confusions.clear()
        self.ml_training_signature=None
        self.ml_model_combo["values"]=[]
        self.eval_ml_combo["values"]=[]
        self.motion_ml_combo["values"]=[]
        self.error_ml_combo["values"]=[]
        self.ml_model_var.set("")

        self.ml_status.set(
            f"FRESH dataset ready: {self.ml_dataset_source_text}"
        )
        self.ml_plot_results()

    def ml_build_model(self,name):
        if name=="KNN":
            return Pipeline([("scale",StandardScaler()),("model",KNeighborsRegressor(n_neighbors=7,weights="distance"))])
        if name=="Random Forest":
            return RandomForestRegressor(n_estimators=160,max_depth=24,random_state=42,n_jobs=-1)
        if name=="Extra Trees":
            return ExtraTreesRegressor(n_estimators=160,random_state=42,n_jobs=-1)
        if name=="Neural Network":
            return Pipeline([("scale",StandardScaler()),("model",MLPRegressor(
                hidden_layer_sizes=(96,96),max_iter=500,early_stopping=True,
                validation_fraction=0.15,n_iter_no_change=20,random_state=42
            ))])
        raise ValueError(name)

    def ml_train_models(self):
        if not self.ml_require():
            return

        # Always rebuild fresh data using the chosen source.
        self.ml_generate_dataset()
        if self.ml_X is None:
            return

        expected_sig=(
            self.ml_environment_signature()
            if self.ml_training_scope=="environment"
            else self.ml_current_signature()
        )
        if self.ml_dataset_signature != expected_sig:
            messagebox.showerror(
                "ML Training",
                "Scenario changed while the dataset was being prepared."
            )
            return

        selected=[n for n,v in self.ml_algo_vars.items() if v.get()]
        if not selected:
            messagebox.showinfo("ML Training","Select at least one ML algorithm.")
            return

        try:
            test_pct=float(self.ml_test_pct.get())
            if not 5<=test_pct<=50:
                raise ValueError
        except Exception:
            messagebox.showerror("ML Training","Test percentage must be 5–50.")
            return

        Xtr,Xte,ytr,yte=train_test_split(
            self.ml_X,self.ml_y,
            test_size=test_pct/100.0,
            random_state=42
        )

        self.ml_models.clear()
        self.ml_metrics.clear()
        self.ml_predictions.clear()
        self.ml_confusions.clear()

        true_cls=np.asarray([
            self.command_class_index(v) for v in yte
        ],dtype=int)

        for k,name in enumerate(selected,1):
            self.status.set(
                f"Training ML-FIRST/current model: {name} ({k}/{len(selected)})..."
            )
            self.update_idletasks()

            model=self.ml_build_model(name)
            model.fit(Xtr,ytr)
            pred=np.asarray(model.predict(Xte),float)
            pred_cls=np.asarray([
                self.command_class_index(v) for v in pred
            ],dtype=int)

            self.ml_models[name]=model
            self.ml_predictions[name]=(yte.copy(),pred.copy())
            self.ml_confusions[name]=confusion_matrix(
                true_cls,pred_cls,labels=[0,1,2,3,4]
            )
            self.ml_metrics[name]={
                "rmse_deg":float(np.sqrt(mean_squared_error(yte,pred))),
                "mae_deg":float(mean_absolute_error(yte,pred)),
                "r2":float(r2_score(yte,pred)),
                "command_accuracy":float(accuracy_score(true_cls,pred_cls))
            }

        best=min(
            self.ml_models,
            key=lambda n:self.ml_metrics[n]["rmse_deg"]
        )
        self.ml_model_choice_name=best
        values=list(self.ml_models.keys())
        self.ml_model_combo["values"]=values
        self.eval_ml_combo["values"]=values
        self.motion_ml_combo["values"]=values
        self.error_ml_combo["values"]=values
        self.singularity_ml_combo["values"]=values
        self.workspace_ml_combo["values"]=values
        self.compare_ml_combo["values"]=values

        self.ml_model_var.set(best)
        self.eval_ml_var.set(best)
        self.ml_training_signature=expected_sig

        self.ml_plot_results()

        m=self.ml_metrics[best]
        self.ml_status.set(
            f"Training complete | Source={self.ml_training_source.get()} | "
            f"Best={best} | RMSE={m['rmse_deg']:.3f}° | "
            f"R²={m['r2']:.4f} | Command accuracy={100*m['command_accuracy']:.2f}%"
        )
        self.status.set(
            "ML model READY. Next: select ML + Path Planning pair, then run "
            "ML command FIRST → path planner SECOND."
        )

    def ml_draw_empty(self):
        pair=self.pair_label()
        titles=[
            "Steering RMSE","Model R²","Predicted vs True Steering",
            "Steering Error Distribution","Training Command Distribution",
            "Command Confusion Matrix","ML Route vs Reference","Command Accuracy"
        ]
        for ax,title in zip(self.ml_axes,titles):
            ax.clear()
            self.set_panel_title(ax,title,pair=pair,fontsize=7.6)
            ax.grid(True,alpha=.3)
        self.set_figure_pair_title(self.ml_fig,label="Evaluation Pair",pair=pair,fontsize=10.5)
        self.ml_canvas.draw_idle()

    def full_executed_head_path(self):
        """
        Authoritative ACTUAL trajectory for every analysis graph.

        During/after a live mission this comes from runtime_body_log, which
        persists across dynamic replans.  It therefore contains:
        original motion + obstacle event + replanned motion + final motion.
        """
        if len(self.runtime_body_log)>=1:
            frames=np.asarray(self.runtime_body_log,dtype=float)
            return frames[:,-1,:].copy()

        if self.ml_last_route is not None:
            return np.asarray(self.ml_last_route,dtype=float).copy()

        if self.current_path is not None:
            return np.asarray(self.current_path,dtype=float).copy()

        return None

    def full_reference_path(self):
        """
        Authoritative ORIGINAL reference trajectory.

        Dynamic replanning replaces self.current_path with only the remaining
        route, so analysis must NOT use current_path once a mission has started.
        """
        if self.mission_reference_path is not None:
            return np.asarray(self.mission_reference_path,dtype=float).copy()

        if self.ml_reference_path is not None:
            return np.asarray(self.ml_reference_path,dtype=float).copy()

        if self.current_path is not None:
            return np.asarray(self.current_path,dtype=float).copy()

        return None

    def full_mission_destinations(self):
        if self.runtime_mission_started and self.mission_original_destinations:
            return [
                np.asarray(p,dtype=float).copy()
                for p in self.mission_original_destinations
            ]
        return [
            np.asarray(p,dtype=float).copy()
            for p in self.destinations
        ]

    def ml_plot_results(self):
        self.ml_draw_empty()
        a1,a2,a3,a4,a5,a6,a7,a8=self.ml_axes
        pair=self.pair_label()

        if self.ml_y is not None:
            a5.clear()
            a5.hist(self.ml_y,bins=35)
            self.set_panel_title(a5,"Training Command Distribution",pair=pair,fontsize=7.6)
            a5.set_xlabel("Δheading (deg)")
            a5.set_ylabel("Count")
            a5.grid(True,axis="y",alpha=.3)

        if self.ml_metrics:
            names=list(self.ml_metrics.keys())
            short=[
                {"Random Forest":"RF","Extra Trees":"ET","Neural Network":"MLP"}.get(n,n)
                for n in names
            ]

            a1.clear()
            a1.bar(short,[self.ml_metrics[n]["rmse_deg"] for n in names])
            self.set_panel_title(a1,"Steering RMSE",pair=pair,fontsize=7.6)
            a1.set_ylabel("Degrees")
            a1.grid(True,axis="y",alpha=.3)

            a2.clear()
            a2.bar(short,[self.ml_metrics[n]["r2"] for n in names])
            self.set_panel_title(a2,"Model R²",pair=pair,fontsize=7.6)
            a2.set_ylabel("R²")
            a2.grid(True,axis="y",alpha=.3)

            chosen=self.eval_ml_var.get()
            if chosen not in self.ml_predictions:
                chosen=self.ml_model_var.get()
            if chosen not in self.ml_predictions:
                chosen=names[0]

            yt,yp=self.ml_predictions[chosen]

            a3.clear()
            a3.scatter(yt,yp,s=7,alpha=.25)
            lo=min(float(np.min(yt)),float(np.min(yp)))
            hi=max(float(np.max(yt)),float(np.max(yp)))
            a3.plot([lo,hi],[lo,hi],"--",lw=1)
            self.set_panel_title(a3,"Predicted vs True",pair=f"{self.eval_planner_var.get()} + {chosen}",fontsize=7.6)
            a3.set_xlabel("True Δheading")
            a3.set_ylabel("Predicted Δheading")
            a3.grid(True,alpha=.3)

            a4.clear()
            a4.hist(yp-yt,bins=35)
            self.set_panel_title(a4,"Steering Error",pair=f"{self.eval_planner_var.get()} + {chosen}",fontsize=7.6)
            a4.set_xlabel("Prediction error (deg)")
            a4.set_ylabel("Count")
            a4.grid(True,axis="y",alpha=.3)

            # Confusion matrix from discretized steering commands.
            a6.clear()
            cm=self.ml_confusions.get(chosen)
            if cm is not None:
                im=a6.imshow(cm,aspect="auto")
                labels=self.command_class_labels()
                a6.set_xticks(range(5))
                a6.set_yticks(range(5))
                a6.set_xticklabels(labels,rotation=35,ha="right",fontsize=7)
                a6.set_yticklabels(labels,fontsize=7)
                a6.set_xlabel("Predicted command")
                a6.set_ylabel("True command")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        a6.text(j,i,str(int(cm[i,j])),ha="center",va="center",fontsize=8)
            self.set_panel_title(a6,"Command Confusion Matrix",pair=f"{self.eval_planner_var.get()} + {chosen}",fontsize=7.4)

            a8.clear()
            a8.bar(
                short,
                [100*self.ml_metrics[n]["command_accuracy"] for n in names]
            )
            self.set_panel_title(a8,"Command Classification Accuracy",pair=pair,fontsize=7.4)
            a8.set_ylabel("Accuracy (%)")
            a8.grid(True,axis="y",alpha=.3)

        a7.clear()

        # Show the environment directly in the ML trajectory panel so the user
        # can interpret the route relative to static and dynamic obstacles.
        plotted_static=False
        plotted_dynamic=False
        for o in self.obstacles:
            is_dynamic=bool(o.get("dynamic",False))
            if o["type"]=="circle":
                patch=Circle(
                    (o["x"],o["y"]),
                    o["r"],
                    alpha=.22 if not is_dynamic else .32,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=1.8 if is_dynamic else None,
                    label=(
                        "Dynamic obstacle"
                        if is_dynamic and not plotted_dynamic
                        else ("Static obstacle" if (not is_dynamic and not plotted_static) else None)
                    )
                )
            else:
                patch=Rectangle(
                    (o["x"]-o["w"]/2,o["y"]-o["h"]/2),
                    o["w"],o["h"],
                    alpha=.22 if not is_dynamic else .32,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=1.8 if is_dynamic else None,
                    label=(
                        "Dynamic obstacle"
                        if is_dynamic and not plotted_dynamic
                        else ("Static obstacle" if (not is_dynamic and not plotted_static) else None)
                    )
                )
            a7.add_patch(patch)
            if is_dynamic:
                plotted_dynamic=True
            else:
                plotted_static=True

        rp=self.full_reference_path()
        ep=self.full_executed_head_path()

        if rp is not None and len(rp)>=2:
            a7.plot(
                rp[:,0],rp[:,1],
                lw=1.6,
                ls="--",
                label="Original reference path"
            )

        if ep is not None and len(ep)>=2:
            a7.plot(
                ep[:,0],ep[:,1],
                lw=1.9,
                label=(
                    "Full executed dynamic path"
                    if len(self.runtime_body_log)>=2
                    else "ML / hybrid route"
                )
            )

        self.set_panel_title(a7,"Overall Trajectory with Dynamic Obstacles",pair=pair,fontsize=7.2)
        a7.set_xlabel("World X")
        a7.set_ylabel("World Y")
        a7.set_aspect("equal",adjustable="datalim")
        a7.grid(True,alpha=.3)
        a7.relim()
        a7.autoscale_view()

        if rp is not None or ep is not None or self.obstacles:
            a7.legend(fontsize=7.0,loc="best")

        self.set_figure_pair_title(self.ml_fig,label="Evaluation Pair",pair=pair,fontsize=10.5)
        self.ml_fig.subplots_adjust(left=0.06,right=0.99,bottom=0.08,top=0.87,hspace=0.52,wspace=0.36)
        self.ml_canvas.draw_idle()

    def current_ml_state(self):
        if self.active_body is not None:
            p=np.asarray(self.active_body[-1],float); heading=self.current_runtime_heading()
        else:
            p=np.asarray(self.start,float); heading=None
        if heading is None:
            heading=vector_heading_deg(np.asarray(self.destinations[0])-p) if self.destinations else 0.0
        if heading is None: heading=0.0
        reached=self.reached_destination_count() if self.active_seq is not None else 0
        target=np.asarray(self.destinations[min(reached,len(self.destinations)-1)],float) if self.destinations else p.copy()
        return p,float(heading),target

    def ml_predict_current_command(self):
        if not self.ml_models:
            messagebox.showinfo(
                "ML Command",
                "Train the current route first."
            )
            return

        if not self.ml_signature_matches_current():
            messagebox.showwarning(
                "ML Command — Stale Model",
                "The current path/environment is different from the scenario "
                "used to train this model. Train again before predicting."
            )
            return

        name=self.ml_model_var.get()
        if name not in self.ml_models:
            name=min(
                self.ml_models,
                key=lambda n:self.ml_metrics[n]["rmse_deg"]
            )
            self.ml_model_var.set(name)

        p,h,target=self.current_ml_state()

        # If a current reference path exists, use a nearby future waypoint
        # instead of only the final destination.
        if self.current_path is not None and len(self.current_path)>=2:
            ref=np.asarray(self.current_path,float)
            nearest=int(np.argmin(
                np.linalg.norm(
                    ref-p[None,:],
                    axis=1
                )
            ))
            try:
                lookahead=max(int(self.ml_lookahead_steps.get()),1)
            except Exception:
                lookahead=5
            target=ref[min(nearest+lookahead,len(ref)-1)]

        delta=float(np.clip(
            self.ml_models[name].predict(
                self.ml_feature(p,h,target)[None,:]
            )[0],
            -100.0,100.0
        ))

        self.ml_command_text.set(
            f"ML command [{name}] | "
            f"scenario={self.ml_signature_id(self.ml_training_signature)} | "
            f"{self.command_label(delta)} | "
            f"Δψ={delta:+.2f}° | heading={h:.2f}°"
        )

    def ml_run_planner(self):
        if not self.ml_models:
            messagebox.showinfo(
                "ML Mode",
                "No current ML model is trained.\n\n"
                "Press 'Train Selected Models'. Training now always uses the "
                "CURRENT planned route."
            )
            return

        current_sig=self.ml_current_signature()
        if self.ml_training_signature != current_sig:
            messagebox.showwarning(
                "ML Mode — Stale Model",
                "The robot/map/destinations/path changed after this ML model "
                "was trained.\n\n"
                "The old model will NOT be used. Press 'Train Selected Models' "
                "again for the current route."
            )
            return

        if not self.destinations:
            messagebox.showinfo("ML Mode","Add destinations first.")
            return

        name=self.ml_model_var.get()
        if name not in self.ml_models:
            name=min(
                self.ml_models,
                key=lambda n:self.ml_metrics[n]["rmse_deg"]
            )
            self.ml_model_var.set(name)

        try:
            step=max(float(self.ml_step_var.get()),4.0)
            max_steps=max(int(self.ml_max_steps.get()),100)
            lookahead=max(int(self.ml_lookahead_steps.get()),1)
        except Exception:
            step,max_steps,lookahead=14.0,1200,5

        model=self.ml_models[name]

        # Save the CURRENT conventional route for research comparison.
        self.ml_reference_path=np.asarray(
            self.current_path,float
        ).copy() if self.current_path is not None else None

        # Use CURRENT planner segments as local reference corridors.
        reference_segments=[
            np.asarray(s,float).copy()
            for s in self.current_segments
            if len(s)>=2
        ]

        if not reference_segments:
            if self.current_path is None or len(self.current_path)<2:
                messagebox.showinfo(
                    "ML Mode",
                    "A current reference planner route is required."
                )
                return
            reference_segments=[np.asarray(self.current_path,float).copy()]

        if len(reference_segments) != len(self.destinations):
            # Reconstruct per-destination references if segment bookkeeping
            # does not match.  This is safer than using an old hidden route.
            rebuilt=[]
            a=np.asarray(self.start,float)
            full=np.asarray(self.current_path,float)
            cursor=0
            for dest in self.destinations:
                dest=np.asarray(dest,float)
                if cursor>=len(full):
                    break
                dists=np.linalg.norm(full[cursor:]-dest[None,:],axis=1)
                end_idx=cursor+int(np.argmin(dists))
                if end_idx<=cursor:
                    end_idx=min(cursor+1,len(full)-1)
                seg=full[cursor:end_idx+1]
                if len(seg)>=2:
                    rebuilt.append(seg.copy())
                cursor=end_idx
            if len(rebuilt)==len(self.destinations):
                reference_segments=rebuilt

        route=[np.asarray(self.start,float).copy()]
        segments=[]
        history=[]

        a=np.asarray(self.start,float).copy()
        heading=None

        effective=self.effective_planning_clearance()
        ds=self.directional_settings()

        for di,target in enumerate(self.destinations):
            target=np.asarray(target,float)

            # Current route reference for THIS destination only.
            if di < len(reference_segments):
                ref_raw=reference_segments[di]
            else:
                ref_raw=np.vstack([a,target])

            ref_len=route_length(ref_raw)
            if ref_len<=1e-9:
                continue

            ref_spacing=max(step*0.65,3.0)
            ref=resample(ref_raw,ref_spacing,1200)
            if len(ref)<2:
                ref=np.vstack([a,target])

            # Force exact destination.
            ref[-1]=target

            if heading is None:
                heading=vector_heading_deg(ref[min(1,len(ref)-1)]-a)
                if heading is None:
                    heading=vector_heading_deg(target-a)
                if heading is None:
                    heading=0.0

            seg=[a.copy()]
            success=False
            ref_progress=0

            for _ in range(max_steps):
                remaining=dist(a,target)

                # Finish exactly at destination when safely possible.
                if remaining<=max(step*1.25,6.0):
                    if not segment_collision(
                        a,target,self.obstacles,effective,
                        max(step/4,2.5)
                    ):
                        seg.append(target.copy())
                        route.append(target.copy())
                        history.append(target.copy())
                        a=target.copy()
                        success=True
                    break

                # Only search reference points at/after previous progress;
                # prevents the ML navigator from selecting a past waypoint.
                search_start=max(ref_progress-1,0)
                future_ref=ref[search_start:]
                nearest_rel=int(np.argmin(
                    np.linalg.norm(
                        future_ref-a[None,:],
                        axis=1
                    )
                ))
                nearest_idx=search_start+nearest_rel
                ref_progress=max(ref_progress,nearest_idx)

                look_idx=min(
                    ref_progress+lookahead,
                    len(ref)-1
                )
                local_target=ref[look_idx]

                feat=self.ml_feature(
                    a,heading,local_target
                )[None,:]

                predicted=float(np.clip(
                    model.predict(feat)[0],
                    -ds["max_turn_deg"],
                    ds["max_turn_deg"]
                ))

                # Safety shield explores nearby steering angles around the ML
                # command but scores them against the CURRENT reference path.
                offsets=[0,8,-8,16,-16,28,-28,42,-42,60,-60]
                candidates=[]

                for off in offsets:
                    if off!=0 and not self.ml_safety_shield.get():
                        continue

                    nh=heading+predicted+off
                    rad=math.radians(nh)
                    cand=a+step*np.array([
                        math.cos(rad),math.sin(rad)
                    ])

                    if not inside_bounds(cand,self.bounds()):
                        continue
                    if point_collision(
                        cand,self.obstacles,effective
                    ):
                        continue
                    if segment_collision(
                        a,cand,self.obstacles,effective,
                        max(step/4,2.5)
                    ):
                        continue

                    if history_segment_blocked(
                        a,cand,
                        np.asarray(history,float)
                        if history else None,
                        ds["retrace_radius"]
                        if ds["enabled"] else 0.0,
                        a,
                        ds["release_radius"]
                        if ds["enabled"] else 0.0,
                        sample_step=max(step/4,2.5)
                    ):
                        continue

                    # Reference-tracking score:
                    # 1) approach local look-ahead
                    # 2) remain close to future reference
                    # 3) avoid large safety-shield override
                    future_window=ref[
                        max(ref_progress-1,0):
                        min(ref_progress+lookahead+8,len(ref))
                    ]
                    if len(future_window):
                        cross_track=float(np.min(
                            np.linalg.norm(
                                future_window-cand[None,:],
                                axis=1
                            )
                        ))
                    else:
                        cross_track=dist(cand,local_target)

                    score=(
                        1.00*dist(cand,local_target)
                        + 1.25*cross_track
                        + 0.20*abs(off)
                        + 0.10*dist(cand,target)
                    )

                    candidates.append(
                        (score,cand,nh)
                    )

                if not candidates:
                    # Final safety fallback: use the CURRENT planner reference
                    # heading, not any past learned value.
                    desired=vector_heading_deg(
                        local_target-a
                    )
                    if (
                        self.ml_safety_shield.get()
                        and desired is not None
                    ):
                        nr=math.radians(desired)
                        cand=a+step*np.array([
                            math.cos(nr),math.sin(nr)
                        ])
                        if (
                            inside_bounds(cand,self.bounds())
                            and not point_collision(
                                cand,self.obstacles,effective
                            )
                            and not segment_collision(
                                a,cand,self.obstacles,effective,
                                max(step/4,2.5)
                            )
                        ):
                            candidates.append(
                                (
                                    dist(cand,local_target),
                                    cand,
                                    desired
                                )
                            )

                if not candidates:
                    break

                _,cand,nh=min(
                    candidates,
                    key=lambda z:z[0]
                )

                a=np.asarray(cand,float)
                heading=float(nh)
                seg.append(a.copy())
                route.append(a.copy())
                history.append(a.copy())

                # Advance reference progress after movement.
                future_ref=ref[ref_progress:]
                if len(future_ref):
                    nearest_rel=int(np.argmin(
                        np.linalg.norm(
                            future_ref-a[None,:],
                            axis=1
                        )
                    ))
                    ref_progress+=nearest_rel

            if not success:
                messagebox.showwarning(
                    "ML Mode",
                    f"Fresh model [{name}] could not safely reach D{di+1}.\n\n"
                    f"Training scenario: {self.ml_signature_id(self.ml_training_signature)}\n"
                    f"Reference look-ahead: {lookahead}\n"
                    f"ML step: {step:.2f}\n\n"
                    "Try a smaller ML step (8–12), larger look-ahead (5–8), "
                    "or generate a smoother planner reference."
                )
                return

            segments.append(
                np.asarray(seg,float)
            )

        route=np.asarray(route,float)

        self.current_path=route
        self.current_segments=segments
        self.current_algorithm=f"ML:{name}"
        self.ml_last_route=route.copy()

        center_cl=route_clearance(
            route,self.obstacles
        )
        est_cl=(
            center_cl-self.full_body_margin()
            if np.isfinite(center_cl)
            else float("inf")
        )

        self.current_result={
            "algorithm":self.current_algorithm,
            "success":True,
            "path":route,
            "segments":segments,
            "nodes":len(route),
            "path_length":route_length(route),
            "smoothness":route_smoothness(route),
            "clearance":est_cl,
            "destinations":len(self.destinations),
            "ml_mode":True,
            "training_scenario":self.ml_signature_id(
                self.ml_training_signature
            )
        }

        self.redraw()
        self.ml_plot_results()

        self.status.set(
            f"ML route generated from FRESH current training | "
            f"{self.ml_signature_id(self.ml_training_signature)}"
        )
        self.ml_command_text.set(
            f"ML MODE | Model={name} | "
            f"scenario={self.ml_signature_id(self.ml_training_signature)} | "
            f"points={len(route)} | "
            f"length={route_length(route):.2f}"
        )

    def run_selected_pair(self):
        """
        Operational sequence required by the GUI:
            1) ML command inference FIRST
            2) selected path planner SECOND

        The ML steering command defines the initial departure heading supplied
        to the directional planner.
        """
        if not self.ml_models:
            messagebox.showinfo(
                "Hybrid Evaluation",
                "Train an ML command model first. With the default ML-FIRST "
                "training source, no path planning is required before training."
            )
            return

        if not self.ml_signature_matches_current():
            messagebox.showwarning(
                "Hybrid Evaluation",
                "The environment/destinations changed after ML training. "
                "Retrain the ML model before running this pair."
            )
            return

        ml_name=self.eval_ml_var.get()
        planner=self.eval_planner_var.get()

        if ml_name not in self.ml_models:
            messagebox.showinfo(
                "Hybrid Evaluation",
                "Select a trained ML algorithm."
            )
            return

        if not self.destinations:
            messagebox.showinfo(
                "Hybrid Evaluation",
                "Add at least one destination."
            )
            return

        # ---------- 1. ML COMMAND FIRST ----------
        p=np.asarray(self.start,float)
        target=np.asarray(self.destinations[0],float)
        base_heading=vector_heading_deg(target-p)
        if base_heading is None:
            base_heading=0.0

        feature=self.ml_feature(
            p,base_heading,target
        )[None,:]
        delta=float(np.clip(
            self.ml_models[ml_name].predict(feature)[0],
            -100.0,100.0
        ))
        ml_departure_heading=base_heading+delta

        self.ml_command_text.set(
            f"STEP 1 — ML [{ml_name}] command: {self.command_label(delta)} | "
            f"Δψ={delta:+.2f}° | departure heading={ml_departure_heading:.2f}°"
        )
        self.status.set(
            f"ML command executed first ({ml_name}); now running {planner}..."
        )
        self.update_idletasks()

        # ---------- 2. PATH PLANNING SECOND ----------
        result=self.solve(
            planner,
            initial_heading_override=ml_departure_heading
        )
        if result is None:
            return
        if not result.get("success"):
            messagebox.showwarning(
                "Hybrid Evaluation",
                f"{planner} could not find a feasible route after the initial "
                f"{ml_name} steering command."
            )
            return

        self.current_algorithm=planner
        self.current_result=result
        self.current_path=np.asarray(result["path"],float)
        self.current_segments=[
            np.asarray(s,float) for s in result.get("segments",[])
        ]

        self.active_pair_ml=ml_name
        self.active_pair_planner=planner
        self.ml_reference_path=self.current_path.copy()
        self.ml_last_route=self.current_path.copy()

        key=(planner,ml_name)
        self.pair_results[key]={
            "path":self.current_path.copy(),
            "segments":[s.copy() for s in self.current_segments],
            "result":dict(result),
            "ml_delta_deg":delta,
            "ml_departure_heading_deg":ml_departure_heading
        }

        self.redraw()
        self.ml_plot_results()

        # Graphs can be calculated from the generated body sequence even before
        # live animation is played.
        self.runtime_body_log=[]
        self.active_seq=None
        self.analysis_cache=None
        self.error_cache=None
        try:self.refresh_motion_analysis()
        except Exception:pass
        try:self.refresh_error_analysis()
        except Exception:pass
        try:self.refresh_singularity_analysis()
        except Exception:pass
        try:self.refresh_workspace_analysis()
        except Exception:pass

        self.status.set(
            f"PAIR READY: {planner} + {ml_name} | "
            f"ML command first → path planner second"
        )

    def load_selected_pair_graphs(self):
        key=self.selected_pair_key()
        if key not in self.pair_results:
            messagebox.showinfo(
                "Pair Graphs",
                f"No stored result exists for {key[0]} + {key[1]}.\n\n"
                "Run the selected pair first."
            )
            return

        data=self.pair_results[key]
        self.current_path=np.asarray(data["path"],float).copy()
        self.current_segments=[
            np.asarray(s,float).copy() for s in data["segments"]
        ]
        self.current_result=dict(data["result"])
        self.current_algorithm=key[0]
        self.active_pair_planner=key[0]
        self.active_pair_ml=key[1]
        self.ml_reference_path=self.current_path.copy()
        self.ml_last_route=self.current_path.copy()

        self.runtime_body_log=[]
        self.active_seq=None
        self.analysis_cache=None
        self.error_cache=None

        self.redraw()
        self.ml_plot_results()
        try:self.refresh_motion_analysis()
        except Exception:pass
        try:self.refresh_error_analysis()
        except Exception:pass
        try:self.refresh_singularity_analysis()
        except Exception:pass
        try:self.refresh_workspace_analysis()
        except Exception:pass

        self.status.set(
            f"Loaded graph pair: {self.pair_label()}"
        )

    def ml_save_model(self):
        if not self.ml_require() or not self.ml_models: return
        name=self.ml_model_var.get()
        if name not in self.ml_models: return
        path=filedialog.asksaveasfilename(
            defaultextension=".joblib",filetypes=[("Joblib model","*.joblib")],
            initialfile=f"snake_ml_command_{name.replace(' ','_')}.joblib"
        )
        if not path: return
        joblib.dump({
            "name":name,
            "model":self.ml_models[name],
            "metrics":self.ml_metrics.get(name,{}),
            "training_signature":self.ml_training_signature,
            "dataset_source":self.ml_dataset_source_text,
            "lookahead_steps":int(self.ml_lookahead_steps.get()),
            "training_scope":self.ml_training_scope
        },path)
        self.status.set(f"ML model saved: {path}")

    def ml_load_model(self):
        if not self.ml_require(): return
        path=filedialog.askopenfilename(filetypes=[("Joblib model","*.joblib")])
        if not path: return
        try:
            payload=joblib.load(path); name=str(payload.get("name","Loaded Model"))
            unique=name; n=2
            while unique in self.ml_models:
                unique=f"{name} {n}"; n+=1
            self.ml_models[unique]=payload["model"]
            self.ml_metrics[unique]=payload.get(
                "metrics",
                {"rmse_deg":float("nan"),"mae_deg":float("nan"),"r2":float("nan")}
            )
            loaded_sig=payload.get("training_signature")
            self.ml_training_signature=loaded_sig
            self.ml_training_scope=payload.get("training_scope","environment")
            self.ml_dataset_source_text=payload.get(
                "dataset_source",
                "Loaded external model"
            )
            if "lookahead_steps" in payload:
                try:self.ml_lookahead_steps.set(int(payload["lookahead_steps"]))
                except Exception:pass
            values=list(self.ml_models.keys())
            self.ml_model_combo["values"]=values
            self.eval_ml_combo["values"]=values
            self.motion_ml_combo["values"]=values
            self.error_ml_combo["values"]=values
            self.singularity_ml_combo["values"]=values
            self.workspace_ml_combo["values"]=values
            self.compare_ml_combo["values"]=values
            self.ml_model_var.set(unique)
            self.eval_ml_var.set(unique)

            if self.ml_signature_matches_current():
                msg=f"Loaded model: {unique} | CURRENT scenario match"
            else:
                msg=(
                    f"Loaded model: {unique} | scenario mismatch — "
                    "retrain before hybrid evaluation"
                )
            self.ml_status.set(msg)
        except Exception as exc:
            messagebox.showerror("Load Model",str(exc))

    def ml_export_csv(self):
        if self.ml_X is None and not self.ml_metrics:
            messagebox.showinfo("ML Export","No ML data available."); return
        folder=filedialog.askdirectory(title="Select folder for ML CSV files")
        if not folder: return
        folder=Path(folder)
        if self.ml_X is not None:
            headers=["goal_dx_norm","goal_dy_norm","goal_distance_norm","sin_heading","cos_heading",
                     "obstacle_clearance_norm","sin_obstacle_bearing","cos_obstacle_bearing","steering_delta_deg"]
            np.savetxt(folder/"ml_command_dataset.csv",np.column_stack([self.ml_X,self.ml_y]),
                       delimiter=",",header=",".join(headers),comments="")
        if self.ml_metrics:
            with open(folder/"ml_model_metrics.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["model","rmse_deg","mae_deg","r2","command_accuracy"])
                for name,m in self.ml_metrics.items():
                    w.writerow([name,m["rmse_deg"],m["mae_deg"],m["r2"],m.get("command_accuracy","")])

        labels=self.command_class_labels()
        for name,cm in self.ml_confusions.items():
            safe=name.replace(" ","_")
            with open(folder/f"confusion_matrix_{safe}.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f)
                w.writerow(["True/Predicted",*labels])
                for label,row in zip(labels,cm):
                    w.writerow([label,*[int(v) for v in row]])

        self.ml_status.set(f"ML CSV files exported to: {folder}")

    def body_to_joint_angles(self,body):
        body=np.asarray(body,float)
        d=np.diff(body,axis=0)
        if len(d)==0: return np.zeros(self.dof)
        headings=np.unwrap(np.arctan2(d[:,1],d[:,0]))
        q=np.zeros(len(headings),float); q[0]=headings[0]
        if len(headings)>1: q[1:]=np.diff(headings)
        return np.degrees(q)

    def jacobian_metrics_from_body(self,body):
        q=np.deg2rad(self.body_to_joint_angles(body))
        cumulative=np.cumsum(q); n=len(q); J=np.zeros((2,n),float); L=float(self.robot.link)
        for j in range(n):
            for k in range(j,n):
                J[0,j]+=-L*math.sin(cumulative[k]); J[1,j]+=L*math.cos(cumulative[k])
        s=np.linalg.svd(J,compute_uv=False)
        smin=float(s[-1]) if len(s) else 0.0
        cond=float("inf") if smin<1e-10 else float(s[0]/smin)
        manip=math.sqrt(max(float(np.linalg.det(J@J.T)),0.0))
        rank=int(np.linalg.matrix_rank(J,tol=1e-8))
        return smin,cond,manip,rank

    def body_clearance_value(self,body):
        if not self.obstacles: return float("inf")
        body=np.asarray(body,float); samples=[]
        for a,b in zip(body[:-1],body[1:]):
            L=np.linalg.norm(b-a); n=max(int(L/max(float(self.robot.link)/10,2.0)),1)
            for k in range(n+1): samples.append(a+(k/n)*(b-a))
        return route_clearance(np.asarray(samples),self.obstacles)-max(float(self.body_radius.get()),0.0)

    def analysis_frames(self):
        if len(self.runtime_body_log)>=2: return np.asarray(self.runtime_body_log,float)
        try: seq=self.anim_sequence()
        except Exception: seq=None
        return None if seq is None else np.asarray(seq["body_frames"],float)

    def compute_motion_cache(self):
        frames=self.analysis_frames()
        if frames is None or len(frames)<2: return None
        try: dt=max(float(self.analysis_dt.get()),1e-4)
        except Exception: dt=0.05
        n=len(frames); t=np.arange(n,dtype=float)*dt
        head=frames[:,-1,:]
        q=np.asarray([self.body_to_joint_angles(b) for b in frames],float)
        vel=np.gradient(head,dt,axis=0); acc=np.gradient(vel,dt,axis=0)
        qvel=np.gradient(q,dt,axis=0); qacc=np.gradient(qvel,dt,axis=0)
        step=np.linalg.norm(np.diff(head,axis=0),axis=1)
        cache={
            "t":t,"head":head,"vel":vel,"acc":acc,
            "speed":np.linalg.norm(vel,axis=1),"accel":np.linalg.norm(acc,axis=1),
            "q":q,"qvel":qvel,"qacc":qacc,
            "cumulative":np.r_[0.0,np.cumsum(step)],
            "displacement":np.linalg.norm(head-head[0],axis=1),
            "frames":frames
        }
        self.analysis_cache=cache; return cache

    def refresh_motion_analysis(self):
        d=self.compute_motion_cache()
        if d is None:
            messagebox.showinfo("Motion Analysis","Plan or animate a route first."); return
        for ax in self.motion_axes: ax.clear()
        a1,a2,a3,a4,a5,a6=self.motion_axes; t=d["t"]

        a1.plot(t,d["head"][:,0]-d["head"][0,0],label="ΔX")
        a1.plot(t,d["head"][:,1]-d["head"][0,1],label="ΔY")
        a1.plot(t,d["cumulative"],label="Travelled")
        self.set_panel_title(a1,"Head Displacement",fontsize=8.0); a1.set_xlabel("Time (s)"); a1.set_ylabel("Distance")
        a1.grid(True,alpha=.3); a1.legend(fontsize=8)

        a2.plot(t,d["speed"],label="Speed"); a2.plot(t,d["vel"][:,0],label="Vx",alpha=.7)
        a2.plot(t,d["vel"][:,1],label="Vy",alpha=.7)
        self.set_panel_title(a2,"Head Velocity",fontsize=8.0); a2.set_xlabel("Time (s)"); a2.set_ylabel("Velocity")
        a2.grid(True,alpha=.3); a2.legend(fontsize=8)

        a3.plot(t,d["accel"],label="Acceleration"); a3.plot(t,d["acc"][:,0],label="Ax",alpha=.7)
        a3.plot(t,d["acc"][:,1],label="Ay",alpha=.7)
        self.set_panel_title(a3,"Head Acceleration",fontsize=8.0); a3.set_xlabel("Time (s)"); a3.set_ylabel("Acceleration")
        a3.grid(True,alpha=.3); a3.legend(fontsize=8)

        for j in range(min(self.dof,d["q"].shape[1])): a4.plot(t,d["q"][:,j],lw=.9,label=f"J{j+1}")
        self.set_panel_title(a4,"Joint Angular Displacement",fontsize=8.0); a4.set_xlabel("Time (s)"); a4.set_ylabel("Angle (deg)")
        a4.grid(True,alpha=.3); a4.legend(ncol=min(4,self.dof),fontsize=6)

        for j in range(min(self.dof,d["qvel"].shape[1])): a5.plot(t,d["qvel"][:,j],lw=.9)
        self.set_panel_title(a5,"Joint Angular Velocity",fontsize=8.0); a5.set_xlabel("Time (s)"); a5.set_ylabel("deg/s"); a5.grid(True,alpha=.3)

        a6.plot(
            d["head"][:,0],
            d["head"][:,1],
            lw=1.7,
            label="Full executed head trajectory"
        )

        p=self.full_reference_path()
        if p is not None and len(p)>=2:
            a6.plot(
                p[:,0],p[:,1],
                "--",
                lw=1.2,
                label="Original reference path"
            )

        # Dynamic obstacles are shown in red here as well.
        static_label_used=False
        dynamic_label_used=False
        for o in self.obstacles:
            is_dynamic=bool(o.get("dynamic",False))
            label=None
            if is_dynamic and not dynamic_label_used:
                label="Dynamic obstacle"
                dynamic_label_used=True
            elif (not is_dynamic) and not static_label_used:
                label="Static obstacle"
                static_label_used=True

            if o["type"]=="circle":
                patch=Circle(
                    (o["x"],o["y"]),
                    o["r"],
                    alpha=.20 if not is_dynamic else .30,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=1.6 if is_dynamic else None,
                    label=label
                )
            else:
                patch=Rectangle(
                    (o["x"]-o["w"]/2,o["y"]-o["h"]/2),
                    o["w"],o["h"],
                    alpha=.20 if not is_dynamic else .30,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=1.6 if is_dynamic else None,
                    label=label
                )
            a6.add_patch(patch)

        self.set_panel_title(a6,"Overall Full-Mission Trajectory",fontsize=8.0)
        a6.set_xlabel("World X")
        a6.set_ylabel("World Y")
        a6.set_aspect("equal",adjustable="datalim")
        a6.grid(True,alpha=.3)
        a6.relim()
        a6.autoscale_view()
        a6.legend(fontsize=7.2)

        self.set_figure_pair_title(self.motion_fig,label="Motion Evaluation",fontsize=10.5)
        self.motion_fig.subplots_adjust(left=0.07,right=0.98,bottom=0.08,top=0.87,hspace=0.46,wspace=0.30); self.motion_canvas.draw_idle()
        self.motion_summary.set(
            f"Samples={len(t)} | Travel={d['cumulative'][-1]:.2f} | "
            f"Peak speed={np.max(d['speed']):.3f} | Peak acceleration={np.max(d['accel']):.3f} | "
            f"Peak joint velocity={np.max(np.abs(d['qvel'])):.3f} deg/s"
        )

    def compute_error_cache(self):
        d=self.compute_motion_cache()
        if d is None:
            return None

        frames=d["frames"]
        n=len(frames)
        t=d["t"]
        heads=frames[:,-1,:]

        planned=self.full_reference_path()
        destinations=self.full_mission_destinations()

        head_error=np.zeros(n)
        heading_error=np.zeros(n)
        sigma_min=np.zeros(n)
        condition=np.zeros(n)
        manipulability=np.zeros(n)
        clearance=np.zeros(n)
        rank=np.zeros(n)
        singularity_ratio=np.zeros(n)
        target_index=np.zeros(n,dtype=int)

        # ------------------------------------------------------------
        # Determine when each ORIGINAL destination is reached using the
        # complete executed head history.  This avoids using shortened
        # post-replan destination lists.
        # ------------------------------------------------------------
        arrival_frames=[]
        search_from=0

        for dest in destinations:
            if search_from>=n:
                arrival_frames.append(n-1)
                continue

            distances=np.linalg.norm(
                heads[search_from:]-dest[None,:],
                axis=1
            )
            rel=int(np.argmin(distances))
            arrival=search_from+rel
            arrival_frames.append(arrival)
            search_from=min(arrival+1,n-1)

        # For each frame, active target is the first original destination whose
        # arrival frame has not yet been passed.
        for i in range(n):
            if destinations:
                di=0
                while (
                    di < len(arrival_frames)-1
                    and i > arrival_frames[di]
                ):
                    di+=1

                target_index[i]=di
                target=destinations[di]
                head_error[i]=dist(heads[i],target)

            body=frames[i]
            head=heads[i]

            # Heading error is referenced to the ORIGINAL planner path rather
            # than the shortened current_path produced by online replanning.
            actual_heading=vector_heading_deg(
                body[-1]-body[-2]
            )
            desired_heading=None

            if planned is not None and len(planned)>=2:
                nearest=int(np.argmin(
                    np.linalg.norm(
                        planned-head[None,:],
                        axis=1
                    )
                ))
                j=min(nearest,len(planned)-2)
                desired_heading=vector_heading_deg(
                    planned[j+1]-planned[j]
                )

            if (
                actual_heading is not None
                and desired_heading is not None
            ):
                heading_error[i]=abs(
                    angle_diff_deg(
                        actual_heading,
                        desired_heading
                    )
                )

            smin,cond,manip,rnk=self.jacobian_metrics_from_body(
                body
            )
            sigma_min[i]=smin
            condition[i]=(
                cond if np.isfinite(cond) else 1e12
            )
            manipulability[i]=manip
            rank[i]=rnk
            clearance[i]=self.body_clearance_value(
                body
            )

            singularity_ratio[i]=(
                0.0
                if not np.isfinite(cond) or cond<=0
                else min(1.0,1.0/cond)
            )

        self.error_cache={
            "t":t,
            "head_error":head_error,
            "heading_error":heading_error,
            "sigma_min":sigma_min,
            "condition":condition,
            "manipulability":manipulability,
            "clearance":clearance,
            "rank":rank,
            "singularity_ratio":singularity_ratio,
            "target_index":target_index,
            "arrival_frames":arrival_frames
        }
        return self.error_cache

    def refresh_error_analysis(self):
        d=self.compute_error_cache()
        if d is None:
            messagebox.showinfo("Error Analysis","Plan or animate a route first."); return
        for ax in self.error_axes: ax.clear()
        a1,a2,a3,a4,a5,a6=self.error_axes; t=d["t"]

        a1.plot(t,d["head_error"]); self.set_panel_title(a1,"Head-to-Destination Error",fontsize=8.0)
        a1.set_xlabel("Time (s)"); a1.set_ylabel("Distance"); a1.grid(True,alpha=.3)

        a2.plot(t,d["heading_error"]); self.set_panel_title(a2,"Head Heading Error",fontsize=8.0)
        a2.set_xlabel("Time (s)"); a2.set_ylabel("Degrees"); a2.grid(True,alpha=.3)

        a3.plot(t,d["sigma_min"]); a3.axhline(1e-6,ls="--",lw=1)
        self.set_panel_title(a3,"Smallest Singular Value",fontsize=8.0); a3.set_xlabel("Time (s)")
        a3.set_ylabel("σmin"); a3.grid(True,alpha=.3)

        a4.semilogy(t,np.maximum(d["condition"],1e-9)); self.set_panel_title(a4,"Jacobian Condition Number",fontsize=8.0)
        a4.set_xlabel("Time (s)"); a4.set_ylabel("Condition number"); a4.grid(True,alpha=.3)

        a5.plot(t,d["manipulability"]); self.set_panel_title(a5,"Manipulability",fontsize=8.0)
        a5.set_xlabel("Time (s)"); a5.set_ylabel("Index"); a5.grid(True,alpha=.3)

        a6.plot(t,d["clearance"]); a6.axhline(0,ls="--",lw=1)
        self.set_panel_title(a6,"Full-Body Obstacle Clearance",fontsize=8.0); a6.set_xlabel("Time (s)")
        a6.set_ylabel("Clearance"); a6.grid(True,alpha=.3)

        self.set_figure_pair_title(self.error_fig,label="Head Error / Safety Evaluation",fontsize=10.5)
        self.error_fig.subplots_adjust(left=0.07,right=0.98,bottom=0.08,top=0.87,hspace=0.46,wspace=0.30); self.error_canvas.draw_idle()
        self.error_summary.set(
            f"Final head error={d['head_error'][-1]:.3f} | "
            f"Max heading error={np.max(d['heading_error']):.3f}° | "
            f"Min singularity ratio={np.min(d['singularity_ratio']):.6f} | "
            f"Min body clearance={np.min(d['clearance']):.3f}"
        )

    def compute_workspace_cache(self):
        """
        Monte-Carlo FK workspace for the snake head with a fixed base.

        Joint samples are uniformly drawn within ±workspace_joint_limit.
        The calculation is vectorized for speed and also computes Jacobian
        singularity/manipulability metrics at every sampled configuration.
        """
        try:
            ns=max(int(self.workspace_samples.get()),500)
            ns=min(ns,50000)

            joint_limit=float(self.workspace_joint_limit.get())
            joint_limit=float(np.clip(joint_limit,1.0,180.0))

            grid_res=max(float(self.workspace_grid_res.get()),1.0)
            seed=int(self.workspace_seed.get())
        except Exception:
            messagebox.showerror(
                "Workspace Analysis",
                "Check Samples, Joint Limit, Area Grid and Seed values."
            )
            return None

        rng=np.random.default_rng(seed)

        # Random joint configurations in radians.
        q_deg=rng.uniform(
            -joint_limit,
            joint_limit,
            size=(ns,self.dof)
        )
        q=np.deg2rad(q_deg)

        # Cumulative planar link orientations.
        theta=np.cumsum(q,axis=1)

        L=float(self.robot.link)

        # Head endpoint coordinates with fixed base at (0,0), heading=0.
        x=L*np.sum(np.cos(theta),axis=1)
        y=L*np.sum(np.sin(theta),axis=1)

        radius=np.sqrt(x*x+y*y)
        polar_angle=np.arctan2(y,x)

        # ------------------------------------------------------------
        # Vectorized 2xn Jacobian metrics.
        #
        # Jx_j = -L * sum_{k=j}^{n-1} sin(theta_k)
        # Jy_j =  L * sum_{k=j}^{n-1} cos(theta_k)
        # ------------------------------------------------------------
        sin_t=np.sin(theta)
        cos_t=np.cos(theta)

        rev_sin=np.flip(
            np.cumsum(
                np.flip(sin_t,axis=1),
                axis=1
            ),
            axis=1
        )
        rev_cos=np.flip(
            np.cumsum(
                np.flip(cos_t,axis=1),
                axis=1
            ),
            axis=1
        )

        Jx=-L*rev_sin
        Jy= L*rev_cos

        a=np.sum(Jx*Jx,axis=1)
        b=np.sum(Jx*Jy,axis=1)
        c=np.sum(Jy*Jy,axis=1)

        disc=np.sqrt(
            np.maximum(
                (a-c)*(a-c)+4.0*b*b,
                0.0
            )
        )

        lam_max=np.maximum(
            0.5*((a+c)+disc),
            0.0
        )
        lam_min=np.maximum(
            0.5*((a+c)-disc),
            0.0
        )

        sigma_max=np.sqrt(lam_max)
        sigma_min=np.sqrt(lam_min)

        singularity_ratio=np.divide(
            sigma_min,
            sigma_max,
            out=np.zeros_like(sigma_min),
            where=sigma_max>1e-12
        )

        condition=np.divide(
            sigma_max,
            sigma_min,
            out=np.full_like(sigma_max,np.inf),
            where=sigma_min>1e-12
        )

        det_jjt=np.maximum(
            a*c-b*b,
            0.0
        )
        manipulability=np.sqrt(det_jjt)

        rank=np.where(
            sigma_min>1e-8,
            2,
            np.where(sigma_max>1e-8,1,0)
        ).astype(int)

        # ------------------------------------------------------------
        # Empirical workspace area by occupied XY grid cells.
        # ------------------------------------------------------------
        gx=np.floor(x/grid_res).astype(np.int64)
        gy=np.floor(y/grid_res).astype(np.int64)

        occupied=np.unique(
            np.column_stack([gx,gy]),
            axis=0
        )

        empirical_area=float(
            len(occupied)*grid_res*grid_res
        )

        theoretical_max_reach=float(
            self.dof*L
        )

        theoretical_disk_area=float(
            math.pi*theoretical_max_reach**2
        )

        coverage_pct=(
            100.0*empirical_area/theoretical_disk_area
            if theoretical_disk_area>0
            else 0.0
        )

        # ------------------------------------------------------------
        # Angular reach envelope.
        # ------------------------------------------------------------
        nbins=72
        edges=np.linspace(
            -math.pi,
            math.pi,
            nbins+1
        )
        centers=0.5*(edges[:-1]+edges[1:])

        env_max=np.full(nbins,np.nan)
        env_min=np.full(nbins,np.nan)

        bin_id=np.digitize(
            polar_angle,
            edges
        )-1

        bin_id=np.clip(
            bin_id,
            0,
            nbins-1
        )

        for bi in range(nbins):
            m=bin_id==bi
            if np.any(m):
                env_max[bi]=float(np.max(radius[m]))
                env_min[bi]=float(np.min(radius[m]))

        cache={
            "q_deg":q_deg,
            "x":x,
            "y":y,
            "radius":radius,
            "polar_angle":polar_angle,
            "sigma_min":sigma_min,
            "sigma_max":sigma_max,
            "singularity_ratio":singularity_ratio,
            "condition":condition,
            "manipulability":manipulability,
            "rank":rank,
            "samples":ns,
            "joint_limit_deg":joint_limit,
            "grid_resolution":grid_res,
            "seed":seed,
            "empirical_area":empirical_area,
            "theoretical_max_reach":theoretical_max_reach,
            "theoretical_disk_area":theoretical_disk_area,
            "coverage_pct":coverage_pct,
            "envelope_angle":centers,
            "envelope_max":env_max,
            "envelope_min":env_min
        }

        self.workspace_cache=cache
        return cache

    def refresh_workspace_analysis(self):
        d=self.compute_workspace_cache()
        if d is None:
            return

        # ------------------------------------------------------------
        # IMPORTANT:
        # A Matplotlib colorbar owns its own Axes. Calling ax.clear()
        # on the six workspace plots does NOT delete that colorbar Axes.
        # Without explicitly removing it, every refresh creates another
        # colorbar on top of the old one.
        # ------------------------------------------------------------
        if self.workspace_singularity_cbar is not None:
            try:
                self.workspace_singularity_cbar.remove()
            except Exception:
                pass
            self.workspace_singularity_cbar=None

        # Defensive cleanup for any orphaned colorbar axes that may already
        # have been created by an older version of the GUI.
        for extra_ax in list(self.workspace_fig.axes):
            if extra_ax not in self.workspace_axes:
                try:
                    self.workspace_fig.delaxes(extra_ax)
                except Exception:
                    pass

        for ax in self.workspace_axes:
            ax.clear()

        a1,a2,a3,a4,a5,a6=self.workspace_axes

        x=d["x"]
        y=d["y"]
        r=d["radius"]
        ratio=d["singularity_ratio"]
        manip=d["manipulability"]

        # 1) Reachable head workspace
        a1.scatter(
            x,
            y,
            s=3,
            alpha=0.22
        )
        a1.scatter(
            [0],[0],
            marker="s",
            s=45,
            label="Fixed base"
        )
        self.set_panel_title(
            a1,
            "Sampled Head Workspace",
            fontsize=8.0
        )
        a1.set_xlabel("Local X")
        a1.set_ylabel("Local Y")
        a1.set_aspect(
            "equal",
            adjustable="datalim"
        )
        a1.grid(True,alpha=.3)
        a1.legend(fontsize=7)

        # 2) Spatial density
        hb=a2.hexbin(
            x,
            y,
            gridsize=42,
            mincnt=1
        )
        self.set_panel_title(
            a2,
            "Workspace Occupancy Density",
            fontsize=8.0
        )
        a2.set_xlabel("Local X")
        a2.set_ylabel("Local Y")
        a2.set_aspect(
            "equal",
            adjustable="datalim"
        )
        a2.grid(True,alpha=.2)

        # 3) Radial reach distribution
        a3.hist(
            r,
            bins=40
        )
        a3.axvline(
            d["theoretical_max_reach"],
            ls="--",
            lw=1.2,
            label="Theoretical max"
        )
        a3.axvline(
            np.max(r),
            ls=":",
            lw=1.2,
            label="Sampled max"
        )
        self.set_panel_title(
            a3,
            "Radial Reach Distribution",
            fontsize=8.0
        )
        a3.set_xlabel("Head radius from fixed base")
        a3.set_ylabel("Samples")
        a3.grid(True,axis="y",alpha=.3)
        a3.legend(fontsize=7)

        # 4) Singularity ratio over workspace
        sc=a4.scatter(
            x,
            y,
            c=ratio,
            s=5,
            alpha=0.45
        )
        self.set_panel_title(
            a4,
            "Workspace Singularity Ratio",
            fontsize=8.0
        )
        a4.set_xlabel("Local X")
        a4.set_ylabel("Local Y")
        a4.set_aspect(
            "equal",
            adjustable="datalim"
        )
        a4.grid(True,alpha=.2)
        self.workspace_singularity_cbar=self.workspace_fig.colorbar(
            sc,
            ax=a4,
            fraction=0.038,
            pad=0.025,
            shrink=0.84,
            aspect=24
        )
        self.workspace_singularity_cbar.set_label(
            "σmin / σmax",
            fontsize=7.5,
            labelpad=5
        )
        self.workspace_singularity_cbar.ax.tick_params(
            labelsize=7,
            pad=2
        )

        # 5) Manipulability against reach
        a5.scatter(
            r,
            manip,
            s=5,
            alpha=0.25
        )
        self.set_panel_title(
            a5,
            "Manipulability vs Radial Reach",
            fontsize=8.0
        )
        a5.set_xlabel("Head radius")
        a5.set_ylabel("Manipulability")
        a5.grid(True,alpha=.3)

        # 6) Angular reach envelope
        ang=d["envelope_angle"]
        rmax=d["envelope_max"]
        rmin=d["envelope_min"]

        valid_max=np.isfinite(rmax)
        valid_min=np.isfinite(rmin)

        if np.any(valid_max):
            a6.plot(
                ang[valid_max],
                rmax[valid_max],
                lw=1.6,
                label="Maximum reach"
            )

        if np.any(valid_min):
            a6.plot(
                ang[valid_min],
                rmin[valid_min],
                lw=1.0,
                label="Minimum sampled reach"
            )

        a6.set_title(
            self.panel_title(
                "Angular Reach Envelope"
            ),
            fontsize=8.0,
            pad=12,
            fontweight="semibold"
        )
        a6.set_theta_zero_location("E")
        a6.set_theta_direction(1)
        a6.grid(True,alpha=.3)
        a6.legend(
            loc="upper right",
            bbox_to_anchor=(1.28,1.13),
            fontsize=6.5
        )

        self.set_figure_pair_title(
            self.workspace_fig,
            label="Workspace Evaluation",
            fontsize=10.5
        )

        self.workspace_fig.subplots_adjust(
            left=0.07,
            right=0.955,
            bottom=0.08,
            top=0.87,
            hspace=0.50,
            wspace=0.42
        )

        self.workspace_canvas.draw_idle()

        near_ratio=0.05
        near_fraction=100.0*float(
            np.mean(ratio<near_ratio)
        )

        self.workspace_summary.set(
            f"DOF={self.dof} | Link={self.robot.link:.2f} | "
            f"Samples={d['samples']:,} | Joint range=±{d['joint_limit_deg']:.1f}° | "
            f"Sampled max reach={np.max(r):.2f} | "
            f"Theoretical max={d['theoretical_max_reach']:.2f} | "
            f"Empirical area≈{d['empirical_area']:.2f} | "
            f"Disk coverage≈{d['coverage_pct']:.2f}% | "
            f"Near-singular ratio<0.05={near_fraction:.2f}%"
        )

        self.status.set(
            f"Workspace analysis complete: {d['samples']:,} FK configurations"
        )

    def export_workspace_csv(self):
        d=self.workspace_cache
        if d is None:
            d=self.compute_workspace_cache()
        if d is None:
            return

        path=filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialfile=(
                f"{self.graph_pair_text().replace(' ','_').replace('+','plus')}"
                "_workspace_samples.csv"
            )
        )
        if not path:
            return

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            w=csv.writer(f)

            headers=[
                "sample_index",
                "head_x",
                "head_y",
                "radial_reach",
                "sigma_min",
                "sigma_max",
                "singularity_ratio",
                "condition_number",
                "manipulability",
                "jacobian_rank"
            ]
            headers += [
                f"q{i+1}_deg"
                for i in range(self.dof)
            ]

            w.writerow(headers)

            for i in range(d["samples"]):
                w.writerow([
                    i,
                    d["x"][i],
                    d["y"][i],
                    d["radius"][i],
                    d["sigma_min"][i],
                    d["sigma_max"][i],
                    d["singularity_ratio"][i],
                    d["condition"][i],
                    d["manipulability"][i],
                    d["rank"][i],
                    *d["q_deg"][i]
                ])

        self.status.set(
            f"Workspace CSV saved: {path}"
        )

    def refresh_singularity_analysis(self):
        d=self.compute_error_cache()
        if d is None:
            messagebox.showinfo(
                "Singularity Analysis",
                "Plan or animate a route first."
            )
            return

        try:
            threshold=float(self.singularity_ratio_threshold.get())
            threshold=float(np.clip(threshold,1e-6,0.5))
        except Exception:
            threshold=0.05
            self.singularity_ratio_threshold.set(threshold)

        for ax in self.singularity_axes:
            ax.clear()

        a1,a2,a3,a4,a5,a6=self.singularity_axes
        t=d["t"]
        ratio=d["singularity_ratio"]
        near=ratio<threshold

        a1.plot(t,d["sigma_min"],lw=1.4)
        self.set_panel_title(
            a1,
            "Smallest Singular Value"
        )
        a1.set_xlabel("Time (s)")
        a1.set_ylabel("σmin")
        a1.grid(True,alpha=.3)

        a2.semilogy(
            t,
            np.maximum(d["condition"],1e-9),
            lw=1.4
        )
        self.set_panel_title(
            a2,
            "Jacobian Condition Number"
        )
        a2.set_xlabel("Time (s)")
        a2.set_ylabel("κ(J)")
        a2.grid(True,alpha=.3)

        a3.plot(t,d["manipulability"],lw=1.4)
        self.set_panel_title(
            a3,
            "Manipulability Index"
        )
        a3.set_xlabel("Time (s)")
        a3.set_ylabel("√det(JJᵀ)")
        a3.grid(True,alpha=.3)

        a4.step(
            t,
            d["rank"],
            where="mid",
            lw=1.4
        )
        a4.axhline(2,ls="--",lw=1)
        self.set_panel_title(
            a4,
            "Planar Jacobian Rank"
        )
        a4.set_xlabel("Time (s)")
        a4.set_ylabel("Rank")
        a4.set_ylim(-0.1,2.3)
        a4.grid(True,alpha=.3)

        a5.plot(t,ratio,lw=1.5)
        a5.axhline(
            threshold,
            ls="--",
            lw=1.2,
            label=f"Threshold={threshold:.3f}"
        )
        if np.any(near):
            a5.scatter(
                t[near],
                ratio[near],
                s=15,
                label="Near-singular"
            )
        self.set_panel_title(
            a5,
            "Normalized Singularity Ratio σmin/σmax"
        )
        a5.set_xlabel("Time (s)")
        a5.set_ylabel("σmin / σmax")
        a5.grid(True,alpha=.3)
        a5.legend(fontsize=7)

        a6.hist(
            ratio,
            bins=30
        )
        a6.axvline(
            threshold,
            ls="--",
            lw=1.2,
            label=f"Threshold={threshold:.3f}"
        )
        self.set_panel_title(
            a6,
            "Singularity-Ratio Distribution"
        )
        a6.set_xlabel("σmin / σmax")
        a6.set_ylabel("Frames")
        a6.grid(True,axis="y",alpha=.3)
        a6.legend(fontsize=7)

        self.set_figure_pair_title(
            self.singularity_fig,
            label="Singularity Evaluation"
        )
        self.singularity_fig.subplots_adjust(
            left=0.07,right=0.98,bottom=0.08,top=0.88,
            hspace=0.48,wspace=0.30
        )
        self.singularity_canvas.draw_idle()

        near_count=int(np.sum(near))
        rank_loss=int(np.sum(d["rank"]<2))
        min_ratio=float(np.min(ratio))
        max_cond=float(np.max(d["condition"]))

        self.singularity_summary.set(
            f"Near-singular frames={near_count}/{len(t)} | "
            f"Rank-loss frames={rank_loss}/{len(t)} | "
            f"Minimum σmin/σmax={min_ratio:.6f} | "
            f"Maximum condition number={max_cond:.3e} | "
            f"Threshold={threshold:.3f}"
        )

    def export_singularity_csv(self):
        d=self.compute_error_cache()
        if d is None:
            return

        try:
            threshold=float(self.singularity_ratio_threshold.get())
        except Exception:
            threshold=0.05

        path=filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")],
            initialfile=(
                f"{self.graph_pair_text().replace(' ','_').replace('+','plus')}"
                "_singularity_analysis.csv"
            )
        )
        if not path:
            return

        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow([
                "time_s",
                "sigma_min",
                "condition_number",
                "manipulability",
                "jacobian_rank",
                "singularity_ratio_sigma_min_over_sigma_max",
                "near_singular"
            ])
            for i in range(len(d["t"])):
                w.writerow([
                    d["t"][i],
                    d["sigma_min"][i],
                    d["condition"][i],
                    d["manipulability"][i],
                    d["rank"][i],
                    d["singularity_ratio"][i],
                    int(d["singularity_ratio"][i] < threshold)
                ])

    def export_motion_csv(self):
        d=self.compute_motion_cache()
        if d is None: return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],
                                          initialfile="snake_motion_analysis.csv")
        if not path: return
        headers=["time_s","head_x","head_y","displacement","cumulative_distance",
                 "vx","vy","speed","ax","ay","acceleration"]
        headers += [f"J{i+1}_deg" for i in range(d["q"].shape[1])]
        headers += [f"J{i+1}_vel_deg_s" for i in range(d["qvel"].shape[1])]
        headers += [f"J{i+1}_acc_deg_s2" for i in range(d["qacc"].shape[1])]
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(headers)
            for i in range(len(d["t"])):
                w.writerow([d["t"][i],d["head"][i,0],d["head"][i,1],d["displacement"][i],d["cumulative"][i],
                            d["vel"][i,0],d["vel"][i,1],d["speed"][i],d["acc"][i,0],d["acc"][i,1],d["accel"][i],
                            *d["q"][i],*d["qvel"][i],*d["qacc"][i]])

    def export_error_csv(self):
        d=self.compute_error_cache()
        if d is None: return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],
                                          initialfile="snake_error_singularity_analysis.csv")
        if not path: return
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["time_s","head_destination_error","heading_error_deg","sigma_min",
                        "condition_number","manipulability","body_clearance","jacobian_rank"])
            for i in range(len(d["t"])):
                w.writerow([d["t"][i],d["head_error"][i],d["heading_error"][i],d["sigma_min"][i],
                            d["condition"][i],d["manipulability"][i],d["clearance"][i],d["rank"][i]])

    def save_figure_generic(self,fig,default_name):
        pair_slug=self.pair_label().replace(" ","_").replace("*","star").replace("+","plus")
        stem=Path(default_name).stem
        suffix=Path(default_name).suffix or ".png"
        paired_default=f"{pair_slug}_{stem}{suffix}"
        path=filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],
            initialfile=paired_default
        )
        if path:
            fig.savefig(path,dpi=300,bbox_inches="tight"); self.status.set(f"Figure saved: {path}")

    def save_all_research_outputs(self):
        folder=filedialog.askdirectory(title="Select folder for research outputs")
        if not folder: return
        folder=Path(folder)
        try:self.refresh_motion_analysis()
        except Exception:pass
        try:self.refresh_error_analysis()
        except Exception:pass
        try:self.refresh_singularity_analysis()
        except Exception:pass
        try:self.refresh_workspace_analysis()
        except Exception:pass
        try:self.ml_plot_results()
        except Exception:pass

        for fig,name in [
            (self.fig,"01_path_planning_simulation.png"),
            (self.ml_fig,"02_ml_training_commands.png"),
            (self.motion_fig,"03_motion_analysis.png"),
            (self.error_fig,"04_head_error_safety.png"),
            (self.singularity_fig,"05_singularity_analysis.png"),
            (self.workspace_fig,"06_workspace_analysis.png"),
            (self.cfig,"07_planner_comparison.png")
        ]:
            try: fig.savefig(folder/name,dpi=300,bbox_inches="tight")
            except Exception: pass

        if self.analysis_cache is not None:
            d=self.analysis_cache
            with open(folder/"motion_summary.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["metric","value"])
                w.writerow(["travel_distance",d["cumulative"][-1]])
                w.writerow(["peak_speed",float(np.max(d["speed"]))])
                w.writerow(["peak_acceleration",float(np.max(d["accel"]))])
                w.writerow(["peak_joint_velocity",float(np.max(np.abs(d["qvel"])))])
                w.writerow(["peak_joint_acceleration",float(np.max(np.abs(d["qacc"])))])

        if self.error_cache is not None:
            d=self.error_cache
            with open(folder/"error_summary.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["metric","value"])
                w.writerow(["final_head_error",d["head_error"][-1]])
                w.writerow(["max_heading_error_deg",float(np.max(d["heading_error"]))])
                w.writerow(["min_sigma",float(np.min(d["sigma_min"]))])
                w.writerow(["max_condition",float(np.max(d["condition"]))])
                w.writerow(["min_body_clearance",float(np.min(d["clearance"]))])

        if self.workspace_cache is not None:
            d=self.workspace_cache
            with open(folder/"workspace_summary.csv","w",newline="",encoding="utf-8") as f:
                w=csv.writer(f)
                w.writerow(["metric","value"])
                w.writerow(["dof",self.dof])
                w.writerow(["link_length",self.robot.link])
                w.writerow(["samples",d["samples"]])
                w.writerow(["joint_limit_deg",d["joint_limit_deg"]])
                w.writerow(["sampled_min_reach",float(np.min(d["radius"]))])
                w.writerow(["sampled_max_reach",float(np.max(d["radius"]))])
                w.writerow(["theoretical_max_reach",d["theoretical_max_reach"]])
                w.writerow(["empirical_workspace_area",d["empirical_area"]])
                w.writerow(["coverage_percent_of_theoretical_disk",d["coverage_pct"]])
                w.writerow(["mean_singularity_ratio",float(np.mean(d["singularity_ratio"]))])
                w.writerow(["mean_manipulability",float(np.mean(d["manipulability"]))])

        self.export_status.set(f"Research outputs saved to: {folder}")
        self.status.set("All research outputs saved")

    # ---------- online / dynamic-obstacle navigation ----------
    def create_dynamic_obstacle(self,p):
        if self.dynamic_shape.get()=="Rectangle":
            w=max(float(self.rect_w.get()),1.0)
            h=max(float(self.rect_h.get()),1.0)
            return {
                "type":"rect",
                "x":float(p[0]),"y":float(p[1]),
                "w":w,"h":h,
                "dynamic":True
            }

        r=max(float(self.circle_r.get()),1.0)
        return {
            "type":"circle",
            "x":float(p[0]),"y":float(p[1]),
            "r":r,
            "dynamic":True
        }

    def pause_runtime_animation(self):
        """Pause without deleting the active route/sequence."""
        self.animating=False
        if self.anim_job is not None:
            try:
                self.after_cancel(self.anim_job)
            except Exception:
                pass
        self.anim_job=None

    def current_runtime_heading(self):
        if len(self.active_head_trail)>=2:
            v=np.asarray(self.active_head_trail[-1])-np.asarray(self.active_head_trail[-2])
            h=vector_heading_deg(v)
            if h is not None:
                return h

        if self.active_body is not None and len(self.active_body)>=2:
            return vector_heading_deg(
                np.asarray(self.active_body[-1])-np.asarray(self.active_body[-2])
            )
        return None

    def reached_destination_count(self):
        if self.active_seq is None:
            return 0
        ids=self.active_seq.get("destination_frame_indices",[])
        # active_index points to the NEXT frame after the one currently visible.
        current_frame=max(self.active_index-1,0)
        return sum(1 for idx in ids if idx <= current_frame)

    def future_route_is_blocked(self):
        """
        Check the remaining HEAD route against the newly updated obstacle map
        using the same conservative full-body planning inflation.
        """
        if self.active_seq is None:
            return True

        route=np.asarray(self.active_seq.get("head_route",[]),dtype=float)
        if len(route)<2:
            return False

        # active_index is the next frame; start checking from the currently
        # displayed head position.
        start_idx=max(min(self.active_index-1,len(route)-1),0)
        future=route[start_idx:]
        if len(future)==0:
            return False

        inflation=self.effective_planning_clearance()
        sample=max(float(self.grid_res.get())/4.0,3.0)

        for p in future:
            if point_collision(p,self.obstacles,inflation):
                return True

        for a,b in zip(future[:-1],future[1:]):
            if segment_collision(a,b,self.obstacles,inflation,sample):
                return True

        return False

    def replan_from_current_state(self):
        """
        Replan only the unfinished mission:
        Current HEAD -> current unfinished destination -> ... -> last destination.

        The incoming heading and already travelled head path are carried into
        the new plan, so the online replan does not reset the robot to Start.
        """
        if self.active_body is None:
            return False

        current_head=np.asarray(self.active_body[-1],dtype=float).copy()
        incoming_heading=self.current_runtime_heading()
        reached=self.reached_destination_count()
        remaining=[np.asarray(p,float).copy() for p in self.destinations[reached:]]

        if not remaining:
            self.dynamic_state.set("GREEN — mission already complete")
            return True

        old_destinations=[np.asarray(p,float).copy() for p in self.destinations]
        old_start=np.asarray(self.start,float).copy()

        # Previous head history acts as a no-retrace corridor.
        history=np.asarray(self.active_head_trail,dtype=float)
        if len(history)>2:
            # Avoid over-constraining the current departure point.
            history=history[:-1]

        alg=self.current_algorithm or self.alg.get()
        route=[current_head.copy()]
        segments=[]
        total_nodes=0
        a=current_head.copy()
        heading=incoming_heading

        for i,b in enumerate(remaining):
            p,n=self.plan_leg(
                alg,a,b,
                seed=7001+self.dynamic_replan_count*503+i*101,
                incoming_heading=heading,
                history_points=np.asarray(route if len(route)>1 else history,dtype=float)
                if len(history)==0 else np.vstack([history,np.asarray(route,dtype=float)])
            )
            total_nodes += n

            if p is None:
                self.start=old_start
                self.destinations=old_destinations
                return False

            p=np.asarray(p,float)
            segments.append(p.copy())
            route.extend(p[1:].tolist())

            if len(p)>=2:
                heading=vector_heading_deg(p[-1]-p[-2])
            a=np.asarray(b,float)

        route=np.asarray(route,float)

        # Online mission now begins from the CURRENT head and contains only
        # unfinished destinations.
        self.start=current_head.copy()
        self.destinations=remaining
        self.current_path=route
        self.current_segments=segments
        self.current_algorithm=alg

        center_cl=route_clearance(route,self.obstacles)
        est_cl=(
            center_cl-self.full_body_margin()
            if np.isfinite(center_cl) else float("inf")
        )

        self.invalidate_ml_training(
            "dynamic obstacle changed the online reference path",
            clear_models=True
        )

        self.current_result={
            "algorithm":alg,
            "success":True,
            "path":route,
            "segments":segments,
            "nodes":total_nodes,
            "path_length":route_length(route),
            "smoothness":route_smoothness(route),
            "clearance":est_cl,
            "destinations":len(remaining),
            "dynamic_replan":True
        }

        # Use the already travelled HEAD history as the body-history prefix so
        # the tail does not teleport when the new plan begins.
        new_seq=self.anim_sequence(prefix_history=self.active_head_trail)
        if new_seq is None:
            self.start=old_start
            self.destinations=old_destinations
            return False

        self.dynamic_replan_count += 1
        self.refresh_dest()
        self.active_seq=new_seq
        self.active_index=0
        self.active_head_trail=list(self.active_head_trail)
        self.active_body=new_seq["body_frames"][0].copy()

        # Keep the original reference for analysis.  The executed trajectory is
        # obtained from runtime_body_log and grows continuously across replans.
        try:
            self.ml_plot_results()
        except Exception:
            pass

        return True

    def handle_dynamic_obstacle_click(self,p):
        """
        GREEN  : new obstacle does not threaten remaining route -> continue.
        YELLOW : future route blocked -> pause and online replan.
        RED    : obstacle intersects current full body -> emergency stop.
        """
        self.dynamic_event_count += 1
        obstacle=self.create_dynamic_obstacle(p)

        # Record the exact executed-animation frame at which the obstacle
        # appeared.  GIF/MP4 export uses this to make the obstacle visible
        # only from this moment onward.
        obstacle["appear_frame"]=len(self.runtime_body_log)
        obstacle["dynamic_event_id"]=self.dynamic_event_count
        self.obstacles.append(obstacle)

        # Freeze first so route/body tests are performed on a definite state.
        self.pause_runtime_animation()

        # Current-body collision is an emergency; replanning cannot undo a
        # collision that already exists at the instant the object appears.
        if self.active_body is not None and self.body_frame_collision(
            self.active_body,
            include_user_clearance=False
        ):
            self.dynamic_state.set("RED — EMERGENCY STOP")
            self.status.set(
                "EMERGENCY STOP: sudden obstacle intersects the current snake body"
            )
            self.redraw((self.active_body,self.active_head_trail,True))
            messagebox.showwarning(
                "Emergency Stop",
                "The new obstacle appeared inside the current snake body/safety envelope.\\n\\n"
                "The robot has stopped. Move/remove the obstacle before continuing."
            )
            return

        blocked=self.future_route_is_blocked()

        if not blocked:
            self.dynamic_state.set("GREEN — route remains safe")
            self.status.set("Dynamic obstacle added — current route remains safe")
            self.redraw((self.active_body,self.active_head_trail,False))
            self.resume_active_sequence()
            return

        self.dynamic_state.set("YELLOW — replanning")
        self.status.set("Dynamic obstacle blocks future route — replanning from current HEAD")
        self.redraw((self.active_body,self.active_head_trail,False))
        self.update_idletasks()

        if not self.dynamic_auto_replan.get():
            messagebox.showinfo(
                "Route Blocked",
                "The new obstacle blocks the remaining route.\\n\\n"
                "Automatic replanning is disabled, so the snake remains paused."
            )
            return

        success=self.replan_from_current_state()

        if not success:
            self.dynamic_state.set("RED — NO SAFE REPLAN")
            self.status.set("Emergency stop: no feasible online replan found")
            messagebox.showwarning(
                "No Safe Replan",
                "The selected algorithm could not find a safe route from the current "
                "head position to the remaining destinations.\\n\\n"
                "The snake remains stopped."
            )
            return

        self.dynamic_state.set("GREEN — replanned")
        self.status.set(
            f"Online replan #{self.dynamic_replan_count} complete — continuing mission"
        )
        self.redraw((self.active_body,self.active_head_trail,False))
        self.resume_active_sequence()

    # ---------- planning ----------
    def settings(self):
        return (
            max(float(self.clearance.get()),0.0),
            max(float(self.grid_res.get()),4.0),
            max(float(self.rrt_step.get()),5.0),
            max(int(self.rrt_iter.get()),100)
        )

    def directional_settings(self):
        if not hasattr(self,"directional_mode") or not self.directional_mode.get():
            return {
                "enabled":False,
                "max_turn_deg":179.0,
                "retrace_radius":0.0,
                "turn_penalty":0.0,
                "release_radius":0.0
            }

        try:
            max_turn=float(np.clip(float(self.max_turn_var.get()),45.0,150.0))
        except Exception:
            max_turn=100.0

        try:
            retrace=max(float(self.retrace_radius_var.get()),0.0)
        except Exception:
            retrace=32.0

        try:
            penalty=max(float(self.turn_penalty_var.get()),0.0)
        except Exception:
            penalty=0.45

        # Allow the route to leave the current destination before the previous
        # corridor becomes forbidden.
        release=max(
            1.75*float(self.robot.link),
            retrace*1.5,
            25.0
        )

        return {
            "enabled":True,
            "max_turn_deg":max_turn,
            "retrace_radius":retrace,
            "turn_penalty":penalty,
            "release_radius":release
        }

    def full_body_margin(self):
        """
        Extra centerline margin used because a finite-length link can cut
        inside a path corner even when its endpoints are on a safe centerline.
        """
        try:
            radius=max(float(self.body_radius.get()),0.0)
        except Exception:
            radius=5.0
        return radius + 0.60*float(self.robot.link)

    def effective_planning_clearance(self):
        user_clearance,_,_,_=self.settings()
        return user_clearance + self.full_body_margin()

    def body_frame_collision(self, points, include_user_clearance=True):
        """Check every body joint and every link segment against obstacles."""
        pts=np.asarray(points,dtype=float)
        try:
            radius=max(float(self.body_radius.get()),0.0)
        except Exception:
            radius=5.0

        extra=radius
        if include_user_clearance:
            extra += max(float(self.clearance.get()),0.0)

        for p in pts:
            # Workspace bounds constrain the HEAD planner.  The body/tail may
            # temporarily extend outside those display/planning bounds,
            # especially at the initial head position.
            if point_collision(p,self.obstacles,extra):
                return True

        sample_step=max(float(self.robot.link)/12.0,2.0)
        for i in range(len(pts)-1):
            if segment_collision(
                pts[i],pts[i+1],self.obstacles,extra,sample_step
            ):
                return True
        return False

    def validate(self):
        if not self.destinations:
            messagebox.showinfo("Planner","Add at least one destination.")
            return False

        effective=self.effective_planning_clearance()
        for i,p in enumerate([self.start]+self.destinations):
            label="Head Start" if i==0 else f"Head Destination D{i}"
            if not inside_bounds(p,self.bounds()):
                messagebox.showerror("Planner",f"{label} is outside the workspace.")
                return False
            if point_collision(p,self.obstacles,effective):
                messagebox.showerror(
                    "Planner",
                    f"{label} is too close to an obstacle for the selected full-body safety margin.\\n\\n"
                    f"Effective planning clearance = {effective:.2f} units."
                )
                return False
        return True

    def plan_leg(self,alg,a,b,seed,incoming_heading=None,history_points=None):
        _,res,step,it=self.settings()
        cl=self.effective_planning_clearance()
        ds=self.directional_settings()

        kwargs=dict(
            start_heading_deg=incoming_heading if ds["enabled"] else None,
            max_turn_deg=ds["max_turn_deg"],
            history_points=history_points if ds["enabled"] else None,
            retrace_radius=ds["retrace_radius"] if ds["enabled"] else 0.0,
            release_radius=ds["release_radius"] if ds["enabled"] else 0.0
        )

        if alg=="Dijkstra":
            path,n=grid_search(
                a,b,self.obstacles,self.bounds(),res,cl,False,
                turn_penalty=ds["turn_penalty"],
                **kwargs
            )
        elif alg=="A*":
            path,n=grid_search(
                a,b,self.obstacles,self.bounds(),res,cl,True,
                turn_penalty=ds["turn_penalty"],
                **kwargs
            )
        elif alg=="RRT":
            path,n=rrt(
                a,b,self.obstacles,self.bounds(),
                step,it,max(step*1.25,20),cl,False,seed,
                **kwargs
            )
        else:
            path,n=rrt(
                a,b,self.obstacles,self.bounds(),
                step,it,max(step*1.25,20),cl,True,seed,
                **kwargs
            )

        if path is not None and self.smooth.get():
            # Smoothing remains collision-safe. In directional mode, we keep
            # more route structure so smoothing cannot recreate a U-turn.
            path=simplify(path,self.obstacles,cl,max(res/4,3))

            if ds["enabled"] and len(path)>=3:
                # Reject smoothed local reversals that violate the same turn limit.
                cleaned=[path[0]]
                prev_heading=incoming_heading
                for k in range(1,len(path)):
                    cand=path[k]
                    new_heading=vector_heading_deg(cand-cleaned[-1])
                    if new_heading is None:
                        continue
                    if prev_heading is not None:
                        if abs(angle_diff_deg(new_heading,prev_heading)) > ds["max_turn_deg"]:
                            # Preserve the unsmoothed route instead; it was
                            # generated under directional constraints.
                            cleaned=None
                            break
                    cleaned.append(cand)
                    prev_heading=new_heading

                if cleaned is not None and len(cleaned)>=2:
                    path=np.asarray(cleaned,float)

        return path,n


    def solve(self,alg,initial_heading_override=None):
        if not self.validate():
            return None

        t0=time.perf_counter()
        route=[self.start.copy()]
        segments=[]
        destination_route_indices=[]
        total_nodes=0
        a=self.start.copy()

        incoming_heading=initial_heading_override
        ds=self.directional_settings()

        for i,b in enumerate(self.destinations):
            history=np.asarray(route,dtype=float) if (ds["enabled"] and i>0) else None

            p,n=self.plan_leg(
                alg,a,b,
                42+i*97+["Dijkstra","A*","RRT","RRT*"].index(alg)*1009,
                incoming_heading=incoming_heading,
                history_points=history
            )
            total_nodes += n

            if p is None:
                return {
                    "algorithm":alg,
                    "success":False,
                    "failed_leg":i+1,
                    "planning_time":time.perf_counter()-t0,
                    "nodes":total_nodes,
                    "segments":segments
                }

            p=np.asarray(p,dtype=float)

            # Final guard: do not accept an explicit near-180° first departure.
            if ds["enabled"] and incoming_heading is not None and len(p)>=2:
                first_heading=vector_heading_deg(p[1]-p[0])
                if first_heading is not None:
                    reversal=abs(angle_diff_deg(first_heading,incoming_heading))
                    if reversal > ds["max_turn_deg"]:
                        return {
                            "algorithm":alg,
                            "success":False,
                            "failed_leg":i+1,
                            "planning_time":time.perf_counter()-t0,
                            "nodes":total_nodes,
                            "segments":segments,
                            "reason":"direction reversal"
                        }

            segments.append(p.copy())
            route.extend(p[1:].tolist())
            destination_route_indices.append(len(route)-1)

            # Carry terminal heading into the next destination leg.
            if len(p)>=2:
                incoming_heading=vector_heading_deg(p[-1]-p[-2])

            a=np.asarray(b,float)

        route=np.asarray(route,float)
        elapsed=time.perf_counter()-t0

        center_cl=route_clearance(route,self.obstacles)
        margin=self.full_body_margin()
        estimated_body_clearance=(
            center_cl-margin if np.isfinite(center_cl) else float("inf")
        )

        # Detect direct path retracing / reversal in the final centerline.
        reverse_events=0
        if len(route)>=3:
            d=np.diff(route,axis=0)
            headings=[
                vector_heading_deg(v) for v in d if np.linalg.norm(v)>1e-9
            ]
            for h1,h2 in zip(headings[:-1],headings[1:]):
                if abs(angle_diff_deg(h2,h1)) > 150.0:
                    reverse_events += 1

        return {
            "algorithm":alg,
            "success":True,
            "failed_leg":None,
            "planning_time":elapsed,
            "nodes":total_nodes,
            "path":route,
            "segments":segments,
            "destination_route_indices":destination_route_indices,
            "path_length":route_length(route),
            "smoothness":route_smoothness(route),
            "clearance":estimated_body_clearance,
            "centerline_clearance":center_cl,
            "effective_inflation":self.effective_planning_clearance(),
            "points":len(route),
            "destinations":len(self.destinations),
            "directional_mode":ds["enabled"],
            "max_turn_deg":ds["max_turn_deg"],
            "retrace_radius":ds["retrace_radius"],
            "reverse_events":reverse_events
        }


    def plan_selected(self):
        alg=self.alg.get()
        self.status.set(f"Planning {alg} full-body-safe HEAD route...")
        self.update_idletasks()

        r=self.solve(alg)
        if r is None:
            return

        self.current_algorithm=alg
        self.current_result=r

        if not r["success"]:
            self.current_path=None
            self.current_segments=[]
            self.redraw()
            messagebox.showwarning(
                "Planner",
                f"{alg} failed at destination D{r['failed_leg']} with the full-body safety margin."
            )
            return

        self.current_path=r["path"]
        self.current_segments=r.get("segments",[])

        # Route-imitation models depend on the exact reference path and must
        # be retrained. ML-FIRST environment models remain valid because path
        # planning is intentionally executed AFTER the ML command.
        if self.ml_training_scope=="route":
            self.invalidate_ml_training(
                f"new {alg} reference path planned",
                clear_models=True
            )

        self.redraw()

        clr="∞" if not np.isfinite(r["clearance"]) else f"{r['clearance']:.2f}"
        direction_txt=(
            f"Directional ON | MaxTurn={r['max_turn_deg']:.0f}° | "
            f"NoRetrace={r['retrace_radius']:.1f}"
            if r.get("directional_mode") else "Directional OFF"
        )
        self.info.set(
            f"{alg} HEAD route | Dests={r['destinations']} | "
            f"Length={r['path_length']:.2f} | T={r['planning_time']:.4f}s | "
            f"Nodes={r['nodes']} | Body clearance={clr} | "
            f"{direction_txt} | Reverse events={r.get('reverse_events',0)}"
        )
        self.status.set(
            f"{alg} route planned | FRESH ML source ready: "
            f"{len(self.current_segments)} segment(s), "
            f"{len(self.current_path)} path point(s) | "
            f"scenario={self.ml_signature_id()}"
        )

    def compare_all(self):
        if not self.validate(): return
        self.compare_results={}
        for alg in ["Dijkstra","A*","RRT","RRT*"]:
            self.status.set(f"Running {alg}..."); self.update_idletasks()
            self.compare_results[alg]=self.solve(alg)
        if self.ml_training_scope=="route":
            self.invalidate_ml_training(
                "planner comparison routes updated",
                clear_models=True
            )
            status_msg="Comparison complete — route-imitation ML must be retrained"
        else:
            status_msg="Comparison complete — ML-FIRST model preserved"
        self.draw_compare()
        self.nb.select(self.comp_tab)
        self.status.set(status_msg)

    # ---------- drawing ----------
    def initial_body_points(self):
        """
        Initial head-led snake pose.

        The planning Start point is the HEAD position.  The initial body is a
        straight chain extending backward from that head.  If D1 already
        exists, the chain is initially aligned toward D1; otherwise it uses
        the +Global-X direction.
        """
        head=np.asarray(self.start,dtype=float)

        heading=0.0
        if self.destinations:
            h=vector_heading_deg(
                np.asarray(self.destinations[0],dtype=float)-head
            )
            if h is not None:
                heading=h

        th=math.radians(heading)
        forward=np.array([math.cos(th),math.sin(th)],dtype=float)

        offsets=np.arange(
            self.dof,-1,-1,dtype=float
        )*float(self.robot.link)

        # tail -> ... -> head
        pts=head[None,:]-offsets[:,None]*forward[None,:]
        pts[-1]=head
        return pts

    def draw_axes_representation(self,body_pts):
        """
        Conventional robotics frame colours:

            X-axis = RED
            Y-axis = GREEN
            Z-axis = BLUE

        The simulator is planar (X-Y), so +Z is perpendicular to the display.
        It is represented by a blue circled-dot symbol at each frame origin.
        The same convention is used for GLOBAL and LOCAL frames.
        """
        if not getattr(self,"show_axes_var",None):
            return
        if not self.show_axes_var.get():
            return

        pts=np.asarray(body_pts,dtype=float)
        if len(pts)<2:
            return

        xmin,xmax=self.ax.get_xlim()
        ymin,ymax=self.ax.get_ylim()
        visible_span=max(min(xmax-xmin,ymax-ymin),1.0)

        global_len=min(
            max(2.6*float(self.robot.link),70.0),
            0.18*visible_span
        )

        try:
            local_scale=max(
                float(self.axis_scale_var.get()),
                0.15
            )
        except Exception:
            local_scale=0.60

        local_len=min(
            max(
                local_scale*float(self.robot.link),
                8.0
            ),
            0.08*visible_span
        )

        # ============================================================
        # GLOBAL FRAME
        # ============================================================
        global_lw=4.5

        # +X_G : RED
        self.ax.annotate(
            "",
            xy=(global_len,0.0),
            xytext=(0.0,0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="red",
                lw=global_lw,
                mutation_scale=17,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=30
        )

        # +Y_G : GREEN
        self.ax.annotate(
            "",
            xy=(0.0,global_len),
            xytext=(0.0,0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="green",
                lw=global_lw,
                mutation_scale=17,
                shrinkA=0,
                shrinkB=0
            ),
            zorder=30
        )

        # +Z_G : BLUE, out of the X-Y screen plane.
        self.ax.scatter(
            [0.0],[0.0],
            s=145,
            facecolors="none",
            edgecolors="blue",
            linewidths=3.2,
            zorder=31
        )
        self.ax.scatter(
            [0.0],[0.0],
            s=24,
            c="blue",
            zorder=32
        )

        self.ax.text(
            global_len,0.0,
            "  X_G",
            color="red",
            fontweight="bold",
            fontsize=10,
            va="center",
            zorder=33
        )
        self.ax.text(
            0.0,global_len,
            "  Y_G",
            color="green",
            fontweight="bold",
            fontsize=10,
            va="bottom",
            zorder=33
        )
        self.ax.text(
            0.0,0.0,
            "  Z_G",
            color="blue",
            fontweight="bold",
            fontsize=10,
            va="bottom",
            zorder=33
        )

        # ============================================================
        # LOCAL JOINT FRAMES
        # ============================================================
        for i,p in enumerate(pts):
            if i < len(pts)-1:
                tangent=pts[i+1]-pts[i]
            else:
                tangent=pts[i]-pts[i-1]

            norm=np.linalg.norm(tangent)
            if norm<1e-12:
                tangent=np.array([1.0,0.0])
            else:
                tangent=tangent/norm

            normal=np.array(
                [-tangent[1],tangent[0]],
                dtype=float
            )

            x_end=p+local_len*tangent
            y_end=p+local_len*normal

            # Local X_i = RED
            self.ax.annotate(
                "",
                xy=(x_end[0],x_end[1]),
                xytext=(p[0],p[1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="red",
                    lw=2.6,
                    mutation_scale=11,
                    shrinkA=0,
                    shrinkB=0,
                    alpha=0.88
                ),
                zorder=24
            )

            # Local Y_i = GREEN
            self.ax.annotate(
                "",
                xy=(y_end[0],y_end[1]),
                xytext=(p[0],p[1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="green",
                    lw=2.6,
                    mutation_scale=11,
                    shrinkA=0,
                    shrinkB=0,
                    alpha=0.88
                ),
                zorder=24
            )

            # Local Z_i = BLUE, represented as out-of-plane circled dot.
            self.ax.scatter(
                [p[0]],[p[1]],
                s=48,
                facecolors="none",
                edgecolors="blue",
                linewidths=1.8,
                alpha=0.90,
                zorder=25
            )
            self.ax.scatter(
                [p[0]],[p[1]],
                s=8,
                c="blue",
                alpha=0.90,
                zorder=26
            )

            # Compact local-axis labels.
            self.ax.text(
                x_end[0],x_end[1],
                f"x{i}",
                color="red",
                fontsize=6.5,
                fontweight="bold",
                zorder=27
            )
            self.ax.text(
                y_end[0],y_end[1],
                f"y{i}",
                color="green",
                fontsize=6.5,
                fontweight="bold",
                zorder=27
            )
            self.ax.text(
                p[0],p[1],
                f" z{i}",
                color="blue",
                fontsize=6.2,
                fontweight="bold",
                va="bottom",
                zorder=27
            )

    def redraw(self,anim=None):
        self.ax.clear()

        # Always have a visible snake.  Before animation starts, show the
        # straight initial pose with the HEAD exactly at Start.
        if anim is not None:
            display_body=np.asarray(anim[0],dtype=float)
        else:
            display_body=self.initial_body_points()

        xmin,xmax,ymin,ymax=self.bounds()

        # Expand view so the full initial/animated snake and its frames are
        # visible even when the HEAD starts near a workspace boundary.
        if len(display_body):
            try:
                axis_scale=max(float(self.axis_scale_var.get()),0.15)
            except Exception:
                axis_scale=0.60

            pad=max(
                float(self.robot.link)*(1.5+axis_scale),
                35.0
            )

            xmin=min(
                xmin,
                float(np.min(display_body[:,0]))-pad
            )
            xmax=max(
                xmax,
                float(np.max(display_body[:,0]))+pad
            )
            ymin=min(
                ymin,
                float(np.min(display_body[:,1]))-pad
            )
            ymax=max(
                ymax,
                float(np.max(display_body[:,1]))+pad
            )

        self.ax.set_xlim(xmin,xmax)
        self.ax.set_ylim(ymin,ymax)
        self.ax.set_aspect("equal",adjustable="box")
        self.ax.grid(True,alpha=.28)
        self.ax.set_xlabel("World X")
        self.ax.set_ylabel("World Y")
        self.ax.set_title(
            f"{self.graph_pair_text()}\n"
            f"Snake Robot Frames: X=Red, Y=Green, Z=Blue | "
            f"events={self.dynamic_event_count} | replans={self.dynamic_replan_count}",
            fontsize=10,
            pad=8
        )

        # Obstacles
        for k,o in enumerate(self.obstacles):
            is_dynamic=bool(o.get("dynamic",False))
            edge_col="red" if is_dynamic else None
            face_col="red" if is_dynamic else None
            label=(
                "Dynamic Obstacle"
                if is_dynamic and k==0
                else ("Obstacle" if k==0 else None)
            )

            if o["type"]=="circle":
                self.ax.add_patch(
                    Circle(
                        (o["x"],o["y"]),
                        o["r"],
                        alpha=.3,
                        edgecolor=edge_col,
                        facecolor=face_col,
                        linewidth=2.0 if is_dynamic else None,
                        label=label
                    )
                )
            else:
                self.ax.add_patch(
                    Rectangle(
                        (
                            o["x"]-o["w"]/2,
                            o["y"]-o["h"]/2
                        ),
                        o["w"],o["h"],
                        alpha=.3,
                        edgecolor=edge_col,
                        facecolor=face_col,
                        linewidth=2.0 if is_dynamic else None,
                        label=label
                    )
                )

        # Start / destinations
        self.ax.scatter(
            [self.start[0]],[self.start[1]],
            marker="s",s=90,label="Head start",zorder=13
        )
        self.ax.text(
            self.start[0],self.start[1],
            " HEAD Start",
            fontsize=8,zorder=14
        )

        for i,p in enumerate(self.destinations,1):
            self.ax.scatter(
                [p[0]],[p[1]],
                marker="*",s=115,zorder=13
            )
            self.ax.text(
                p[0],p[1],
                f" D{i} (HEAD)",
                fontsize=8,zorder=14
            )

        # Reference / current head routes.
        # Once dynamic motion starts, current_path may contain only the
        # remaining replanned segment; keep the original route visible too.
        ref=self.full_reference_path()

        if ref is not None and len(ref)>=2:
            self.ax.plot(
                ref[:,0],ref[:,1],
                "--",
                lw=1.8,
                alpha=.75,
                label="Original reference path",
                zorder=4
            )

        if self.current_path is not None:
            p=np.asarray(self.current_path)
            show_current=(
                ref is None
                or len(p)!=len(ref)
                or not np.allclose(
                    p,
                    ref[:len(p)]
                    if len(ref)>=len(p)
                    else p,
                    atol=1e-6
                )
            )
            if show_current:
                self.ax.plot(
                    p[:,0],p[:,1],
                    lw=2.1,
                    label="Current / replanned path",
                    zorder=5
                )

        # Snake body: shown even at initial state.
        self.ax.plot(
            display_body[:,0],
            display_body[:,1],
            "-o",
            lw=3.0,
            ms=5,
            label="Snake body",
            zorder=10
        )

        self.ax.scatter(
            [display_body[-1,0]],
            [display_body[-1,1]],
            marker="*",
            s=125,
            label="HEAD",
            zorder=15
        )
        self.ax.scatter(
            [display_body[0,0]],
            [display_body[0,1]],
            marker="s",
            s=75,
            label="TAIL",
            zorder=15
        )

        # Global and all local joint coordinate frames.
        self.draw_axes_representation(display_body)

        # Animation-only information.
        if anim is not None:
            body_pts,htrail,unsafe=anim

            if len(htrail)>1:
                h=np.asarray(htrail)
                self.ax.plot(
                    h[:,0],h[:,1],
                    lw=1.2,
                    alpha=.55,
                    label="Head trail",
                    zorder=6
                )

            if unsafe:
                self.ax.text(
                    0.02,0.98,
                    "WARNING: body-clearance violation",
                    transform=self.ax.transAxes,
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                    zorder=30
                )
        else:
            self.ax.text(
                0.02,0.98,
                "Initial Snake Pose — X Red | Y Green | Z Blue",
                transform=self.ax.transAxes,
                va="top",
                fontsize=10,
                fontweight="bold",
                zorder=30
            )

        self.ax.legend(
            loc="upper right",
            fontsize=8
        )
        self.canvas.draw_idle()

    def draw_compare(self):
        for a in self.caxes:a.clear()
        if not self.compare_results:
            self.ccanvas.draw_idle(); self.comp_text.set("No comparison yet."); return
        names=[n for n in ["Dijkstra","A*","RRT","RRT*"] if self.compare_results.get(n,{}).get("success")]
        if names:
            vals=[
                [self.compare_results[n]["path_length"] for n in names],
                [self.compare_results[n]["planning_time"] for n in names],
                [self.compare_results[n]["nodes"] for n in names],
                [self.compare_results[n]["smoothness"] for n in names]
            ]
            titles=["Total Path Length","Planning Time","Explored / Generated Nodes","Smoothness Cost"]
            yl=["Length","Seconds","Nodes","Total heading change (deg)"]
            ml_name=self.eval_ml_var.get() or "Untrained ML"
            for ax,v,t,y in zip(self.caxes,vals,titles,yl):
                ax.bar(names,v)
                self.set_panel_title(
                    ax,
                    f"Planner Comparison — {t}",
                    pair=f"ML: {ml_name}",
                    fontsize=8.2
                )
                ax.set_ylabel(y)
                ax.grid(True,axis="y",alpha=.3)
        self.cfig.suptitle(
            f"Planner Comparison | ML: {self.graph_pair_text(self.eval_ml_var.get() or 'Untrained ML')}",
            fontsize=10.5,
            fontweight="semibold",
            y=0.98
        )
        self.cfig.subplots_adjust(
            left=0.08,right=0.98,bottom=0.09,top=0.87,
            hspace=0.44,wspace=0.28
        )
        self.ccanvas.draw_idle()

        lines=[]
        for n in ["Dijkstra","A*","RRT","RRT*"]:
            if n not in self.compare_results:continue
            r=self.compare_results[n]
            if r["success"]:
                c="∞" if not np.isfinite(r["clearance"]) else f"{r['clearance']:.2f}"
                lines.append(f"{n:9s} SUCCESS | L={r['path_length']:.2f} | T={r['planning_time']:.4f}s | Nodes={r['nodes']} | Smooth={r['smoothness']:.2f}° | Clearance={c}")
            else:
                lines.append(f"{n:9s} FAILED at D{r['failed_leg']} | T={r['planning_time']:.4f}s | Nodes={r['nodes']}")
        self.comp_text.set("\n".join(lines))

    # ---------- animation ----------
    def anim_sequence(self, prefix_history=None):
        """
        HEAD-LED FOLLOW-THE-LEADER ANIMATION

        - The planned route is the HEAD trajectory.
        - Every intermediate destination is preserved exactly.
        - Tail/body joints are sampled one link length behind each other along
          the already traversed route.
        - A small travelling lateral wave is superimposed only when it remains
          collision-safe. Near obstacles the wave is automatically reduced.
        - Every body link is collision-checked, not just the tail/base.
        """
        if self.current_path is None:
            self.plan_selected()
        if self.current_path is None:
            return None

        try:
            spacing=max(float(self.anim_spacing.get()),2.0)
            amp_deg=abs(float(self.amp_var.get()))
            phase_deg=float(self.phase_var.get())
            wavelength=max(float(self.wave_var.get()),20.0)
            user_clearance=max(float(self.clearance.get()),0.0)
        except Exception:
            spacing,amp_deg,phase_deg,wavelength,user_clearance=8.0,22.0,40.0,120.0,20.0

        # Resample EACH route leg separately, so D1/D2/... are exact frames.
        source_segments=self.current_segments
        if not source_segments:
            source_segments=[np.asarray(self.current_path,dtype=float)]

        dense_parts=[]
        destination_frame_indices=[]
        total_frames=0

        for seg_i,segment in enumerate(source_segments):
            dense=resample(np.asarray(segment,dtype=float),spacing,500)
            if len(dense)<2:
                continue

            # Guarantee exact segment endpoints.
            dense[0]=np.asarray(segment[0],dtype=float)
            dense[-1]=np.asarray(segment[-1],dtype=float)

            if dense_parts:
                dense=dense[1:]

            dense_parts.append(dense)
            total_frames += len(dense)
            destination_frame_indices.append(total_frames-1)

        if not dense_parts:
            return None

        head_route=np.vstack(dense_parts)
        if len(head_route)<2:
            return None

        # Cumulative distance along the HEAD route.
        route_seg=np.linalg.norm(np.diff(head_route,axis=0),axis=1)
        route_s=np.r_[0.0,np.cumsum(route_seg)]

        body_length=self.dof*float(self.robot.link)

        # Build body-history centerline behind the current head.
        # For an online replan, use the already traversed HEAD history so the
        # body/tail remain continuous.  For a normal start, use a straight
        # virtual extension behind the initial head.
        first_vec=head_route[1]-head_route[0]
        first_norm=np.linalg.norm(first_vec)
        if first_norm<1e-9:
            first_unit=np.array([1.0,0.0])
        else:
            first_unit=first_vec/first_norm

        extension_length=body_length+2.0*float(self.robot.link)

        history_prefix=None
        if prefix_history is not None:
            hp=np.asarray(prefix_history,dtype=float)
            if len(hp)>=2:
                # Keep only approximately one body length of historical path.
                hp_seg=np.linalg.norm(np.diff(hp,axis=0),axis=1)
                hp_s=np.r_[0.0,np.cumsum(hp_seg)]
                keep_from=max(hp_s[-1]-extension_length,0.0)
                first_keep=int(np.searchsorted(hp_s,keep_from,side="left"))
                history_prefix=hp[max(first_keep-1,0):].copy()

                # Avoid duplicating the current head.
                if len(history_prefix) and np.linalg.norm(
                    history_prefix[-1]-head_route[0]
                )<1e-6:
                    history_prefix=history_prefix[:-1]

        if history_prefix is not None and len(history_prefix)>=2:
            centerline=np.vstack([history_prefix,head_route])

            cl_seg=np.linalg.norm(np.diff(centerline,axis=0),axis=1)
            cl_s=np.r_[0.0,np.cumsum(cl_seg)]

            # Distance from centerline start to current head_route[0].
            prefix_seg=np.linalg.norm(np.diff(
                np.vstack([history_prefix,head_route[0]]),axis=0
            ),axis=1)
            start_s=float(np.sum(prefix_seg))
        else:
            extension_start=head_route[0]-first_unit*extension_length
            centerline=np.vstack([extension_start,head_route])
            cl_seg=np.linalg.norm(np.diff(centerline,axis=0),axis=1)
            cl_s=np.r_[0.0,np.cumsum(cl_seg)]
            start_s=extension_length

        # Interpolation helper returns point and local unit tangent.
        def interp_at(sval):
            sval=float(np.clip(sval,cl_s[0],cl_s[-1]))
            j=int(np.searchsorted(cl_s,sval,side="right")-1)
            j=max(0,min(j,len(cl_seg)-1))
            denom=cl_s[j+1]-cl_s[j]
            u=0.0 if denom<1e-12 else (sval-cl_s[j])/denom
            point=centerline[j]+u*(centerline[j+1]-centerline[j])

            tangent=centerline[j+1]-centerline[j]
            norm=np.linalg.norm(tangent)
            if norm<1e-12:
                tangent=np.array([1.0,0.0])
            else:
                tangent=tangent/norm
            return point,tangent

        # Convert requested joint-wave angle to a restrained WORLD lateral
        # animation amplitude. It is visual only; actual safety is validated.
        visual_amp=min(
            0.22*float(self.robot.link),
            0.35*float(self.robot.link)*math.sin(math.radians(min(amp_deg,70.0)))
        )

        frames=[]
        unsafe_frames=0
        used_wave_scales=[]

        # Tail -> ... -> Head sample offsets.
        behind=np.arange(self.dof,-1,-1,dtype=float)*float(self.robot.link)

        for frame_i,(head_point,travelled) in enumerate(zip(head_route,route_s)):
            head_s=start_s+travelled

            base_points=[]
            normals=[]
            for dbehind in behind:
                p,tangent=interp_at(head_s-dbehind)
                normal=np.array([-tangent[1],tangent[0]],dtype=float)
                base_points.append(p)
                normals.append(normal)

            base_points=np.asarray(base_points,dtype=float)
            normals=np.asarray(normals,dtype=float)

            # Head must be EXACTLY on the planned route / destination.
            base_points[-1]=head_point

            # Travelling-wave phase progresses with head travel.
            temporal_phase=-360.0*travelled/wavelength

            # Try full wave; reduce automatically near obstacles.
            safe_points=None
            chosen_scale=0.0
            for scale in (1.0,0.75,0.50,0.25,0.0):
                pts=base_points.copy()

                # Zero at tail and head, maximum near the middle.
                for j in range(1,self.dof):
                    taper=math.sin(math.pi*j/self.dof)
                    wave_phase=temporal_phase+j*phase_deg
                    offset=(
                        scale*visual_amp*taper*
                        math.sin(math.radians(wave_phase))
                    )
                    pts[j]=pts[j]+normals[j]*offset

                pts[-1]=head_point

                if (
                    not self.body_frame_collision(
                        pts,
                        include_user_clearance=True
                    )
                    and not body_self_intersection(pts)
                ):
                    safe_points=pts
                    chosen_scale=scale
                    break

            if safe_points is None:
                # This should be rare because planning already used a
                # conservative full-body margin. Keep the zero-wave body for
                # diagnostic display and count it as unsafe.
                safe_points=base_points.copy()
                safe_points[-1]=head_point
                unsafe_frames += 1

            frames.append(safe_points)
            used_wave_scales.append(chosen_scale)

        frames=np.asarray(frames,dtype=float)

        # Verify every destination is exactly a HEAD frame.
        for dest_i,frame_idx in enumerate(destination_frame_indices):
            if 0 <= frame_idx < len(frames):
                frames[frame_idx,-1]=self.destinations[dest_i]

        avg_scale=float(np.mean(used_wave_scales)) if used_wave_scales else 0.0

        return {
            "head_route":head_route,
            "body_frames":frames,
            "destination_frame_indices":destination_frame_indices,
            "unsafe_frames":unsafe_frames,
            "average_wave_scale":avg_scale
        }

    def animate_route(self):
        try:
            seq=self.anim_sequence()
        except Exception as exc:
            messagebox.showerror(
                "Animation",
                f"Could not create the head-led full-body animation.\\n\\n{exc}"
            )
            return

        if seq is None:
            messagebox.showinfo("Animation","Plan a valid route first.")
            return

        if seq["unsafe_frames"]>0:
            messagebox.showwarning(
                "Full-body clearance",
                f"{seq['unsafe_frames']} frame(s) could not satisfy the requested "
                "whole-body safety clearance.\\n\\n"
                "Increase clearance or adjust the environment."
            )

        self.stop()

        # Store the mission as it existed BEFORE any online replanning.
        # Dynamic replanning later replaces self.start/self.destinations with
        # the remaining mission, so these originals are required for a full
        # exported GIF/MP4.
        self.mission_original_start=np.asarray(self.start,float).copy()
        self.mission_original_destinations=[
            np.asarray(p,float).copy() for p in self.destinations
        ]
        self.mission_reference_path=(
            np.asarray(self.current_path,float).copy()
            if self.current_path is not None else None
        )
        self.runtime_mission_started=True

        # Any dynamic obstacle that was already present before motion begins
        # is visible from frame 0.
        for o in self.obstacles:
            if o.get("dynamic",False) and "appear_frame" not in o:
                o["appear_frame"]=0

        self.active_seq=seq
        self.active_index=0
        self.active_head_trail=[]
        self.active_body=seq["body_frames"][0].copy()
        self.runtime_body_log=[]
        self.runtime_replan_flags=[]
        self.analysis_cache=None
        self.error_cache=None
        self.dynamic_state.set("GREEN — moving")
        self.resume_active_sequence()

    def resume_active_sequence(self):
        if self.active_seq is None:
            return

        body_frames=self.active_seq["body_frames"]
        self.animating=True

        def step():
            if not self.animating:
                return

            if self.active_index>=len(body_frames):
                self.animating=False
                self.dynamic_state.set("GREEN — mission complete")
                self.status.set(
                    "Animation complete — HEAD reached all remaining destinations"
                )
                if len(body_frames):
                    self.active_body=body_frames[-1].copy()
                    self.redraw((self.active_body,self.active_head_trail,False))
                try:self.refresh_motion_analysis()
                except Exception:pass
                try:self.refresh_error_analysis()
                except Exception:pass
                try:self.refresh_singularity_analysis()
                except Exception:pass
                try:self.ml_plot_results()
                except Exception:pass
                return

            body=body_frames[self.active_index]
            self.active_body=body.copy()

            # The active index corresponds to this currently displayed frame.
            self.active_head_trail.append(body[-1].copy())
            self.runtime_body_log.append(body.copy())
            self.runtime_replan_flags.append(self.dynamic_replan_count)
            self.analysis_cache=None
            self.error_cache=None

            self.redraw((body,self.active_head_trail,False))

            self.active_index += 1

            try:
                speed=max(float(self.speed_var.get()),0.05)
            except Exception:
                speed=1.0

            delay=max(int(45/speed),1)
            self.anim_job=self.after(delay,step)

        self.status.set(
            f"Dynamic-capable snake animation running | re-plans={self.dynamic_replan_count}"
        )
        step()

    def stop(self):
        self.animating=False
        if self.anim_job is not None:
            try:
                self.after_cancel(self.anim_job)
            except Exception:
                pass
        self.anim_job=None
        if hasattr(self,"dynamic_state") and self.dynamic_state.get().startswith("GREEN"):
            self.dynamic_state.set("IDLE / PAUSED")

    def save_anim(self,kind):
        """
        Save the COMPLETE mission.

        Priority:
        1) If a live/dynamic mission has executed, use runtime_body_log so the
           export contains everything before and after every online replan.
        2) If the mission is currently still running/paused, append the known
           remaining frames from active_seq.
        3) If no runtime mission exists, fall back to a freshly generated
           animation sequence.

        Dynamic obstacles are revealed at their recorded appear_frame.
        """
        if kind=="gif" and not PillowWriter.isAvailable():
            messagebox.showerror(
                "GIF export",
                "Pillow is not available. Install it with:\n\npip install pillow"
            )
            return

        if kind=="mp4" and not FFMpegWriter.isAvailable():
            messagebox.showerror(
                "MP4 export",
                "FFmpeg is not available on PATH.\nUse GIF export or install FFmpeg."
            )
            return

        # ------------------------------------------------------------
        # Build COMPLETE animation frames.
        # ------------------------------------------------------------
        using_runtime=False
        source_indices=None

        if len(self.runtime_body_log)>0:
            using_runtime=True
            executed=np.asarray(self.runtime_body_log,float)

            frames=[b.copy() for b in executed]

            # If the user saves while the mission is still paused/running,
            # append all currently known remaining frames so the exported file
            # still reaches the end of the present mission.
            if self.active_seq is not None:
                remaining=np.asarray(
                    self.active_seq.get("body_frames",[]),
                    dtype=float
                )
                next_idx=max(int(self.active_index),0)

                if len(remaining)>next_idx:
                    for b in remaining[next_idx:]:
                        frames.append(b.copy())

            body_frames=np.asarray(frames,float)

            if len(body_frames)<2:
                messagebox.showinfo(
                    "Animation save",
                    "Not enough executed animation frames are available yet."
                )
                return

            # Original source-frame index for dynamic-obstacle appearance.
            source_indices=np.arange(len(body_frames),dtype=int)

            # Build complete head route directly from body history.
            head_route=body_frames[:,-1,:].copy()

            # A runtime sequence has already passed live safety checks.
            unsafe_frames=0

        else:
            try:
                seq=self.anim_sequence()
            except Exception as exc:
                messagebox.showerror(
                    "Animation save",
                    f"Could not create the head-led animation.\n\n{exc}"
                )
                return

            if seq is None:
                messagebox.showinfo(
                    "Animation save",
                    "Plan a valid route first."
                )
                return

            unsafe_frames=int(seq.get("unsafe_frames",0))
            if unsafe_frames>0:
                messagebox.showerror(
                    "Animation save",
                    f"Animation contains {unsafe_frames} frame(s) with a full-body "
                    "clearance violation.\n\nIncrease Safety clearance and re-plan first."
                )
                return

            body_frames=np.asarray(seq["body_frames"],float)
            head_route=np.asarray(seq["head_route"],float)
            source_indices=np.arange(len(body_frames),dtype=int)

        # ------------------------------------------------------------
        # File dialog.
        # ------------------------------------------------------------
        ext=".mp4" if kind=="mp4" else ".gif"

        pair_slug=(
            f"{self.eval_planner_var.get()}_"
            f"{self.eval_ml_var.get().replace(' ','_')}"
        )

        path=filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[
                ("MP4 Video","*.mp4")
            ] if kind=="mp4" else [
                ("GIF Animation","*.gif")
            ],
            initialfile=f"snake_FULL_{pair_slug}{ext}"
        )
        if not path:
            return

        # ------------------------------------------------------------
        # Preserve the FULL mission.  We permit many more GIF frames than the
        # old 320-frame cap.  Extremely long runs are uniformly sampled, but
        # the first and last frames and the complete mission span are retained.
        # ------------------------------------------------------------
        original_frame_count=len(body_frames)
        max_frames=900 if kind=="gif" else 1600

        if len(body_frames)>max_frames:
            ids=np.linspace(
                0,
                len(body_frames)-1,
                max_frames
            ).round().astype(int)
            ids=np.unique(ids)

            body_frames=body_frames[ids]
            head_route=head_route[ids]
            source_indices=source_indices[ids]

        export_frame_count=len(body_frames)

        self.status.set(
            f"Saving FULL {kind.upper()} mission: "
            f"{export_frame_count}/{original_frame_count} frames..."
        )
        self.update_idletasks()

        # ------------------------------------------------------------
        # Figure and complete extents.
        # ------------------------------------------------------------
        fig=Figure(figsize=(8.6,6.2),dpi=100)
        ax=fig.add_subplot(111)

        xmin,xmax,ymin,ymax=self.bounds()

        all_body_xy=body_frames.reshape(-1,2)
        if len(all_body_xy):
            pad=max(float(self.robot.link)*1.7,35.0)
            xmin=min(xmin,float(np.min(all_body_xy[:,0]))-pad)
            xmax=max(xmax,float(np.max(all_body_xy[:,0]))+pad)
            ymin=min(ymin,float(np.min(all_body_xy[:,1]))-pad)
            ymax=max(ymax,float(np.max(all_body_xy[:,1]))+pad)

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_aspect("equal",adjustable="box")
        ax.grid(True,alpha=.28)
        ax.set_xlabel("World X")
        ax.set_ylabel("World Y")
        ax.set_title(
            f"{self.graph_pair_text()}\n"
            f"FULL Dynamic Mission — {self.dof}-DOF Snake",
            fontsize=10,
            pad=8
        )

        # ------------------------------------------------------------
        # Static obstacles are visible from frame 0.
        # Dynamic obstacles get patches whose visibility is changed frame by
        # frame based on appear_frame.
        # ------------------------------------------------------------
        dynamic_patch_records=[]

        for o in self.obstacles:
            is_dynamic=bool(o.get("dynamic",False))

            if o["type"]=="circle":
                patch=Circle(
                    (o["x"],o["y"]),
                    o["r"],
                    alpha=.30 if not is_dynamic else .38,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=2.2 if is_dynamic else None
                )
            else:
                patch=Rectangle(
                    (
                        o["x"]-o["w"]/2,
                        o["y"]-o["h"]/2
                    ),
                    o["w"],
                    o["h"],
                    alpha=.30 if not is_dynamic else .38,
                    edgecolor="red" if is_dynamic else None,
                    facecolor="red" if is_dynamic else None,
                    linewidth=2.2 if is_dynamic else None
                )

            ax.add_patch(patch)

            if is_dynamic:
                appear_frame=int(o.get("appear_frame",0))
                # Hide until its recorded event frame.
                patch.set_visible(
                    int(source_indices[0]) >= appear_frame
                )
                dynamic_patch_records.append(
                    (patch,appear_frame)
                )

        # ------------------------------------------------------------
        # Original mission markers, not shortened post-replan markers.
        # ------------------------------------------------------------
        if self.runtime_mission_started:
            display_start=np.asarray(
                self.mission_original_start,
                float
            )
            display_destinations=[
                np.asarray(p,float)
                for p in self.mission_original_destinations
            ]
        else:
            display_start=np.asarray(self.start,float)
            display_destinations=[
                np.asarray(p,float)
                for p in self.destinations
            ]

        ax.scatter(
            [display_start[0]],
            [display_start[1]],
            marker="s",
            s=80,
            label="Original Head Start"
        )

        for i,p in enumerate(display_destinations,1):
            ax.scatter(
                [p[0]],[p[1]],
                marker="*",
                s=110
            )
            ax.text(
                p[0],p[1],
                f" D{i}"
            )

        # Show BOTH trajectories in the exported animation:
        #   1) the original planner/reference path
        #   2) the executed dynamic route
        reference_path=None
        if self.mission_reference_path is not None:
            reference_path=np.asarray(self.mission_reference_path,float)
        elif self.ml_reference_path is not None:
            reference_path=np.asarray(self.ml_reference_path,float)
        elif self.current_path is not None:
            reference_path=np.asarray(self.current_path,float)

        if reference_path is not None and len(reference_path)>=2:
            ax.plot(
                reference_path[:,0],
                reference_path[:,1],
                lw=1.2,
                ls="--",
                alpha=.70,
                label="Original reference path"
            )

        ax.plot(
            head_route[:,0],
            head_route[:,1],
            lw=1.2,
            alpha=.38,
            label="Executed dynamic route"
        )

        body_line,=ax.plot(
            [],[],
            "-o",
            lw=2.5,
            ms=4,
            label="Snake body"
        )
        head_marker,=ax.plot(
            [],[],
            marker="*",
            linestyle="None",
            ms=10,
            label="HEAD"
        )
        tail_marker,=ax.plot(
            [],[],
            marker="s",
            linestyle="None",
            ms=6,
            label="TAIL"
        )
        head_trail_line,=ax.plot(
            [],[],
            lw=1.2,
            alpha=.65,
            label="Executed head trail"
        )

        event_text=ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            fontweight="bold"
        )

        ax.legend(
            loc="upper right",
            fontsize=7.5
        )

        hx=[]
        hy=[]

        # ------------------------------------------------------------
        # Animation update.
        # ------------------------------------------------------------
        def upd(i):
            body=body_frames[i]
            src_frame=int(source_indices[i])

            body_line.set_data(
                body[:,0],
                body[:,1]
            )
            head_marker.set_data(
                [body[-1,0]],
                [body[-1,1]]
            )
            tail_marker.set_data(
                [body[0,0]],
                [body[0,1]]
            )

            hx.append(float(body[-1,0]))
            hy.append(float(body[-1,1]))
            head_trail_line.set_data(
                hx,
                hy
            )

            visible_dynamic=0
            for patch,appear_frame in dynamic_patch_records:
                visible=src_frame>=appear_frame
                patch.set_visible(visible)
                if visible:
                    visible_dynamic+=1

            # Replan count at this runtime frame when available.
            replan_count=0
            if using_runtime and len(self.runtime_replan_flags):
                runtime_idx=min(
                    src_frame,
                    len(self.runtime_replan_flags)-1
                )
                replan_count=int(
                    self.runtime_replan_flags[runtime_idx]
                )

            event_text.set_text(
                f"Frame {i+1}/{len(body_frames)} | "
                f"Ref path + executed path | "
                f"Dynamic obstacles active: {visible_dynamic} | "
                f"Replans: {replan_count}"
            )

            artists=[
                body_line,
                head_marker,
                tail_marker,
                head_trail_line,
                event_text
            ]
            artists.extend(
                [p for p,_ in dynamic_patch_records]
            )
            return tuple(artists)

        # Slightly lower FPS for GIF keeps a long full mission practical.
        fps=16 if kind=="gif" else 24

        anim=FuncAnimation(
            fig,
            upd,
            frames=len(body_frames),
            interval=1000/fps,
            blit=False,
            repeat=False
        )

        try:
            writer=(
                FFMpegWriter(
                    fps=fps,
                    bitrate=2400
                )
                if kind=="mp4"
                else PillowWriter(
                    fps=fps
                )
            )

            anim.save(
                path,
                writer=writer,
                dpi=100
            )

            if (
                not Path(path).exists()
                or Path(path).stat().st_size==0
            ):
                raise RuntimeError(
                    "No valid animation file was created."
                )

            self.status.set(
                f"FULL animation saved: {path}"
            )

            source_text=(
                "complete executed runtime history"
                if using_runtime
                else "planned animation sequence"
            )

            messagebox.showinfo(
                "Full Animation Saved",
                f"{kind.upper()} saved successfully.\n\n"
                f"Source: {source_text}\n"
                f"Mission frames available: {original_frame_count}\n"
                f"Frames written: {export_frame_count}\n"
                f"Dynamic replans recorded: {self.dynamic_replan_count}\n\n"
                f"{path}"
            )

        except Exception as exc:
            self.status.set(
                "Animation save failed"
            )
            messagebox.showerror(
                "Animation save",
                f"Animation could not be saved.\n\n{exc}"
            )

    # ---------- export ----------
    def save_path_csv(self):
        if self.current_path is None:self.plan_selected()
        if self.current_path is None:return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],initialfile=f"{self.current_algorithm}_multi_destination_path.csv")
        if not path:return
        p=np.asarray(self.current_path); cum=np.zeros(len(p))
        if len(p)>1:cum[1:]=np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["head_path_index","head_x","head_y","cumulative_distance"])
            for i,pt in enumerate(p):w.writerow([i,pt[0],pt[1],cum[i]])

    def save_compare_csv(self):
        if not self.compare_results:self.compare_all()
        if not self.compare_results:return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],initialfile="path_planning_algorithm_comparison.csv")
        if not path:return
        fields=["algorithm","success","failed_leg","destinations","path_length","planning_time_s","nodes","smoothness_deg","min_clearance","route_points"]
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for n in ["Dijkstra","A*","RRT","RRT*"]:
                if n not in self.compare_results:continue
                r=self.compare_results[n]
                w.writerow({"algorithm":n,"success":r["success"],"failed_leg":r.get("failed_leg"),"destinations":len(self.destinations),
                            "path_length":r.get("path_length",""),"planning_time_s":r.get("planning_time",""),"nodes":r.get("nodes",""),
                            "smoothness_deg":r.get("smoothness",""),"min_clearance":r.get("clearance",""),"route_points":r.get("points","")})

    def save_map_fig(self):
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],initialfile="multi_destination_path.png")
        if path:self.fig.savefig(path,dpi=300,bbox_inches="tight")

    def save_compare_fig(self):
        path=filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],initialfile="path_planning_comparison.png")
        if path:self.cfig.savefig(path,dpi=300,bbox_inches="tight")


if __name__=="__main__":
    App().mainloop()
