import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    import keyboard 
    HAS_KEYBOARD = True
except Exception:
    HAS_KEYBOARD = False


class AsteroidStatic:
    """
    Inertial Goal-Reaching with Obstacles (fully custom).
    Continuous state/action 2D inertial point-mass with circular obstacles and circular goal.

    State:  s = [x, y, vx, vy]
    Action: a = [ax, ay], each in [-a_max, +a_max]

    Key features:
    - Procedural generation per reset: random start, random goal, random non-overlapping obstacles
    - Constraints: no obstacle overlap; no obstacle in goal region; start-goal minimum separation
    - Reward shaping: Gaussian "warmth" signal increasing near goal
    - Plotting and optional manual control (if keyboard works)
    - Headless mode saves figures (trajectory/gif) instead of opening a GUI window
    """

    def __init__(
        self,
        # ----------------------
        # Physics / dynamics
        # ----------------------
        dt=0.1,                 # Integration time step (seconds) for Euler update
        a_max=1.0,              # Max absolute acceleration per axis; actions clipped to [-a_max, +a_max]
        v_max=4.0,              # Max speed magnitude; velocity is scaled down if ||v|| > v_max
        noise_std=0.2,          # Std-dev of Gaussian noise added to acceleration (stochastic dynamics)
        max_steps=500,          # Episode horizon; truncates when step_idx >= max_steps
        agent_radius=0.10,      # Agent collision radius (used for obstacle/goal/bounds collision checks)

        # ----------------------
        # World / procedural generation
        # ----------------------
        world_bounds=(-0.5, 4.5, -0.5, 4.5),  # Rectangle bounds: (xmin, xmax, ymin, ymax)
        n_obstacles=25,         # Number of circular obstacles to place in the scene
        obs_radius_range=(0.15, 0.5),  # Obstacle radius sampling range (r_lo, r_hi)
        margin=0.12,            # Extra clearance margin used in non-overlap/clearance constraints
        spawn_clearance=0.2,    # Minimum clearance around start (prevents start too close to goal/obstacles)
        goal_radius=0.2,        # Goal region radius (goal is reached when dist <= goal_radius + agent_radius)
        goal_clearance=0.1,     # Extra clearance around goal for obstacle placement (keeps goal region free)
        min_start_goal_dist=3.0,# Hard minimum Euclidean distance between start and goal centers
        max_gen_tries=10000,    # Max rejection-sampling attempts per element (start/goal/obstacle placement)
    
        # ----------------------
        # Reward shaping (goal / termination)
        # ----------------------
        terminal_goal_bonus=10.0,       # Bonus added on reaching the goal (terminal success reward)
        terminal_collision_penalty=100.0,# Penalty on collision or out-of-bounds (terminal failure cost)
    
        # ----------------------
        # Control / rendering / reproducibility
        # ----------------------
        allow_keyboard_control=True, # Enable manual control if keyboard backend is available
        seed=None,                   # RNG seed for reproducible layout generation and noise
        fixed_layout: bool = True,   # If True: generate layout once per instance; if False: regenerate each reset
    ):
        # Physics
        self.dt = float(dt)
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.noise_std = float(noise_std)
        self.max_steps = int(max_steps)
        self.agent_radius = float(agent_radius)

        # World / generation
        self.world_bounds = tuple(world_bounds)
        self.n_obstacles = int(n_obstacles)
        self.obs_radius_range = tuple(obs_radius_range)
        self.margin = float(margin)
        self.spawn_clearance = float(spawn_clearance)
        self.goal_radius = float(goal_radius)
        self.goal_clearance = float(goal_clearance)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.max_gen_tries = int(max_gen_tries)

        # Reward
        self.terminal_goal_bonus = float(terminal_goal_bonus)
        self.terminal_collision_penalty = float(terminal_collision_penalty)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Control flags
        self.allow_keyboard_control = bool(allow_keyboard_control) and HAS_KEYBOARD

        # Runtime state
        self.step_idx = 0
        self.state = np.zeros(4, dtype=np.float64)

        # Generated episode layout
        self.start_pos = np.zeros(2, dtype=np.float64)
        self.goal_center = np.zeros(2, dtype=np.float64)
        self.obstacles = np.zeros((0, 3), dtype=np.float64)  # (x, y, r)

        # Plot reuse
        self._fig = None
        self._ax = None

        self.prev_d = None

        self.fixed_layout = bool(fixed_layout)
        self._layout_ready = False

        self.reset()

    # ----------------------
    # Public API
    # ----------------------

    def get_observation_dim(self):
        return 4

    def get_action_dim(self):
        return 2

    def get_action_limits(self):
        return (-self.a_max, self.a_max)

    def reset(self):
        self.step_idx = 0

        if (not self.fixed_layout) or (not self._layout_ready):
            self._generate_episode_layout()
            self._layout_ready = True

        self.prev_d = self._dist(self.start_pos, self.goal_center)
        vx, vy = self.rng.normal(0.0, 0.05, size=2)
        self.state[:] = (self.start_pos[0], self.start_pos[1], vx, vy)
        return self.state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -self.a_max, self.a_max)

        # single transition
        self._calc_next_state(action)
    
        obs = self.state.copy()
        reward, info = self._calc_reward_and_info()
        return obs, reward, info["done"], info


    # ----------------------
    # Procedural generation
    # ----------------------

    def _sample_point_in_bounds(self):
        xmin, xmax, ymin, ymax = self.world_bounds
        x = self.rng.uniform(xmin, xmax)
        y = self.rng.uniform(ymin, ymax)
        return np.array([x, y], dtype=np.float64)

    def _dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def _generate_episode_layout(self):
        """
        Requirements implemented:
        1) No obstacle in goal region
        2) Each reset is unique (random positions/radii)
        3) Obstacles cannot overlap
        4) Start random; goal random but not near start (spawn_clearance and min_start_goal_dist)
        """

        xmin, xmax, ymin, ymax = self.world_bounds
        # Keep points away from edges a bit (optional but helps visuals)
        edge_pad = max(self.agent_radius, 0.05)

        def sample_valid_point():
            for _ in range(self.max_gen_tries):
                p = self._sample_point_in_bounds()
                if (p[0] < xmin + edge_pad) or (p[0] > xmax - edge_pad):
                    continue
                if (p[1] < ymin + edge_pad) or (p[1] > ymax - edge_pad):
                    continue
                return p
            raise RuntimeError("Failed to sample a valid point within bounds.")

        # 1) Sample start
        start = sample_valid_point()

        # 2) Sample goal far enough from start
        for _ in range(self.max_gen_tries):
            goal = sample_valid_point()
            if self._dist(goal, start) < self.min_start_goal_dist:
                continue
            # Stronger exclusion: prevent goal near spawn within spawn_clearance
            if self._dist(goal, start) < self.spawn_clearance:
                continue
            break
        else:
            raise RuntimeError("Failed to sample goal with sufficient separation from start.")

        # 3) Sample obstacles with non-overlap and clearance constraints
        obstacles = []
        r_lo, r_hi = self.obs_radius_range

        for _ in range(self.n_obstacles):
            placed = False
            for _try in range(self.max_gen_tries):
                r = float(self.rng.uniform(r_lo, r_hi))
                c = sample_valid_point()

                # Clearance from start (avoid trivial immediate collision)
                if self._dist(c, start) < (r + self.agent_radius + self.spawn_clearance + self.margin):
                    continue

                # Clearance from goal region (forbidden area)
                if self._dist(c, goal) < (r + self.goal_radius + self.goal_clearance + self.margin):
                    continue

                # Non-overlap with existing obstacles
                ok = True
                for (ox, oy, orad) in obstacles:
                    if self._dist(c, np.array([ox, oy])) < (r + orad + self.margin):
                        ok = False
                        break
                if not ok:
                    continue

                obstacles.append((c[0], c[1], r))
                placed = True
                break

            if not placed:
                raise RuntimeError("Failed to place all obstacles without overlap/violations.")

        self.start_pos = start
        self.goal_center = goal
        self.obstacles = np.array(obstacles, dtype=np.float64) if obstacles else np.zeros((0, 3), dtype=np.float64)

    # ----------------------
    # Dynamics
    # ----------------------

    def _calc_next_state(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(2)
        a = np.clip(a, -self.a_max, self.a_max)

        if self.noise_std > 0.0:
            a = a + self.rng.normal(0.0, self.noise_std, size=2)

        # Euler integration
        vx, vy = self.state[2], self.state[3]
        vx = vx + a[0] * self.dt
        vy = vy + a[1] * self.dt

        # Velocity clip
        v = np.array([vx, vy], dtype=np.float64)
        speed = np.linalg.norm(v)
        if speed > self.v_max:
            v *= (self.v_max / (speed + 1e-12))
        vx, vy = v

        x, y = self.state[0], self.state[1]
        x = x + vx * self.dt
        y = y + vy * self.dt

        self.state[:] = (x, y, vx, vy)
        self.step_idx += 1

    # ----------------------
    # Termination / reward
    # ----------------------

    def _is_goal(self):
        pos = self.state[:2]
        return self._dist(pos, self.goal_center) <= (self.goal_radius + self.agent_radius)

    def _is_collision(self):
        pos = self.state[:2]
        if self.obstacles.shape[0] == 0:
            return False
        centers = self.obstacles[:, :2]
        radii = self.obstacles[:, 2]
        d = np.linalg.norm(centers - pos[None, :], axis=1)
        return bool(np.any(d <= (radii + self.agent_radius)))

    def _calc_reward_and_info(self):
        pos = self.state[:2]
        d = self._dist(pos, self.goal_center)
    
        reached = self._is_goal()
        collided = self._is_collision()
        out = self._is_out_of_bounds()
        timeout = self.step_idx >= self.max_steps
        
        # Progress shaping (prevents moving away)
        if self.prev_d is None:
            self.prev_d = d
        progress = self.prev_d - d
        self.prev_d = d
        reward = 0.5 * float(progress)

        # Terminal shaping
        if reached:
            reward += self.terminal_goal_bonus
        if collided:
            reward -= self.terminal_collision_penalty
        if out:
            reward -= self.terminal_collision_penalty

        done = reached or collided or out or timeout

        info = {
            "dist_to_goal": float(d),
            "reached_goal": bool(reached),
            "collision": bool(collided),
            "out_of_bounds": bool(out),
            "timeout": bool(timeout),
            "done": bool(done),
            "step_idx": int(self.step_idx),
        }
        return reward, info


    # ----------------------
    # Control
    # ----------------------

    def manual_control(self):
        """
        Arrow keys accelerate; space = brake (damps velocity).
        Only active if keyboard import works and allow_keyboard_control=True.
        """
        a = np.zeros(2, dtype=np.float64)
        if not self.allow_keyboard_control:
            return a

        import keyboard  # local import (avoids hard dependency)

        if keyboard.is_pressed("up"):
            a[1] += self.a_max
        if keyboard.is_pressed("down"):
            a[1] -= self.a_max
        if keyboard.is_pressed("right"):
            a[0] += self.a_max
        if keyboard.is_pressed("left"):
            a[0] -= self.a_max

        if keyboard.is_pressed("space"):
            self.state[2:] *= 0.7

        return np.clip(a, -self.a_max, self.a_max)

    # ----------------------
    # Rendering (GUI) and headless export
    # ----------------------

    def plot(self):
        """Interactive plot (requires display). Use save_* methods for headless Docker."""
        if self._fig is None:
            self._fig, self._ax = plt.subplots(num=1, clear=True)

        ax = self._ax
        ax.clear()

        xmin, xmax, ymin, ymax = self.world_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Obstacles (black)
        for ox, oy, r in self.obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, fill=False, linewidth=2))

        # Goal (green dashed)
        ax.add_patch(
            plt.Circle(
                self.goal_center,
                self.goal_radius,
                fill=False,
                linewidth=3,
                linestyle="--",
                edgecolor="green",
            )
        )
        ax.text(self.goal_center[0], self.goal_center[1] + 0.45, "Goal", color="green", ha="center")

        # Start marker
        ax.scatter(self.start_pos[0], self.start_pos[1], s=50, marker="o")

        # Agent
        x, y, vx, vy = self.state
        ax.add_patch(plt.Circle((x, y), self.agent_radius, facecolor="w", edgecolor="k", linewidth=2))

        # Velocity arrow
        ax.arrow(x, y, vx * 0.3, vy * 0.3, head_width=0.08, length_includes_head=True)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"step={self.step_idx}  dist={self._dist(self.state[:2], self.goal_center):.2f}")
        plt.pause(0.001)

    def _render_static_axes(self, ax):
        """Draw obstacles + goal + start onto an axis (for trajectory export)."""
        xmin, xmax, ymin, ymax = self.world_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        for ox, oy, r in self.obstacles:
            ax.add_patch(plt.Circle((ox, oy), r, fill=False, linewidth=2))

        ax.add_patch(
            plt.Circle(
                self.goal_center,
                self.goal_radius,
                fill=False,
                linewidth=3,
                linestyle="--",
                edgecolor="green",
            )
        )
        ax.text(self.goal_center[0], self.goal_center[1] + 0.45, "Goal", color="green", ha="center")
        ax.scatter(self.start_pos[0], self.start_pos[1], s=50, marker="o")
        ax.set_aspect("equal", adjustable="box")

    def save_frame(self, path="frame.png", dpi=160):
        """Headless-safe: saves a single frame of the current state."""
        fig, ax = plt.subplots()
        self._render_static_axes(ax)

        x, y, vx, vy = self.state
        ax.add_patch(plt.Circle((x, y), self.agent_radius, facecolor="w", edgecolor="k", linewidth=2))
        ax.arrow(x, y, vx * 0.3, vy * 0.3, head_width=0.08, length_includes_head=True)

        ax.set_title(f"step={self.step_idx}  dist={self._dist(self.state[:2], self.goal_center):.2f}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {path}")

    def _is_out_of_bounds(self) -> bool:
        xmin, xmax, ymin, ymax = self.world_bounds
        x, y = float(self.state[0]), float(self.state[1])
        r = self.agent_radius
        return (x < xmin + r) or (x > xmax - r) or (y < ymin + r) or (y > ymax - r)


if __name__ == "__main__":
    env = AsteroidStatic()

    if env.allow_keyboard_control:
        # Interactive play (requires display + keyboard support)
        plt.ion()
        done = False
        env.reset()
        while not done:
            action = env.manual_control()
            _, _, done, info = env.step(action)
            env.plot()
        print("Episode finished:", info)
        plt.ioff()
        plt.show()

