"""Generate the Task 2 project report as a PDF (Arial). Run: python make_report.py"""
from fpdf import FPDF
from fpdf.enums import XPos, YPos

FONT = "Arial"
FDIR = "C:/Windows/Fonts/"
NX, NY = XPos.LMARGIN, YPos.NEXT          # always return cursor to the left margin


class Report(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font(FONT, "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0)

    def _mc(self, h, txt, size, style="", color=(0, 0, 0)):
        self.set_x(self.l_margin)
        self.set_font(FONT, style, size)
        self.set_text_color(*color)
        self.multi_cell(0, h, txt, new_x=NX, new_y=NY)
        self.set_text_color(0)

    def maintitle(self, txt):
        self._mc(8, txt, 16, "B")

    def sub(self, txt):
        self._mc(5.2, txt, 10.5, "", (90, 90, 90))

    def h1(self, txt):
        self.ln(2)
        self._mc(7, txt, 13, "B", (20, 40, 80))
        self.ln(0.5)

    def h2(self, txt):
        self.ln(1)
        self._mc(6, txt, 11, "B")

    def body(self, txt):
        self._mc(5.4, txt, 10.5)
        self.ln(1)

    def bullet(self, txt):
        self._mc(5.4, "-  " + txt, 10.5)
        self.ln(0.8)

    def caption(self, txt):
        self._mc(5.4, txt, 10, "B")


pdf = Report(format="A4")
pdf.set_auto_page_break(auto=True, margin=18)
pdf.set_margins(20, 18, 20)
pdf.add_font(FONT, "", FDIR + "arial.ttf")
pdf.add_font(FONT, "B", FDIR + "arialbd.ttf")
pdf.add_font(FONT, "I", FDIR + "ariali.ttf")
pdf.add_font(FONT, "BI", FDIR + "arialbi.ttf")
pdf.alias_nb_pages()
pdf.add_page()

# ---------------------------------------------------------------- title
pdf.maintitle("Solving AsteroidStatic with Deep Reinforcement Learning")
pdf.sub("Task 2 - Deep Reinforcement Learning (Prof. Thomas Nierhoff, OTH Amberg-Weiden)")
pdf.sub("Author: Himanshu Mohan Nikam")
pdf.ln(3)

# ---------------------------------------------------------------- 1. intro
pdf.h1("1. Introduction and Problem")
pdf.body(
    "This project trains reinforcement-learning agents to solve the provided AsteroidStatic "
    "environment. The agent is a point-mass spaceship in a 2D arena populated with 25 circular "
    "asteroids. It is driven by a continuous acceleration action and drifts due to inertia. An "
    "episode succeeds when the ship reaches within 0.30 of the goal, and fails on collision with "
    "an asteroid, on leaving the arena, or on reaching the 500-step time limit.")
pdf.body(
    "Observation: obs = [x, y, vx, vy] (position and velocity only - the goal and asteroid "
    "positions are NOT observable). Action: a = [ax, ay], each a real value in [-1, 1]. "
    "Reward (already shaped by the environment): +0.5 x progress toward the goal each step, "
    "+10 for reaching the goal, and -100 for a crash. The agent learns purely from observations, "
    "rewards and info - it may not read obstacle positions or plan against the geometry. All "
    "provided files are used unmodified, and only the random seed is varied to obtain different "
    "arenas.")
pdf.body(
    "Two graded objectives are measured, each averaged over ten arenas (seeds 0-9): "
    "(1) sample efficiency - the fewest environment transition calls until the goal is first "
    "reached; and (2) speed - the fewest time steps taken to reach the goal once trained.")

# ---------------------------------------------------------------- 2. approaches
pdf.h1("2. Approaches")
pdf.body(
    "Because the environment is a continuous-control navigation task, we investigated two "
    "complementary families of algorithms and implemented a progression within each.")

pdf.h2("2.1 Value-based methods (DQN family)")
pdf.body(
    "We first implemented and tried the DQN approach. Deep Q-Networks learn a value Q(s, a) over "
    "a finite set of actions, so the continuous action is discretized into 9 bang-bang actions - "
    "every combination of {-1, 0, +1} for ax and ay. Full-throttle pushes are also the fastest "
    "way to move a mass, which helps the speed objective. Starting from vanilla DQN we added the "
    "standard improvements: Double DQN (DDQN), which decouples action selection from evaluation "
    "to reduce value over-estimation, and Dueling DDQN, which splits the network into a "
    "state-value stream V(s) and an advantage stream A(s, a). All three share experience replay, "
    "a target network, and epsilon-greedy exploration.")

pdf.h2("2.2 Policy-based methods (continuous control)")
pdf.body(
    "The second family acts directly on the continuous action, avoiding discretization. A "
    "deterministic actor mu(s) proposes an action and a critic Q(s, a) evaluates it. We "
    "implemented a progression of three: a base Actor-Critic; DDPG, which stabilizes it with "
    "soft (Polyak) target updates and temporally-correlated Ornstein-Uhlenbeck exploration noise; "
    "and TD3, which adds twin critics with a clipped-minimum target (to curb over-estimation), "
    "target-policy smoothing, and delayed actor updates. Continuous control offers fine, graded "
    "control - including gentle braking - which the fixed full-throttle actions cannot express, "
    "and is therefore expected to produce shorter, smoother paths.")

pdf.h2("2.3 Methods considered")
pdf.body(
    "We also considered but did not focus on: Deep SARSA, an on-policy method that is naturally "
    "more risk-averse near the large -100 crash penalty; and Soft Actor-Critic (SAC) and PPO as "
    "alternative continuous-control algorithms with stronger, entropy-driven exploration. These "
    "are discussed as future directions.")

# ---------------------------------------------------------------- 3. techniques
pdf.h1("3. Key Implementation Techniques")
pdf.body(
    "Several practical measures were decisive in getting the methods to learn in this "
    "obstacle-dense, sparse-goal environment:")
pdf.bullet(
    "Input normalization. Positions and velocities are rescaled to a comparable range using the "
    "known environment bounds. This is a simple input scaling (not analysis of the environment) "
    "and was the single largest contributor to stable training.")
pdf.bullet(
    "Reward handling. The -100 crash penalty dominates the small +0.5 progress-shaping term by "
    "roughly two orders of magnitude, which collapses the learned value estimates and leaves no "
    "usable signal to distinguish actions. We apply an asymmetric reward clip that floors the "
    "crash penalty while keeping the +10 goal bonus large and salient, so the rare successful "
    "transitions strongly influence learning. This deliberate reshaping is noted as a design "
    "choice.")
pdf.bullet(
    "Exploration. The task is hard-exploration: a purely random agent reaches the goal in only "
    "about 0.2% of episodes and never at all on several arenas, because the goal lies behind "
    "dense obstacles at least three units away. We therefore keep exploration strong and "
    "persistent - full-magnitude random action injection for the value-based agents, and "
    "correlated Ornstein-Uhlenbeck noise for the continuous agents - so the agent keeps "
    "discovering the goal rather than collapsing into a trivial 'do nothing' or 'flee to "
    "safety' policy.")
pdf.bullet(
    "Stability. Gradient clipping, target networks, and a replay warm-up period were used "
    "throughout.")

# ---------------------------------------------------------------- 4. setup
pdf.h1("4. Experimental Setup")
pdf.body(
    "All agents are implemented in PyTorch and use the environment's default settings; only the "
    "seed is changed to obtain different arenas. Hyperparameters are optimized with Optuna. "
    "Following the task rule, tuning is performed on seeds that are disjoint from the evaluation "
    "seeds: candidate seeds 100-119 were screened, and the two most tractable (seeds 115 and 119, "
    "on which the goal is at least occasionally reachable) were used for the search; the final "
    "numbers are reported on seeds 0-9. The tuning objective minimizes calls-to-first-solve, with "
    "a distance-based fallback so that configurations which do not solve within the episode budget "
    "are still ranked by how close they came. For each run we record calls-to-first-solve "
    "(Metric 1), the shortest successful episode length (Metric 2), and the calls associated with "
    "that shortest episode.")

# ---------------------------------------------------------------- 5. results
pdf.h1("5. Results")
pdf.body(
    "The hyperparameter search and final evaluation are still in progress; the tables below will "
    "be populated once tuning completes. All values are averages over seeds 0-9.")

pdf.caption("Table 1. Method comparison on the two graded metrics.")
pdf.set_font(FONT, "", 9.5)
with pdf.table(width=165, col_widths=(45, 45, 40, 35), text_align="CENTER",
               first_row_as_headings=True) as table:
    hdr = table.row()
    for c in ("Method", "Metric 1 (calls)", "Metric 2 (steps)", "Assoc. calls"):
        hdr.cell(c)
    for method in ("DQN", "DDQN", "Dueling DDQN", "Actor-Critic", "DDPG", "TD3"):
        row = table.row()
        row.cell(method)
        row.cell("TBD"); row.cell("TBD"); row.cell("TBD")
pdf.ln(3)

pdf.caption("Table 2. Tuned hyperparameters (best Optuna configuration per method).")
pdf.set_font(FONT, "", 9.5)
with pdf.table(width=165, col_widths=(45, 120), text_align="LEFT",
               first_row_as_headings=True) as table:
    hdr = table.row()
    hdr.cell("Method"); hdr.cell("Best configuration")
    for method in ("DQN family", "Actor-Critic", "DDPG", "TD3"):
        row = table.row()
        row.cell(method); row.cell("TBD (to be filled from Optuna study)")
pdf.ln(2)
pdf.body(
    "Learning curves (episode length and success rate versus number of calls) and a trajectory "
    "plot of the trained policy overlaid on the arena will be added here to demonstrate that the "
    "arenas are solved.")

# ---------------------------------------------------------------- 6. discussion
pdf.h1("6. Discussion")
pdf.body(
    "The dominant difficulty is exploration, not the learning rule. Because the goal sits behind "
    "a dense obstacle field, the straight-line progress reward can actively lure an agent into the "
    "asteroids just in front of the goal; states nearer the goal then carry higher crash risk and "
    "lower value, and an agent can rationally learn to flee to open space or, for the continuous "
    "deterministic policy, to output a near-zero 'do nothing' action and simply survive to the "
    "time limit. Overcoming this required exploration strong enough to actually reach the goal "
    "often enough for the value function to gain a positive anchor.")
pdf.body(
    "Two things that did not work are worth reporting. Symmetric reward clipping to [-1, 1] left "
    "the crash penalty large enough to collapse all value estimates. And weak Gaussian exploration "
    "around an already-collapsed deterministic policy only jitters locally and never travels far "
    "enough to find the goal; strong, correlated or full-magnitude exploration was necessary. In "
    "our experiments the value-based agents, whose random exploration uses full-throttle actions, "
    "tended to discover the goal more readily, while the continuous agents produced smoother, "
    "shorter paths once they did learn.")

# ---------------------------------------------------------------- 7. conclusion
pdf.h1("7. Conclusion")
pdf.body(
    "We implemented and compared two families of deep RL agents on the AsteroidStatic task - a "
    "value-based DQN / DDQN / Dueling DDQN progression and a continuous-control Actor-Critic / "
    "DDPG / TD3 progression - together with the exploration and reward-shaping techniques needed "
    "to make them learn in this hard-exploration setting. The final averaged metrics and tuned "
    "configurations will be inserted once the Optuna searches and seed-0-9 evaluations complete; "
    "the stronger variants (Dueling DDQN and TD3) are expected to give the best sample efficiency "
    "and shortest paths respectively.")

pdf.output("Task2_Report.pdf")
print("wrote Task2_Report.pdf  (pages:", pdf.page_no(), ")")
