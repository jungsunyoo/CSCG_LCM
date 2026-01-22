import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


# ----------------------------
# Continuous 2D environment (continuous actions)
# Cue-crossing affects ONLY how terminal is labeled (reward vs no-reward)
# ----------------------------
class CuedContinuousBox:
    """
    Continuous state x=(x,y) in [0,1]^2.
    Continuous action a_raw=(a_x,a_y) in [-1,1]^2, scaled by step_size.

    Cue crossing: entering a cue disk at ANY time.
    Terminal: reaching lower-right goal disk OR max_steps.

    Terminal outcome (for CoDA contingency):
      reward (k=0)     if reached_goal AND crossed_cue
      no-reward (k=1)  otherwise (including reached_goal without cue, or timeout)

    NOTE: observation returned is only (x,y). Cue-crossing is hidden (in info only).
    """
    def __init__(self,
                 step_size=0.08,
                 noise=0.01,
                 goal_radius=0.08,
                 cue_radius=0.08,
                 max_steps=40,
                 seed=0):
        self.step_size = float(step_size)
        self.noise = float(noise)
        self.goal_radius = float(goal_radius)
        self.cue_radius = float(cue_radius)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        self.GOAL = np.array([0.90, 0.15], dtype=np.float32)  # lower-right-ish
        self.CUE  = np.array([0.55, 0.50], dtype=np.float32)  # gate

        self.reset()

    def reset(self):
        self.t = 0
        self.x = np.array([0.0, 1.0], dtype=np.float32)
        self.crossed_cue = False
        return self.obs()

    def obs(self):
        return (float(self.x[0]), float(self.x[1]))

    def _intersects_circle(self, p1, p2, center, r):
        # check if segment p1-p2 intersects circle(center, r)
        # 1. Check endpoints
        if np.linalg.norm(p1 - center) < r: return True
        if np.linalg.norm(p2 - center) < r: return True
        
        # 2. Check closest point on segment
        d = p2 - p1
        if np.allclose(d, 0): return False
        f = p1 - center
        
        # t for closest point P = p1 + t*d
        # (p1 + t*d - center) . d = 0 => (f + t*d) . d = 0 => f.d + t*(d.d) = 0
        t = -np.dot(f, d) / np.dot(d, d)
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * d
        return np.linalg.norm(closest - center) < r

    def _update_cue_crossing(self, x_prev):
        # check if path from x_prev to self.x crossed cue
        if not self.crossed_cue:
            if self._intersects_circle(x_prev, self.x, self.CUE, self.cue_radius):
                self.crossed_cue = True

    def _in_goal(self, x_prev):
        # check if path from x_prev to self.x crossed goal
        return self._intersects_circle(x_prev, self.x, self.GOAL, self.goal_radius)

    def step(self, a_raw):
        a_raw = np.asarray(a_raw, dtype=np.float32)
        a_raw = np.clip(a_raw, -1.0, 1.0)

        a_vec = (self.step_size * a_raw).astype(np.float32)
        dx = a_vec + self.rng.normal(0.0, self.noise, size=2).astype(np.float32)

        x_prev = self.x.copy()
        self.x = np.clip(self.x + dx, 0.0, 1.0)
        self.t += 1

        self._update_cue_crossing(x_prev)

        reached_goal = self._in_goal(x_prev)
        timeout = (self.t >= self.max_steps)
        done = reached_goal or timeout

        # outcome for your contingency learner (binary):
        # k=0 reward if reached goal AND crossed cue, else k=1
        if reached_goal and self.crossed_cue:
            outcome_k = 0
        else:
            outcome_k = 1

        info = {
            "x_prev": x_prev,
            "x_next": self.x.copy(),
            "a_vec": a_vec,
            "a_raw": a_raw,
            "crossed_cue": bool(self.crossed_cue),
            "reached_goal": bool(reached_goal),
            "timeout": bool(timeout),
        }
        return self.obs(), done, outcome_k, info


# ----------------------------
# RBF features
# ----------------------------
class RBF2D:
    def __init__(self, n_per_dim=6, sigma=0.18):
        xs = np.linspace(0.0, 1.0, n_per_dim)
        ys = np.linspace(0.0, 1.0, n_per_dim)
        centers = np.array([(x, y) for x in xs for y in ys], dtype=np.float32)
        self.centers = centers
        self.sigma = float(sigma)
        self.M = centers.shape[0]

    def phi(self, x, y):
        xy = np.array([x, y], dtype=np.float32)
        d2 = np.sum((self.centers - xy[None, :]) ** 2, axis=1)
        return np.exp(-0.5 * d2 / (self.sigma ** 2)).astype(np.float32)  # (M,)


class TerminalRBFEncoder:
    """
    This is the key idea you asked for.

    Normal states:      phi = fe_normal.phi(x,y)
    Terminal w/ cue:    phi = fe_normal.phi(x,y)   (same as usual)
    Terminal w/o cue:   phi = fe_nocue.phi(x,y)    (different kernel or perturbed input)
    """
    def __init__(self, n_per_dim=6, sigma_normal=0.18, sigma_nocue=0.08, 
                 noncue_factor=0):
        self.fe_normal = RBF2D(n_per_dim=n_per_dim, sigma=sigma_normal)
        # same centers (because same n_per_dim), but different sigma => different kernel
        # self.fe_nocue = RBF2D(n_per_dim=n_per_dim, sigma=sigma_nocue)
        self.M = self.fe_normal.M
        self.noncue_factor = float(noncue_factor)
        # self.rng = np.random.default_rng()

    def phi(self, x, y, *, terminal=False, crossed_cue=False):
        if terminal and (not crossed_cue):
            # Scaled normal feature
            return self.fe_normal.phi(x, y) * self.noncue_factor
        else:
            return self.fe_normal.phi(x, y)


# ----------------------------
# Parametric transition model: linear delta model with ground-truth W
# ----------------------------
class LinearDeltaModel:
    """
    delta_hat = W @ [x, y, a_x, a_y, 1]
    Ground truth (mean dynamics ignoring noise/clipping): delta = a_vec
    """
    def __init__(self, W=None):
        if W is None:
            self.W = np.array([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ], dtype=np.float32)
        else:
            self.W = np.asarray(W, dtype=np.float32)

    def featurize(self, x, a_vec):
        return np.array([x[0], x[1], a_vec[0], a_vec[1], 1.0], dtype=np.float32)

    def predict_delta(self, x, a_vec):
        u = self.featurize(x, a_vec)
        return (self.W @ u).astype(np.float32)


# ----------------------------
# CoDA-like contingency learner + optional transition-PE salience
# ----------------------------
class CoDAContingencyContinuous:
    """
    Eligibility over features:
      e <- gamma*lambda*e + phi(x_t)

    Outcome-conditioned feature counts:
      C[k] += e at episode end (k in {0,1})

    NOTE: we will make sure the terminal phi gets added before end_episode().
    """
    def __init__(self, n_outcomes, feat_dim, transition_model, gamma=0.98, lam=0.9, eps=1e-8, 
                 obs_dim=0, obs_lr=0.05, use_obs_trace=False):
        self.K = int(n_outcomes)
        self.M = int(feat_dim)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.eps = float(eps)
        self.trans = transition_model

        # Only ONE eligibility trace needed online
        self.e = np.zeros(self.M, dtype=np.float32)
        self.C = np.zeros((self.K, self.M), dtype=np.float32)

        # optional: salience from transition PE
        self.e_s = np.zeros(self.M, dtype=np.float32)
        self.S = np.zeros(self.M, dtype=np.float32)

    def reset_episode(self):
        self.e[:] = 0.0
        self.e_s[:] = 0.0

    def step_features(self, phi_x):
        # Update single trace
        g = self.gamma * self.lam
        self.e *= g
        self.e += phi_x
        
        # Update salience trace
        self.e_s *= g
        self.e_s += phi_x

    def observe_transition_and_update_salience(self, x_prev, a_vec, x_next):
        delta_obs = (x_next - x_prev).astype(np.float32)
        delta_hat = self.trans.predict_delta(x_prev, a_vec)
        pe = float(np.linalg.norm(delta_obs - delta_hat))
        self.S += pe * self.e_s
        return pe

    def end_episode(self, outcome_k):
        # Add the single eligibility trace to the specific outcome count
        self.C[int(outcome_k)] += self.e

    def p_outcome_given_xy(self, phi_xy):
        scores = self.C @ phi_xy
        scores = np.maximum(scores, 0.0) + self.eps
        return scores / np.sum(scores)

    def entropy_xy(self, phi_xy):
        return float(entropy(self.p_outcome_given_xy(phi_xy)))

    def relevance_xy(self, phi_xy):
        total_C = np.sum(self.C, axis=0)
        return float(total_C @ phi_xy)

    def salience_xy(self, phi_xy):
        return float(self.S @ phi_xy)


# ----------------------------
# Training (random continuous policy)
# ----------------------------
def train(seed=0, n_episodes=8000):
    env = CuedContinuousBox(seed=seed, cue_radius=0.08, goal_radius=0.08)
    fe = TerminalRBFEncoder(n_per_dim=6, sigma_normal=0.18, sigma_nocue=0.08)
    trans_model = LinearDeltaModel()
    agent = CoDAContingencyContinuous(n_outcomes=2, feat_dim=fe.M, transition_model=trans_model)

    rng = np.random.default_rng(seed + 123)
    pe_running = []

    for ep in range(n_episodes):
        obs = env.reset()
        agent.reset_episode()

        while True:
            x, y = obs

            # normal (nonterminal) features
            phi_xy = fe.phi(x, y, terminal=False, crossed_cue=False)
            agent.step_features(phi_xy)

            a_raw = rng.uniform(0.0, 1.0, size=(2,)).astype(np.float32)
            obs_next, done, outcome_k, info = env.step(a_raw)

            # optional salience update
            pe_running.append(
                agent.observe_transition_and_update_salience(
                    x_prev=info["x_prev"],
                    a_vec=info["a_vec"],
                    x_next=info["x_next"]
                )
            )

            obs = obs_next

            if done:
                # IMPORTANT: add TERMINAL state's feature into eligibility BEFORE ending episode
                xT, yT = obs
                phi_T = fe.phi(
                    xT, yT,
                    terminal=True,
                    crossed_cue=info["crossed_cue"]  # <-- picks different kernel only if noncue terminal
                )
                agent.step_features(phi_T)

                agent.end_episode(outcome_k)
                break

    return env, fe, agent, float(np.mean(pe_running))


# if __name__ == "__main__":
#     env, fe, agent, pe_mean = train(seed=1, n_episodes=12000)
#     print(f"Mean one-step transition PE during training: {pe_mean:.4f}")
#     print(f"Cue center={env.CUE.tolist()} cue_radius={env.cue_radius:.3f}")
#     print(f"Goal center={env.GOAL.tolist()} goal_radius={env.goal_radius:.3f}")
#     print(f"Terminal kernels: sigma_normal={fe.fe_normal.sigma:.3f}, sigma_nocue={fe.fe_nocue.sigma:.3f}")

#     probes = [
#         (0.15, 0.50),
#         (0.55, 0.50),
#         (0.85, 0.20),
#         (0.90, 0.15),  # goal center
#     ]

#     print("\nProbe p(outcome|x,y): [k=0 reward(cue-crossed+goal), k=1 otherwise]")
#     for (x, y) in probes:
#         # probing nonterminal representation
#         phi_xy = fe.phi(x, y, terminal=False, crossed_cue=False)
#         p = agent.p_outcome_given_xy(phi_xy)
#         H = agent.entropy_xy(phi_xy)
#         rel = agent.relevance_xy(phi_xy)
#         sal = agent.salience_xy(phi_xy)
#         print(f"(x={x:.2f}, y={y:.2f}) p=[{p[0]:.2f},{p[1]:.2f}] H={H:.2f} rel={rel:.2f} sal={sal:.3f}")

#     # show how terminal encoding differs at the SAME (x,y)
#     xg, yg = env.GOAL
#     phi_term_cue   = fe.phi(xg, yg, terminal=True, crossed_cue=True)
#     phi_term_nocue = fe.phi(xg, yg, terminal=True, crossed_cue=False)
#     print("\nTerminal feature difference at the same goal location:")
#     print("  ||phi(goal, cue) - phi(goal, no-cue)|| =", float(np.linalg.norm(phi_term_cue - phi_term_nocue)))
