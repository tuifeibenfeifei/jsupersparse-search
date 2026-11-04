# -*- coding: utf-8 -*-
# isort:skip_file
"""algos.py - collection of old/new algorithms, all subclass BaseAgent and observe presence only via query_cell."""
from __future__ import annotations
import math
import numpy as np
from common import BaseAgent  # noqa: E402

# 8-neighbourhood
NEI8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _rand_in_bounds(rng, R, C):
    return int(rng.integers(0, R)), int(rng.integers(0, C))


# ============================ old algorithms ============================
class RandomAgent(BaseAgent):
    """Uniform random sampling without replacement: draw uniformly from unvisited cells."""
    name = "random"
    kind = "old"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.unvisited = {(i, j) for i in range(self.R) for j in range(self.C)}

    def propose(self):
        if not self.unvisited:
            return _rand_in_bounds(self.rng, self.R, self.C)
        for _ in range(50):
            i, j = _rand_in_bounds(self.rng, self.R, self.C)
            if (i, j) in self.unvisited:
                return i, j
        return next(iter(self.unvisited))

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        self.unvisited.discard((i, j))


class GridAgent(BaseAgent):
    """Fixed-stride sequential scan in row-major order, stride 1. Deterministic."""
    name = "grid"
    kind = "old"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.ii, self.jj = 0, 0

    def propose(self):
        return self.ii, self.jj

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        self.jj += 1
        if self.jj >= self.C:
            self.jj = 0
            self.ii += 1


class HillClimbAgent(BaseAgent):
    """Random-restart hill climbing: greedily move to the 8-neighbour with the highest guidance score; score = guidance (covariate similarity)."""
    name = "hill_climb"
    kind = "old"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.cur = _rand_in_bounds(rng, self.R, self.C)
        self.stuck = 0

    def _score(self, i, j):
        s = self.guide.guide(i, j)
        if (i, j) in self.visited:
            s -= 0.5
        return s

    def propose(self):
        if not self.found:
            return _rand_in_bounds(self.rng, self.R, self.C)
        best, best_s = None, -1e9
        for di, dj in NEI8:
            ni, nj = self.cur[0] + di, self.cur[1] + dj
            if self.in_bounds(ni, nj):
                s = self._score(ni, nj)
                if s > best_s:
                    best_s, best = s, (ni, nj)
        if best is None or best_s <= self._score(*self.cur) - 1e-6:
            self.stuck += 1
            if self.stuck >= 3 or self.rng.random() < 0.15:
                if self.found and self.rng.random() < 0.6:
                    fi, fj = self.found[self.rng.integers(0, len(self.found))]
                    self.cur = (int(np.clip(fi + int(self.rng.integers(-8, 9)), 0, self.R - 1)),
                                int(np.clip(fj + int(self.rng.integers(-8, 9)), 0, self.C - 1)))
                else:
                    self.cur = _rand_in_bounds(self.rng, self.R, self.C)
                self.stuck = 0
                return self.cur
            return best if best else self.cur
        self.stuck = 0
        self.cur = best
        return best


class SimAnnealAgent(BaseAgent):
    """Simulated annealing: neighbourhood / random jumps, accept worse moves with exp(-delta/T); objective = guidance score."""
    name = "simulated_annealing"
    kind = "old"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.cur = _rand_in_bounds(rng, self.R, self.C)
        self.T = 1.0
        self.step = 0

    def propose(self):
        self.step += 1
        self.T = max(0.02, 1.0 * (0.9995 ** self.step))
        if not self.found:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if self.rng.random() < 0.7:
            di, dj = NEI8[self.rng.integers(0, 8)]
            cand = (self.cur[0] + di, self.cur[1] + dj)
        else:
            cand = _rand_in_bounds(self.rng, self.R, self.C)
        if not self.in_bounds(*cand):
            cand = _rand_in_bounds(self.rng, self.R, self.C)
        sc = self.guide.guide(*cand) - (0.5 if cand in self.visited else 0.0)
        ss = self.guide.guide(*self.cur) - (0.5 if self.cur in self.visited else 0.0)
        if sc >= ss or self.rng.random() < math.exp(min(0.0, (sc - ss) / max(self.T, 1e-6))):
            self.cur = cand
        return self.cur


class GAAgent(BaseAgent):
    """Genetic algorithm: individual = coordinate; fitness = guidance score; tournament selection + crossover + mutation."""
    name = "ga"
    kind = "old"
    POP = 30

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.pop = [np.array(_rand_in_bounds(rng, self.R, self.C), dtype=float) for _ in range(self.POP)]
        self.queue = []

    def _fit(self, x):
        i, j = int(round(x[0])), int(round(x[1]))
        if not self.in_bounds(i, j):
            return -1.0
        s = self.guide.guide(i, j)
        if (i, j) in self.visited:
            s -= 0.5
        return s

    def _evolve(self):
        fits = np.array([self._fit(x) for x in self.pop])
        new_pop = []
        for _ in range(self.POP):
            def tour():
                c = self.rng.integers(0, self.POP, 3)
                return self.pop[int(c[np.argmax(fits[c])])].copy()
            a, b = tour(), tour()
            c = 0.5 * (a + b) + self.rng.normal(0, 1.5, 2)
            new_pop.append(c)
        self.pop = new_pop
        fits = np.array([self._fit(x) for x in self.pop])
        order = np.argsort(-fits)
        for k in order:
            x = self.pop[k]
            i = int(np.clip(int(round(x[0])), 0, self.R - 1))
            j = int(np.clip(int(round(x[1])), 0, self.C - 1))
            cell = (i, j)
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= 25:
                break

    def propose(self):
        if not self.found:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._evolve()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()


class PSOAgent(BaseAgent):
    """Particle swarm: particle = coordinate, velocity toward personal/global best (guidance score)."""
    name = "pso"
    kind = "old"
    POP = 25

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.pos = np.array([_rand_in_bounds(rng, self.R, self.C) for _ in range(self.POP)], dtype=float)
        self.vel = self.rng.normal(0, 1.0, (self.POP, 2))
        self.pbest = self.pos.copy()
        self.pbest_f = np.full(self.POP, -1e9)
        self.gbest = self.pos[0].copy()
        self.gbest_f = -1e9
        self.queue = []

    def _fit(self, i, j):
        s = self.guide.guide(int(i), int(j))
        if (int(i), int(j)) in self.visited:
            s -= 0.5
        return s

    def _step(self):
        w, c1, c2 = 0.5, 1.2, 1.2
        r1, r2 = self.rng.random((self.POP, 2)), self.rng.random((self.POP, 2))
        self.vel = (w * self.vel
                    + c1 * r1 * (self.pbest - self.pos)
                    + c2 * r2 * (self.gbest[None, :] - self.pos))
        self.pos = np.clip(self.pos + self.vel, [0, 0], [self.R - 1, self.C - 1])
        for k in range(self.POP):
            f = self._fit(self.pos[k, 0], self.pos[k, 1])
            if f > self.pbest_f[k]:
                self.pbest_f[k] = f; self.pbest[k] = self.pos[k].copy()
            if f > self.gbest_f:
                self.gbest_f = f; self.gbest = self.pos[k].copy()
        fits = np.array([self._fit(p[0], p[1]) for p in self.pos])
        order = np.argsort(-fits)
        for k in order:
            i = int(round(self.pos[k, 0])); j = int(round(self.pos[k, 1]))
            cell = (i, j)
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= 25:
                break

    def propose(self):
        if not self.found:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._step()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()


# ============================ new algorithms ============================
class GPUCBAgent(BaseAgent):
    """Bayesian optimisation GP-UCB: inputs = normalised coordinates, labels = presence; batched refitting + candidate-pool acquisition."""
    name = "gp_ucb"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        self.GP = GaussianProcessRegressor
        self.kernel = ConstantKernel(1.0) * RBF(0.08) + WhiteKernel(0.1)
        self.X, self.y = [], []
        self.every = 250
        self.model = None
        self.cand_n = 1200
        self.boot = 50
        self.queue = []

    def _fit_model(self):
        if len(self.X) >= 30:
            X = np.array(self.X[-250:]); y = np.array(self.y[-250:])
            self.model = self.GP(kernel=self.kernel, optimizer=None, normalize_y=False)
            try:
                self.model.fit(X, y)
            except Exception:
                self.model = None

    def _refill(self):
        self._fit_model()
        ci = self.rng.integers(0, self.R, self.cand_n)
        cj = self.rng.integers(0, self.C, self.cand_n)
        Xc = np.column_stack([ci / self.R, cj / self.C])
        if self.model is not None:
            mu, std = self.model.predict(Xc, return_std=True)
            acq = mu + 1.5 * std
        else:
            acq = np.array([self.guide.guide(int(a), int(b)) for a, b in zip(ci, cj)])
        order = np.argsort(-acq)
        for k in order:
            cell = (int(ci[k]), int(cj[k]))
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= self.every:
                break

    def propose(self):
        if self.oracle.query_count < self.boot:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._refill()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        if presence is not None:
            self.X.append([i / self.R, j / self.C])
            self.y.append(1.0 if presence else 0.0)


class TurboAgent(BaseAgent):
    """Simplified trust-region BO (Eriksson 2019): sample within a box of side L centred on the current best, plus UCB;
    expand L on consecutive successes, shrink on failure; restart when L becomes too small."""
    name = "turbo"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        self.GP = GaussianProcessRegressor
        self.kernel = ConstantKernel(1.0) * RBF(0.08) + WhiteKernel(0.1)
        self.X, self.y = [], []
        self.center = np.array([self.R / 2.0, self.C / 2.0])
        self.L = float(min(self.R, self.C)) / 2.0
        self.success = 0; self.failure = 0
        self.model = None
        self.boot = 50
        self.queue = []

    def _fit(self):
        if len(self.X) >= 30:
            X = np.array(self.X[-250:]); y = np.array(self.y[-250:])
            self.model = self.GP(kernel=self.kernel, optimizer=None, normalize_y=False)
            try:
                self.model.fit(X, y)
            except Exception:
                self.model = None

    def _refill(self):
        self._fit()
        nb = 800
        off = self.rng.uniform(-self.L, self.L, (nb, 2))
        pos = self.center[None, :] + off
        ci = np.clip(pos[:, 0].astype(int), 0, self.R - 1)
        cj = np.clip(pos[:, 1].astype(int), 0, self.C - 1)
        gi = self.rng.integers(0, self.R, 800)
        gj = self.rng.integers(0, self.C, 800)
        ci = np.concatenate([ci, gi]); cj = np.concatenate([cj, gj])
        Xc = np.column_stack([ci / self.R, cj / self.C])
        if self.model is not None:
            mu, std = self.model.predict(Xc, return_std=True)
            acq = mu + 1.0 * std
        else:
            acq = np.array([self.guide.guide(a, b) for a, b in zip(ci, cj)])
        order = np.argsort(-acq)
        for k in order:
            cell = (int(ci[k]), int(cj[k]))
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= 150:
                break

    def propose(self):
        if self.oracle.query_count < self.boot:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._refill()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        if presence is not None:
            self.X.append([i / self.R, j / self.C])
            self.y.append(1.0 if presence else 0.0)
        if is_new_found:
            self.center = np.array([i, j], dtype=float)
            self.success += 1; self.failure = 0
            if self.success >= 3:
                self.L = min(self.L * 2.0, max(self.R, self.C))
                self.success = 0
        elif presence is False:
            self.failure += 1
            if self.failure >= 5:
                self.L *= 0.5; self.failure = 0
                if self.L < 2.0:
                    self.L = float(min(self.R, self.C)) / 2.0
                    self.center = np.array([self.R / 2.0, self.C / 2.0])


class GoExploreAgent(BaseAgent):
    """Go-Explore (Ecoffet 2019): archive = discovered points; periodically return to an archive point and random-walk around it."""
    name = "go_explore"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.archived = []
        self.cur = _rand_in_bounds(rng, self.R, self.C)
        self.walk = 0
        self.teleport_every = 25
        self.walk_r = 3

    def propose(self):
        if not self.archived:
            return _rand_in_bounds(self.rng, self.R, self.C)
        self.walk += 1
        if (self.walk % self.teleport_every == 0 or
                (self.cur in self.visited and self.rng.random() < 0.1)):
            self.cur = self.archived[self.rng.integers(0, len(self.archived))]
        di = int(self.rng.integers(-self.walk_r, self.walk_r + 1))
        dj = int(self.rng.integers(-self.walk_r, self.walk_r + 1))
        ni = int(np.clip(self.cur[0] + di, 0, self.R - 1))
        nj = int(np.clip(self.cur[1] + dj, 0, self.C - 1))
        self.cur = (ni, nj)
        return ni, nj

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        if is_new_found:
            self.archived.append((i, j))


class MapElitesAgent(BaseAgent):
    """MAP-Elites (Mouret & Clune 2015): bin the study area as a behaviour descriptor, keep the best individual per bin, mutation = random jump / perturbation."""
    name = "map_elites"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        self.GX, self.GY = 5, 6
        self.niche = {}

    def _bidx(self, i, j):
        gx = min(self.GX - 1, int(i / self.R * self.GX))
        gy = min(self.GY - 1, int(j / self.C * self.GY))
        return gx, gy

    def propose(self):
        if not self.found and self.oracle.query_count < 40:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if self.niche and self.rng.random() < 0.8:
            keys = list(self.niche.keys())
            gx, gy = keys[self.rng.integers(0, len(keys))]
            ei, ej, _ = self.niche[(gx, gy)]
            if self.rng.random() < 0.5:
                ni = int(np.clip(ei + int(self.rng.integers(-5, 6)), 0, self.R - 1))
                nj = int(np.clip(ej + int(self.rng.integers(-5, 6)), 0, self.C - 1))
            else:
                ni, nj = _rand_in_bounds(self.rng, self.R, self.C)
        else:
            ni, nj = _rand_in_bounds(self.rng, self.R, self.C)
        return ni, nj

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        score = self.guide.guide(i, j) + (1.0 if is_new_found else 0.0)
        b = self._bidx(i, j)
        if b not in self.niche or score > self.niche[b][2]:
            self.niche[b] = (i, j, score)


class RNDNoveltyAgent(BaseAgent):
    """RND / curiosity-driven (2018-2019): Ridge ensemble predicting covs -> presence; ensemble variance = curiosity score; batched acquisition."""
    name = "rnd_novelty"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        from sklearn.linear_model import Ridge
        self.Ridge = Ridge
        self.models = []
        self.Xc, self.y = [], []
        self.boot = 40
        self.queue = []

    def _fit(self):
        if len(self.Xc) >= 40:
            X = np.array(self.Xc[-500:]); y = np.array(self.y[-500:])
            self.models = []
            n = len(X)
            for _ in range(3):
                idx = self.rng.integers(0, n, n)
                m = self.Ridge(alpha=1.0)
                try:
                    m.fit(X[idx], y[idx]); self.models.append(m)
                except Exception:
                    pass

    def _refill(self):
        self._fit()
        ci = self.rng.integers(0, self.R, 2000)
        cj = self.rng.integers(0, self.C, 2000)
        Xc = self.guide.zcov[ci, cj]
        if self.models:
            preds = np.column_stack([m.predict(Xc) for m in self.models])
            unc = preds.var(axis=1)
        else:
            unc = np.array([self.guide.guide(a, b) for a, b in zip(ci, cj)])
        order = np.argsort(-unc)
        for k in order:
            cell = (int(ci[k]), int(cj[k]))
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= 80:
                break

    def propose(self):
        if self.oracle.query_count < self.boot:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._refill()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        if presence is not None:
            self.Xc.append(self.guide.zcov[i, j])
            self.y.append(1.0 if presence else 0.0)


class AdaptiveSamplingAgent(BaseAgent):
    """Adaptive sampling / information path planning: GP predictive variance (an information-gain proxy) as the acquisition function, pure exploration of high-variance regions, batched acquisition."""
    name = "adaptive_sampling"
    kind = "new"

    def __init__(self, oracle, rng, guide):
        super().__init__(oracle, rng, guide)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        self.GP = GaussianProcessRegressor
        self.kernel = ConstantKernel(1.0) * RBF(0.1) + WhiteKernel(0.1)
        self.X, self.y = [], []
        self.model = None
        self.boot = 50
        self.queue = []

    def _fit(self):
        if len(self.X) >= 30:
            X = np.array(self.X[-250:]); y = np.array(self.y[-250:])
            self.model = self.GP(kernel=self.kernel, optimizer=None, normalize_y=False)
            try:
                self.model.fit(X, y)
            except Exception:
                self.model = None

    def _refill(self):
        self._fit()
        ci = self.rng.integers(0, self.R, 1200)
        cj = self.rng.integers(0, self.C, 1200)
        Xc = np.column_stack([ci / self.R, cj / self.C])
        if self.model is not None:
            _, std = self.model.predict(Xc, return_std=True)
            acq = std
        else:
            acq = np.array([self.guide.guide(a, b) for a, b in zip(ci, cj)])
        order = np.argsort(-acq)
        for k in order:
            cell = (int(ci[k]), int(cj[k]))
            if cell not in self.visited and cell not in self.queue:
                self.queue.append(cell)
            if len(self.queue) >= 150:
                break

    def propose(self):
        if self.oracle.query_count < self.boot:
            return _rand_in_bounds(self.rng, self.R, self.C)
        if not self.queue:
            self._refill()
        if not self.queue:
            return _rand_in_bounds(self.rng, self.R, self.C)
        return self.queue.pop()

    def update(self, i, j, presence, cov, is_new_found):
        super().update(i, j, presence, cov, is_new_found)
        if presence is not None:
            self.X.append([i / self.R, j / self.C])
            self.y.append(1.0 if presence else 0.0)


AGENTS = {
    "random": RandomAgent,
    "grid": GridAgent,
    "hill_climb": HillClimbAgent,
    "simulated_annealing": SimAnnealAgent,
    "ga": GAAgent,
    "pso": PSOAgent,
    "gp_ucb": GPUCBAgent,
    "turbo": TurboAgent,
    "go_explore": GoExploreAgent,
    "map_elites": MapElitesAgent,
    "rnd_novelty": RNDNoveltyAgent,
    "adaptive_sampling": AdaptiveSamplingAgent,
}
