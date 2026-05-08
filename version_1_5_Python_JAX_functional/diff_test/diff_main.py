"""Python+JAX JSON-line CLI bridge for the v1.5 diff test.

Mirrors `version_1_5_LEAN/Main.lean` exactly — same dispatch tags, same
JSON shapes — so the Julia driver can stub-pipe to either Lean4 binary
or Python interpreter and exercise the same 121 randomised cases.

Reads one JSON object per line on stdin, writes one JSON object per
line on stdout. Dispatch tags:

  drift         : (state, phi, params_v1)               → deriv [dB, dF, dA]
  diffusion     : (state, params_v1)                    → sigma [σ_B, σ_F, σ_A]
  emStep        : (state, params_v1, noise, phi, dt, n_substeps)
                                                          → next_state [B, F, A]
  paramsV15ToV1 : (params_v15)                          → params_v1 (14 fields)
  reflectUnit   : (x : Float)                           → x' : Float
  plantStep     : (state, phi, params_v15, obs_params, dt,
                   sde_noise, obs_noise)                → (next_state, obs)
  obsLogWeight  : (B, F, A, obs, obs_params)            → log_weight
  applyPrior    : (kind ∈ {logNormal, normal}, mu, sigma, u) → x

Run from `version_1_5_Python_JAX/` with PYTHONPATH=.:..:

    PYTHONPATH=.:.. JAX_ENABLE_X64=True python diff_test/diff_main.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import json
import math
import sys

import numpy as np

from models.fsa_high_res._dynamics import (
    drift_jax, diffusion_state_dep, em_step_substepped,
)
from models.fsa_high_res.simulation import params_v15_to_v1
from models.fsa_high_res._plant import _reflect_unit, PlantState, plant_step
from models.fsa_high_res.estimation import obs_log_weight_one


def _to_float(x):
    return float(x)


def _state_arr(arr):
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _params_v1_dict(d):
    keys = ('tau_B', 'tau_F', 'kappa_B', 'kappa_F', 'epsilon_A',
            'lambda_A', 'mu_0', 'mu_B', 'mu_F', 'mu_FF', 'eta',
            'sigma_B', 'sigma_F', 'sigma_A')
    return {k: float(d[k]) for k in keys}


def _params_v15_dict(d):
    keys = ('tau_B', 'tau_F', 'B_inf', 'F_inf', 'epsilon_A',
            'lambda_A', 'mu_0', 'mu_B', 'mu_F', 'mu_FF', 'eta',
            'sigma_B', 'sigma_F', 'sigma_A')
    return {k: float(d[k]) for k in keys}


def _obs_params(d):
    return dict(
        sigma_B_obs=float(d['sigma_B_obs']),
        sigma_F_obs=float(d['sigma_F_obs']),
        sigma_A_obs=float(d['sigma_A_obs']),
    )


def handle(req: dict) -> dict:
    fn = req['fn']

    if fn == 'drift':
        y = _state_arr(req['state'])
        p = _params_v1_dict(req['params'])
        d = drift_jax(y, p, float(req['phi']))
        return {'deriv': [float(d[0]), float(d[1]), float(d[2])]}

    if fn == 'diffusion':
        y = _state_arr(req['state'])
        p = _params_v1_dict(req['params'])
        s = diffusion_state_dep(y, p)
        return {'sigma': [float(s[0]), float(s[1]), float(s[2])]}

    if fn == 'emStep':
        y = _state_arr(req['state'])
        p = _params_v1_dict(req['params'])
        noise = np.array([float(x) for x in req['noise']], dtype=np.float64)
        phi = float(req['phi'])
        dt = float(req['dt'])
        n_sub = int(req['n_substeps'])
        # em_step_substepped is JAX-side, takes JAX arrays — fine to pass
        # numpy, JAX coerces.
        y_next = em_step_substepped(y, p, noise, phi, dt, n_substeps=n_sub)
        return {'next_state': [float(y_next[0]), float(y_next[1]),
                                float(y_next[2])]}

    if fn == 'paramsV15ToV1':
        p15 = _params_v15_dict(req['params'])
        p1 = params_v15_to_v1(p15)
        return {'params_v1': {k: float(v) for k, v in p1.items()}}

    if fn == 'reflectUnit':
        x = float(req['x'])
        # _reflect_unit returns a JAX scalar; coerce.
        return {'x': float(_reflect_unit(x))}

    if fn == 'plantStep':
        bfa = _state_arr(req['state'])
        # PlantState wraps the (B, F, A) triple; t_bin is reset to 0.
        s = PlantState(bfa=bfa, t_bin=0)
        phi = float(req['phi'])
        p15 = _params_v15_dict(req['params'])
        # Merge obs-noise σ's so plant_step can build obs.
        op = _obs_params(req['obs_params'])
        params = {**p15, **op}
        dt = float(req['dt'])
        sde_noise = np.array([float(x) for x in req['sde_noise']], dtype=np.float64)
        obs_noise = np.array([float(x) for x in req['obs_noise']], dtype=np.float64)
        s_next, obs = plant_step(s, phi, params, dt,
                                   sde_noise=sde_noise, obs_noise=obs_noise)
        return {
            'next_state': [float(s_next.bfa[0]), float(s_next.bfa[1]),
                            float(s_next.bfa[2])],
            'obs': [float(obs['obs_B']), float(obs['obs_F']),
                     float(obs['obs_A'])],
        }

    if fn == 'obsLogWeight':
        B = float(req['B']); F = float(req['F']); A = float(req['A'])
        obs = req['obs']
        op = _obs_params(req['obs_params'])
        lw = obs_log_weight_one(B, F, A,
                                  dict(obs_B=float(obs['obs_B']),
                                        obs_F=float(obs['obs_F']),
                                        obs_A=float(obs['obs_A'])),
                                  op)
        return {'log_w': float(lw)}

    if fn == 'applyPrior':
        kind = req['kind']
        mu = float(req['mu']); sigma = float(req['sigma']); u = float(req['u'])
        if kind == 'logNormal':
            x = math.exp(min(20.0, max(-20.0, u)))
        else:
            x = mu + sigma * u
        return {'x': float(x)}

    return {'error': f'unknown fn: {fn}'}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle(req)
        except Exception as e:                       # noqa: BLE001
            resp = {'error': f'{type(e).__name__}: {e}'}
        # Match Lean4's extended-JSON convention for ±inf/NaN so
        # JSON3.read in Julia can parse with allow_inf=true.
        sys.stdout.write(json.dumps(resp, allow_nan=True))
        sys.stdout.write('\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
