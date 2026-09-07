"""Torch Dryden turbulence must reproduce the scalar CPU model step for step.

Both implementations are explicit-Euler LTI filters with identical A/B/C matrices, so driving
them with the SAME white-noise sequence has to give bit-close gusts -- this is an exact parity
test, not a statistical one. Guards the torch backend against drifting from the CPU/warp wind
model (the torch backend previously had no wind at all).
"""
import numpy as np
import pytest
import torch

from falcons.sim.cpu.physics.dryden import DrydenTurbulenceModel
from falcons.sim.torch.wind import TorchDryden

DT, B, STEPS = 0.01, 5.0, 300


@pytest.fixture
def f64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _run_pair(intensity, va_traj, h_traj, seed=0):
    """Step both models with one shared noise sequence; return (cpu, torch) gust histories."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((len(va_traj), 6))

    cpu = DrydenTurbulenceModel(dt=DT, b=B, intensity=intensity)
    tor = TorchDryden(1, dt=DT, b=B, intensity=intensity, device="cpu")

    cpu_lin, cpu_ang, tor_lin, tor_ang = [], [], [], []
    for k, (va, h) in enumerate(zip(va_traj, h_traj)):
        # feed the CPU model the same draw it would have taken from np.random.randn
        saved = np.random.randn
        np.random.randn = lambda *a, _n=noise[k]: _n
        try:
            lin, ang = cpu.update(va, h)
        finally:
            np.random.randn = saved
        cpu_lin.append(lin); cpu_ang.append(ang)

        tl, ta = tor.update(torch.tensor([va]), torch.tensor([h]),
                            noise=torch.tensor(noise[k]).reshape(1, 6))
        tor_lin.append(tl[0].numpy()); tor_ang.append(ta[0].numpy())

    return (np.array(cpu_lin), np.array(cpu_ang)), (np.array(tor_lin), np.array(tor_ang))


@pytest.mark.parametrize("intensity", ["light", "moderate", "severe"])
def test_linear_gusts_match_cpu(f64, intensity):
    """u/v/w gusts are what actually perturb the airspeed vector -- these must match exactly."""
    va = np.full(STEPS, 22.0)
    h = np.full(STEPS, 50.0)
    (cpu_lin, _), (tor_lin, _) = _run_pair(intensity, va, h)
    assert np.abs(cpu_lin).max() > 0.1, "degenerate test: CPU produced no turbulence"
    np.testing.assert_allclose(tor_lin, cpu_lin, rtol=1e-10, atol=1e-12)


def test_angular_gusts_match_cpu(f64):
    """p/q/r channels (generated but not yet fed to the aero model) must also agree."""
    va = np.full(STEPS, 22.0)
    h = np.full(STEPS, 50.0)
    (_, cpu_ang), (_, tor_ang) = _run_pair("light", va, h)
    np.testing.assert_allclose(tor_ang, cpu_ang, rtol=1e-10, atol=1e-12)


def test_matches_across_altitude_and_airspeed(f64):
    """Scale lengths/intensities are re-evaluated every step, including the 1000 ft branch switch."""
    t = np.arange(STEPS)
    h = np.linspace(20.0, 500.0, STEPS)          # crosses 1000 ft (304.8 m)
    va = 22.0 + 5.0 * np.sin(t / 30.0)
    (cpu_lin, cpu_ang), (tor_lin, tor_ang) = _run_pair("moderate", va, h)
    np.testing.assert_allclose(tor_lin, cpu_lin, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(tor_ang, cpu_ang, rtol=1e-10, atol=1e-12)


def test_batch_envs_are_independent(f64):
    """N envs stepped together must equal N single-env runs (no cross-env leakage)."""
    n = 4
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((STEPS, n, 6))
    va = np.linspace(15.0, 30.0, n)
    h = np.linspace(30.0, 120.0, n)

    batch = TorchDryden(n, dt=DT, b=B, intensity="light", device="cpu")
    singles = [TorchDryden(1, dt=DT, b=B, intensity="light", device="cpu") for _ in range(n)]
    for k in range(STEPS):
        lin_b, _ = batch.update(torch.tensor(va), torch.tensor(h), noise=torch.tensor(noise[k]))
        lin_s = [s.update(torch.tensor([va[i]]), torch.tensor([h[i]]),
                          noise=torch.tensor(noise[k, i]).reshape(1, 6))[0][0] for i, s in enumerate(singles)]
        np.testing.assert_allclose(lin_b.numpy(), torch.stack(lin_s).numpy(), rtol=1e-12, atol=1e-14)


def test_reset_zeros_selected_rows(f64):
    n = 3
    d = TorchDryden(n, dt=DT, b=B, intensity="severe", device="cpu")
    for _ in range(50):
        d.update(torch.full((n,), 22.0), torch.full((n,), 50.0))
    assert d.xu.abs().sum() > 0
    d.reset(torch.tensor([True, False, True]))
    assert d.xu[0] == 0 and d.xu[2] == 0 and d.xu[1] != 0
