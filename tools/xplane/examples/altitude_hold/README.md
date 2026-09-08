# PPO altitude hold in X-Plane

Flies a FALCON-S altitude-keeping PPO checkpoint in X-Plane 12 over the XPlaneConnect (`xpc`)
bridge. One file, one control loop.

The bridge is the easy half. The policy reads eleven numbers, two of them the states of the
elevator servo and the throttle rather than anything the aeroplane reports, so the script rebuilds
the observation scale for scale from `falcons/sim/warp/altitude_obs.py`, the reference ramp and
clamped integral from `falcons/envs/altitude.py`, and the actuator states with the plant's own
`ActuatorSystem`. Every constant comes from the airframe JSON or the env config that trained the
checkpoint.

## Without X-Plane

```bash
.venv/bin/python tools/xplane/examples/altitude_hold/altitude_hold.py --mock \
    --plane Volantex_Ranger --start 40 --target-delta 10 --seconds 30
```

`--mock` puts the CPU plant where X-Plane would be, through the same code path — no X-Plane, no
GPU — and is the test that the shim is right, the policy being back on the dynamics it trained on:
the Volantex climbs 40 → 50 m along its 2 m/s reference and holds within ±0.3 m. It tests no
X-Plane convention; the state arrives already in FALCON-S's frame.

## In X-Plane

Put the `XPlaneConnect` folder in `X-Plane 12/Resources/plugins/` (UDP 49009) and get an aeroplane
flying level at altitude — the script takes over elevator and throttle, it does not take off or
trim for you. Disengage the autopilot; it will fight the loop.

```bash
.venv/bin/python tools/xplane/examples/altitude_hold/altitude_hold.py \
    --plane Airship_V7 --target-delta 15 --elevator-bias 0.82 --seconds 120 --log flight.csv
```

`--target-delta 15` climbs fifteen metres from wherever the aeroplane is when the loop takes over,
which is usually what you want since MSL depends on where you took off. `--target 500` sets an
absolute altitude; with neither, the present altitude is held. `--help` has the rest.

## Why `--elevator-bias`, and how to find yours

The policy's altitude channel is proportional — 0.073 of elevator per metre, measured — with no
useful integral: sweeping its integral observation from 0 to 250 m·s moves the elevator by 0.06.
It cannot absorb a constant disturbance, and an aeroplane needing a different stick than its
training airframe to fly level is exactly that. It converges beautifully to the wrong altitude.

Two runs give you the number: fly at `--elevator-bias 0`, note the settled error, fly again with a
guess, interpolate. On an airship-like custom airframe with `Airship_V7`:

| bias | settled error |
| --- | --- |
| 0 | +18.87 m |
| 1.37 | −12.85 m |
| **0.82** | **−0.06 m** |

−23.2 m of error per unit of bias. Do not compute it from the trim stick instead: the stick an
aeroplane needs depends on its speed, and the loop settles on a curve through altitude *and*
airspeed, not at a fixed trim point — that reasoning produced the 1.37 that overshot.

One bias per airframe, not per flight: it survives a different start altitude and delta, but
shifts if the settled airspeed does. At 0.82 what remains is a lightly damped phugoid, ±3 m over
a 65 s period with airspeed trading 24.6 ↔ 33.1 m/s. A slow integrator driving the bias from
accumulated error would make it self-tuning; that is not in here.

## It is a transfer test, not a validation

Only elevator and throttle are driven, as in training, so nothing holds the wings level. X-Plane
flies whichever aeroplane is loaded, so a flight is a transfer test, not a validation. The script
sends X-Plane the servo's *output position*, and `--mock` the *command*, since the plant models
the servo itself — the wrong one puts two lags in series.
