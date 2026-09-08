# PPO altitude hold in X-Plane

Flies a FALCON-S altitude-keeping PPO checkpoint in X-Plane 12 through the XPlaneConnect (`xpc`)
UDP bridge. One file, one control loop, no dashboard.

The interesting part is not the bridge — it is what has to sit between the network and the
aeroplane. The policy was trained against the Warp plant, and it reads eleven numbers, two of
which are the *states of the elevator servo and the throttle*, not the aeroplane's. So this
example rebuilds:

* the eleven-element observation, scale for scale and clip for clip, from
  `falcons/sim/warp/altitude_obs.py`;
* the rate-limited altitude reference and the clamped integral term, from
  `falcons/envs/altitude.py` and `falcons/envs/configs.py`;
* the actuator models whose states two of the observations report — using the plant's own
  `ActuatorSystem`, not a second implementation of it.

Nothing here is a re-tuning: every constant comes from the airframe JSON or the env config that
trained the checkpoint.

## Run it without X-Plane first

```bash
.venv/bin/python tools/xplane/examples/altitude_hold/altitude_hold.py --mock \
    --plane Volantex_Ranger --target 50 --start 40 --seconds 30
```

`--mock` puts the FALCON-S CPU plant where X-Plane would be, through the same code path. No
X-Plane, no GPU. It is also the test that the shim is right: the policy is back on the dynamics
it was trained on, so if altitude hold works here, the observation assembly and the actuator
emulation are correct. It does:

```
     t        h  sub tgt      Va   pitch    elev    thr
   0.0    40.00    40.02   15.00    0.00  -0.000  0.015
   4.0    46.81    48.02   12.64    8.83  -0.305  0.575
  10.0    49.76    50.00   14.20    1.90  -0.157  0.465
  20.0    50.20    50.00   15.87    0.88  -0.106  0.374
  28.0    50.16    50.00   16.00    1.24  -0.080  0.354
```

It climbs 40 → 50 m along the 2 m/s rate-limited reference and holds within ±0.3 m. The airship
(`--plane Airship_V7 --target 30 --start 20`) and a descent (`--target 30 --start 45`) both work
too. What `--mock` does *not* test is any X-Plane convention: the state arrives already in
FALCON-S's frame, so a sign error in the bridge would not show up here.

## Then in X-Plane

1. **Install the XPlaneConnect plugin** in X-Plane: drop the `XPlaneConnect` folder into
   `X-Plane 12/Resources/plugins/` and restart. It listens on UDP 49009.
2. **The `xpc` client** is already at `tools/xplane/xpc/`; the script puts it on `sys.path`
   itself, so nothing to install.
3. **Load an aeroplane and get it flying** — level, trimmed, at a sensible speed, a few hundred
   metres up. The script does not take off, reposition or trim for you; it takes over the
   elevator and throttle of an aeroplane already in the air.
4. Run it:

```bash
.venv/bin/python tools/xplane/examples/altitude_hold/altitude_hold.py \
    --plane Volantex_Ranger --target 500 --seconds 120 --log flight.csv
```

`--target` is **altitude MSL in metres**, matching X-Plane's POSI. The loop runs in real time at
the airframe's control period (10 ms), reads state, sends stick and throttle, and prints a line
every `--report` seconds. Ctrl-C returns the controls to neutral.

`--help` lists the rest: `--seed` for a different checkpoint of the same cell, `--host`/`--port`
for a remote X-Plane, `--device cuda` to run the network on the GPU (it is a small MLP; CPU is
fine), `--dt` to change the control period.

## Read this before trusting a flight

**The X-Plane conventions in `XPlaneBridge` are not flight-tested.** The velocity mapping
(X-Plane local x=east, y=up, z=south → NED → body through the attitude), the `P`/`Q`/`R` body-rate
signs, and `ELEVATOR_SIGN` are derived from the DREF documentation and the earlier rig in
`WIG_Plane_RL_Control/src/xplane/io.py`, and they have never been checked against a real X-Plane
session. Expect to fix one of them on the first flight; they are all in one place at the top of
the file, and `--log` gives you the trace to diagnose from. A reversed elevator sign will be
obvious immediately and violently.

**It is a transfer test, not a validation.** X-Plane flies whichever aeroplane you have loaded,
not the FALCON-S airframe the checkpoint was trained on. `Volantex_Ranger` is a 1.4 kg model
aeroplane trimmed at 15 m/s and `Airship_V7` an airship at 28 m/s; nothing in X-Plane's default
hangar is either of those. Pick the closest thing you have — a slow ultralight for the Volantex —
and read the result as "how far does this policy carry to a different aeroplane", not as
"does the policy work".

**Only the elevator and the throttle are driven.** That is exactly how the checkpoint was
trained: aileron and rudder are commanded zero and nothing holds the wings level. In X-Plane,
where the aeroplane is not the one the policy learned on, a slow roll-off is likely and the
policy will not correct it. Fly with the autopilot's wing leveller on, or accept the roll.

**The observation includes emulated actuator states, and X-Plane does not know about them.** The
script sends the servo's *output position* to X-Plane, because X-Plane has no model of the
FALCON-S servo; in `--mock` it sends the *command* instead, because the plant models the servo
itself. Sending the wrong one puts two lags in series — it was the first bug this example had,
and `--mock` is what caught it.

## Files

| file | role |
| --- | --- |
| `altitude_hold.py` | the whole example: bridge, mock plant, deployment shim, loop |
| `../../xpc/` | the XPlaneConnect UDP client (third-party, unchanged) |
