# nbody-webgpu

A gravitational N-body simulation that runs on the GPU, in the browser. Every
body pulls on every other one: a brute-force O(N²) sum, recomputed every frame
in a WebGPU compute shader and drawn as points. Nothing is precomputed and
there is no server. The page opens and the GPU does the physics live.

Live at [ethan-puyaubreau.github.io/nbody-webgpu](https://ethan-puyaubreau.github.io/nbody-webgpu/).
It needs WebGPU: Chrome and Edge expose it. See Limitations for Firefox.

## How it works

The force sum is tiled. A naive kernel has every body loop over all the others
in global memory, which is bandwidth-bound. Here each workgroup loads a block
of bodies into shared memory, waits at a barrier, and runs its inner loop on
that on-chip copy. This is the standard GPU Gems N-body layout. The kernel is
in [`src/shaders/nbody.wgsl`](src/shaders/nbody.wgsl). The tile size is the
`TILE` override constant, set once from `WORKGROUP` in
[`src/simulation.ts`](src/simulation.ts).

The time stepper is leapfrog (kick-drift) with the velocity kept half a step
behind the position. Leapfrog is symplectic: the energy oscillates around its
initial value instead of drifting away, as it does with plain Euler. The first
step does a half kick to set up the offset.

A change of `dt` breaks the offset, because the old half step no longer matches
the new one. The kick is scaled by `kickScale = (dtPrev + dt) / (2 dt)`: 0.5 on
the first step, 1 while `dt` is constant. A step with a changed `dt` runs as
one dispatch, because the scale lives in the uniform shared by all substeps of
a pass.

Close encounters use Plummer softening: ε² is added to each squared distance,
which keeps the denominator away from zero when two bodies nearly overlap. The
self term is zero for a different reason: the offset r between a body and
itself is exactly 0, so the numerator vanishes.

Positions live in two buffers. Each step reads one and writes the other, then
they swap. WebGPU orders dispatches within a pass, so the read-after-write
needs no manual barrier. Velocities sit in a single buffer, because each
thread touches only its own.

WebGPU point topology draws 1 px dots, so each body is an instanced quad
expanded to a fixed pixel size, with a soft radial falloff and additive
blending. Color is keyed to speed: the fast inner disk runs hot and the slow
rim stays dark.

The initial state is a rotating disk around a heavy central mass. Radii are
sampled as `rInner + (rOuter - rInner) * u^1.4`, so the core is denser than the
rim. The speed of each body is the circular speed for the enclosed mass: the
central mass plus the disk mass times `x^(1/1.4)`, with `x` the normalized
radius. The same exponent that places the bodies gives the mass inside each
radius. An 8 percent speed jitter breaks the ring symmetry. The seed is fixed,
so the same galaxy comes back on every reload.

## HUD

- `fps`: frames per second, smoothed.
- `gpu ms`: time of the compute pass of one frame, from `timestamp-query`,
  smoothed. Without that feature the label reads `frame ms` and the value is
  the frame interval.
- `bodies`: the body count.
- `pairwise / s`: N² times the dispatches of the frame, divided by the GPU
  time, or multiplied by the frame rate when no GPU time is available.
- `energy drift`: (E - E0) / |E0| in percent, with E0 the first sample after a
  reset. The page samples the energy about every 500 ms. The kinetic term uses
  the velocity half a step behind the position, so the reading oscillates
  around the true value.

## Measured

Not measured yet. Open the page in a browser with WebGPU and read `gpu ms`,
`pairwise / s` and `energy drift` from the HUD.

## Running

```bash
pnpm install
pnpm dev      # http://localhost:5173
pnpm build    # type-check, then bundle into dist/
pnpm check    # lint, type-check, tests and the style script
```

Node 22 is required. There are no runtime dependencies. Vite and TypeScript
run at build time, and the fonts are bundled at build time.

## Deploying to GitHub Pages

`.github/workflows/deploy.yml` builds on every push to `main` and publishes
`dist/`. One manual step remains, once, in the repository settings: set Pages,
then Source, then GitHub Actions. The Vite `base` is relative, so the build
works from a project page or a custom domain without changes. For a custom
domain, set it in the repository settings under Pages and point the DNS at it.
A workflow deploy ignores a `CNAME` file.

## Parameters

The panel updates live.

Bodies: the cost is N², so 65 536 bodies is 16 times the work of 16 384. The
`pairwise / s` readout reacts right away.

Gravity and time step: past a certain value the integrator comes apart. More
softening recovers some of that range.

Substeps: more physics per rendered frame, so the galaxy evolves faster at the
same frame rate, as long as the GPU keeps up.

## Why WebGPU and not WebGL

This is a compute problem, not a rendering one. WebGL fakes compute with
render-to-texture tricks. WebGPU has compute shaders and shared memory, which
is what the simulation needs. The cost is reach: where WebGPU is missing, the
page shows a notice instead of a dead canvas.

## Limitations

- Positions and velocities are float32. The energy sum on the CPU is float64.
- The force sum is O(N²). The page offers up to 65 536 bodies, and that is the
  ceiling.
- The disk is not momentum-balanced, so the center of mass can drift.
- Changing G or the softening changes the definition of the energy, so the
  page rebaselines the drift on the next sample.
- `timestamp-query` is optional, and some browsers coarsen its timestamps.
- Firefox: WebGPU is on by default on Windows since version 141 and on macOS 26
  with Apple Silicon since version 145. Elsewhere it needs the
  `dom.webgpu.enabled` flag (caniuse, checked on 2026-10-18).

## Ideas for later

- Barnes-Hut or an FMM to break the O(N²) ceiling and reach into the millions.
- A proper color map and an HDR bloom pass.
- Seed two disks and let them collide.

---

Built by Ethan Puyaubreau. MIT licensed.
