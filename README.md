# nbody-webgpu

A gravitational N-body simulation that runs on the GPU, in the browser. Every
body pulls on every other one (a brute-force O(N²) sum), recomputed every frame
in a WebGPU compute shader and drawn as a few tens of thousands of points.
Nothing is precomputed and there's no server. Open the page and the GPU does the
physics live.

Live at [ethan-puyaubreau.github.io/nbody-webgpu](https://ethan-puyaubreau.github.io/nbody-webgpu/). Works in Chrome, Edge, and Safari 18+.

## How it works

The force sum is tiled. A naive kernel has every body loop over all the others
straight out of global memory, which works but is bandwidth-bound. Instead each
workgroup loads a block of bodies into shared memory, syncs, and runs its inner
loop against that on-chip copy. It's the standard GPU-Gems N-body layout, and
where most of the throughput comes from. The kernel is in
[`src/shaders/nbody.wgsl`](src/shaders/nbody.wgsl).

The time stepper is leapfrog (kick-drift), with velocity kept half a step behind
position. Leapfrog is symplectic, so the energy wobbles but doesn't drift off,
and the disk holds its shape over millions of steps instead of slowly unwinding
or collapsing the way plain Euler does. The first step does a half kick to set up
the offset.

Close encounters use Plummer softening: a small ε² added to each squared distance
keeps the 1/r² force finite when two bodies nearly overlap. It also zeroes out a
body's pull on itself, so the self term needs no special case.

Positions live in two buffers. Each step reads one and writes the other, then
they swap. WebGPU orders dispatches within a pass, so the read-after-write is safe
without manual barriers. Velocities sit in a single buffer, since each thread
only ever touches its own.

WebGPU's point topology only draws 1px dots, so each body is an instanced quad
blown up to a fixed pixel size, with a soft radial falloff and additive blending.
Colour is keyed to speed, so the fast inner disk runs hot and the slow rim stays
dark.

The initial state is a rotating disk around a heavy central mass, with orbital
speeds set from the enclosed mass so it starts roughly balanced. The seed is
fixed, so the same galaxy comes back on every reload.

## Running

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check, then bundle into dist/
```

No runtime dependencies. Just Vite and TypeScript at build time.

## Deploying to GitHub Pages

`.github/workflows/deploy.yml` builds on every push to `main` and publishes
`dist/`. The only manual step is once, in the repo settings: set Pages → Source →
GitHub Actions. The Vite `base` is relative, so it works from a project page or a
custom domain without any changes. For a custom domain, drop a `CNAME` file in
`public/` and point the DNS at it.

## Parameters

The panel updates live.

Bodies: the cost is N², so 65 536 bodies is 16 times the work of 16 384. The
pairwise/s readout reacts right away.

Gravity and time step: push either one far enough and the integrator comes apart.
More softening buys some of that back.

Substeps: more physics per rendered frame, so the galaxy evolves faster without
the frame rate dropping.

## Why WebGPU and not WebGL

This is a compute problem, not a rendering one. WebGL can fake compute with
render-to-texture tricks, but WebGPU just has compute shaders and shared memory,
which is what the simulation actually needs. The trade-off is reach: WebGPU is
solid on Chromium and recent Safari but isn't everywhere yet, so where it's
missing the page shows a "WebGPU required" notice instead of a dead canvas.

## Ideas for later

- Barnes-Hut or an FMM to break the O(N²) ceiling and reach into the millions.
- A proper colour map and an HDR bloom pass.
- Seed two disks and let them collide.

---

Built by Ethan Puyaubreau. MIT licensed.
