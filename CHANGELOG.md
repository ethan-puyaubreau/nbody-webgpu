# Changelog

## [Unreleased]

### Added

- The project builds with pnpm, Node 22, Vite 7, Biome and Vitest. `pnpm check`
  runs the lint, the type-check, the tests and the style script.
- The fonts are bundled at build time, so the page makes no request to a font
  host.
- `Simulation.gpuMs` reports the smoothed GPU time of the compute pass, read
  from `timestamp-query`. It is `null` when the device lacks the feature.
- `Simulation.energy()` returns the kinetic and potential energy, computed by
  a second compute entry point and summed in float64 on the CPU.
- The HUD shows `gpu ms` (or `frame ms` without `timestamp-query`) and
  `energy drift`, the relative change of the total energy from the first
  sample.
- `Simulation.validate()` throws the shader compilation errors with file, line
  and column.

### Changed

- The HUD takes its time and rate from the GPU timing when it is available.
- The workgroup size comes from one constant, `WORKGROUP`, passed to the
  shader as the `TILE` override.
- The step and camera uniform writes reuse scratch arrays.

### Fixed

- `buildGalaxy` no longer underestimates the enclosed mass: the disk mass
  inside a radius follows `x^(1/1.4)`, the same exponent as the radial
  sampling.
- A change of `dt` no longer breaks the leapfrog stagger: the kick is scaled by
  `(dtPrev + dt) / (2 dt)` and the step runs as one dispatch.
- A pointer that leaves the window, or a canceled pointer, no longer leaves
  the camera stuck in a drag: `pointercancel` and `lostpointercapture` end it.
- A GPU validation error or a shader compilation error now shows its message
  on the page instead of leaving a black canvas.
- The entrance animation and the value pulse are off under
  `prefers-reduced-motion`.
