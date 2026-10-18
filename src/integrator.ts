// The stored velocity sits half a step behind the position. When dt changes
// from dtPrev to dt, the kick that restores the stagger is
// a * (dtPrev + dt) / 2, which is a * dt * kickScale.
//
// dtPrev = 0 gives 0.5: the initial half kick. dtPrev = dt gives 1: the
// steady leapfrog kick.
export function kickScale(dtPrev: number, dt: number): number {
  return (dtPrev + dt) / (2 * dt);
}

// The step uniform is shared by every substep of a compute pass, so a step
// whose kickScale differs from 1 runs as a single dispatch.
export function substepsFor(dtPrev: number, dt: number, requested: number): number {
  if (dtPrev !== dt) {
    return 1;
  }
  return Math.max(1, requested);
}
