import type { Vec3 } from "./math";

export interface GalaxyParams {
  count: number;
  g: number;
  centralMass: number;
  diskMass: number;
  rInner: number;
  rOuter: number;
  thickness: number;
  seed: number;
}

export interface InitialState {
  positions: Float32Array; // vec4 per body: xyz, w = mass
  velocities: Float32Array; // vec4 per body: xyz, w unused
}

export const DEFAULTS: Omit<GalaxyParams, "count"> = {
  g: 1.0,
  centralMass: 2.0,
  diskMass: 1.0,
  rInner: 0.4,
  rOuter: 6.0,
  thickness: 0.05,
  seed: 0x9e3779b9,
};

// The radius is r = rInner + (rOuter - rInner) * u^RADIAL_BIAS with u uniform
// in [0, 1). An exponent above 1 moves bodies inward, so the core is denser
// than the rim. The same exponent gives the enclosed disk mass: the fraction
// of bodies inside r is x^(1/RADIAL_BIAS) with x = (r - rInner) / (rOuter - rInner).
export const RADIAL_BIAS = 1.4;

// Central mass plus the expected disk mass inside radius r.
export function enclosedMass(params: GalaxyParams, r: number): number {
  const { centralMass, diskMass, rInner, rOuter } = params;
  const x = (r - rInner) / (rOuter - rInner);
  return centralMass + diskMass * x ** (1 / RADIAL_BIAS);
}

// Body 0 is the central mass at rest at the origin. The other bodies form a
// disk on nearly circular orbits. The speed comes from the enclosed mass, so
// the disk starts in approximate rotational balance instead of falling in at
// once. An 8 percent speed jitter breaks the clockwork ring.
export function buildGalaxy(params: GalaxyParams): InitialState {
  const { count, g, centralMass, diskMass, rInner, rOuter, thickness, seed } = params;
  const rand = mulberry32(seed);

  const positions = new Float32Array(count * 4);
  const velocities = new Float32Array(count * 4);

  const bodyMass = diskMass / Math.max(1, count - 1);

  positions[3] = centralMass; // w of body 0

  for (let i = 1; i < count; i++) {
    const r = rInner + (rOuter - rInner) * rand() ** RADIAL_BIAS;
    const theta = rand() * Math.PI * 2;
    const cx = Math.cos(theta);
    const cz = Math.sin(theta);

    const x = r * cx;
    const z = r * cz;
    const y = gaussian(rand) * thickness;

    const enclosed = enclosedMass(params, r);
    const vCirc = Math.sqrt((g * enclosed) / r);

    // Tangent in the disk plane, 90 degrees from the radial direction.
    const tangent: Vec3 = [-cz, 0, cx];
    const jitter = 1 + (rand() - 0.5) * 0.08;

    const o = i * 4;
    positions[o + 0] = x;
    positions[o + 1] = y;
    positions[o + 2] = z;
    positions[o + 3] = bodyMass;

    velocities[o + 0] = tangent[0] * vCirc * jitter;
    velocities[o + 1] = gaussian(rand) * vCirc * 0.02;
    velocities[o + 2] = tangent[2] * vCirc * jitter;
  }

  return { positions, velocities };
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform. One value of the pair is enough for the scatter.
function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-7);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
