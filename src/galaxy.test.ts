import { describe, expect, it } from "vitest";
import { buildGalaxy, DEFAULTS } from "./galaxy";

const params = { ...DEFAULTS, count: 2000 };

describe("buildGalaxy", () => {
  it("returns identical arrays for the same seed", () => {
    const a = buildGalaxy(params);
    const b = buildGalaxy(params);
    expect(a.positions).toEqual(b.positions);
    expect(a.velocities).toEqual(b.velocities);
  });

  it("places the central mass at rest at the origin", () => {
    const { positions, velocities } = buildGalaxy(params);
    expect(Array.from(positions.slice(0, 3))).toEqual([0, 0, 0]);
    expect(positions[3]).toBe(params.centralMass);
    expect(Array.from(velocities.slice(0, 4))).toEqual([0, 0, 0, 0]);
  });

  it("conserves the total mass", () => {
    const { positions } = buildGalaxy(params);
    let total = 0;
    for (let i = 0; i < params.count; i++) {
      total += positions[i * 4 + 3];
    }
    expect(Math.abs(total - (params.centralMass + params.diskMass))).toBeLessThan(1e-5);
  });

  it("gives the outermost bodies about the circular speed of the full mass", () => {
    const { positions, velocities } = buildGalaxy(params);
    const bodies: { r: number; speed: number }[] = [];
    for (let i = 1; i < params.count; i++) {
      const r = Math.hypot(positions[i * 4], positions[i * 4 + 2]);
      const speed = Math.hypot(velocities[i * 4], velocities[i * 4 + 2]);
      bodies.push({ r, speed });
    }
    bodies.sort((p, q) => q.r - p.r);
    const totalMass = params.centralMass + params.diskMass;
    for (const { r, speed } of bodies.slice(0, 10)) {
      const expected = Math.sqrt((params.g * totalMass) / r);
      expect(Math.abs(speed / expected - 1)).toBeLessThan(0.08);
    }
  });

  it("follows the sampled enclosed mass at mid radius", () => {
    // The linear formula runs 2.5 percent slow here. The per-body jitter is
    // +-4 percent and averages out over the band.
    const big = { ...DEFAULTS, count: 200_000 };
    const { positions, velocities } = buildGalaxy(big);
    const bodies: { r: number; speed: number }[] = [];
    for (let i = 1; i < big.count; i++) {
      const r = Math.hypot(positions[i * 4], positions[i * 4 + 2]);
      const speed = Math.hypot(velocities[i * 4], velocities[i * 4 + 2]);
      bodies.push({ r, speed });
    }
    bodies.sort((p, q) => p.r - q.r);
    const bodyMass = big.diskMass / (big.count - 1);
    const span = big.rOuter - big.rInner;
    let sum = 0;
    let n = 0;
    bodies.forEach(({ r, speed }, index) => {
      const x = (r - big.rInner) / span;
      if (x >= 0.2 && x <= 0.4) {
        const enclosed = big.centralMass + bodyMass * index;
        sum += speed / Math.sqrt((big.g * enclosed) / r);
        n++;
      }
    });
    expect(Math.abs(sum / n - 1)).toBeLessThan(0.01);
  });

  it("produces only finite values", () => {
    const { positions, velocities } = buildGalaxy(params);
    expect(positions.every(Number.isFinite)).toBe(true);
    expect(velocities.every(Number.isFinite)).toBe(true);
  });
});
