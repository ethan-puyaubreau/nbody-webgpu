import { describe, expect, it } from "vitest";
import { kickScale, substepsFor } from "./integrator";

describe("kickScale", () => {
  it("is 0.5 on the first step", () => {
    expect(kickScale(0, 0.01)).toBe(0.5);
  });

  it("is 1 when dt does not change", () => {
    expect(kickScale(0.01, 0.01)).toBe(1);
  });

  it("averages the old and the new dt when dt grows", () => {
    expect(kickScale(0.01, 0.02)).toBeCloseTo(0.75, 12);
  });
});

describe("substepsFor", () => {
  it("runs one dispatch when dt changed", () => {
    expect(substepsFor(0.01, 0.02, 8)).toBe(1);
  });

  it("runs one dispatch on the first step", () => {
    expect(substepsFor(0, 0.01, 8)).toBe(1);
  });

  it("runs the requested count when dt is steady", () => {
    expect(substepsFor(0.01, 0.01, 8)).toBe(8);
  });

  it("never runs fewer than one dispatch", () => {
    expect(substepsFor(0.01, 0.01, 0)).toBe(1);
  });
});
