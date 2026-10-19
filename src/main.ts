import "@fontsource-variable/fraunces/opsz.css";
import "@fontsource-variable/fraunces/opsz-italic.css";
import "@fontsource/space-grotesk/latin-300.css";
import "@fontsource/space-grotesk/latin-400.css";
import "@fontsource/space-grotesk/latin-500.css";
import "./style.css";
import { OrbitCamera } from "./camera";
import { Simulation, type StepParams } from "./simulation";

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const fallback = document.getElementById("fallback") as HTMLElement;

boot().catch((err) => {
  console.error(err);
  const detail = err instanceof Error ? err.message.split("\n")[0] : String(err);
  showFallback(`The GPU setup failed: ${detail}`);
});

async function boot(): Promise<void> {
  if (!navigator.gpu) {
    showFallback("This page needs WebGPU, and this browser does not expose it. Chrome and Edge support it.");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    showFallback("No WebGPU adapter is available. The GPU is headless or restricted. Try another browser or machine.");
    return;
  }

  const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  const device = await adapter.requestDevice({ requiredFeatures });
  let deviceAlive = true;
  let energyTimer: number | undefined;
  // Stops the frame loop and the energy sampler: after a GPU failure both only repeat the error.
  const stopRunning = (): void => {
    deviceAlive = false;
    clearInterval(energyTimer);
  };
  device.addEventListener("uncapturederror", (event) => {
    const message = (event as GPUUncapturedErrorEvent).error.message;
    console.error(message);
    stopRunning();
    showFallback(`A GPU error occurred: ${message}`);
  });
  device.lost.then((info) => {
    stopRunning();
    if (info.reason !== "destroyed") {
      showFallback("The GPU device was lost. Reload the page to recover it.");
    }
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    showFallback("The canvas did not return a WebGPU context.");
    return;
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const sim = new Simulation(device, format);
  await sim.validate();
  const camera = new OrbitCamera();
  camera.attach(canvas);

  const ui = wireControls();
  const params: StepParams = { g: 0, softening: 0, dt: 0 };
  sim.setParams(ui.params(params));

  // Incremented on every reset: an energy readback that straddles a reset is discarded.
  let resetEpoch = 0;
  let baseline: { e0: number; g: number; softening: number } | null = null;
  const resetSim = (n: number): void => {
    resetEpoch += 1;
    baseline = null;
    sim.reset(n);
  };
  resetSim(ui.bodies());
  ui.onBodies(resetSim);

  const hud = {
    fps: byId("hud-fps"),
    ms: byId("hud-ms"),
    msLabel: byId("hud-ms-label"),
    bodies: byId("hud-bodies"),
    rate: byId("hud-rate"),
    drift: byId("hud-drift"),
  };

  const sampleEnergy = async (): Promise<void> => {
    const epoch = resetEpoch;
    const g = params.g;
    const softening = params.softening;
    try {
      const { kinetic, potential } = await sim.energy();
      if (epoch !== resetEpoch) return;
      const total = kinetic + potential;
      if (baseline === null || baseline.g !== g || baseline.softening !== softening) {
        baseline = { e0: total, g, softening };
      }
      const drift = ((total - baseline.e0) / Math.abs(baseline.e0)) * 100;
      setTextAnimated(hud.drift, `${drift >= 0 ? "+" : "-"}${Math.abs(drift).toFixed(3)}%`);
    } catch (err) {
      // A reset during the readback rejects the promise: that sample is stale.
      if (epoch !== resetEpoch) return;
      console.error(err);
    }
  };
  let energyPending = false;
  energyTimer = window.setInterval(() => {
    if (ui.paused() || energyPending) return;
    energyPending = true;
    sampleEnergy().finally(() => {
      energyPending = false;
    });
  }, 500);

  let viewport: [number, number] = [1, 1];
  const resize = (): void => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (w !== canvas.width || h !== canvas.height) {
      canvas.width = w;
      canvas.height = h;
    }
    viewport = [w, h];
  };
  new ResizeObserver(resize).observe(canvas);
  resize();

  const fps = new Ema(0.1);
  let last = performance.now();
  let hudAt = 0;

  const frame = (now: number): void => {
    if (!deviceAlive) return;

    const interval = now - last;
    last = now;
    const dt = Math.min(interval / 1000, 0.05);
    fps.push(1000 / Math.max(interval, 1));

    camera.update(dt);
    sim.setParams(ui.params(params));

    const encoder = device.createCommandEncoder();
    const substeps = ui.substeps();
    if (!ui.paused()) {
      sim.step(encoder, substeps);
    }
    const vp = camera.viewProjection(viewport[0] / viewport[1]);
    sim.render(encoder, context.getCurrentTexture().createView(), vp, viewport, ui.pointSize());
    device.queue.submit([encoder.finish()]);

    if (now - hudAt > 220) {
      hudAt = now;
      const rate = fps.value;
      const steps = ui.paused() ? 0 : sim.lastSteps;
      const pairs = sim.count * sim.count * steps;
      const gpuMs = sim.gpuMs;
      const pairsPerSecond = gpuMs !== null ? pairs / (gpuMs / 1000) : pairs * rate;
      setTextAnimated(hud.fps, rate.toFixed(0));
      setTextAnimated(hud.ms, (gpuMs ?? 1000 / Math.max(rate, 1)).toFixed(1));
      hud.msLabel.textContent = gpuMs !== null ? "gpu ms" : "frame ms";
      setTextAnimated(hud.bodies, formatCount(sim.count));
      setTextAnimated(hud.rate, formatRate(pairsPerSecond));
    }

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

interface Controls {
  params(out: StepParams): StepParams;
  bodies(): number;
  substeps(): number;
  pointSize(): number;
  paused(): boolean;
  onBodies(cb: (n: number) => void): void;
}

function wireControls(): Controls {
  const bodies = byId<HTMLSelectElement>("ctrl-bodies");
  const g = bound("ctrl-g", "val-g", (v) => v.toFixed(2));
  const soft = bound("ctrl-soft", "val-soft", (v) => v.toFixed(3));
  const dt = bound("ctrl-dt", "val-dt", (v) => v.toFixed(3));
  const size = bound("ctrl-size", "val-size", (v) => v.toFixed(2));
  const sub = bound("ctrl-sub", "val-sub", (v) => v.toFixed(0));

  let isPaused = false;
  const pauseBtn = byId<HTMLButtonElement>("btn-pause");
  pauseBtn.addEventListener("click", () => {
    isPaused = !isPaused;
    pauseBtn.textContent = isPaused ? "resume" : "pause";
    pauseBtn.classList.toggle("is-active", isPaused);
  });

  let bodiesCb: (n: number) => void = () => {};
  const resetBtn = byId<HTMLButtonElement>("btn-reset");
  resetBtn.addEventListener("click", () => bodiesCb(parseInt(bodies.value, 10)));
  bodies.addEventListener("change", () => bodiesCb(parseInt(bodies.value, 10)));

  // The panel collapses on small screens or on demand.
  const panel = byId<HTMLElement>("panel");
  byId<HTMLButtonElement>("panel-toggle").addEventListener("click", () => {
    panel.classList.toggle("collapsed");
  });

  return {
    params: (out) => {
      out.g = g();
      out.softening = soft();
      out.dt = dt();
      return out;
    },
    bodies: () => parseInt(bodies.value, 10),
    substeps: () => Math.round(sub()),
    pointSize: () => size(),
    paused: () => isPaused,
    onBodies: (cb) => {
      bodiesCb = cb;
    },
  };
}

function bound(inputId: string, labelId: string, fmt: (v: number) => string): () => number {
  const input = byId<HTMLInputElement>(inputId);
  const label = byId<HTMLElement>(labelId);
  const sync = (): void => {
    label.textContent = fmt(parseFloat(input.value));
    const pct =
      ((parseFloat(input.value) - parseFloat(input.min)) / (parseFloat(input.max) - parseFloat(input.min))) * 100;
    input.style.setProperty("--val", pct.toFixed(1));
  };
  input.addEventListener("input", sync);
  sync();
  return () => parseFloat(input.value);
}

class Ema {
  value = 60;
  constructor(private readonly a: number) {}
  push(x: number): void {
    if (Number.isFinite(x)) this.value += this.a * (x - this.value);
  }
}

function formatCount(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n % 1000 ? 1 : 0)}k` : `${n}`;
}

function formatRate(perSec: number): string {
  if (perSec >= 1e9) return `${(perSec / 1e9).toFixed(2)} G/s`;
  if (perSec >= 1e6) return `${(perSec / 1e6).toFixed(1)} M/s`;
  return `${perSec.toFixed(0)} /s`;
}

// Two class names map to two identically shaped keyframes: swapping them
// changes the animation name, which restarts the pulse without reading layout.
function setTextAnimated(el: HTMLElement, text: string): void {
  el.textContent = text;
  const next = el.classList.contains("pulse-a") ? "pulse-b" : "pulse-a";
  el.classList.remove("pulse-a", "pulse-b");
  el.classList.add(next);
}

function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
}

// The first message wins: later failures are usually consequences of the first.
function showFallback(message: string): void {
  if (!fallback.hidden) return;
  fallback.querySelector("p")!.textContent = message;
  fallback.hidden = false;
  canvas.style.display = "none";
}
