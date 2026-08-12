import "./style.css";
import { Simulation, type StepParams } from "./simulation";
import { OrbitCamera } from "./camera";

const canvas = document.getElementById("scene") as HTMLCanvasElement;
const fallback = document.getElementById("fallback") as HTMLElement;

boot().catch((err) => {
  console.error(err);
  showFallback("Something went wrong spinning up the GPU. The console has the details.");
});

async function boot(): Promise<void> {
  if (!navigator.gpu) {
    showFallback("This demo runs on WebGPU, which your browser doesn't expose yet. Chrome, Edge or Safari 18+ will render it.");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    showFallback("No WebGPU adapter available. That usually means a headless or locked-down GPU. Try a different browser or machine.");
    return;
  }

  const device = await adapter.requestDevice();
  let deviceAlive = true;
  device.lost.then((info) => {
    deviceAlive = false;
    if (info.reason !== "destroyed") {
      showFallback("The GPU device was lost. A reload normally brings it back.");
    }
  });

  const context = canvas.getContext("webgpu");
  if (!context) {
    showFallback("Couldn't get a WebGPU canvas context.");
    return;
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const sim = new Simulation(device, format);
  const camera = new OrbitCamera();
  camera.attach(canvas);

  const ui = wireControls();
  sim.setParams(ui.params());
  sim.reset(ui.bodies());
  ui.onBodies((n) => sim.reset(n));

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
  window.addEventListener("resize", resize);
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
    resize();

    camera.update(dt);
    sim.setParams(ui.params());

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
      const interactions = sim.count * sim.count * (ui.paused() ? 0 : substeps) * rate;
      setText("hud-fps", rate.toFixed(0));
      setText("hud-ms", (1000 / Math.max(rate, 1)).toFixed(1));
      setText("hud-bodies", formatCount(sim.count));
      setText("hud-rate", formatRate(interactions));
    }

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

interface Controls {
  params(): StepParams;
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

  // collapse the panel on small screens / on demand
  const panel = byId<HTMLElement>("panel");
  byId<HTMLButtonElement>("panel-toggle").addEventListener("click", () => {
    panel.classList.toggle("collapsed");
  });

  return {
    params: () => ({ g: g(), softening: soft(), dt: dt() }),
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

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing element #${id}`);
  return el as T;
}

function showFallback(message: string): void {
  fallback.querySelector("p")!.textContent = message;
  fallback.hidden = false;
  canvas.style.display = "none";
}
