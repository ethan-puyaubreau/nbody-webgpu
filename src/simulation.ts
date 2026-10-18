import { buildGalaxy, DEFAULTS, type GalaxyParams } from "./galaxy";
import { kickScale, substepsFor } from "./integrator";
import nbodyWgsl from "./shaders/nbody.wgsl?raw";
import renderWgsl from "./shaders/render.wgsl?raw";

const WORKGROUP = 256;
const GPU_MS_SMOOTHING = 0.1;

export interface StepParams {
  g: number;
  softening: number;
  dt: number;
}

export interface EnergySample {
  kinetic: number;
  potential: number;
}

interface Timing {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  staging: GPUBuffer;
  // True from the step that records a sample until the readback unmaps.
  busy: boolean;
}

export class Simulation {
  count = 0;
  vMax = 1;
  // Smoothed GPU time in milliseconds of the compute pass of one step() call.
  // null when the device has no timestamp-query or no sample has arrived yet.
  gpuMs: number | null = null;
  // Number of integration dispatches recorded by the last step() call. A step
  // that restores the stagger after a dt change records one, whatever the
  // requested substeps.
  lastSteps = 0;

  private workgroups = 0;
  private current = 0;
  // The dt of the previous step. 0 after reset: the first kick is a half kick.
  private dtPrev = 0;

  private posBuffers: [GPUBuffer, GPUBuffer] | null = null;
  private velBuffer: GPUBuffer | null = null;
  private energyBuffer: GPUBuffer | null = null;
  private energyStaging: GPUBuffer | null = null;
  private energyPending: Promise<EnergySample> | null = null;

  private timing: Timing | null = null;
  private timingArmed = false;

  private readonly simBuffer: GPUBuffer;
  private readonly energyParamsBuffer: GPUBuffer;
  private readonly camBuffer: GPUBuffer;
  private readonly computeBind: [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private readonly energyBind: [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private readonly renderBind: [GPUBindGroup, GPUBindGroup] = [null!, null!];

  // Scratch memory reused every frame to avoid allocations.
  private readonly camScratch = new Float32Array(20);
  private readonly simScratch = new ArrayBuffer(32);
  private readonly simFloats = new Float32Array(this.simScratch);
  private readonly simUints = new Uint32Array(this.simScratch);

  private readonly computeModule: GPUShaderModule;
  private readonly renderModule: GPUShaderModule;
  private readonly computePipeline: GPUComputePipeline;
  private readonly energyPipeline: GPUComputePipeline;
  private readonly renderPipeline: GPURenderPipeline;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly renderLayout: GPUBindGroupLayout;

  private params: StepParams = { g: DEFAULTS.g, softening: 0.12, dt: 0.01 };

  constructor(
    private readonly device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.simBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // The energy pass has its own uniform: writing the step uniform from
    // energy() could change the kick of a step that is encoded but not yet
    // submitted.
    this.energyParamsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.camBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.computeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    this.renderLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.computeModule = device.createShaderModule({ code: nbodyWgsl });
    const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
    this.computePipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: {
        module: this.computeModule,
        entryPoint: "integrate",
        constants: { TILE: WORKGROUP },
      },
    });
    this.energyPipeline = device.createComputePipeline({
      layout: computeLayout,
      compute: {
        module: this.computeModule,
        entryPoint: "energy",
        constants: { TILE: WORKGROUP },
      },
    });

    this.renderModule = device.createShaderModule({ code: renderWgsl });
    this.renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: { module: this.renderModule, entryPoint: "vs" },
      fragment: {
        module: this.renderModule,
        entryPoint: "fs",
        targets: [
          {
            format,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    if (device.features.has("timestamp-query")) {
      this.timing = {
        querySet: device.createQuerySet({ type: "timestamp", count: 2 }),
        resolve: device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        staging: device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        }),
        busy: false,
      };
    }
  }

  // Throws when a shader module has compilation errors. The message lists each
  // error as "file:line:column message". Does nothing when there are none.
  async validate(): Promise<void> {
    const modules: [string, GPUShaderModule][] = [
      ["nbody.wgsl", this.computeModule],
      ["render.wgsl", this.renderModule],
    ];
    const errors: string[] = [];
    for (const [name, module] of modules) {
      const info = await module.getCompilationInfo();
      for (const message of info.messages) {
        if (message.type === "error") {
          errors.push(`${name}:${message.lineNum}:${message.linePos} ${message.message}`);
        }
      }
    }
    if (errors.length > 0) {
      throw new Error(errors.join("\n"));
    }
  }

  setParams(p: StepParams): void {
    this.params = p;
  }

  reset(count: number): void {
    this.posBuffers?.forEach((b) => {
      b.destroy();
    });
    this.velBuffer?.destroy();
    this.energyBuffer?.destroy();
    this.energyStaging?.destroy();
    // A readback in flight targets the destroyed buffers and rejects. Drop it
    // so the next energy() call samples the new galaxy.
    this.energyPending = null;

    this.count = count;
    this.workgroups = Math.ceil(count / WORKGROUP);
    this.current = 0;
    this.dtPrev = 0;

    const galaxy: GalaxyParams = { ...DEFAULTS, count, g: this.params.g };
    const { positions, velocities } = buildGalaxy(galaxy);
    this.vMax = Math.sqrt((galaxy.g * galaxy.centralMass) / galaxy.rInner) * 1.2;

    const bytes = count * 4 * 4;
    const a = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(a.getMappedRange()).set(positions);
    a.unmap();

    const b = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const vel = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vel.getMappedRange()).set(velocities);
    vel.unmap();

    // One vec2<f32> per body: kinetic and potential energy.
    const energyBytes = count * 2 * 4;
    const energy = this.device.createBuffer({
      size: energyBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.energyStaging = this.device.createBuffer({
      size: energyBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.posBuffers = [a, b];
    this.velBuffer = vel;
    this.energyBuffer = energy;

    const mk = (uniform: GPUBuffer, posIn: GPUBuffer, posOut: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: posIn } },
          { binding: 2, resource: { buffer: posOut } },
          { binding: 3, resource: { buffer: vel } },
          { binding: 4, resource: { buffer: energy } },
        ],
      });
    this.computeBind[0] = mk(this.simBuffer, a, b);
    this.computeBind[1] = mk(this.simBuffer, b, a);
    this.energyBind[0] = mk(this.energyParamsBuffer, a, b);
    this.energyBind[1] = mk(this.energyParamsBuffer, b, a);

    const mkRender = (pos: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout: this.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.camBuffer } },
          { binding: 1, resource: { buffer: pos } },
          { binding: 2, resource: { buffer: vel } },
        ],
      });
    this.renderBind[0] = mkRender(a);
    this.renderBind[1] = mkRender(b);
  }

  // Successive dispatches in one compute pass are ordered by WebGPU, so the
  // ping-pong read-after-write between substeps needs no manual barrier.
  step(encoder: GPUCommandEncoder, substeps: number): void {
    // The previous call has been submitted by now: start reading its sample.
    if (this.timing !== null && this.timingArmed) {
      this.timingArmed = false;
      void this.readTimestamps(this.timing);
    }

    const dt = this.params.dt;
    this.writeSimParams(this.simBuffer, kickScale(this.dtPrev, dt));
    const steps = substepsFor(this.dtPrev, dt, substeps);
    this.lastSteps = steps;

    // One sample in flight at a time: the staging buffer is mapped while the
    // readback is pending, and a copy into a mapped buffer is invalid.
    const timing = this.timing !== null && !this.timing.busy ? this.timing : null;
    const pass = encoder.beginComputePass(
      timing === null
        ? undefined
        : {
            timestampWrites: {
              querySet: timing.querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          },
    );
    pass.setPipeline(this.computePipeline);
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, this.computeBind[this.current]);
      pass.dispatchWorkgroups(this.workgroups);
      this.current ^= 1; // the buffer written by this dispatch becomes the current one
    }
    pass.end();

    if (timing !== null) {
      encoder.resolveQuerySet(timing.querySet, 0, 2, timing.resolve, 0);
      encoder.copyBufferToBuffer(timing.resolve, 0, timing.staging, 0, 16);
      timing.busy = true;
      this.timingArmed = true;
    }

    this.dtPrev = dt;
  }

  // Total kinetic and potential energy, summed in float64 on the CPU.
  // Call it outside the frame, not between step() and queue.submit(): it
  // submits its own pass on the current position buffer. A call while one is
  // pending returns the same promise. The promise rejects when reset() runs
  // during the readback: the caller must catch it and discard the sample.
  energy(): Promise<EnergySample> {
    if (this.energyPending !== null) {
      return this.energyPending;
    }
    const pending = this.readEnergy().finally(() => {
      // reset() may have replaced or cleared the pending sample already.
      if (this.energyPending === pending) {
        this.energyPending = null;
      }
    });
    this.energyPending = pending;
    return pending;
  }

  render(
    encoder: GPUCommandEncoder,
    view: GPUTextureView,
    camera: Float32Array,
    viewport: [number, number],
    pointSize: number,
  ): void {
    const cam = this.camScratch;
    cam.set(camera, 0);
    cam[16] = viewport[0];
    cam[17] = viewport[1];
    cam[18] = pointSize;
    cam[19] = this.vMax;
    this.device.queue.writeBuffer(this.camBuffer, 0, cam);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.011, g: 0.012, b: 0.018, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBind[this.current]);
    pass.draw(6, this.count);
    pass.end();
  }

  private writeSimParams(target: GPUBuffer, kick: number): void {
    this.simFloats[0] = this.params.g;
    this.simFloats[1] = this.params.softening * this.params.softening;
    this.simFloats[2] = this.params.dt;
    this.simFloats[3] = kick;
    this.simUints[4] = this.count;
    this.device.queue.writeBuffer(target, 0, this.simScratch);
  }

  private async readTimestamps(timing: Timing): Promise<void> {
    try {
      await timing.staging.mapAsync(GPUMapMode.READ);
      const ticks = new BigUint64Array(timing.staging.getMappedRange());
      const start = ticks[0];
      const end = ticks[1];
      timing.staging.unmap();
      // Timestamps are in nanoseconds. A non-positive span means the counter
      // was discontinuous: drop the sample.
      if (end > start) {
        const ms = Number(end - start) / 1e6;
        this.gpuMs = this.gpuMs === null ? ms : this.gpuMs + GPU_MS_SMOOTHING * (ms - this.gpuMs);
      }
      timing.busy = false;
    } catch (error) {
      console.error("GPU timing disabled: the timestamp readback failed.", error);
      this.timing = null;
      this.gpuMs = null;
    }
  }

  private async readEnergy(): Promise<EnergySample> {
    const energy = this.energyBuffer;
    const staging = this.energyStaging;
    if (energy === null || staging === null) {
      throw new Error("energy() was called before reset().");
    }

    this.writeSimParams(this.energyParamsBuffer, 1);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.energyPipeline);
    pass.setBindGroup(0, this.energyBind[this.current]);
    pass.dispatchWorkgroups(this.workgroups);
    pass.end();
    encoder.copyBufferToBuffer(energy, 0, staging, 0, this.count * 2 * 4);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    try {
      const values = new Float32Array(staging.getMappedRange());
      let kinetic = 0;
      let potential = 0;
      for (let i = 0; i < this.count; i++) {
        kinetic += values[i * 2];
        potential += values[i * 2 + 1];
      }
      // The stored velocity sits half a step behind the position, so the
      // kinetic energy is evaluated half a step early. The offset oscillates
      // and does not accumulate.
      return { kinetic, potential };
    } finally {
      staging.unmap();
    }
  }
}
