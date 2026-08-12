import nbodyWgsl from "./shaders/nbody.wgsl?raw";
import renderWgsl from "./shaders/render.wgsl?raw";
import { buildGalaxy, DEFAULTS, type GalaxyParams } from "./galaxy";

const WORKGROUP = 256;

export interface StepParams {
  g: number;
  softening: number;
  dt: number;
}

export class Simulation {
  count = 0;
  private workgroups = 0;
  private current = 0;
  private primed = false;

  private posBuffers: [GPUBuffer, GPUBuffer] | null = null;
  private velBuffer: GPUBuffer | null = null;

  private readonly simBuffer: GPUBuffer;
  private readonly camBuffer: GPUBuffer;
  private readonly computeBind: [GPUBindGroup, GPUBindGroup] = [null!, null!];
  private readonly renderBind: [GPUBindGroup, GPUBindGroup] = [null!, null!];

  private readonly computePipeline: GPUComputePipeline;
  private readonly renderPipeline: GPURenderPipeline;
  private readonly computeLayout: GPUBindGroupLayout;
  private readonly renderLayout: GPUBindGroupLayout;

  private params: StepParams = { g: DEFAULTS.g, softening: 0.12, dt: 0.01 };
  vMax = 1;

  constructor(
    private readonly device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.simBuffer = device.createBuffer({
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
      ],
    });

    this.renderLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    const computeModule = device.createShaderModule({ code: nbodyWgsl });
    this.computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }),
      compute: { module: computeModule, entryPoint: "integrate" },
    });

    const renderModule = device.createShaderModule({ code: renderWgsl });
    this.renderPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderLayout] }),
      vertex: { module: renderModule, entryPoint: "vs" },
      fragment: {
        module: renderModule,
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
  }

  setParams(p: StepParams): void {
    this.params = p;
  }

  reset(count: number): void {
    this.posBuffers?.forEach((b) => b.destroy());
    this.velBuffer?.destroy();

    this.count = count;
    this.workgroups = Math.ceil(count / WORKGROUP);
    this.current = 0;
    this.primed = false;

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

    this.posBuffers = [a, b];
    this.velBuffer = vel;

    const mk = (posIn: GPUBuffer, posOut: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout: this.computeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.simBuffer } },
          { binding: 1, resource: { buffer: posIn } },
          { binding: 2, resource: { buffer: posOut } },
          { binding: 3, resource: { buffer: vel } },
        ],
      });
    this.computeBind[0] = mk(a, b);
    this.computeBind[1] = mk(b, a);

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
  // ping-pong read-after-write between substeps is safe without manual barriers.
  step(encoder: GPUCommandEncoder, substeps: number): void {
    const kick = this.primed ? 1.0 : 0.5;
    this.writeSim(kick);
    const steps = this.primed ? Math.max(1, substeps) : 1; // first ever step only primes

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    for (let s = 0; s < steps; s++) {
      pass.setBindGroup(0, this.computeBind[this.current]);
      pass.dispatchWorkgroups(this.workgroups);
      this.current ^= 1; // freshly written buffer becomes current
    }
    pass.end();
    this.primed = true;
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView, camera: Float32Array, viewport: [number, number], pointSize: number): void {
    const cam = new Float32Array(20);
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

  private writeSim(kick: number): void {
    const buf = new ArrayBuffer(32);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f[0] = this.params.g;
    f[1] = this.params.softening * this.params.softening;
    f[2] = this.params.dt;
    f[3] = kick;
    u[4] = this.count;
    this.device.queue.writeBuffer(this.simBuffer, 0, buf);
  }
}
