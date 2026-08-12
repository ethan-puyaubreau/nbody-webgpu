// O(N^2) gravitational N-body, one body per invocation.
//
// Forces are summed in tiles: each workgroup cooperatively stages a block of
// bodies into shared memory, so the inner loop reads from on-chip storage
// instead of hammering global memory N times per body. This is the classic
// GPU-Gems tiling trick and it's where most of the speedup over a naive
// global-memory loop comes from.
//
// Integration is leapfrog (kick-drift). The velocity is staggered half a step
// behind position, which is what makes it second-order and (more importantly
// for a thing that runs forever) symplectic: energy wobbles but doesn't drift,
// so the disk holds together instead of either flying apart or collapsing into
// the softening floor. kickScale = 0.5 on the very first step performs the
// initial half-kick that sets up the stagger; 1.0 thereafter.

const TILE: u32 = 256u;

struct SimParams {
  g: f32,
  soften2: f32,   // squared Plummer softening length
  dt: f32,
  kickScale: f32, // 0.5 on the first step, 1.0 after
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read> posIn: array<vec4<f32>>;   // xyz = position, w = mass
@group(0) @binding(2) var<storage, read_write> posOut: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;

var<workgroup> shared_pos: array<vec4<f32>, TILE>;

@compute @workgroup_size(256)
fn integrate(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  let n = sim.n;

  // Out-of-range threads still have to march through the tile loop so their
  // workgroupBarrier() calls line up with everyone else's; they just never
  // write anything back.
  var self_pos = vec3<f32>(0.0);
  if (i < n) {
    self_pos = posIn[i].xyz;
  }

  var acc = vec3<f32>(0.0);
  let tiles = (n + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < tiles; t = t + 1u) {
    let load = t * TILE + lid.x;
    var p = vec4<f32>(0.0); // padding bodies carry mass 0 -> zero force
    if (load < n) {
      p = posIn[load];
    }
    shared_pos[lid.x] = p;
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let other = shared_pos[k];
      let r = other.xyz - self_pos;
      // Plummer softening also handles the self-term for free: when other == self
      // the separation is zero, so the contribution is zero regardless.
      let dist2 = dot(r, r) + sim.soften2;
      let inv = inverseSqrt(dist2);
      acc = acc + sim.g * other.w * r * (inv * inv * inv);
    }
    workgroupBarrier();
  }

  if (i >= n) {
    return;
  }

  var v = vel[i].xyz + acc * sim.dt * sim.kickScale; // kick
  let next = self_pos + v * sim.dt;                  // drift

  vel[i] = vec4<f32>(v, 0.0);
  posOut[i] = vec4<f32>(next, posIn[i].w); // mass rides along in w
}
