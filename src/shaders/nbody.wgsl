// O(N^2) gravitational N-body, one body per invocation.
//
// Forces are summed in tiles: each workgroup stages a block of bodies into
// shared memory, so the inner loop reads on-chip storage instead of global
// memory. Each body is then read from global memory once per tile instead of
// once per pair.
//
// Integration is leapfrog (kick-drift). The velocity is staggered half a step
// behind the position. This makes the scheme second-order and symplectic: the
// energy oscillates around its initial value instead of drifting. kickScale
// is (dtPrev + dt) / (2 dt): 0.5 on the first step, which performs the
// initial half kick, 1.0 while dt is constant, below 1.0 when dt grows and
// above 1.0 when dt shrinks.
//
// The workgroup size is the override TILE. It has no default: the host must
// pass it through the pipeline constants, so the value exists in one place.

override TILE: u32;

struct SimParams {
  g: f32,
  soften2: f32,   // squared Plummer softening length
  dt: f32,
  kickScale: f32, // (dtPrev + dt) / (2 dt)
  n: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

@group(0) @binding(0) var<uniform> sim: SimParams;
@group(0) @binding(1) var<storage, read> posIn: array<vec4<f32>>;   // xyz = position, w = mass
@group(0) @binding(2) var<storage, read_write> posOut: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> energyOut: array<vec2<f32>>; // per body: x = kinetic, y = potential

var<workgroup> shared_pos: array<vec4<f32>, TILE>;

@compute @workgroup_size(TILE)
fn integrate(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  let n = sim.n;

  // Out-of-range threads still run the tile loop: their workgroupBarrier()
  // calls must match those of the other threads. They never write back.
  var self_pos = vec3<f32>(0.0);
  if (i < n) {
    self_pos = posIn[i].xyz;
  }

  var acc = vec3<f32>(0.0);
  let tiles = (n + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < tiles; t = t + 1u) {
    let load = t * TILE + lid.x;
    var p = vec4<f32>(0.0); // padding bodies carry mass 0, so they add no force
    if (load < n) {
      p = posIn[load];
    }
    shared_pos[lid.x] = p;
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let other = shared_pos[k];
      let r = other.xyz - self_pos;
      // The self term is zero because r = 0 makes the numerator zero.
      // Softening only keeps the denominator away from zero.
      let dist2 = dot(r, r) + sim.soften2;
      let inv = inverseSqrt(dist2);
      acc = acc + sim.g * other.w * r * (inv * inv * inv);
    }
    workgroupBarrier();
  }

  if (i >= n) {
    return;
  }

  let v = vel[i].xyz + acc * sim.dt * sim.kickScale; // kick
  let next = self_pos + v * sim.dt;                  // drift

  vel[i] = vec4<f32>(v, 0.0);
  posOut[i] = vec4<f32>(next, posIn[i].w); // mass rides along in w
}

// Per-body energy, read back by the host and summed in float64.
//
// Kinetic: 0.5 m |v|^2 with the stored velocity, which sits half a step
// behind the position.
// Potential: 0.5 m_i phi_i with phi_i = sum over j != i of
// -G m_j / sqrt(|r|^2 + soften2). The same softened form as the force, so
// the energy matches the dynamics. Every pair is counted from both sides,
// hence the factor 0.5. The self term does not vanish here: at r = 0
// it equals -G m_i / soften, which is finite, so the pair (i, i) is excluded
// by comparing indices.
@compute @workgroup_size(TILE)
fn energy(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let i = gid.x;
  let n = sim.n;

  var self_pos = vec3<f32>(0.0);
  var self_mass = 0.0;
  var self_vel = vec3<f32>(0.0);
  if (i < n) {
    let p = posIn[i];
    self_pos = p.xyz;
    self_mass = p.w;
    self_vel = vel[i].xyz;
  }

  var phi = 0.0;
  let tiles = (n + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < tiles; t = t + 1u) {
    let load = t * TILE + lid.x;
    var p = vec4<f32>(0.0);
    if (load < n) {
      p = posIn[load];
    }
    shared_pos[lid.x] = p;
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let j = t * TILE + k;
      if (j != i && j < n) {
        let other = shared_pos[k];
        let r = other.xyz - self_pos;
        phi = phi - sim.g * other.w * inverseSqrt(dot(r, r) + sim.soften2);
      }
    }
    workgroupBarrier();
  }

  if (i >= n) {
    return;
  }

  energyOut[i] = vec2<f32>(
    0.5 * self_mass * dot(self_vel, self_vel),
    0.5 * self_mass * phi,
  );
}
