// Point sprites, the WebGPU way. point-list topology only ever gives you 1px
// dots, so instead each body is an instanced two-triangle quad expanded in clip
// space to a constant pixel size, with a soft radial falloff in the fragment
// stage. Drawn additively over near-black, dense regions pile up and bloom
// toward white the same way a long exposure of a star field does.

struct Camera {
  view_proj: mat4x4<f32>,
  viewport: vec2<f32>, // pixels
  point_size: f32,     // pixels
  v_max: f32,          // speed that maps to the hottest colour
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec4<f32>>;

struct VsOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) heat: f32,
};

@vertex
fn vs(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VsOut {
  // two triangles, corners in [-1, 1]; local var so the dynamic index is
  // portable across WGSL backends
  var quad = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );

  let body = pos[ii];
  let clip = cam.view_proj * vec4<f32>(body.xyz, 1.0);

  let corner = quad[vi];
  // Multiply the NDC offset by clip.w so it survives the perspective divide as a
  // constant pixel size regardless of depth.
  let offset = corner * (vec2<f32>(cam.point_size) / cam.viewport) * 2.0 * clip.w;

  var out: VsOut;
  out.clip = vec4<f32>(clip.xy + offset, clip.z, clip.w);
  out.uv = corner;
  out.heat = clamp(length(vel[ii].xyz) / cam.v_max, 0.0, 1.0);
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) {
    discard;
  }
  let glow = pow(1.0 - d, 2.2);

  let ember = vec3<f32>(0.55, 0.19, 0.03);
  let flare = vec3<f32>(1.0, 0.82, 0.50);
  let tint = mix(ember, flare, in.heat);

  // premultiplied: rgb already scaled by alpha for the additive blend
  return vec4<f32>(tint * glow, glow);
}
