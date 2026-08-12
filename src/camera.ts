import { lookAt, multiply, perspective, type Mat4, type Vec3 } from "./math";

export class OrbitCamera {
  private azimuth = 0.6;
  private elevation = 0.42;
  private distance = 15;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  drift = 0.06; // radians/sec when idle
  readonly fov = (48 * Math.PI) / 180;

  attach(target: HTMLElement): void {
    target.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      target.setPointerCapture(e.pointerId);
    });
    target.addEventListener("pointerup", (e) => {
      this.dragging = false;
      target.releasePointerCapture(e.pointerId);
    });
    target.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      this.azimuth -= (e.clientX - this.lastX) * 0.006;
      this.elevation += (e.clientY - this.lastY) * 0.006;
      this.elevation = clamp(this.elevation, -1.45, 1.45);
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
    target.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.distance = clamp(this.distance * Math.exp(e.deltaY * 0.0012), 3.5, 45);
      },
      { passive: false },
    );
  }

  update(dt: number): void {
    if (!this.dragging) {
      this.azimuth += this.drift * dt;
    }
  }

  viewProjection(aspect: number): Mat4 {
    const ce = Math.cos(this.elevation);
    const eye: Vec3 = [
      this.distance * ce * Math.sin(this.azimuth),
      this.distance * Math.sin(this.elevation),
      this.distance * ce * Math.cos(this.azimuth),
    ];
    const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const proj = perspective(this.fov, aspect, 0.1, 120);
    return multiply(proj, view);
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
