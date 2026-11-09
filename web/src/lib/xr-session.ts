import * as THREE from "three";

type ControlsLike = {
  enabled: boolean;
  target: THREE.Vector3;
  update(): void;
};

export type XRSessionManagerOptions = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: ControlsLike;
  /** Root group holding the grabbable visualization content. Must be a direct child of `scene`. */
  content: THREE.Group;
  /** Where the content floats when the session starts (local-floor space, meters). */
  vrPosition?: THREE.Vector3;
  /** Uniform content scale applied on session start. */
  vrScale?: number;
  onSessionChange?: (active: boolean) => void;
};

const DEFAULT_VR_POSITION = new THREE.Vector3(0, 1.3, -1.2);
const DEFAULT_VR_SCALE = 0.35;
const RAY_LENGTH = 0.3;
const MIN_SCALE_FACTOR = 0.1;
const MAX_SCALE_FACTOR = 8;

type SavedState = {
  cameraPos: THREE.Vector3;
  cameraQuat: THREE.Quaternion;
  target: THREE.Vector3;
  contentPos: THREE.Vector3;
  contentQuat: THREE.Quaternion;
  contentScale: THREE.Vector3;
};

/**
 * Retrofits an immersive-vr WebXR session onto an existing vanilla-Three
 * scene. On enter: floats `content` in front of the user and wires both
 * controllers for grab (squeeze = Quest grip, select = trigger / Vision Pro
 * pinch) and two-hand scaling. On exit (page button or headset system UI):
 * restores the desktop camera, controls, and content transform.
 *
 * The owning component must drive its render loop via
 * `renderer.setAnimationLoop` and call `update()` once per frame.
 */
export class XRSessionManager {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: ControlsLike;
  private content: THREE.Group;
  private vrPosition: THREE.Vector3;
  private vrScale: number;
  private onSessionChange?: (active: boolean) => void;

  private session: XRSession | null = null;
  private controllers: THREE.Group[] = [];
  private grabbers: THREE.Group[] = []; // controllers currently grabbing, in grab order
  private twoHand: { initialDist: number; initialScale: number } | null = null;
  private saved: SavedState | null = null;
  private controllerCleanups: Array<() => void> = [];

  constructor(opts: XRSessionManagerOptions) {
    this.renderer = opts.renderer;
    this.scene = opts.scene;
    this.camera = opts.camera;
    this.controls = opts.controls;
    this.content = opts.content;
    this.vrPosition = opts.vrPosition?.clone() ?? DEFAULT_VR_POSITION.clone();
    this.vrScale = opts.vrScale ?? DEFAULT_VR_SCALE;
    this.onSessionChange = opts.onSessionChange;
  }

  get presenting(): boolean {
    return this.renderer.xr.isPresenting;
  }

  /** Must be called directly from a click handler (user activation). */
  async enter(): Promise<void> {
    if (this.session) return;
    const xr = typeof navigator !== "undefined" ? navigator.xr : undefined;
    if (!xr) return;

    const session = await xr.requestSession("immersive-vr", {
      optionalFeatures: ["local-floor", "hand-tracking"],
    });
    this.session = session;

    this.saved = {
      cameraPos: this.camera.position.clone(),
      cameraQuat: this.camera.quaternion.clone(),
      target: this.controls.target.clone(),
      contentPos: this.content.position.clone(),
      contentQuat: this.content.quaternion.clone(),
      contentScale: this.content.scale.clone(),
    };
    this.controls.enabled = false;

    session.addEventListener("end", this.handleSessionEnd);

    // Three has no internal fallback if the reference space is unavailable,
    // so pick it from what the session actually granted.
    const hasFloor = session.enabledFeatures?.includes("local-floor") ?? true;
    this.renderer.xr.enabled = true;
    this.renderer.xr.setReferenceSpaceType(hasFloor ? "local-floor" : "local");

    try {
      await this.renderer.xr.setSession(session);
    } catch (err) {
      session.removeEventListener("end", this.handleSessionEnd);
      this.session = null;
      this.renderer.xr.enabled = false;
      this.restoreDesktopState();
      throw err;
    }

    this.content.position.copy(this.vrPosition);
    this.content.quaternion.identity();
    this.content.scale.setScalar(this.vrScale);
    if (!hasFloor) {
      // "local" origin is at head height, not the floor — drop content to eye level.
      this.content.position.y -= 1.3;
    }

    this.setupControllers();
    this.onSessionChange?.(true);
  }

  exit(): void {
    this.session?.end().catch(() => {});
  }

  /** Call once per frame from the component's animation loop. */
  update(): void {
    if (!this.presenting || !this.twoHand || this.grabbers.length !== 2) return;
    const dist = this.controllerDistance();
    if (dist <= 0 || this.twoHand.initialDist <= 0) return;
    const scale = THREE.MathUtils.clamp(
      this.twoHand.initialScale * (dist / this.twoHand.initialDist),
      this.vrScale * MIN_SCALE_FACTOR,
      this.vrScale * MAX_SCALE_FACTOR,
    );
    this.content.scale.setScalar(scale);
  }

  /** Idempotent; safe under React StrictMode double-invoke. Call before renderer.dispose(). */
  dispose(): void {
    const session = this.session;
    if (!session) return;
    // The component is being torn down — run the restore path synchronously
    // rather than waiting for the async "end" event.
    session.removeEventListener("end", this.handleSessionEnd);
    this.session = null;
    this.teardownControllers();
    session.end().catch(() => {});
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private handleSessionEnd = (): void => {
    this.session?.removeEventListener("end", this.handleSessionEnd);
    this.session = null;
    this.teardownControllers();
    this.renderer.xr.enabled = false;
    this.restoreDesktopState();
    this.onSessionChange?.(false);
  };

  private restoreDesktopState(): void {
    if (this.content.parent !== this.scene) this.scene.attach(this.content);
    const s = this.saved;
    if (s) {
      this.content.position.copy(s.contentPos);
      this.content.quaternion.copy(s.contentQuat);
      this.content.scale.copy(s.contentScale);
      this.camera.position.copy(s.cameraPos);
      this.camera.quaternion.copy(s.cameraQuat);
      this.controls.target.copy(s.target);
      this.saved = null;
    }
    this.controls.enabled = true;
    this.controls.update();
  }

  private setupControllers(): void {
    for (let i = 0; i < 2; i++) {
      const controller = this.renderer.xr.getController(i);

      const rayGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, -RAY_LENGTH),
      ]);
      const rayMat = new THREE.LineBasicMaterial({
        color: 0x9ec5ff,
        transparent: true,
        opacity: 0.6,
      });
      const ray = new THREE.Line(rayGeo, rayMat);
      const tipGeo = new THREE.SphereGeometry(0.012, 12, 8);
      const tipMat = new THREE.MeshBasicMaterial({ color: 0xbcd6ff });
      const tip = new THREE.Mesh(tipGeo, tipMat);
      controller.add(ray, tip);

      // Squeeze = grip button; select = trigger on Quest and the only event
      // transient-pointer (Vision Pro pinch) input ever fires.
      const onGrabStart = () => this.startGrab(controller);
      const onGrabEnd = () => this.endGrab(controller);
      controller.addEventListener("squeezestart", onGrabStart);
      controller.addEventListener("squeezeend", onGrabEnd);
      controller.addEventListener("selectstart", onGrabStart);
      controller.addEventListener("selectend", onGrabEnd);

      this.scene.add(controller);
      this.controllers.push(controller);
      this.controllerCleanups.push(() => {
        controller.removeEventListener("squeezestart", onGrabStart);
        controller.removeEventListener("squeezeend", onGrabEnd);
        controller.removeEventListener("selectstart", onGrabStart);
        controller.removeEventListener("selectend", onGrabEnd);
        controller.remove(ray, tip);
        rayGeo.dispose();
        rayMat.dispose();
        tipGeo.dispose();
        tipMat.dispose();
        this.scene.remove(controller);
      });
    }
  }

  private teardownControllers(): void {
    if (this.grabbers.length > 0 && this.content.parent !== this.scene) {
      this.scene.attach(this.content);
    }
    this.grabbers = [];
    this.twoHand = null;
    for (const cleanup of this.controllerCleanups) cleanup();
    this.controllerCleanups = [];
    this.controllers = [];
  }

  private startGrab(controller: THREE.Group): void {
    if (this.grabbers.includes(controller)) return;
    this.grabbers.push(controller);
    if (this.grabbers.length === 1) {
      controller.attach(this.content);
    } else if (this.grabbers.length === 2) {
      this.twoHand = {
        initialDist: this.controllerDistance(),
        initialScale: this.content.scale.x,
      };
    }
  }

  private endGrab(controller: THREE.Group): void {
    const idx = this.grabbers.indexOf(controller);
    if (idx === -1) return;
    const wasHolder = idx === 0;
    this.grabbers.splice(idx, 1);
    this.twoHand = null;
    if (this.grabbers.length === 0) {
      this.scene.attach(this.content);
    } else if (wasHolder) {
      // Hand the content to the remaining grabbing controller.
      this.grabbers[0]!.attach(this.content);
    }
  }

  private controllerDistance(): number {
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    this.grabbers[0]!.getWorldPosition(a);
    this.grabbers[1]!.getWorldPosition(b);
    return a.distanceTo(b);
  }
}
