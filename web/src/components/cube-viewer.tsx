"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type CubeViewerProps = {
  faces: (string | null)[]; // length 6, DYAMOND face indices 0..5
  altText?: string;
};

const AUTOROTATE_SPEED = (8 * Math.PI) / 180; // rad/s
const SPHERE_RADIUS = 0.82;
const SEG = 36; // subdivision segments per face edge — higher = smoother sphere

// ── Face mapping ─────────────────────────────────────────────────────────────
// Three.js BoxGeometry material slots: +X(0) -X(1) +Y(2) -Y(3) +Z(4) -Z(5)
//   +Y = top, -Y = bottom in Three.js (Y-up convention)
//
// DYAMOND GEOS face convention (from _equirect_from_cubed_faces in query_runner.py):
//   0 = +X  equatorial (lon ~0°)
//   1 = +Y  equatorial (lon ~+90°)
//   2 = -X  equatorial (lon ~180°)
//   3 = -Y  equatorial (lon ~-90°)
//   4 = +Z  North Pole
//   5 = -Z  South Pole
//
// Desired Three.js layout: North(4)→top(+Y,2)  South(5)→bottom(-Y,3)
//   dyamond[0] (+X equat.) → three.js slot 0 (+X)
//   dyamond[1] (+Y equat.) → three.js slot 4 (+Z front)
//   dyamond[2] (-X equat.) → three.js slot 1 (-X)
//   dyamond[3] (-Y equat.) → three.js slot 5 (-Z back)
//   dyamond[4] (N Pole)    → three.js slot 2 (+Y top)
//   dyamond[5] (S Pole)    → three.js slot 3 (-Y bottom)
const DYAMOND_TO_THREEJS: number[] = [0, 4, 1, 5, 2, 3];

// Labels to place at pole faces (world-space direction × 1.15, attached to the sphere mesh)
const POLE_LABELS = [
  { text: "N  face 4", dir: new THREE.Vector3(0, 1, 0), color: "#60a5fa" },
  { text: "S  face 5", dir: new THREE.Vector3(0, -1, 0), color: "#f472b6" },
];

function makeTextSprite(text: string, color: string): THREE.Sprite {
  const W = 320, H = 72;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  // Semi-transparent dark pill background
  ctx.fillStyle = "rgba(0,0,0,0.72)";
  const r = 14;
  ctx.beginPath();
  ctx.moveTo(r, 2);
  ctx.lineTo(W - r, 2);
  ctx.quadraticCurveTo(W - 2, 2, W - 2, r);
  ctx.lineTo(W - 2, H - r);
  ctx.quadraticCurveTo(W - 2, H - 2, W - r, H - 2);
  ctx.lineTo(r, H - 2);
  ctx.quadraticCurveTo(2, H - 2, 2, H - r);
  ctx.lineTo(2, r);
  ctx.quadraticCurveTo(2, 2, r, 2);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = color;
  ctx.font = "bold 32px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, W / 2, H / 2);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  // Scale so the label appears at reasonable size relative to SPHERE_RADIUS
  sprite.scale.set(0.75, 0.75 * (H / W), 1);
  return sprite;
}

export function CubeViewer({ faces, altText }: CubeViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const materialsRef = useRef<THREE.MeshBasicMaterial[] | null>(null);
  const texturesRef = useRef<(THREE.Texture | null)[]>([null, null, null, null, null, null]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width = container.clientWidth || 1;
    let height = container.clientHeight || 1;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(2.2, 1.6, 2.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    // ── 6 materials in Three.js BoxGeometry slot order ───────────────────────
    const materials: THREE.MeshBasicMaterial[] = Array.from({ length: 6 }, () =>
      new THREE.MeshBasicMaterial({ color: 0x1c1c2a, side: THREE.FrontSide }),
    );
    materialsRef.current = materials;

    // ── Cubed-sphere geometry ────────────────────────────────────────────────
    // Start from a subdivided unit cube, then normalize each vertex onto the sphere.
    // UV coordinates stay per-face (BoxGeometry assigns them that way), so each
    // face texture maps cleanly onto its curved patch.
    const geometry = new THREE.BoxGeometry(1, 1, 1, SEG, SEG, SEG);
    const pos = geometry.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).normalize().multiplyScalar(SPHERE_RADIUS);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    pos.needsUpdate = true;
    geometry.computeVertexNormals();

    const sphere = new THREE.Mesh(geometry, materials);
    scene.add(sphere);

    // ── Subtle lat/lon wireframe overlay ────────────────────────────────────
    const wireGeo = new THREE.SphereGeometry(SPHERE_RADIUS * 1.002, 24, 12);
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x4aa3ff,
      wireframe: true,
      transparent: true,
      opacity: 0.10,
    });
    sphere.add(new THREE.Mesh(wireGeo, wireMat));

    // ── Pole labels ──────────────────────────────────────────────────────────
    const spriteCleanup: (() => void)[] = [];
    for (const { text, dir, color } of POLE_LABELS) {
      const sprite = makeTextSprite(text, color);
      const d = SPHERE_RADIUS * 1.28;
      sprite.position.set(dir.x * d, dir.y * d, dir.z * d);
      sphere.add(sprite);
      spriteCleanup.push(() => {
        (sprite.material as THREE.SpriteMaterial).map?.dispose();
        (sprite.material as THREE.SpriteMaterial).dispose();
      });
    }

    // ── Orbit controls ───────────────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2.0;
    controls.maxDistance = 8;

    let autoRotate = true;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    const pauseAuto = () => {
      autoRotate = false;
      if (idleTimer) clearTimeout(idleTimer);
    };
    const resumeSoon = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { autoRotate = true; }, 2000);
    };
    renderer.domElement.addEventListener("pointerdown", pauseAuto);
    window.addEventListener("pointerup", resumeSoon);
    renderer.domElement.addEventListener("wheel", () => { pauseAuto(); resumeSoon(); }, { passive: true });

    // ── Resize ───────────────────────────────────────────────────────────────
    const resize = () => {
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      if (w === width && h === height) return;
      width = w; height = h;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    // ── Render loop ──────────────────────────────────────────────────────────
    let rafId = 0;
    let last = performance.now();
    const animate = () => {
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      if (autoRotate) {
        sphere.rotation.y += AUTOROTATE_SPEED * dt;
        sphere.rotation.x += AUTOROTATE_SPEED * dt * 0.3;
      }
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(animate);
    };
    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      if (idleTimer) clearTimeout(idleTimer);
      ro.disconnect();
      controls.dispose();
      geometry.dispose();
      wireGeo.dispose();
      wireMat.dispose();
      spriteCleanup.forEach((fn) => fn());
      materials.forEach((m) => m.dispose());
      texturesRef.current.forEach((t) => t?.dispose());
      texturesRef.current = [null, null, null, null, null, null];
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      materialsRef.current = null;
    };
  }, []);

  // ── Sync textures: apply DYAMOND → Three.js face remapping ─────────────────
  useEffect(() => {
    const mats = materialsRef.current;
    if (!mats) return;
    const loader = new THREE.TextureLoader();
    faces.forEach((url, dyamondIdx) => {
      if (dyamondIdx > 5) return;
      const slot = DYAMOND_TO_THREEJS[dyamondIdx];
      const mat = mats[slot];
      if (!mat) return;
      if (!url) {
        texturesRef.current[slot]?.dispose();
        texturesRef.current[slot] = null;
        mat.map = null;
        mat.color.set(0x1c1c2a);
        mat.needsUpdate = true;
        return;
      }
      loader.load(url, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 4;
        texturesRef.current[slot]?.dispose();
        texturesRef.current[slot] = tex;
        mat.map = tex;
        mat.color.set(0xffffff);
        mat.needsUpdate = true;
      });
    });
  }, [faces]);

  const loadedCount = faces.filter(Boolean).length;

  return (
    <div className="relative mx-auto aspect-square w-full max-w-[760px]">
      <div
        ref={containerRef}
        aria-label={altText ?? "Interactive cubed-sphere viewer"}
        className="h-full w-full rounded-lg border border-border/50 bg-gradient-to-b from-background to-muted/40"
      />
      {loadedCount < 6 && (
        <div className="pointer-events-none absolute bottom-2 right-2 rounded-md border border-border/50 bg-background/80 px-2 py-1 font-mono text-[10px] text-muted-foreground">
          {loadedCount}/6 faces loaded
        </div>
      )}
    </div>
  );
}
