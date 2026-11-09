"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { XRSessionManager } from "@/lib/xr-session";
import { useXRSupport } from "@/lib/use-xr-support";
import { VRButton } from "@/components/vr-button";

type GlobeViewerProps = {
  textureUrl: string | null;
  altText?: string;
  /** px from the container's right edge — pass panel width + gap when a floating side panel overlaps the viewer. */
  vrButtonRight?: number;
  /** px from the container's bottom edge — pass extra clearance when a playback bar overlaps the viewer's bottom. */
  vrButtonBottom?: number;
};

const AUTOROTATE_SPEED = 0.5; // OrbitControls autoRotateSpeed — ~120 s/orbit
const IDLE_RESUME_MS = 2500;

// Fresnel atmosphere — glows at limb, transparent at center
const ATM_VERT = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vNormal   = normalize(normalMatrix * normal);
    vViewDir  = normalize(-mvPos.xyz);
    gl_Position = projectionMatrix * mvPos;
  }
`;
const ATM_FRAG = /* glsl */ `
  varying vec3 vNormal;
  varying vec3 vViewDir;
  void main() {
    float rim = 1.0 - abs(dot(vNormal, vViewDir));
    float fresnel = pow(rim, 2.2);
    // cool blue near horizon, warmer blue towards peak
    vec3 col = mix(vec3(0.25, 0.55, 1.0), vec3(0.55, 0.82, 1.0), rim);
    gl_FragColor = vec4(col, fresnel * 0.22);
  }
`;

function buildStars(count = 2400): THREE.Points {
  const pos = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    // uniformly distributed on a large sphere shell
    const phi   = Math.acos(2 * Math.random() - 1);
    const theta = 2 * Math.PI * Math.random();
    const r     = 80 + Math.random() * 40;
    pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i * 3 + 2] = r * Math.cos(phi);
    sizes[i] = 0.6 + Math.random() * 1.6;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
  const mat = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.18,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.85,
  });
  return new THREE.Points(geo, mat);
}

export function GlobeViewer({ textureUrl, altText, vrButtonRight = 12, vrButtonBottom = 12 }: GlobeViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const textureRef   = useRef<THREE.Texture | null>(null);
  const sphereMatRef = useRef<THREE.MeshBasicMaterial | null>(null);
  const xrRef        = useRef<XRSessionManager | null>(null);
  const xrSupported  = useXRSupport();
  const [xrActive, setXrActive] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width  = container.clientWidth  || 1;
    let height = container.clientHeight || 1;

    // ── Renderer ────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.cssText = "width:100%;height:100%;display:block;";
    container.appendChild(renderer.domElement);

    // ── Scene / camera ───────────────────────────────────────────────────────
    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 500);
    // z must keep the radius-1 globe inside the 40° vertical FOV (needs > ~2.75)
    // or it overfills the viewport into a white wall.
    camera.position.set(0, 0, 2.85);

    // Grabbable root for VR — globe + atmosphere live here; stars stay on the
    // scene so they act as a fixed skybox when the globe is moved/scaled.
    const content = new THREE.Group();
    scene.add(content);

    // ── Globe sphere — flat material (no lighting/shadows) ───────────────────
    const geo = new THREE.SphereGeometry(1, 96, 64);
    const mat = new THREE.MeshBasicMaterial();
    sphereMatRef.current = mat;
    const sphere = new THREE.Mesh(geo, mat);
    content.add(sphere);

    // ── Fresnel atmosphere (custom shader, Back-side) ────────────────────────
    const atmoGeo = new THREE.SphereGeometry(1.055, 48, 32);
    const atmoMat = new THREE.ShaderMaterial({
      vertexShader:   ATM_VERT,
      fragmentShader: ATM_FRAG,
      transparent: true,
      blending:    THREE.AdditiveBlending,
      side:        THREE.BackSide,
      depthWrite:  false,
    });
    content.add(new THREE.Mesh(atmoGeo, atmoMat));

    // ── Stars ────────────────────────────────────────────────────────────────
    const stars = buildStars(2400);
    scene.add(stars);

    // ── Controls ─────────────────────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan    = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance  = 1.35;
    controls.maxDistance  = 8;
    controls.rotateSpeed  = 0.45;
    controls.zoomSpeed    = 0.65;
    controls.autoRotate   = true;
    controls.autoRotateSpeed = AUTOROTATE_SPEED;

    const xr = new XRSessionManager({
      renderer, scene, camera, controls, content,
      vrPosition: new THREE.Vector3(0, 1.35, -1.2),
      vrScale: 0.35, // radius-1 sphere → ~0.7 m globe at arm's length
      onSessionChange: setXrActive,
    });
    xrRef.current = xr;

    let idleTimer: ReturnType<typeof setTimeout> | null = null;

    const pauseAuto   = () => { controls.autoRotate = false; if (idleTimer) clearTimeout(idleTimer); };
    const schedResume = () => { if (idleTimer) clearTimeout(idleTimer); idleTimer = setTimeout(() => { controls.autoRotate = true; }, IDLE_RESUME_MS); };

    renderer.domElement.addEventListener("pointerdown", pauseAuto);
    window.addEventListener("pointerup", schedResume);
    renderer.domElement.addEventListener("wheel", () => { pauseAuto(); schedResume(); }, { passive: true });

    // ── Resize ───────────────────────────────────────────────────────────────
    const ro = new ResizeObserver(() => {
      if (renderer.xr.isPresenting) return; // XR framebuffer is session-sized
      const w = container.clientWidth  || 1;
      const h = container.clientHeight || 1;
      if (w === width && h === height) return;
      width = w; height = h;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    });
    ro.observe(container);

    // ── Render loop — setAnimationLoop is required for WebXR; three falls
    // back to requestAnimationFrame when no session is presenting ────────────
    const animate = () => {
      // OrbitControls must not touch the head-tracked camera (this also
      // pauses auto-rotate in VR; user rotates by grabbing instead).
      if (!renderer.xr.isPresenting) controls.update();
      xr.update();
      renderer.render(scene, camera);
    };
    renderer.setAnimationLoop(animate);

    return () => {
      xr.dispose();
      xrRef.current = null;
      renderer.setAnimationLoop(null);
      if (idleTimer) clearTimeout(idleTimer);
      ro.disconnect();
      renderer.domElement.removeEventListener("pointerdown", pauseAuto);
      window.removeEventListener("pointerup", schedResume);
      controls.dispose();
      geo.dispose(); mat.dispose();
      atmoGeo.dispose(); atmoMat.dispose();
      (stars.geometry as THREE.BufferGeometry).dispose();
      (stars.material as THREE.PointsMaterial).dispose();
      if (textureRef.current) { textureRef.current.dispose(); textureRef.current = null; }
      renderer.dispose();
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
      sphereMatRef.current = null;
    };
  }, []);

  // ── Swap texture when URL changes ─────────────────────────────────────────
  useEffect(() => {
    const m = sphereMatRef.current;
    if (!m) return;
    if (!textureUrl) {
      if (textureRef.current) { textureRef.current.dispose(); textureRef.current = null; }
      m.map = null; m.needsUpdate = true;
      return;
    }
    const loader = new THREE.TextureLoader();
    loader.load(
      textureUrl,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 8;
        if (textureRef.current) textureRef.current.dispose();
        textureRef.current = tex;
        m.map = tex; m.needsUpdate = true;
      },
      undefined,
      () => { /* ignore load errors */ },
    );
  }, [textureUrl]);

  return (
    // Dark "space" container — the globe floats in it
    <div
      ref={containerRef}
      aria-label={altText ?? "Interactive 3D globe"}
      className="relative w-full h-full min-h-[380px] rounded-xl overflow-hidden bg-[#030a18]"
      style={{
        // Soft blue-white glow around the whole canvas, simulating bloom
        boxShadow: textureUrl
          ? "0 0 60px 4px rgba(80,140,255,0.08)"
          : "none",
      }}
    >
      {!textureUrl && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-3 text-center">
          {/* Dim wireframe globe placeholder */}
          <svg viewBox="0 0 120 120" width="90" height="90" className="opacity-10">
            <circle cx="60" cy="60" r="54" fill="none" stroke="white" strokeWidth="1.2" />
            <ellipse cx="60" cy="60" rx="54" ry="22" fill="none" stroke="white" strokeWidth="0.8" />
            <ellipse cx="60" cy="60" rx="22" ry="54" fill="none" stroke="white" strokeWidth="0.8" />
            <line x1="6" y1="60" x2="114" y2="60" stroke="white" strokeWidth="0.8" />
            <line x1="60" y1="6" x2="60" y2="114" stroke="white" strokeWidth="0.8" />
          </svg>
          <p className="text-[11px] text-white/30 font-mono tracking-wider">
            Globe renders after query completes
          </p>
        </div>
      )}
      <VRButton
        supported={xrSupported}
        active={xrActive}
        onEnter={() => { void xrRef.current?.enter(); }}
        onExit={() => xrRef.current?.exit()}
        className="absolute z-10"
        style={{ right: vrButtonRight, bottom: vrButtonBottom }}
      />
    </div>
  );
}
