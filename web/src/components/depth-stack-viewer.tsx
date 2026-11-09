"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { XRSessionManager } from "@/lib/xr-session";
import { useXRSupport } from "@/lib/use-xr-support";
import { VRButton } from "@/components/vr-button";

export type DepthSlice = {
  idx: number;
  level: number;
  meters: number;
  dataUrl: string;
};

type Props = {
  slices: DepthSlice[];
  altText?: string;
  unitLabel?: string; // "m" for ocean depth, "km" for atmosphere altitude, etc.
  /** px from the container's right edge — pass panel width + gap when a floating side panel overlaps the viewer. */
  vrButtonRight?: number;
  /** px from the container's bottom edge — pass extra clearance when a playback bar overlaps the viewer's bottom. */
  vrButtonBottom?: number;
};

// Plane size: 2:1 aspect matches equirectangular textures
const PLANE_W = 4.2;
const PLANE_D = 2.1;
const STACK_H = 2.6;

function makeLabelSprite(text: string, color = "#cfd8ff"): THREE.Sprite {
  const W = 256, H = 56;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(8, 14, 30, 0.78)";
  const r = 10;
  ctx.beginPath();
  ctx.moveTo(r, 4);
  ctx.lineTo(W - r, 4);
  ctx.quadraticCurveTo(W - 4, 4, W - 4, r + 4);
  ctx.lineTo(W - 4, H - r - 4);
  ctx.quadraticCurveTo(W - 4, H - 4, W - r, H - 4);
  ctx.lineTo(r, H - 4);
  ctx.quadraticCurveTo(4, H - 4, 4, H - r - 4);
  ctx.lineTo(4, r + 4);
  ctx.quadraticCurveTo(4, 4, r, 4);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "rgba(140, 180, 255, 0.35)";
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.font = "600 22px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, W / 2, H / 2 + 2);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.85, 0.85 * (H / W), 1);
  return sprite;
}

export function DepthStackViewer({ slices, altText, unitLabel = "m", vrButtonRight = 12, vrButtonBottom = 12 }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const groupRef = useRef<THREE.Group | null>(null);
  const xrRef = useRef<XRSessionManager | null>(null);
  const xrSupported = useXRSupport();
  const [xrActive, setXrActive] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let width = container.clientWidth || 1;
    let height = container.clientHeight || 1;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height, false);
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.cssText = "width:100%;height:100%;display:block;";
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
    camera.position.set(4.8, 3.4, 5.3);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 1.0));

    // Grabbable root for VR — wireframe, floor, and slab stack move as a unit.
    const content = new THREE.Group();
    scene.add(content);

    // Wireframe box surrounding the stack
    const boxGeo = new THREE.BoxGeometry(PLANE_W * 1.04, STACK_H * 1.08, PLANE_D * 1.04);
    const edgeGeo = new THREE.EdgesGeometry(boxGeo);
    const edgeMat = new THREE.LineBasicMaterial({
      color: 0xbcd6ff,
      transparent: true,
      opacity: 0.28,
    });
    const wireframe = new THREE.LineSegments(edgeGeo, edgeMat);
    content.add(wireframe);

    // Subtle "ground" reflection
    const floorGeo = new THREE.PlaneGeometry(PLANE_W * 1.6, PLANE_D * 1.6);
    const floorMat = new THREE.MeshBasicMaterial({
      color: 0x081224,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -STACK_H * 0.7;
    content.add(floor);

    // Group that holds the plane slabs (rotates as a unit)
    const stack = new THREE.Group();
    groupRef.current = stack;
    content.add(stack);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.minDistance = 4.0;
    controls.maxDistance = 14;
    controls.maxPolarAngle = Math.PI * 0.55; // keep view above the horizon

    const xr = new XRSessionManager({
      renderer, scene, camera, controls, content,
      vrPosition: new THREE.Vector3(0, 1.2, -1.4),
      vrScale: 0.3, // 4.2-unit-wide stack → ~1.26 m
      onSessionChange: setXrActive,
    });
    xrRef.current = xr;

    let autoRotate = true;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    const pauseAuto = () => { autoRotate = false; if (idleTimer) clearTimeout(idleTimer); };
    const resumeSoon = () => {
      if (idleTimer) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { autoRotate = true; }, 2500);
    };
    renderer.domElement.addEventListener("pointerdown", pauseAuto);
    window.addEventListener("pointerup", resumeSoon);
    renderer.domElement.addEventListener("wheel", () => { pauseAuto(); resumeSoon(); }, { passive: true });

    const ro = new ResizeObserver(() => {
      if (renderer.xr.isPresenting) return; // XR framebuffer is session-sized
      const w = container.clientWidth || 1;
      const h = container.clientHeight || 1;
      if (w === width && h === height) return;
      width = w; height = h;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    });
    ro.observe(container);

    let last = performance.now();
    const ROT_SPEED = 0.13;
    const animate = () => {
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      // In VR the user rotates by grabbing; OrbitControls must not touch the
      // head-tracked camera.
      if (!renderer.xr.isPresenting) {
        if (autoRotate) {
          stack.rotation.y += ROT_SPEED * dt;
          wireframe.rotation.y += ROT_SPEED * dt;
        }
        controls.update();
      }
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
      window.removeEventListener("pointerup", resumeSoon);
      controls.dispose();
      boxGeo.dispose();
      edgeGeo.dispose();
      edgeMat.dispose();
      floorGeo.dispose();
      floorMat.dispose();
      stack.traverse((obj) => {
        if ((obj as THREE.Mesh).isMesh) {
          const m = obj as THREE.Mesh;
          m.geometry.dispose();
          const mat = m.material as THREE.Material & { map?: THREE.Texture | null };
          if (mat.map) mat.map.dispose();
          mat.dispose();
        }
        if ((obj as THREE.Sprite).isSprite) {
          const sp = obj as THREE.Sprite;
          (sp.material as THREE.SpriteMaterial).map?.dispose();
          (sp.material as THREE.SpriteMaterial).dispose();
        }
      });
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      groupRef.current = null;
    };
  }, []);

  // Rebuild slabs whenever the slices list changes
  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;

    // Clear previous children + dispose
    while (group.children.length > 0) {
      const obj = group.children[0]!;
      group.remove(obj);
      if ((obj as THREE.Mesh).isMesh) {
        const m = obj as THREE.Mesh;
        m.geometry.dispose();
        const mat = m.material as THREE.Material & { map?: THREE.Texture | null };
        if (mat.map) mat.map.dispose();
        mat.dispose();
      } else if ((obj as THREE.Sprite).isSprite) {
        const sp = obj as THREE.Sprite;
        (sp.material as THREE.SpriteMaterial).map?.dispose();
        (sp.material as THREE.SpriteMaterial).dispose();
      }
    }

    if (slices.length === 0) return;

    const sorted = [...slices].sort((a, b) => a.meters - b.meters);
    const maxM = Math.max(...sorted.map((s) => s.meters), 1);
    const loader = new THREE.TextureLoader();

    sorted.forEach((slice) => {
      const yNorm = slice.meters / maxM;
      const y = STACK_H / 2 - yNorm * STACK_H; // surface (≈0 m) → top

      const geo = new THREE.PlaneGeometry(PLANE_W, PLANE_D);
      const mat = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0.92,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      loader.load(slice.dataUrl, (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 6;
        mat.map = tex;
        mat.needsUpdate = true;
      });
      const plane = new THREE.Mesh(geo, mat);
      plane.rotation.x = -Math.PI / 2;
      plane.position.y = y;
      group.add(plane);

      // Depth label on the left side of each slab
      const depthText = slice.meters >= 1000
        ? `${(slice.meters / 1000).toFixed(1)} km`
        : `${slice.meters.toFixed(0)} ${unitLabel}`;
      const sprite = makeLabelSprite(depthText);
      sprite.position.set(-PLANE_W / 2 - 0.55, y, 0);
      group.add(sprite);
    });
  }, [slices, unitLabel]);

  return (
    <div
      ref={containerRef}
      aria-label={altText ?? "3D depth-stack viewer"}
      className="relative h-full w-full min-h-[380px] overflow-hidden rounded-xl bg-[#030a18]"
      style={{
        boxShadow: "0 0 70px 6px rgba(80,140,255,0.16), 0 0 140px 18px rgba(40,80,200,0.08)",
      }}
    >
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
