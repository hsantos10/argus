"use client";

/**
 * GPU ray-marching volume viewer for LLC4320 3D slab exploration.
 * Adapted from VolumeViewer.tsx (SlabScene) by TODO-ATTRIBUTION.
 * Simplified: VR in-scene histogram panels replaced with HTML overlay.
 * Fetches from the FastAPI backend on port 8000.
 */

import { useEffect, useRef, useState, useMemo, type Dispatch, type SetStateAction } from "react";
import * as THREE from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { VOLUME_VERT, VOLUME_FRAG } from "@/shaders/volume-shaders";
import { makeColormapTexture } from "@/lib/transfer-functions";
import { oceanFetchVolume, oceanGetMetadata, type VolumePayload, type OceanDatasetMetadata } from "@/lib/ocean-api";

// ─── Types ────────────────────────────────────────────────────────────────────

type VolumeKey = "theta" | "salt" | "w";

type VolumeClipState = {
  pan: [number, number, number];
  zoom: [number, number, number];
};

type ClipsState = { theta: VolumeClipState; salt: VolumeClipState; w: VolumeClipState };

const DEFAULT_CLIP: VolumeClipState = { pan: [0, 0, 0], zoom: [1, 1, 1] };

// ─── VolumeMesh ───────────────────────────────────────────────────────────────

type VolumeMeshProps = {
  nameKey: VolumeKey;
  colormap: string;
  steps: number;
  pan: [number, number, number];
  zoom: [number, number, number];
  gridWidth: number;
  gridHeight: number;
  gridDepth: number;
  thetaTex: THREE.Data3DTexture | null;
  saltTex: THREE.Data3DTexture | null;
  wTex: THREE.Data3DTexture | null;
  volumes: Record<VolumeKey, VolumePayload | null>;
  thresholds: Record<VolumeKey, [number, number]>;
  onClipChange: (pan: [number, number, number], zoom: [number, number, number]) => void;
};

function VolumeMesh({
  nameKey, steps, pan, zoom,
  gridWidth, gridHeight, gridDepth,
  thetaTex, saltTex, wTex, volumes, thresholds,
  onClipChange,
}: VolumeMeshProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const matRef = useRef<THREE.ShaderMaterial>(null);
  const outlineGeom = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), []);

  // Pick colormap from nameKey
  const colormapName = nameKey === "theta" ? "coolwarm" : nameKey === "salt" ? "viridis" : "puor";
  const transferTex = useMemo(() => makeColormapTexture(colormapName as any), [colormapName]);

  // Scale mesh to match grid aspect ratio
  useEffect(() => {
    if (!meshRef.current) return;
    const maxXY = Math.max(gridWidth, gridHeight);
    meshRef.current.scale.set(gridWidth / maxXY, gridHeight / maxXY, (gridDepth / 90.0) * 0.25);
  }, [gridWidth, gridHeight, gridDepth]);

  // Sync textures
  useEffect(() => {
    if (!matRef.current) return;
    const u = matRef.current.uniforms;
    const primaryTex = nameKey === "theta" ? thetaTex : nameKey === "salt" ? saltTex : wTex;
    u.uVolume.value = primaryTex;
    u.uVolumeTheta.value = thetaTex;
    u.uVolumeSalt.value = saltTex;
    u.uVolumeW.value = wTex;
    u.uHasTheta.value = !!thetaTex;
    u.uHasSalt.value = !!saltTex;
    u.uHasW.value = !!wTex;
  }, [thetaTex, saltTex, wTex, nameKey]);

  // Sync colormaps and ranges
  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uTransfer.value = transferTex;
  }, [transferTex]);

  // Sync ranges and thresholds
  useEffect(() => {
    if (!matRef.current) return;
    const u = matRef.current.uniforms;
    const getRange = (k: VolumeKey): [number, number] => {
      const vol = volumes[k];
      return vol ? [vol.min, vol.max] : [0, 1];
    };
    const [tMin, tMax] = getRange("theta");
    const [sMin, sMax] = getRange("salt");
    const [wMin, wMax] = getRange("w");
    u.uRangeTheta.value.set(tMin, tMax);
    u.uRangeSalt.value.set(sMin, sMax);
    u.uRangeW.value.set(wMin, wMax);
    const [rMin, rMax] = getRange(nameKey);
    u.uRange.value.set(rMin, rMax);

    const normThreshold = (k: VolumeKey): [number, number] => {
      const vol = volumes[k];
      if (!vol) return [0, 1];
      const [r0, r1] = getRange(k);
      const span = r1 - r0 || 1;
      const [t0, t1] = thresholds[k];
      const physMin = vol.min + t0 * (vol.max - vol.min);
      const physMax = vol.min + t1 * (vol.max - vol.min);
      return [
        Math.max(0, Math.min(1, (physMin - r0) / span)),
        Math.max(0, Math.min(1, (physMax - r0) / span)),
      ];
    };
    const [nt0, nt1] = normThreshold("theta");
    const [ns0, ns1] = normThreshold("salt");
    const [nw0, nw1] = normThreshold("w");
    u.uThresholdTheta.value.set(nt0, nt1);
    u.uThresholdSalt.value.set(ns0, ns1);
    u.uThresholdW.value.set(nw0, nw1);
    const [np0, np1] = normThreshold(nameKey);
    u.uThresholdRange.value.set(np0, np1);
  }, [volumes, thresholds, nameKey]);

  // Sync pan/zoom/steps
  useEffect(() => {
    if (!matRef.current) return;
    matRef.current.uniforms.uSteps.value = steps;
    matRef.current.uniforms.uPan.value.set(pan[0], pan[1]);
    matRef.current.uniforms.uZoom.value.set(zoom[0], zoom[1]);
  }, [steps, pan, zoom]);

  useEffect(() => () => { outlineGeom.dispose(); }, [outlineGeom]);

  const initialUniforms = useMemo(() => ({
    uVolume: { value: null as any },
    uTransfer: { value: null as any },
    uRange: { value: new THREE.Vector2(0, 1) },
    uSteps: { value: steps },
    uClipMin: { value: new THREE.Vector3(-0.5, -0.5, -0.5) },
    uClipMax: { value: new THREE.Vector3(0.5, 0.5, 0.5) },
    uPan: { value: new THREE.Vector2(0, 0) },
    uZoom: { value: new THREE.Vector2(1, 1) },
    uThresholdRange: { value: new THREE.Vector2(0, 1) },
    uVolumeTheta: { value: null as any },
    uVolumeSalt: { value: null as any },
    uVolumeW: { value: null as any },
    uHasTheta: { value: false },
    uHasSalt: { value: false },
    uHasW: { value: false },
    uRangeTheta: { value: new THREE.Vector2(0, 1) },
    uRangeSalt: { value: new THREE.Vector2(0, 1) },
    uRangeW: { value: new THREE.Vector2(0, 1) },
    uThresholdTheta: { value: new THREE.Vector2(0, 1) },
    uThresholdSalt: { value: new THREE.Vector2(0, 1) },
    uThresholdW: { value: new THREE.Vector2(0, 1) },
  }), []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <mesh ref={meshRef} name={nameKey}>
      <boxGeometry args={[1, 1, 1]} />
      <shaderMaterial
        ref={matRef}
        glslVersion={THREE.GLSL3}
        vertexShader={VOLUME_VERT}
        fragmentShader={VOLUME_FRAG}
        uniforms={initialUniforms}
        transparent
        side={THREE.BackSide}
        depthWrite={false}
      />
      <lineSegments geometry={outlineGeom}>
        <lineBasicMaterial attach="material" color={0x3a4252} />
      </lineSegments>
    </mesh>
  );
}

// ─── SlabScene ────────────────────────────────────────────────────────────────

function SlabScene({
  volumes,
  metas,
  clips,
  setClips,
  thresholds,
  zMax,
}: {
  volumes: Record<VolumeKey, VolumePayload | null>;
  metas: Record<VolumeKey, OceanDatasetMetadata | null>;
  clips: ClipsState;
  setClips: Dispatch<SetStateAction<ClipsState>>;
  thresholds: Record<VolumeKey, [number, number]>;
  zMax: number;
}) {
  const steps = 120;
  const groupRef = useRef<THREE.Group>(null);

  const thetaTexRef = useRef<THREE.Data3DTexture | null>(null);
  const saltTexRef  = useRef<THREE.Data3DTexture | null>(null);
  const wTexRef     = useRef<THREE.Data3DTexture | null>(null);

  const makeTex = (vol: VolumePayload | null, ref: React.MutableRefObject<THREE.Data3DTexture | null>) => {
    if (!vol) { ref.current?.dispose(); ref.current = null; return; }
    const [d, h, w] = vol.shape;
    const tex = new THREE.Data3DTexture(vol.data as unknown as Float32Array<ArrayBuffer>, w, h, d);
    tex.format = THREE.RedFormat;
    tex.type = THREE.FloatType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapR = tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;
    ref.current?.dispose();
    ref.current = tex;
  };

  useEffect(() => makeTex(volumes.theta, thetaTexRef), [volumes.theta]);
  useEffect(() => makeTex(volumes.salt,  saltTexRef),  [volumes.salt]);
  useEffect(() => makeTex(volumes.w,     wTexRef),     [volumes.w]);
  useEffect(() => () => { thetaTexRef.current?.dispose(); saltTexRef.current?.dispose(); wTexRef.current?.dispose(); }, []);

  const handleClip = (key: VolumeKey) => (p: [number, number, number], z: [number, number, number]) => {
    setClips((prev) => ({ ...prev, [key]: { pan: p, zoom: z } }));
  };

  const getMetaDims = (k: VolumeKey): [number, number] =>
    [metas[k]?.dims[0] ?? 17280, metas[k]?.dims[1] ?? 12960];

  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 10, 3]} intensity={1.5} />
      <gridHelper args={[30, 30, "#4f46e5", "#1e293b"]} position={[0, 0, 0]} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#020617" roughness={0.8} metalness={0.1} />
      </mesh>
      <group ref={groupRef} position={[0, 0.8, -3]}>
        {(["theta", "salt", "w"] as VolumeKey[]).map((k, i) => {
          const [gw, gh] = getMetaDims(k);
          const xOffset = (i - 1) * 1.05;
          const rotY = (i - 1) * -0.35;
          return metas[k] ? (
            <group key={k} position={[xOffset, 0, 0]} rotation={[0, rotY, 0]}>
              <VolumeMesh
                nameKey={k}
                colormap={k === "theta" ? "coolwarm" : k === "salt" ? "viridis" : "puor"}
                steps={steps}
                pan={clips[k].pan}
                zoom={clips[k].zoom}
                onClipChange={handleClip(k)}
                gridWidth={gw}
                gridHeight={gh}
                gridDepth={zMax}
                thetaTex={thetaTexRef.current}
                saltTex={saltTexRef.current}
                wTex={wTexRef.current}
                volumes={volumes}
                thresholds={thresholds}
              />
            </group>
          ) : null;
        })}
      </group>
      <OrbitControls enableDamping dampingFactor={0.08} />
    </>
  );
}

// ─── HTML controls overlay ────────────────────────────────────────────────────

const KEY_COLOR: Record<VolumeKey, string> = { theta: "#f97316", salt: "#06b6d4", w: "#a855f7" };
const KEY_LABEL: Record<VolumeKey, string> = { theta: "Theta (Temp)", salt: "Salt", w: "W (Vert. vel.)" };

function VolumeControls({
  quality, setQuality,
  thresholds, setThresholds,
  volumes, loading,
}: {
  quality: number;
  setQuality: (q: number) => void;
  thresholds: Record<VolumeKey, [number, number]>;
  setThresholds: Dispatch<SetStateAction<Record<VolumeKey, [number, number]>>>;
  volumes: Record<VolumeKey, VolumePayload | null>;
  loading: Record<VolumeKey, boolean>;
}) {
  const qBtn = (v: number) =>
    `px-2 py-0.5 text-[10px] font-mono border rounded transition-colors ` +
    (quality === v
      ? "border-[#00c4a8]/60 bg-[#00c4a8]/15 text-[#00c4a8]"
      : "border-white/[0.08] text-white/35 hover:text-white/60");

  return (
    <div className="absolute top-4 right-4 w-52 pointer-events-auto">
      <div className="rounded-xl border border-white/[0.08] bg-black/75 backdrop-blur-md p-3 space-y-3">
        <p className="font-mono text-[9px] uppercase tracking-widest text-white/35">3D Volume — LLC4320</p>

        {/* Quality */}
        <div>
          <p className="font-mono text-[9px] uppercase tracking-widest text-white/30 mb-1.5">Resolution</p>
          <div className="flex gap-1 flex-wrap">
            {[[-15, "Coarse"], [-12, "Low"], [-8, "Med"], [-4, "High"]].map(([v, l]) => (
              <button key={v} type="button" onClick={() => setQuality(v as number)} className={qBtn(v as number)}>{l}</button>
            ))}
          </div>
        </div>

        {/* Per-variable filter range + status */}
        {(["theta", "salt", "w"] as VolumeKey[]).map((k) => {
          const vol = volumes[k];
          const [t0, t1] = thresholds[k];
          return (
            <div key={k}>
              <div className="flex items-center gap-1.5 mb-1">
                <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ background: KEY_COLOR[k] }} />
                <span className="font-mono text-[10px] text-white/55">{KEY_LABEL[k]}</span>
                {loading[k] && <span className="ml-auto font-mono text-[8px] text-amber-400 animate-pulse">loading…</span>}
              </div>
              {vol ? (
                <>
                  <div className="flex justify-between font-mono text-[9px] text-white/30 mb-0.5">
                    <span>{(vol.min + t0 * (vol.max - vol.min)).toFixed(2)}</span>
                    <span>{(vol.min + t1 * (vol.max - vol.min)).toFixed(2)}</span>
                  </div>
                  <div className="flex gap-1">
                    <input type="range" min={0} max={1} step={0.01} value={t0}
                      onChange={(e) => setThresholds((p) => ({ ...p, [k]: [Number(e.target.value), Math.max(Number(e.target.value), p[k][1])] as [number, number] }))}
                      className="w-full accent-current" style={{ accentColor: KEY_COLOR[k] }} />
                    <input type="range" min={0} max={1} step={0.01} value={t1}
                      onChange={(e) => setThresholds((p) => ({ ...p, [k]: [Math.min(p[k][0], Number(e.target.value)), Number(e.target.value)] as [number, number] }))}
                      className="w-full accent-current" style={{ accentColor: KEY_COLOR[k] }} />
                  </div>
                </>
              ) : (
                <p className="font-mono text-[9px] text-white/20">{loading[k] ? "fetching…" : "unavailable"}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── VolumeViewer — public component ─────────────────────────────────────────

export function VolumeViewer() {
  const [quality, setQualityState] = useState(-15);
  const [volumes, setVolumes] = useState<Record<VolumeKey, VolumePayload | null>>({ theta: null, salt: null, w: null });
  const [metas, setMetas] = useState<Record<VolumeKey, OceanDatasetMetadata | null>>({ theta: null, salt: null, w: null });
  const [loading, setLoading] = useState<Record<VolumeKey, boolean>>({ theta: false, salt: false, w: false });
  const [clips, setClips] = useState<ClipsState>({ theta: DEFAULT_CLIP, salt: DEFAULT_CLIP, w: DEFAULT_CLIP });
  const [thresholds, setThresholds] = useState<Record<VolumeKey, [number, number]>>({
    theta: [0, 1], salt: [0, 1], w: [0, 1],
  });
  const [error, setError] = useState<string | null>(null);
  const qualityRef = useRef(quality);

  const setQuality = (q: number) => { qualityRef.current = q; setQualityState(q); };

  // Fetch metadata once
  useEffect(() => {
    const keys: VolumeKey[] = ["theta", "salt", "w"];
    Promise.all(keys.map((k) => oceanGetMetadata(k).catch(() => null)))
      .then((results) => {
        setMetas({ theta: results[0], salt: results[1], w: results[2] });
      })
      .catch((e) => setError(`Ocean backend offline: ${e.message}`));
  }, []);

  // Fetch volumes when quality changes
  useEffect(() => {
    const keys: VolumeKey[] = ["theta", "salt", "w"];
    setLoading({ theta: true, salt: true, w: true });
    setError(null);
    keys.forEach((k) => {
      oceanFetchVolume(k, { quality })
        .then((payload) => {
          setVolumes((prev) => ({ ...prev, [k]: payload }));
          setLoading((prev) => ({ ...prev, [k]: false }));
        })
        .catch((e) => {
          setLoading((prev) => ({ ...prev, [k]: false }));
          setError(`Failed to load ${k}: ${e.message}`);
        });
    });
  }, [quality]);

  const zMax = Math.max(
    ...["theta", "salt", "w"].map((k) => metas[k as VolumeKey]?.dims[2] ?? 90),
  );

  return (
    <div className="relative w-full h-full" style={{ background: "#020617" }}>
      {error && (
        <div className="absolute inset-x-0 top-0 z-20 bg-red-900/80 px-4 py-2 text-center font-mono text-[11px] text-red-200 backdrop-blur-sm">
          {error} — make sure the ocean backend is running on port 8000
        </div>
      )}

      <div className="absolute inset-0 z-10 pointer-events-none">
        <VolumeControls
          quality={quality}
          setQuality={setQuality}
          thresholds={thresholds}
          setThresholds={setThresholds}
          volumes={volumes}
          loading={loading}
        />
      </div>

      <Canvas
        gl={{ antialias: true }}
        camera={{ position: [0, 2, 4], fov: 50 }}
        style={{ width: "100%", height: "100%" }}
      >
        <color attach="background" args={["#020617"]} />
        <SlabScene
          volumes={volumes}
          metas={metas}
          clips={clips}
          setClips={setClips}
          thresholds={thresholds}
          zMax={zMax}
        />
      </Canvas>
    </div>
  );
}
