import * as THREE from "three";

export type ColormapName = "viridis" | "coolwarm" | "puor" | "sst";

// CSS hex stops (cold → hot) for gradient legends
export const COLORMAP_CSS_STOPS: Record<ColormapName, string[]> = {
  viridis:  ["#440154", "#30678d", "#35b778", "#fde724"],
  coolwarm: ["#3b4cc0", "#88c5fd", "#f5f5f5", "#f08c5f", "#b40426"],
  puor:     ["#7f3b08", "#e08214", "#fee0b6", "#d8daeb", "#762a83"],
  // ice-blue → sky-blue → purple → coral → orange → cream/yellow
  sst:      ["#c8ebfc", "#55b9eb", "#804bb9", "#e46250", "#f4a337", "#ffe9a5"],
};

export function makeColormapTexture(name: ColormapName, size = 256): THREE.DataTexture {
  const data = new Uint8Array(size * 4);

  const stops: [number, number, number][] =
    name === "viridis"
      ? [[68,1,84],[70,50,127],[54,92,141],[39,127,142],[31,161,135],[74,194,109],[159,218,58],[253,231,37]]
      : name === "coolwarm"
      // blue (cold/low) → white (neutral) → red (hot/high)
      ? [[59,76,192],[99,149,231],[245,245,245],[233,107,72],[180,4,38]]
      : name === "sst"
      // ice-blue → sky-blue → purple → coral → orange → cream/yellow
      ? [[200,235,252],[85,185,235],[128,75,185],[228,98,80],[244,163,55],[255,233,165]]
      : [[127,59,8],[179,88,6],[224,130,20],[253,184,99],[254,224,182],[216,218,235],[178,171,210],[128,115,172],[84,39,136]];

  for (let i = 0; i < size; i++) {
    const t = i / (size - 1);
    const idx = t * (stops.length - 1);
    const lo = Math.floor(idx), hi = Math.min(stops.length - 1, lo + 1);
    const frac = idx - lo;
    const [r0, g0, b0] = stops[lo];
    const [r1, g1, b1] = stops[hi];
    data[i * 4 + 0] = Math.round(r0 + (r1 - r0) * frac);
    data[i * 4 + 1] = Math.round(g0 + (g1 - g0) * frac);
    data[i * 4 + 2] = Math.round(b0 + (b1 - b0) * frac);
    data[i * 4 + 3] = 255;
  }

  const tex = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}
