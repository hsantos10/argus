"use client";

import { useEffect, useState } from "react";

/**
 * True when the browser can start an immersive-vr WebXR session.
 * Resolves false on insecure contexts (plain-http LAN), so callers can
 * simply hide their "Enter VR" UI until the page is served over HTTPS.
 */
export function useXRSupport(): boolean {
  const [supported, setSupported] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const xr = typeof navigator !== "undefined" ? navigator.xr : undefined;
    if (!xr?.isSessionSupported) return;
    xr.isSessionSupported("immersive-vr")
      .then((ok) => {
        if (!cancelled) setSupported(ok);
      })
      .catch(() => {
        /* insecure context or permissions policy — leave unsupported */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return supported;
}
