"use client";

import type { CSSProperties } from "react";
import { Glasses } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type VRButtonProps = {
  supported: boolean;
  active: boolean;
  onEnter: () => void;
  onExit: () => void;
  className?: string;
  style?: CSSProperties;
};

/**
 * "Enter VR" overlay button for the 3D viewers. Renders nothing when WebXR
 * immersive-vr is unavailable (non-XR browser or insecure context).
 */
export function VRButton({ supported, active, onEnter, onExit, className, style }: VRButtonProps) {
  // When WebXR immersive-vr isn't available (plain desktop browsers), show a
  // disabled "headset required" button so the feature is still discoverable,
  // rather than hiding it entirely.
  if (!supported) {
    return (
      <Button
        variant="outline"
        size="sm"
        disabled
        title="Open in the Meta Quest Browser or Apple Vision Pro Safari to enter VR"
        style={style}
        className={cn(
          "border-white/10 bg-black/30 font-mono text-[11px] uppercase tracking-widest text-white/40 backdrop-blur-sm",
          className,
        )}
      >
        <Glasses />
        VR · headset required
      </Button>
    );
  }
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={active ? onExit : onEnter}
      style={style}
      className={cn(
        "border-white/10 bg-black/40 font-mono text-[11px] uppercase tracking-widest text-white/80 backdrop-blur-sm hover:bg-black/60 hover:text-white",
        className,
      )}
    >
      <Glasses />
      {active ? "Exit VR" : "Enter VR"}
    </Button>
  );
}
