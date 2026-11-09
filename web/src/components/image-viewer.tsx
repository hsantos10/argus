"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ImgHTMLAttributes,
  type ReactNode,
} from "react";
import { cn } from "@/lib/utils";

type ViewerState = {
  open: boolean;
  src: string | null;
  alt: string | null;
  caption?: string | null;
};

type ViewerContextValue = {
  state: ViewerState;
  openImage: (src: string, alt?: string, caption?: string) => void;
  closeImage: () => void;
};

const ImageViewerContext = createContext<ViewerContextValue | null>(null);

export function ImageViewerProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ViewerState>({
    open: false,
    src: null,
    alt: null,
    caption: null,
  });

  const openImage = useCallback((src: string, alt?: string, caption?: string) => {
    setState({ open: true, src, alt: alt ?? "", caption: caption ?? null });
  }, []);

  const closeImage = useCallback(() => {
    setState((s) => ({ ...s, open: false }));
  }, []);

  const value = useMemo<ViewerContextValue>(
    () => ({ state, openImage, closeImage }),
    [state, openImage, closeImage],
  );

  return (
    <ImageViewerContext.Provider value={value}>
      {children}
    </ImageViewerContext.Provider>
  );
}

export function useImageViewer() {
  const ctx = useContext(ImageViewerContext);
  if (!ctx) {
    // Graceful fallback: noop opener — avoids runtime crash if provider missing.
    return {
      open: () => {},
    };
  }
  return { open: ctx.openImage };
}

type ClickableImageProps = ImgHTMLAttributes<HTMLImageElement> & {
  caption?: string;
};

export function ClickableImage({ className, caption, onClick, ...props }: ClickableImageProps) {
  const ctx = useContext(ImageViewerContext);

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      {...props}
      alt={props.alt ?? ""}
      className={cn("cursor-zoom-in transition-opacity hover:opacity-95", className)}
      onClick={(e) => {
        onClick?.(e);
        if (!e.defaultPrevented && props.src && ctx) {
          ctx.openImage(String(props.src), props.alt ?? "", caption);
        }
      }}
    />
  );
}

export function ImageViewerDialog() {
  const ctx = useContext(ImageViewerContext);

  useEffect(() => {
    if (!ctx?.state.open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") ctx.closeImage();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [ctx]);

  if (!ctx || !ctx.state.open || !ctx.state.src) return null;

  const { src, alt, caption } = ctx.state;

  return (
    <div
      className="fixed inset-0 z-[1000] flex flex-col items-center justify-center bg-black/85 backdrop-blur-sm p-6"
      onClick={ctx.closeImage}
      role="dialog"
      aria-modal="true"
      aria-label={alt ?? "Image viewer"}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          ctx.closeImage();
        }}
        className="absolute right-5 top-5 rounded-full border border-white/20 bg-black/50 px-3 py-1 text-xs font-mono tracking-widest text-white hover:bg-black/70"
        aria-label="Close image viewer"
      >
        CLOSE · ESC
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt ?? ""}
        className="max-h-[88vh] max-w-[92vw] rounded-lg object-contain shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      />
      {(caption || alt) && (
        <p className="mt-4 max-w-3xl text-center text-xs leading-relaxed text-white/80">
          {caption || alt}
        </p>
      )}
    </div>
  );
}
