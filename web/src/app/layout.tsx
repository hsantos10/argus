import type { Metadata } from "next";
import Link from "next/link";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import { ImageViewerProvider, ImageViewerDialog } from "@/components/image-viewer";
import "katex/dist/katex.min.css";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  weight: ["400", "500"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ARGUS",
  description: "A live globe for petabyte-scale climate data, streamed into the browser and VR.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${spaceGrotesk.variable} ${ibmPlexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        {/* Hidden-tab capture shim: browsers suspend requestAnimationFrame in
            hidden tabs, freezing the WebGL globe in embedded/headless preview
            panes (and blanking canvas captures). If the tab starts hidden,
            drive rAF from timers instead; __afterFrame lets capture tooling
            hook the moment after a frame renders. No effect in visible tabs. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `if (document.visibilityState === "hidden") { window.requestAnimationFrame = function (cb) { return window.setTimeout(function () { cb(performance.now()); if (window.__afterFrame) { var f = window.__afterFrame; window.__afterFrame = null; f(); } }, 50); }; window.cancelAnimationFrame = window.clearTimeout.bind(window); }`,
          }}
        />
        <ThemeProvider>
          <ImageViewerProvider>
            <div className="sticky top-0 z-20 border-b border-border/40 bg-background/90 backdrop-blur-md">
              <div className="mx-auto flex h-12 w-full max-w-6xl items-center justify-between px-6">
                <div className="flex items-center gap-6">
                  <Link href="/" className="flex items-baseline gap-2">
                    <span className="font-mono text-[10px] font-bold tracking-widest text-primary">
                      ARGUS
                    </span>
                    <span className="text-xs font-semibold tracking-wide text-foreground/70">
                      SciVis 2026
                    </span>
                  </Link>
                  <nav className="flex items-center gap-4">
                    <Link href="/dashboard" className="text-xs font-semibold tracking-wide text-foreground/60 hover:text-foreground transition-colors">
                      Dashboard
                    </Link>
                  </nav>
                </div>
                <ThemeToggle />
              </div>
            </div>
            {children}
            <ImageViewerDialog />
          </ImageViewerProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
