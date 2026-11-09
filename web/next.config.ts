import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // LAN/Tailscale subnets used to reach the HTTPS dev server from VR headsets
  // (Quest, Vision Pro) during WebXR testing.
  allowedDevOrigins: ["10.0.*.*", "192.168.*.*"],
  async rewrites() {
    return [
      {
        source: "/ocean-api/:path*",
        // Override with ARGUS_BACKEND_URL if the FastAPI backend runs elsewhere.
        destination: `${process.env.ARGUS_BACKEND_URL ?? "http://127.0.0.1:8000"}/api/:path*`,
      },
    ];
  },
  turbopack: {
    // Keep Turbopack rooted in web/ rather than walking up to the repo root.
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
