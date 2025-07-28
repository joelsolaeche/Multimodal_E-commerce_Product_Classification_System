import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://multimodale-commerceproductclassificationsys-production-139f.up.railway.app',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'https://multimodale-commerceproductclassificationsys-production-139f.up.railway.app'}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
