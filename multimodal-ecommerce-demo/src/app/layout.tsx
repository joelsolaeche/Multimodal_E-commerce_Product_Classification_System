import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Multimodal E-commerce Classification System",
  description: "Advanced AI-powered product classification using computer vision and NLP",
  keywords: ["machine learning", "computer vision", "NLP", "multimodal", "e-commerce", "product classification"],
  authors: [{ name: "Your Name" }],
  openGraph: {
    title: "Multimodal E-commerce Classification System",
    description: "Advanced AI-powered product classification using computer vision and NLP",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${inter.className} bg-gray-50 text-gray-900`}>
        <div id="root">{children}</div>
      </body>
    </html>
  );
}
