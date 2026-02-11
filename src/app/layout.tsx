import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BEREAN Protocol | Computational Antibody Discovery Platform",
  description: "Bioinformatic Exploration & Rational Engineering of Antibodies via Neural Networks - Built on IBM MAMMAL",
  icons: { icon: "/favicon.ico" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-slate-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
