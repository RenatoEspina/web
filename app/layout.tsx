import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Bridge Chat",
  description: "Interfaz mínima para conversar con un modelo local mediante un gateway seguro.",
  other: {
    "codex-preview": "development",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body className="antialiased">{children}</body>
    </html>
  );
}
