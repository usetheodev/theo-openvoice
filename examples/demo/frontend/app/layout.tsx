import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Theo OpenVoice Demo",
  description: "Dashboard interativo do scheduler Theo",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
