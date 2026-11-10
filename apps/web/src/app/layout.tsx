import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SeatCraft — JoSAA Decision Support",
  description:
    "Research-oriented JoSAA choice-list decision support with rank-aware recommendations and explicit uncertainty.",
  keywords: ["JoSAA", "JEE", "seat allotment", "counselling", "prediction", "IIT", "NIT"],
  openGraph: {
    title: "SeatCraft — JoSAA Decision Support",
    description: "Rank-aware JoSAA recommendations with explicit uncertainty and limitations.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="bg-zinc-950 font-sans text-white antialiased">
        {children}
      </body>
    </html>
  );
}
