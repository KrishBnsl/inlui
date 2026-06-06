import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SeatCraft — JoSAA Seat Allotment Probability Engine",
  description:
    "Statistically rigorous JoSAA seat allotment predictions using Monte Carlo simulation over 8 years of historical cutoff data. Know your true odds before counselling day.",
  keywords: ["JoSAA", "JEE", "seat allotment", "counselling", "prediction", "IIT", "NIT"],
  openGraph: {
    title: "SeatCraft — JoSAA Probability Engine",
    description: "Monte Carlo powered JoSAA seat prediction engine.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} antialiased bg-zinc-950 text-white`}>
        {children}
      </body>
    </html>
  );
}
