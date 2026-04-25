"use client";

import dynamic from "next/dynamic";

const MapWithNoSSR = dynamic(() => import("@/components/MapClient"), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen w-screen items-center justify-center bg-zinc-900">
      <div className="text-xl text-white">Loading map...</div>
    </div>
  ),
});

export default function Home() {
  return (
    <div className="flex h-screen w-screen flex-col">
      <header className="bg-zinc-900 p-4">
        <h1 className="text-2xl font-bold text-white">NYC Healthcare/Fire ML Heatmap</h1>
        <p className="text-sm text-zinc-400">Fire Risk | Hospital Overpopulation | Emergency Response | Bed Demand | Ambulance Dispatch</p>
      </header>
      <MapWithNoSSR />
    </div>
  );
}