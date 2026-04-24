"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const DEFAULT_MARKER_URL = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png";
const DEFAULT_MARKER_2X_URL = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png";
const DEFAULT_SHADOW_URL = "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png";

delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: DEFAULT_MARKER_URL,
  iconRetinaUrl: DEFAULT_MARKER_2X_URL,
  shadowUrl: DEFAULT_SHADOW_URL,
});

interface Category {
  id: string;
  label: string;
  color: string;
  description: string;
}

const CATEGORIES: Category[] = [
  { id: "fire", label: "Fire", color: "#ff0040", description: "Fire emergency incidents" },
  { id: "medical", label: "Medical", color: "#0080ff", description: "Medical emergencies" },
  { id: "hazmat", label: "Hazmat", color: "#ff8800", description: "Hazardous materials incidents" },
  { id: "rescue", label: "Rescue", color: "#00ff80", description: "Rescue operations" },
];

function getColor(category?: string): string {
  const cat = CATEGORIES.find((c) => c.id === category);
  return cat?.color || "#808080";
}

function getLabel(category?: string): string {
  const cat = CATEGORIES.find((c) => c.id === category);
  return cat?.label || "Unknown";
}

interface MapLayersProps {
  data: any;
  activeCategories: string[];
}

function MapLayers({ data, activeCategories }: MapLayersProps) {
  const map = useMap();
  const layerRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (layerRef.current) {
      layerRef.current.clearLayers();
    }

    layerRef.current = L.layerGroup().addTo(map);

    if (!data?.features) return;

    data.features.forEach((feature: any) => {
      const { category, weight } = feature.properties || {};
      const coords = feature.geometry?.coordinates;
      
      if (!coords || !category) return;
      if (!activeCategories.includes(category)) return;

      const [lng, lat] = coords;
      const radius = Math.max(weight * 4, 6);
      const color = getColor(category);
      const label = getLabel(category);

      const circle = L.circleMarker([lat, lng], {
        radius,
        fillColor: color,
        fillOpacity: 0.75,
        color: "#fff",
        weight: 1.5,
        opacity: 0.9,
      });

      circle.bindPopup(`
        <div style="font-family: Arial, sans-serif; min-width: 150px;">
          <strong style="color: ${color}; font-size: 14px;">${label}</strong>
          <hr style="margin: 6px 0; border: none; border-top: 1px solid #ddd;" />
          <div style="font-size: 12px; color: #333;">
            <div><strong>Priority:</strong> ${weight > 0.7 ? "HIGH" : weight > 0.4 ? "MEDIUM" : "LOW"}</div>
            <div><strong>Score:</strong> ${(weight * 100).toFixed(0)}%</div>
            <div><strong>Location:</strong> ${lat.toFixed(4)}, ${lng.toFixed(4)}</div>
          </div>
        </div>
      `);

      layerRef.current?.addLayer(circle);
    });

    return () => {
      if (layerRef.current) {
        layerRef.current.clearLayers();
        map.removeLayer(layerRef.current);
      }
    };
  }, [map, data, activeCategories]);

  return null;
}

interface ToggleButtonProps {
  category: Category;
  isActive: boolean;
  onClick: () => void;
}

function ToggleButton({ category, isActive, onClick }: ToggleButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        padding: "8px 14px",
        margin: "4px 0",
        border: `2px solid ${isActive ? category.color : "#ddd"}`,
        borderRadius: "24px",
        background: isActive ? "white" : "#f5f5f5",
        cursor: "pointer",
        fontSize: "12px",
        fontWeight: 700,
        color: isActive ? "#111" : "#aaa",
        opacity: isActive ? 1 : 0.35,
        width: "100%",
        transition: "all 0.2s ease",
        boxShadow: isActive ? `0 3px 8px ${category.color}40` : "none",
        transform: isActive ? "scale(1.02)" : "scale(1)",
      }}
    >
      <span
        style={{
          width: "12px",
          height: "12px",
          borderRadius: "50%",
          background: category.color,
          marginRight: "10px",
          border: "1px solid #333",
          boxShadow: "0 0 4px " + category.color,
        }}
      />
      {category.label}
    </button>
  );
}

interface LegendProps {
  activeCategories: string[];
  onToggle: (id: string) => void;
}

function Legend({ activeCategories, onToggle }: LegendProps) {
  return (
    <div
      style={{
        position: "absolute",
        bottom: "30px",
        right: "15px",
        background: "white",
        padding: "14px",
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.35)",
        zIndex: 1000,
        minWidth: "110px",
      }}
    >
      <div
        style={{
          margin: "0 0 12px 0",
          fontSize: "14px",
          fontWeight: 800,
          color: "#111",
          borderBottom: "2px solid #eee",
          paddingBottom: "10px",
          textAlign: "center",
          letterSpacing: "0.5px",
        }}
      >
        NYC 911 CALLS
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {CATEGORIES.map((cat) => (
          <ToggleButton
            key={cat.id}
            category={cat}
            isActive={activeCategories.includes(cat.id)}
            onClick={() => onToggle(cat.id)}
          />
        ))}
      </div>
    </div>
  );
}

export default function MapClient() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCategories, setActiveCategories] = useState<string[]>(
    CATEGORIES.map((c) => c.id)
  );

  useEffect(() => {
    fetch("/data/heatmap.geojson")
      .then((res) => res.json())
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  const handleToggle = useCallback((id: string) => {
    console.log("Toggling:", id, "was active:", activeCategories.includes(id));
    setActiveCategories((prev) => {
      const next = prev.includes(id)
        ? prev.filter((c) => c !== id)
        : [...prev, id];
      console.log("Categories now:", next);
      return next;
    });
  }, [activeCategories]);

  if (loading) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", background: "#0f0f1a" }}>
        <div style={{ fontSize: "20px", color: "white", fontWeight: 600 }}>Loading NYC incidents...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh", background: "#0f0f1a" }}>
        <div style={{ fontSize: "18px", color: "#ff4444" }}>Error: {error}</div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", height: "100vh", width: "100vw" }}>
      <MapContainer
        center={[40.758, -73.985]}
        zoom={13}
        style={{ height: "100%", width: "100%" }}
        zoomControl={true}
      >
        <TileLayer
          attribution='© OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          maxZoom={19}
        />
        <MapLayers data={data} activeCategories={activeCategories} />
      </MapContainer>
      <Legend activeCategories={activeCategories} onToggle={handleToggle} />
    </div>
  );
}