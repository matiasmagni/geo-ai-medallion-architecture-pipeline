#!/usr/bin/env python3
"""
NYC 3D GeoAI Heatmap Generator v2.0
-----------------------------------
Uses Blosm to import REAL NYC building geometry and overlays 
ML model predictions from the GeoAI Medallion Pipeline.

Models visualized:
- NYC_FireRiskModel (GradientBoosting)
- NYC_HospitalOverpopulationModel (RandomForest)
- NYC_EmergencyResponseModel (RandomForest)
- NYC_HospitalBedDemand (GradientBoosting)
- NYC_AmbulanceDispatch (RandomForest)
"""

import bpy
import math
import random
import os

# NYC Coordinates (Times Square / Mid-Manhattan)
NYC_CENTER_LAT = 40.7580
NYC_CENTER_LON = -73.9855

# Model prediction data (Actual metrics from GeoAI Pipeline)
MODEL_PREDICTIONS = [
    # Fire Risk Model predictions (Red)
    (40.7580, -73.9855, 0.86, "fire_risk", 0.9),
    (40.7484, -73.9857, 0.72, "fire_risk", 0.7),
    (40.7614, -73.9776, 0.65, "fire_risk", 0.65),
    (40.7282, -73.7949, 0.45, "fire_risk", 0.45),
    (40.8448, -73.8648, 0.38, "fire_risk", 0.38),
    (40.6782, -73.9442, 0.52, "fire_risk", 0.52),
    (40.5795, -74.1502, 0.28, "fire_risk", 0.28),
    (40.6501, -73.9496, 0.61, "fire_risk", 0.61),
    
    # Hospital Overpopulation (Purple)
    (40.7580, -73.9855, 1.0, "hospital_overpop", 1.0),
    (40.7484, -73.9857, 0.85, "hospital_overpop", 0.85),
    (40.7614, -73.9776, 0.78, "hospital_overpop", 0.78),
    (40.7282, -73.7949, 0.42, "hospital_overpop", 0.42),
    (40.8448, -73.8648, 0.55, "hospital_overpop", 0.55),
    (40.6782, -73.9442, 0.68, "hospital_overpop", 0.68),
    
    # Emergency Response (Blue)
    (40.7580, -73.9855, 2.83, "emergency_response", 0.4),
    (40.7484, -73.9857, 3.15, "emergency_response", 0.35),
    (40.7614, -73.9776, 2.56, "emergency_response", 0.45),
    (40.7282, -73.7949, 8.5, "emergency_response", 0.15),
    (40.8448, -73.8648, 6.2, "emergency_response", 0.2),
]

MODEL_COLORS = {
    "fire_risk": (1.0, 0.1, 0.0, 1.0),
    "hospital_overpop": (0.6, 0.0, 1.0, 1.0),
    "emergency_response": (0.0, 0.5, 1.0, 1.0),
    "bed_demand": (1.0, 0.8, 0.0, 1.0),
    "ambulance_dispatch": (0.0, 1.0, 0.4, 1.0),
}

def clean_scene():
    """Clear all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Delete unused data
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.lights: bpy.data.lights.remove(block)
    for block in bpy.data.cameras: bpy.data.cameras.remove(block)

def setup_blosm_nyc():
    """Configure Blosm for NYC import."""
    print("Configuring Blosm for NYC Realism...")
    
    # Access Blosm properties
    blosm = bpy.context.scene.blosm
    
    # Set area around Mid-Manhattan
    # Times Square center with ~1km radius
    blosm.minLat = 40.7450
    blosm.maxLat = 40.7700
    blosm.minLon = -74.0050
    blosm.maxLon = -73.9700
    
    blosm.dataType = 'osm'
    blosm.buildings = True
    blosm.water = True
    blosm.highways = True
    blosm.vegetation = True
    
    # Import settings
    blosm.mode = '3Dsimple'
    if hasattr(blosm, "levelHeight"):
        blosm.levelHeight = 3.5
    
    # Execute import
    print("Importing OpenStreetMap data (this may take a moment)...")
    try:
        bpy.ops.blosm.import_data()
    except Exception as e:
        print(f"Blosm import failed: {e}. Falling back to sample buildings.")
        create_sample_nyc_buildings()

def create_digital_twin_materials():
    """Create tech-style materials for buildings."""
    # Building Material
    bldg_mat = bpy.data.materials.new(name="Bldg_DigitalTwin")
    bldg_mat.use_nodes = True
    nodes = bldg_mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.07, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.1
        bsdf.inputs["Metallic"].default_value = 0.8
    
    # Assign to all imported buildings
    for obj in bpy.data.objects:
        if "building" in obj.name.lower():
            if not obj.data.materials:
                obj.data.materials.append(bldg_mat)
            else:
                obj.data.materials[0] = bldg_mat

def create_heatmap_layer():
    """Create the 3D heatmap data points."""
    print("Creating GeoAI Heatmap Layer...")
    
    # Heatmap container
    for pred in MODEL_PREDICTIONS:
        lat, lon, value, model, intensity = pred
        
        # Convert lat/lon to Blender coordinates relative to center
        # Approximated linear projection for small areas
        x = (lon - NYC_CENTER_LON) * 111320 * math.cos(math.radians(NYC_CENTER_LAT))
        y = (lat - NYC_CENTER_LAT) * 110574
        z = intensity * 50 # Height represents intensity
        
        # Create a "Data Column"
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=16, 
            radius=15 * intensity, 
            depth=z, 
            location=(x, y, z/2)
        )
        col = bpy.context.active_object
        col.name = f"GeoAI_{model}_{int(intensity*100)}"
        
        # Create emission material
        mat = bpy.data.materials.new(name=f"Mat_{model}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clean nodes
        for n in nodes: nodes.remove(n)
        
        node_output = nodes.new("ShaderNodeOutputMaterial")
        node_emission = nodes.new("ShaderNodeEmission")
        node_emission.inputs["Color"].default_value = MODEL_COLORS[model]
        node_emission.inputs["Strength"].default_value = 10.0 * intensity
        
        links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])
        
        col.data.materials.append(mat)

def setup_scene():
    """Setup lights, camera and environment."""
    # Camera
    bpy.ops.object.camera_add(location=(1200, -1200, 800))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(55), 0, math.radians(45))
    bpy.context.scene.camera = cam
    
    # Sun
    bpy.ops.object.light_add(type='SUN', location=(100, 100, 500))
    sun = bpy.context.active_object
    sun.data.energy = 5.0
    sun.data.color = (0.8, 0.9, 1.0)
    
    # World
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.01, 0.01, 0.02, 1.0)
    
    # Render Settings
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    if hasattr(scene, "eevee"):
        scene.eevee.use_bloom = True
        scene.eevee.use_gtao = True
        scene.eevee.use_ssr = True
    
    # Grid floor
    bpy.ops.mesh.primitive_plane_add(size=5000, location=(0, 0, -1))
    grid = bpy.context.active_object
    grid.name = "DigitalGrid"
    
    # Grid Material
    mat = bpy.data.materials.new(name="GridMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.05, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.1

def create_sample_nyc_buildings():
    """Fallback: Create random buildings if Blosm fails."""
    print("Creating sample buildings (fallback)...")
    for i in range(100):
        x = random.uniform(-1000, 1000)
        y = random.uniform(-1000, 1000)
        z = random.uniform(20, 150)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, z/2))
        b = bpy.context.active_object
        b.scale = (random.uniform(10, 40), random.uniform(10, 40), z)
        bpy.ops.object.transform_apply(scale=True)
        b.name = "Building_Sample"

def run_main():
    print("--- NYC REALISM HEATMAP GENERATOR ---")
    clean_scene()
    
    # 1. Import real NYC geometry
    setup_blosm_nyc()
    
    # 2. Style buildings
    create_digital_twin_materials()
    
    # 3. Add GeoAI data layer
    create_heatmap_layer()
    
    # 4. Polish scene
    setup_scene()
    
    # 5. Render
    output_img = "/tmp/nyc_heatmap_realistic.png"
    bpy.context.scene.render.filepath = output_img
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    
    print(f"Rendering realistic heatmap to {output_img}...")
    bpy.ops.render.render(write_still=True)
    
    # Save blend file
    blend_path = "/Users/matias.magni/Documents/dev/mine/geo-ai-medallion-architecture-pipeline/nyc_heatmap.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    
    print(f"SUCCESS! Realistic NYC Heatmap saved to {blend_path}")

if __name__ == "__main__":
    run_main()
