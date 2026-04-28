import bpy
import math
import os
import pandas as pd
import random

# =============================================================================
# 1. SETUP & CONFIG
# =============================================================================
PARQUET_PATH = "./ml_predictions_gold.parquet"
REF_LAT, REF_LON = 40.7075, -74.0065 # FiDi Center
CLIP_DISTANCE = 100000.0

# VIVID COLOR PALETTE (Viewport & Render)
PALETTE = {
    "FIRE": (1.0, 0.0, 0.0, 1.0),      # RED
    "HOSPITAL": (1.0, 0.0, 1.0, 1.0),  # MAGENTA
    "AMBULANCE": (0.0, 0.5, 1.0, 1.0), # BLUE
    "RESCUE": (1.0, 0.5, 0.0, 1.0),    # ORANGE
    "TEXT": (1.0, 1.0, 1.0, 1.0)       # WHITE
}

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for m in bpy.data.meshes: bpy.data.meshes.remove(m)
    for m in bpy.data.materials: bpy.data.materials.remove(m)

def latlon_to_xyz(lat, lon, alt=0):
    R = 6371000 
    x = R * math.radians(lon - REF_LON) * math.cos(math.radians(REF_LAT))
    y = R * math.radians(lat - REF_LAT)
    return (x, y, alt)

# =============================================================================
# 2. THE SKY BILLBOARD (GIGANTIC LEGEND)
# =============================================================================
def create_sky_legend():
    """Creates a massive billboard at (0, 0, 500) that is impossible to miss."""
    print("Creating GIGANTIC Sky Legend...")
    
    legend_col = bpy.data.collections.new("LEGEND_LOOK_UP")
    bpy.context.scene.collection.children.link(legend_col)
    
    # Text items
    items = [
        ("--- GEOAI NYC HEATMAP LEGEND ---", PALETTE["TEXT"]),
        ("RED SPHERES: Fire Risk Predicted > 50%", PALETTE["FIRE"]),
        ("MAGENTA SPHERES: Hospital (Size = Bed Demand)", PALETTE["HOSPITAL"]),
        ("BLUE SPHERES: Ambulance Dispatch / Road", PALETTE["AMBULANCE"]),
        ("ORANGE SPHERES: Rescue / Emergency Infrastructure", PALETTE["RESCUE"])
    ]
    
    base_z = 500
    base_y = 0
    
    for i, (text, col) in enumerate(items):
        z_pos = base_z - (i * 40)
        
        # Create Text
        bpy.ops.object.text_add(location=(0, base_y, z_pos))
        txt = bpy.context.active_object
        txt.data.body = text
        txt.data.size = 20 # 20 Meters tall!
        txt.rotation_euler = (math.radians(90), 0, 0)
        
        # Material
        mat = bpy.data.materials.new(name=f"LegendMat_{i}")
        mat.use_nodes = True
        node_em = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        node_em.inputs['Color'].default_value = col
        node_em.inputs['Strength'].default_value = 20.0
        node_out = mat.node_tree.nodes.get("Material Output")
        mat.node_tree.links.new(node_em.outputs['Emission'], node_out.inputs['Surface'])
        
        # Viewport Color (Crucial for Solid Mode)
        mat.diffuse_color = col
        txt.data.materials.append(mat)
        
        # Link to collection
        for c in txt.users_collection: c.objects.unlink(txt)
        legend_col.objects.link(txt)

# =============================================================================
# 3. HEATMAP (SPHERES ONLY)
# =============================================================================
def generate_heatmap():
    if not os.path.exists(PARQUET_PATH): 
        print(f"WARNING: {PARQUET_PATH} not found. Skipping heatmap generation.")
        return
    df = pd.read_parquet(PARQUET_PATH)
    
    print(f"Generating heatmap for {len(df)} points...")
    
    for i, row in df.iterrows():
        loc = latlon_to_xyz(row['latitude'], row['longitude'])
        
        target_color = (1,1,1,1)
        radius = 8
        
        f_type = str(row.get('facility_type', '')).lower()
        
        if f_type in ['building', 'fire', 'fire_station']:
            # Fire risk markers
            risk = row.get('fire_risk_prob', row.get('severity', 0))
            # If it's a probability, check > 0.5. If it's severity (1-10), check > 5.
            if risk < 0.5 and risk > 1: # If it's 0-1 range
                if risk < 0.5: continue
            elif risk <= 5 and risk > 0: # If it's 1-10 range
                if risk <= 5: continue
                
            target_color = PALETTE["FIRE"]
            radius = 12 # Make fire risk more prominent
            
        elif f_type in ['hospital', 'clinic', 'medical']:
            target_color = PALETTE["HOSPITAL"]
            radius = row.get('bed_demand', 10) # Size driven by demand
            
        elif f_type in ['road', 'ambulance', 'traffic']:
            target_color = PALETTE["AMBULANCE"]
            radius = 6
            
        elif f_type in ['rescue', 'police', 'emergency', 'safety']:
            target_color = PALETTE["RESCUE"]
            radius = 10
        else:
            # Generic point
            target_color = (0.5, 0.5, 0.5, 1.0)
            radius = 4
        
        bpy.ops.mesh.primitive_ico_sphere_add(radius=radius, subdivisions=2, location=loc)
        sphere = bpy.context.active_object
        
        # Material
        mat = bpy.data.materials.new(name=f"DataMat_{i}")
        mat.use_nodes = True
        node_em = mat.node_tree.nodes.new(type='ShaderNodeEmission')
        node_em.inputs['Color'].default_value = target_color
        node_em.inputs['Strength'].default_value = 15.0
        node_out = mat.node_tree.nodes.get("Material Output")
        mat.node_tree.links.new(node_em.outputs['Emission'], node_out.inputs['Surface'])
        mat.diffuse_color = target_color
        sphere.data.materials.append(mat)

# =============================================================================
# 4. ENVIRONMENT & CAMERA
# =============================================================================
def setup_env():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Fix Clipping Everywhere
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.clip_end = CLIP_DISTANCE

    # Main Cinematic Camera
    bpy.ops.object.camera_add(location=(1500, -1500, 1000))
    cam = bpy.context.active_object
    cam.name = "Cinematic_Cam"
    cam.data.clip_end = CLIP_DISTANCE
    cam.rotation_euler = (math.radians(55), 0, math.radians(45))
    scene.camera = cam

    # Black World + Slight Fog
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    node_out = nodes.new(type='ShaderNodeOutputWorld')
    node_vol = nodes.new(type='ShaderNodeVolumePrincipled')
    node_vol.inputs['Density'].default_value = 0.01
    world.node_tree.links.new(node_vol.outputs['Volume'], node_out.inputs['Volume'])

def setup_blosm():
    """Dummy or Real BLOSM setup."""
    if hasattr(bpy.context.scene, "blosm"):
        print("Importing real NYC buildings...")
        b = bpy.context.scene.blosm
        b.minLat, b.maxLat = 40.7000, 40.7150
        b.minLon, b.maxLon = -74.0180, -73.9950
        b.dataType, b.mode = 'osm', '3Dsimple'
        b.buildings, b.highways, b.water = True, True, True
        try: bpy.ops.blosm.import_data()
        except: pass

# =============================================================================
# EXECUTE
# =============================================================================
if __name__ == "__main__":
    clean_scene()
    setup_env()
    setup_blosm()
    generate_heatmap()
    create_sky_legend()
    
    path = "/Users/matias.magni/Documents/dev/mine/geo-ai-medallion-architecture-pipeline/nyc_ml_heatmap_v4.blend"
    bpy.ops.wm.save_as_mainfile(filepath=path)
    print(f"\nSUCCESS! File saved to: {path}")
    print("INSTRUCTIONS: Look up! The legend is 500m above the origin.")
