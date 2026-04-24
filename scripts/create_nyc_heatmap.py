import bpy
import math
import os
import pandas as pd

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Paths to your Gold Layer Parquet files (Relative to CWD)
HAZARD_DATA_PATH = "./fact_hazard_events.parquet"
INFRA_DATA_PATH = "./dim_infrastructure.parquet"

# Lower Manhattan / Financial District Bounding Box
MANHATTAN_BOUNDS = {
    "min_lat": 40.7000,
    "max_lat": 40.7150,
    "min_lon": -74.0180,
    "max_lon": -73.9950
}

# Coordinate Projection Center (Reference point for XYZ 0,0,0)
REF_LAT = 40.7075
REF_LON = -74.0065

# Aesthetic Settings
BUILDING_COLOR = (0.02, 0.02, 0.03, 1.0)  # Dark Obsidian
HOSPITAL_COLOR = (0.0, 0.5, 1.0, 1.0)     # Cyber Blue
FOG_DENSITY = 0.02

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def setup_environment():
    """Clear scene and configure Cycles."""
    print("Initializing environment...")
    # Select and delete all
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Delete unused data blocks
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)

    # Set Render Engine to Cycles
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    # Try to enable GPU
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        # Use first available device type
        for dt in ['METAL', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI']:
            try:
                prefs.compute_device_type = dt
                print(f"Enabled GPU: {dt}")
                scene.cycles.device = 'GPU'
                break
            except:
                continue
    except:
        print("Falling back to CPU")
        scene.cycles.device = 'CPU'
    
    # 4K Resolution
    scene.render.resolution_x = 3840
    scene.render.resolution_y = 2160

def import_blosm_city():
    """Automates the BLOSM addon to fetch Fidi geometry."""
    print("Importing Lower Manhattan geometry via BLOSM...")
    
    if not hasattr(bpy.context.scene, "blosm"):
        print("CRITICAL ERROR: BLOSM addon not found or not enabled!")
        return

    blosm = bpy.context.scene.blosm
    
    blosm.minLat = MANHATTAN_BOUNDS["min_lat"]
    blosm.maxLat = MANHATTAN_BOUNDS["max_lat"]
    blosm.minLon = MANHATTAN_BOUNDS["min_lon"]
    blosm.maxLon = MANHATTAN_BOUNDS["max_lon"]
    
    blosm.dataType = 'osm'
    blosm.buildings = True
    blosm.water = True
    blosm.highways = True
    blosm.mode = '3Dsimple'
    
    # Set building material to dark glass/grey
    bldg_mat = create_architectural_material("Bldg_Dark", BUILDING_COLOR, 0.1)
    
    # Execute the import
    bpy.ops.blosm.import_data()
    
    # Assign material to all imported buildings
    for obj in bpy.data.objects:
        if "building" in obj.name.lower():
            if not obj.data.materials:
                obj.data.materials.append(bldg_mat)
            else:
                obj.data.materials[0] = bldg_mat

def latlon_to_xyz(lat, lon, alt=0):
    """Converts WGS84 coordinates to Blender meters relative to REF point."""
    # Approximate Earth radius in meters
    R = 6371000 
    
    x = R * math.radians(lon - REF_LON) * math.cos(math.radians(REF_LAT))
    y = R * math.radians(lat - REF_LAT)
    z = alt
    return (x, y, z)

def create_architectural_material(name, color, roughness):
    """Creates a realistic dark material for buildings."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = 0.9
    return mat

def create_emission_material(severity):
    """Procedural material that maps 1-10 severity to color/strength."""
    mat_name = f"Severity_{severity}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
        
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Remove default BSDF
    for n in nodes: nodes.remove(n)
    
    # Map Severity 1-10: Yellow (1) to Red (10)
    factor = (severity - 1) / 9.0
    color = (1.0, 1.0 - factor, 0.0, 1.0)
    
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_em = nodes.new(type='ShaderNodeEmission')
    
    node_em.inputs['Color'].default_value = color
    node_em.inputs['Strength'].default_value = severity * 15.0
    
    links.new(node_em.outputs['Emission'], node_out.inputs['Surface'])
    return mat

def generate_heatmap_points():
    """Iterates through parquet and spawns 3D markers."""
    print("Generating 3D Heatmap from Gold Layer...")
    
    if not os.path.exists(HAZARD_DATA_PATH):
        print(f"ERROR: {HAZARD_DATA_PATH} not found!")
        return

    df = pd.read_parquet(HAZARD_DATA_PATH)
    
    # Filter data to our bounding box
    df = df[
        (df['latitude'] >= MANHATTAN_BOUNDS["min_lat"]) & 
        (df['latitude'] <= MANHATTAN_BOUNDS["max_lat"]) &
        (df['longitude'] >= MANHATTAN_BOUNDS["min_lon"]) &
        (df['longitude'] <= MANHATTAN_BOUNDS["max_lon"])
    ]

    for index, row in df.iterrows():
        coords = latlon_to_xyz(row['latitude'], row['longitude'])
        
        # Instantiate an Icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=2.0, 
            subdivisions=2, 
            location=coords
        )
        point = bpy.context.active_object
        point.name = f"Hazard_{index}"
        
        # Assign procedural material
        point.data.materials.append(create_emission_material(row['final_severity']))

def generate_infrastructure_beacons():
    """Creates tall blue beacons for hospitals."""
    print("Generating infrastructure beacons...")
    if not os.path.exists(INFRA_DATA_PATH):
        print(f"ERROR: {INFRA_DATA_PATH} not found!")
        return

    df = pd.read_parquet(INFRA_DATA_PATH)
    
    # Hospital Beacon Material
    mat = create_emission_material(10)
    # Customize color to blue
    mat.node_tree.nodes["Emission"].inputs['Color'].default_value = HOSPITAL_COLOR
    
    for index, row in df.iterrows():
        if row['facility_type'] == 'hospital':
            coords = latlon_to_xyz(row['facility_lat'], row['facility_lon'], alt=100)
            
            bpy.ops.mesh.primitive_cylinder_add(
                radius=1.0, 
                depth=200, 
                location=coords
            )
            beacon = bpy.context.active_object
            beacon.name = f"Hospital_{index}"
            beacon.data.materials.append(mat)

def setup_cinematic_camera():
    """Positions camera for a dramatic overlook."""
    print("Setting up camera and volumetrics...")
    
    # Add Camera
    bpy.ops.object.camera_add(location=(1200, -1200, 800))
    cam = bpy.context.active_object
    cam.data.lens = 50
    cam.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = cam
    
    # World Volumetrics
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear world nodes
    for n in nodes: nodes.remove(n)
    
    node_out = nodes.new(type='ShaderNodeOutputWorld')
    node_vol = nodes.new(type='ShaderNodeVolumePrincipled')
    node_vol.inputs['Density'].default_value = FOG_DENSITY
    node_vol.inputs['Emission Strength'].default_value = 0.005
    
    links.new(node_vol.outputs['Volume'], node_out.inputs['Volume'])

def run_main():
    try:
        setup_environment()
        import_blosm_city()
        generate_heatmap_points()
        generate_infrastructure_beacons()
        setup_cinematic_camera()
        
        # Save blend file
        blend_path = "/Users/matias.magni/Documents/dev/mine/geo-ai-medallion-architecture-pipeline/nyc_heatmap_cinematic.blend"
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        
        print("\n" + "="*40)
        print(f"SCENE GENERATION COMPLETE: {blend_path}")
        print("Ready for 4K Cycles Render.")
        print("="*40)
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")

if __name__ == "__main__":
    run_main()
