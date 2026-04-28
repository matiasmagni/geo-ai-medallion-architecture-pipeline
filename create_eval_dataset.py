import mlflow
from mlflow.genai import create_dataset

mlflow.set_tracking_uri('http://localhost:5001')

dataset_name = "geo_ai_eval_dataset"

experiments = ["1", "2", "3", "4"]

test_data = [
    {"inputs": {"query": "What is 1+1?"}, "outputs": "The sum of 1 and 1 is 2."},
    {"inputs": {"query": "What is the capital of France?"}, "outputs": "The capital of France is Paris."},
    {"inputs": {"query": "What is 2+2?"}, "outputs": "2 plus 2 equals 4."},
    {"inputs": {"query": "What causes earthquakes?"}, "outputs": "Earthquakes are caused by the sudden release of energy in the Earth's crust."},
    {"inputs": {"query": "What is the severity of a magnitude 5 earthquake?"}, "outputs": "A magnitude 5 earthquake is considered moderate, causing light damage to buildings."},
    {"inputs": {"query": "Identify hazard types in NYC data"}, "outputs": "Hazard types include fire, medical, weather, and infrastructure emergencies."},
    {"inputs": {"query": "What is flight risk level high?"}, "outputs": "High flight risk indicates significant probability of flight delays or cancellations."},
    {"inputs": {"query": "Classify this emergency as low severity"}, "outputs": "This emergency is classified as low severity based on incident type and response time."},
    {"inputs": {"query": "Extract the hazard type from: Building fire on 5th Avenue"}, "outputs": '{"hazard_type": "fire", "location": "5th Avenue"}'},
    {"inputs": {"query": "Identify severity: Minor car accident with no injuries"}, "outputs": '{"severity": "low", "emergency_type": "traffic"}'},
    {"inputs": {"query": "What is the response time for a heart attack emergency?"}, "outputs": "The target response time for a heart attack emergency is under 5 minutes for life-threatening conditions."},
    {"inputs": {"query": "Classify hospital bed availability as high risk"}, "outputs": "Hospital bed availability is classified as high risk when occupancy exceeds 85% capacity."},
    {"inputs": {"query": "What weather conditions cause flight delays?"}, "outputs": "Weather conditions causing flight delays include thunderstorms, heavy snow, fog, and high winds."},
    {"inputs": {"query": "Identify infrastructure hazard: fallen power line"}, "outputs": '{"hazard_type": "infrastructure", "severity": "high", "response_needed": "utility_company"}'},
    {"inputs": {"query": "What is the NYC land mask boundary?"}, "outputs": "The NYC land mask boundary uses the NTA (Neighborhood Tabulation Areas) from NYC Open Data."},
]

print(f"Creating dataset '{dataset_name}' in experiments {experiments}...")
dataset = create_dataset(
    name=dataset_name,
    experiment_id=experiments,
    tags={
        "pipeline": "medallion",
        "judge_model": "mistral:7b",
        "extraction_model": "llama3.2:1b",
        "layer": "silver_to_gold",
        "domain": "geospatial_hazard_analysis",
    },
)

print(f"Dataset created!")
print(f"  Name: {dataset.name}")
print(f"  ID: {dataset.dataset_id}")
print(f"  Experiments: {dataset.experiment_ids}")

print(f"\nMerging {len(test_data)} test records...")
dataset.merge_records(test_data)
print(f"  ✓ Merged {len(test_data)} records")

print(f"\nDataset profile: {dataset.profile}")
print(f"\nView at: http://localhost:5001/#/experiments/3/datasets")
print(f"\nDone!")