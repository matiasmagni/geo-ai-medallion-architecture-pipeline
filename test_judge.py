import mlflow
from mlflow.genai import evaluate, get_scorer

mlflow.set_tracking_uri('http://localhost:5001')

test_data = [
    {"inputs": {"query": "What is 1+1?"}, "outputs": "The sum of 1 and 1 is 2."},
    {"inputs": {"query": "What is the capital of France?"}, "outputs": "The capital of France is Paris."},
    {"inputs": {"query": "What is 2+2?"}, "outputs": "2 plus 2 equals 4."},
]

# Use the judge registered in experiment 3
scorer = get_scorer(name='mistral_judge_hdi', experiment_id='3')
print(f"Using scorer: {scorer.name}")

result = evaluate(
    data=test_data,
    scorers=[scorer],
)

print("\nEvaluation completed!")
print("Run ID:", result.run_id)
print("\nMetrics:", result.metrics)

# Try to get tables if available
if hasattr(result, 'tables'):
    print("\nTables available:", list(result.tables.keys()) if isinstance(result.tables, dict) else "dict")