import mlflow
from mlflow.genai import make_judge

mlflow.set_tracking_uri('http://localhost:5000')

# Try with host.docker.internal to see if that works
judge = make_judge(
    name='mistral_judge_hdi',
    instructions='Score 1-5. Input: {{ inputs }}\nOutput: {{ outputs }}\n',
    model='ollama:/mistral',
    inference_params={
        'base_url': 'http://host.docker.internal:11434/v1',
    },
)

print(f"Created judge: {judge.name}, model: {judge.model}")

# Register to all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    if exp.experiment_id == '0':
        continue
    print(f"\nRegistering to {exp.experiment_id}: {exp.name}...")
    try:
        judge.register(experiment_id=exp.experiment_id)
        print(f"  ✓ Registered")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\nDone!")