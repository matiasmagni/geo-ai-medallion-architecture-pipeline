import mlflow
from mlflow.genai import make_judge

mlflow.set_tracking_uri('http://localhost:5001')

judge = make_judge(
    name='mistral_judge',
    instructions='You are a helpful assistant that evaluates the quality of AI-generated outputs. Score the output from 1-5 based on accuracy, relevance, and completeness. Input: {{ inputs }}\nOutput: {{ outputs }}\nProvide a score and brief rationale.',
    model='ollama:/mistral',
    inference_params={
        'base_url': 'http://host.docker.internal:11434/v1',
    },
)

print(f"Created judge: {judge.name}")

experiments = mlflow.search_experiments()
for exp in experiments:
    if exp.experiment_id == '0':
        continue
    print(f"Registering to {exp.experiment_id}: {exp.name}...")
    judge.register(experiment_id=exp.experiment_id)
    print(f"  ✓ Registered")

print("\nDone! Use from Mac with local Ollama at localhost:11434")
