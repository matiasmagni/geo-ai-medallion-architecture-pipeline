"""
Evaluate using the registered MLflow evaluation dataset.

Usage:
    ./venv/bin/python evaluate_from_dataset.py

This script loads the geo_ai_eval_dataset from MLflow and runs
LLM-as-a-Judge evaluation using the registered Mistral judge.
"""
import mlflow
from mlflow.genai import evaluate, get_dataset, get_scorer

mlflow.set_tracking_uri('http://localhost:5001')

def main():
    dataset_name = "geo_ai_eval_dataset"
    judge_name = "mistral_judge_hdi"
    experiment_id = "3"

    print(f"Loading dataset '{dataset_name}'...")
    dataset = get_dataset(dataset_name)
    print(f"  ID: {dataset.dataset_id}")
    print(f"  Records: {dataset.profile}")

    df = dataset.to_df()
    print(f"  Loaded {len(df)} test records")

    test_data = df.to_dict(orient="records")
    print(f"\nLoading judge '{judge_name}' from experiment {experiment_id}...")
    scorer = get_scorer(name=judge_name, experiment_id=experiment_id)
    print(f"  Judge: {scorer.name}")

    print(f"\nRunning evaluation on {len(test_data)} test cases...")
    result = evaluate(
        data=test_data,
        scorers=[scorer],
    )

    print(f"\n✨ Evaluation completed!")
    print(f"  Run ID: {result.run_id}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Tables: {list(result.tables.keys())}")

    tables = result.tables
    if "eval_results" in tables:
        scores_df = tables["eval_results"]
        score_col = f"{judge_name}/value"
        if score_col in scores_df.columns:
            avg_score = scores_df[score_col].mean()
            print(f"\n  Average score: {avg_score:.2f} / 5.0")
            print(f"  Scores:")
            for idx, row in scores_df.iterrows():
                score = row[score_col]
                inputs = row.get("request", row.get("inputs", {}))
                query = inputs.get("query", str(inputs))[:50]
                print(f"    {score}/5 - {query}...")

    print(f"\nView results at: http://localhost:5001/#/experiments/{experiment_id}/evaluation-runs?selectedRunUuid={result.run_id}")


if __name__ == "__main__":
    main()