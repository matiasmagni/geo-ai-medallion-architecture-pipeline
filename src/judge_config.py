"""
MLflow LLM-as-a-Judge configuration for Aviation_Flight_Delay_Prediction experiment.

This module creates a judge using Llama 3.2 (via Ollama) to evaluate GenAI traces.

Usage:
    from judge_config import create_judge
    judge = create_judge()
    
    # Use with mlflow.genai.evaluate()
    result = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[judge]
    )
"""

from mlflow.genai import make_judge


def create_judge():
    """
    Create a judge using Llama 3.2 via Ollama for evaluating GenAI traces.
    
    The judge evaluates responses on a 1-5 scale based on:
    - Accuracy
    - Completeness  
    - Relevance
    
    Returns:
        A Judge object that can be used with mlflow.genai.evaluate()
    """
    judge = make_judge(
        name="Llama3.2 Judge",
        instructions=(
            "Evaluate the quality of the AI response. Score 1-5 where:\n"
            "5 = Excellent - Response is accurate, complete, and highly relevant\n"
            "4 = Good - Response is mostly accurate and relevant with minor gaps\n"
            "3 = Acceptable - Response is adequate but has notable issues\n"
            "2 = Poor - Response has significant problems\n"
            "1 = Very Poor - Response is inaccurate, incomplete, or irrelevant\n\n"
            "Input: {{ inputs }}\n"
            "Output: {{ outputs }}\n"
            "Provide a score and brief rationale."
        ),
        model="ollama:/mistral",
        inference_params={
            "base_url": "http://ollama:11434/v1",
            "extra_headers": {},
        },
    )
    return judge


if __name__ == "__main__":
    judge = create_judge()
    print(f"Judge created: {judge.name}")
    print(f"Model: {judge.model}")
    print("Use with: mlflow.genai.evaluate(data=..., scorers=[judge])")
