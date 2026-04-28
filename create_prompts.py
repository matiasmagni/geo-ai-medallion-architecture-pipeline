import mlflow
from mlflow.genai import register_prompt

mlflow.set_tracking_uri('http://localhost:5001')

print("Creating prompts for LLM-as-a-Judge evaluation...\n")

prompts = [
    {
        "name": "geoai_judge_prompt",
        "template": """You are an expert judge evaluating AI-generated outputs for geospatial hazard analysis.

Evaluate the following input and output for accuracy, relevance, and completeness.

Input: {{inputs}}
Output: {{outputs}}

Score the output on a scale of 1-5:
- 5: Excellent - accurate, complete, relevant
- 4: Good - mostly accurate with minor gaps
- 3: Adequate - partially accurate but missing details
- 2: Poor - contains errors or missing key information
- 1: Failed - completely incorrect or irrelevant

Provide your response as a JSON object with:
{
  "score": <1-5>,
  "rationale": "<brief explanation>"
}

Respond ONLY with valid JSON.""",
        "tags": {
            "pipeline": "medallion",
            "judge_model": "mistral",
            "layer": "silver_quality_audit",
            "purpose": "llm_judge_evaluation",
        },
    },
    {
        "name": "geoai_extraction_prompt",
        "template": """You are an expert at extracting structured data from geospatial hazard reports.

Extract the following information from the input text:
- hazard_type: The type of hazard (fire, medical, weather, infrastructure, traffic, other)
- severity: The severity level (critical, high, medium, low)
- location: The location description if mentioned
- response_type: Recommended response (emergency, urgent, routine)

Input text: {{inputs}}

Respond ONLY with valid JSON in this exact format:
{
  "hazard_type": "<type>",
  "severity": "<severity>",
  "location": "<location or null>",
  "response_type": "<response>"
}""",
        "tags": {
            "pipeline": "medallion",
            "extraction_model": "llama3.2:1b",
            "layer": "silver_enrichment",
            "purpose": "structured_extraction",
        },
    },
    {
        "name": "geoai_hazard_classification_prompt",
        "template": """You are an expert at classifying hazard events for emergency management.

Classify the following hazard event description:

Event: {{inputs}}

Provide a classification with:
- category: Primary hazard category (fire, medical, weather, infrastructure, natural, human_caused)
- subcategory: More specific type
- risk_level: Risk assessment (critical, high, medium, low)
- recommended_actions: List of recommended response actions

Respond ONLY with valid JSON.""",
        "tags": {
            "pipeline": "medallion",
            "extraction_model": "llama3.2:1b",
            "layer": "silver_enrichment",
            "purpose": "hazard_classification",
        },
    },
    {
        "name": "geoai_flight_risk_prompt",
        "template": """You are an aviation expert evaluating flight risk conditions.

Analyze the following conditions and assess flight risk:

Conditions: {{inputs}}

Provide:
- risk_level: Overall risk (high, medium, low)
- factors: Key risk factors contributing to assessment
- recommendations: Operational recommendations

Respond ONLY with valid JSON.""",
        "tags": {
            "pipeline": "medallion",
            "extraction_model": "llama3.2:1b",
            "layer": "gold_aviation",
            "purpose": "flight_risk_assessment",
        },
    },
    {
        "name": "geoai_quality_check_prompt",
        "template": """You are a quality assurance specialist for geospatial AI data.

Verify if the following extraction is accurate and hallucination-free:

Input: {{inputs}}
Extracted Output: {{outputs}}

Check for:
1. Factual accuracy - does the output match the input?
2. Hallucinations - any invented information?
3. Completeness - all required fields present?
4. Format validity - is it valid JSON?

Respond with:
{
  "is_accurate": true/false,
  "has_hallucinations": true/false,
  "issues": ["list of issues or empty list"],
  "quality_score": <1-5>
}

Respond ONLY with valid JSON.""",
        "tags": {
            "pipeline": "medallion",
            "judge_model": "mistral",
            "layer": "silver_quality_audit",
            "purpose": "quality_verification",
        },
    },
]

for prompt_spec in prompts:
    name = prompt_spec["name"]
    template = prompt_spec["template"]
    tags = prompt_spec.pop("tags")
    
    print(f"Registering prompt: {name}")
    try:
        prompt = register_prompt(
            name=name,
            template=template,
            commit_message=f"Initial version of {name}",
            tags=tags,
        )
        print(f"  ✓ Registered: {name} (version {prompt.version})")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()

print("\nDone! View prompts at: http://localhost:5001/#/experiments/3/prompts")