import requests

resp = requests.post(
    "http://localhost:5000/gateway/routes",
    json={
        "name": "ollama-mistral",
        "model": {
            "name": "mistral",
            "provider": "ollama",
            "config": {"base_url": "http://ollama:11434/v1"}
        }
    },
    headers={"Content-Type": "application/json"},
    timeout=30
)

print("Status:", resp.status_code)
print("Response:", resp.text[:1000])