import json
from dotenv import load_dotenv
from cruise_llm import LLM
import datetime

load_dotenv()
available_models = LLM().get_models()
with open("litellm_rankings.json", 'r') as f:
    litellm_rankings = json.load(f)

available_slug_to_model = {model.split('/')[-1]: model for model in available_models}
model_rankings = {}
for key, rankings in litellm_rankings.items():
    model_rankings[key] = []
    for litellm_slug in rankings:
        if litellm_slug in available_slug_to_model:
            full_model_name = available_slug_to_model[litellm_slug]
            if 'codex' not in full_model_name:
                model_rankings[key].append(full_model_name)   
today_str = datetime.datetime.now().strftime('%Y-%m-%d')
filename = f"test_static_rankings_{today_str}.json"
with open(filename, 'w') as f:
    json.dump(model_rankings, f)
print(f"Saved rankings to {filename}")