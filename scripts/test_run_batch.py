"""Test run_batch across different batch sizes and providers."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spaceshift import LLM
from dotenv import load_dotenv
load_dotenv()

classifier = LLM().sys("Classify sentiment. Return JSON with key 'sentiment' (positive/negative/neutral).").user("{text}")

inputs_2 = [{"text": "I love this!"}, {"text": "This is terrible"}]
inputs_4 = inputs_2 + [{"text": "It's okay I guess"}, {"text": "Best day ever!"}]
inputs_8 = inputs_4 + [{"text": "Worst experience"}, {"text": "Pretty decent"}, {"text": "Absolutely amazing"}, {"text": "Not great, not terrible"}]

# Top model from each provider (best ranking)
models = ["claude-opus-4-6", "gpt-5.2-2025-12-11", "gemini/gemini-3-pro-preview"]

for model in models:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    for label, inputs in [("n=2", inputs_2), ("n=4", inputs_4), ("n=8", inputs_8)]:
        try:
            llm = LLM(model=model, v=True).sys(
                "Classify sentiment. Return JSON with key 'sentiment' (positive/negative/neutral)."
            ).user("{text}")

            results = llm.run_batch(inputs, return_errors=True)
            errors = [r for r in results if "error" in r]
            print(f"  {label}: {len(results)} results, {len(errors)} errors, cost=${llm.total_cost():.6f}")
            for i, r in enumerate(results):
                print(f"    [{i}] {r}")
        except Exception as e:
            print(f"  {label}: FAILED - {e}")

    llm2 = LLM(model=model).sys("Return JSON with key 'test': 'ok'").user("{x}")
    r = llm2.run_batch([{"x": "hi"}])
    print(f"  run_batch: {r}")

print("\nAll tests complete.")
