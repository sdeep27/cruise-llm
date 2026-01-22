"""Test models from each category."""
import sys
sys.path.insert(0, '../src')

from cruise_llm import LLM

def test_category_selection():
    """Test that category selection works correctly."""
    print("=" * 60)
    print("Testing Category Selection")
    print("=" * 60)

    # Test default (should use optimal)
    llm = LLM(v=False)
    print(f"\nDefault (None): {llm.model}")

    # Test each category
    categories = ['best', 'fast', 'cheap', 'open', 'optimal', 'codex']
    for cat in categories:
        try:
            llm = LLM(model=cat, v=False)
            print(f"{cat}: {llm.model} (reasoning_effort: {llm.reasoning_effort if llm.reasoning_enabled else 'N/A'})")
        except Exception as e:
            print(f"{cat}: ERROR - {e}")


def test_deterministic_selection():
    """Test deterministic selection with numeric suffix."""
    print("\n" + "=" * 60)
    print("Testing Deterministic Selection (best0, best1, cheap0, etc.)")
    print("=" * 60)

    test_cases = ['best0', 'best1', 'best2', 'fast0', 'cheap0', 'optimal0', 'optimal1', 'codex0']
    for case in test_cases:
        try:
            llm = LLM(model=case, v=False)
            effort_str = f", reasoning_effort={llm.reasoning_effort}" if llm.reasoning_enabled else ""
            print(f"{case}: {llm.model}{effort_str}")
        except Exception as e:
            print(f"{case}: ERROR - {e}")


def test_model_inference():
    """Test actual inference with models from different categories."""
    print("\n" + "=" * 60)
    print("Testing Model Inference")
    print("=" * 60)

    test_prompt = "What is 2+2? Reply with just the number."

    # Test one model from each category
    test_models = ['best0', 'fast0', 'cheap0', 'optimal0']

    for model_spec in test_models:
        try:
            print(f"\n{model_spec}:")
            llm = LLM(model=model_spec, stream=False, v=False, max_tokens=50)
            result = llm.user(test_prompt).res()
            print(f"  Model: {llm.model}")
            print(f"  Response: {result.strip()[:100]}")
            print(f"  Cost: ${llm.last_cost(warn=False):.6f}")
        except Exception as e:
            print(f"  ERROR: {e}")


def test_get_models_for_category():
    """Test get_models_for_category returns correct format."""
    print("\n" + "=" * 60)
    print("Testing get_models_for_category")
    print("=" * 60)

    llm = LLM(v=False)
    for cat in ['best', 'fast', 'cheap', 'open', 'optimal', 'codex']:
        models = llm.get_models_for_category(cat)
        print(f"\n{cat} (top {len(models)}):")
        for i, m in enumerate(models[:5]):
            print(f"  {i}: {m}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")


if __name__ == "__main__":
    test_category_selection()
    test_deterministic_selection()
    test_get_models_for_category()

    # Only run inference tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        test_model_inference()
    else:
        print("\n(Run with --inference to test actual model calls)")
