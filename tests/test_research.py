"""
Tests for spaceshift research toolkit features.
These tests make real API calls — they are not mocked.
"""
import os
import sys
import shutil
import tempfile

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TEST_DIR, '../src'))

from spaceshift import (
    LLM, pairwise_evaluate,
    prompt_transform, list_transforms, language_transform,
    subprompt, superprompt, sideprompt, prompt_tree,
    research_tree, prompt_probe, compare_models, grid_search,
    to_md,
)


PROMPT = "What are the second-order effects of AI on labor markets?"


class TestPromptTransform:

    def test_list_transforms_returns_all(self):
        transforms = list_transforms(v=False)
        assert len(transforms) >= 20
        assert "inverse" in transforms
        assert "abstract_up" in transforms
        assert "translate_korean" in transforms

    def test_apply_single_transform(self):
        result = prompt_transform(PROMPT, "inverse")
        assert isinstance(result, str)
        assert len(result) > 10
        assert result != PROMPT

    def test_apply_abstract_up(self):
        result = prompt_transform(PROMPT, "abstract_up")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_apply_translation_transform(self):
        result = prompt_transform(PROMPT, "translate_french")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            prompt_transform(PROMPT, "nonexistent_transform_xyz")


class TestSubprompt:

    def test_basic_decomposition(self):
        results = subprompt(PROMPT, n=3, v=False)
        assert len(results) == 3
        for item in results:
            assert "prompt" in item
            assert "depth" in item
            assert "parent" in item
            assert isinstance(item["prompt"], str)
            assert len(item["prompt"]) > 10

    def test_multi_level(self):
        results = subprompt(PROMPT, n=[2, 2], v=False)
        assert len(results) == 4
        assert all(item["depth"] == 1 for item in results)


class TestSuperprompt:

    def test_basic_superprompt(self):
        results = superprompt(PROMPT, n=3, v=False)
        assert len(results) == 3
        for item in results:
            assert "prompt" in item
            assert isinstance(item["prompt"], str)
            assert len(item["prompt"]) > 10

    def test_multi_level(self):
        results = superprompt(PROMPT, n=[2, 2], v=False)
        assert len(results) == 4


class TestSideprompt:

    def test_basic_sideprompt(self):
        results = sideprompt(PROMPT, n=3, v=False)
        assert len(results) == 3
        for item in results:
            assert "prompt" in item
            assert isinstance(item["prompt"], str)
            assert len(item["prompt"]) > 10


class TestPromptTree:

    def test_all_three_directions(self):
        tree = prompt_tree(PROMPT, sub_n=[2], super_n=[2], side_n=[2], v=False, viz=False)
        assert tree["prompt"] == PROMPT
        assert "sub" in tree
        assert "super" in tree
        assert "side" in tree
        assert len(tree["sub"]) == 2
        assert len(tree["super"]) == 2
        assert len(tree["side"]) == 2

    def test_sub_only(self):
        tree = prompt_tree(PROMPT, sub_n=[3], v=False, viz=False)
        assert "sub" in tree
        assert "super" not in tree
        assert "side" not in tree
        assert len(tree["sub"]) == 3

    def test_viz_generates_graph(self):
        tree = prompt_tree(PROMPT, sub_n=[2], super_n=[2], v=False, viz=True)
        assert "graph" in tree
        assert tree["graph"] is not None

    def test_requires_at_least_one_direction(self):
        with pytest.raises(ValueError):
            prompt_tree(PROMPT, v=False)


class TestLanguageTransform:

    def test_korean(self):
        result = language_transform(PROMPT, language="korean", v=False)
        assert "original_prompt" in result
        assert "translated_prompt" in result
        assert "translated_response" in result
        assert "output_response" in result
        assert result["language"] == "korean"
        assert result["original_prompt"] == PROMPT
        assert len(result["output_response"]) > 50

    def test_save_to_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lang_test.md")
            result = language_transform(PROMPT, language="french", save=path, v=False)
            assert "saved" in result
            assert os.path.exists(result["saved"])


class TestCompareModels:

    def test_compare_two_models(self):
        result = compare_models(
            "Explain why the sky is blue",
            models=[1, 2],
            v=False,
        )
        assert "top_output" in result
        assert "top_model" in result
        assert "rankings" in result
        assert "scores" in result
        assert len(result["responses"]) == 2
        assert len(result["rankings"]) == 2

    def test_compare_without_evaluation(self):
        result = compare_models(
            "Explain gravity in one sentence",
            models=[1, 2],
            evaluate=False,
            v=False,
        )
        assert "responses" in result
        assert "top_output" not in result
        assert len(result["responses"]) == 2

    def test_save_to_md(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compare_test")
            result = compare_models(
                "What is dark matter?",
                models=[1, 2],
                save=path,
                v=False,
            )
            assert "saved" in result
            assert len(result["saved"]) == 2
            assert all(os.path.exists(p) for p in result["saved"])


class TestPromptProbe:

    def test_basic_probe(self):
        result = prompt_probe(
            "Explain why the sky is blue",
            n=3,
            v=False,
        )
        assert "top_output" in result
        assert "top_transform" in result
        assert "top_prompt" in result
        assert "rankings_transforms" in result
        assert len(result["responses"]) == 4  # 3 transforms + original
        assert "original" in result["transforms"]

    def test_specific_transforms(self):
        result = prompt_probe(
            "Explain gravity",
            transforms=["inverse", "abstract_up"],
            v=False,
        )
        assert len(result["responses"]) == 3  # 2 transforms + original
        assert "inverse" in result["transforms"]
        assert "abstract_up" in result["transforms"]

    def test_probe_without_evaluation(self):
        result = prompt_probe(
            "What is quantum computing?",
            n=2,
            evaluate=False,
            v=False,
        )
        assert "responses" in result
        assert "top_output" not in result


class TestGridSearch:

    def test_small_grid(self):
        result = grid_search(
            "Explain why the sky is blue",
            models=[1, 2],
            n_transforms=2,
            v=False,
        )
        assert "top_output" in result
        assert "top_model" in result
        assert "top_transform" in result
        # 2 transforms + original = 3 prompts x 2 models = 6 cells
        assert len(result["grid"]) == 6
        assert len(result["responses"]) == 6
        assert result["grid"][0]["rank"] == 1  # grid is sorted by rank

    def test_grid_without_evaluation(self):
        result = grid_search(
            "Explain gravity",
            models=[1, 2],
            n_transforms=2,
            evaluate=False,
            v=False,
        )
        assert "grid" in result
        assert "top_output" not in result


class TestPairwiseEvaluate:

    def test_evaluate_canned_responses(self):
        responses = [
            "The sky is blue because of Rayleigh scattering.",
            "Blue light has a shorter wavelength and scatters more in the atmosphere, which is why we see a blue sky.",
            "It just is blue.",
        ]
        result = pairwise_evaluate(
            prompt="Why is the sky blue?",
            results=responses,
            v=False,
        )
        assert len(result["rankings"]) == 3
        assert all(0 <= s <= 1 for s in result["scores"].values())
        assert sorted(result["rankings"]) == [0, 1, 2]

    def test_custom_metrics(self):
        responses = [
            "Quantum computing uses qubits.",
            "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
        ]
        result = pairwise_evaluate(
            results=responses,
            metrics=["How thorough is the explanation?", "How accessible to a beginner?"],
            v=False,
        )
        assert len(result["rankings"]) == 2
        assert len(result["raw"]["metrics_used"]) == 2

    def test_evaluate_last(self):
        llm = LLM(v=False, max_tokens=200).user("Write a haiku about recursion").chat()
        score = llm.evaluate_last(
            metrics={"How well does this follow 5-7-5 structure?": "1-10"},
            v=False,
        )
        assert 0 <= score["score"] <= 1
        assert "metric_scores" in score

    def test_single_item(self):
        result = pairwise_evaluate(results=["Only one response"], v=False)
        assert result["rankings"] == [0]
        assert result["scores"][0] == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pairwise_evaluate(results=[], v=False)


class TestResearchTree:

    def test_basic_research_tree(self):
        result = research_tree(
            PROMPT,
            sub_n=[2], super_n=[2], side_n=[2],
            save=False,
            v=False,
        )
        assert result["prompt"] == PROMPT
        assert "root_output" in result
        assert len(result["root_output"]) > 50
        assert "tree" in result
        assert "outputs" in result
        # root + 2 sub + 2 super + 2 side = 7
        assert len(result["outputs"]) == 7
        for entry in result["outputs"]:
            assert "prompt" in entry
            assert "response" in entry
            assert "direction" in entry
            assert len(entry["response"]) > 0

    def test_research_tree_saves(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "research_test")
            result = research_tree(
                "What is dark matter?",
                sub_n=[2], super_n=None, side_n=None,
                save=save_dir,
                v=False,
            )
            assert "saved" in result
            assert len(result["saved"]) > 0
            assert all(os.path.exists(p) for p in result["saved"])

    def test_reuse_existing_tree(self):
        tree = prompt_tree(PROMPT, sub_n=[2], v=False, viz=True)
        result = research_tree(tree, save=False, v=False)
        assert result["prompt"] == PROMPT
        assert len(result["outputs"]) == 3  # root + 2 sub


class TestSaveToMarkdown:

    def test_save_raw_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.md")
            result = to_md("Hello world", path)
            assert os.path.exists(result)
            with open(result) as f:
                assert "Hello world" in f.read()

    def test_save_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            result = to_md(["Response A", "Response B"], path)
            assert len(result) == 2
            assert all(os.path.exists(p) for p in result)

    def test_passthrough_without_path(self):
        assert to_md("just text") == "just text"
        assert to_md(["a", "b"]) == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
