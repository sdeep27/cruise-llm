# Changelog

**cruise-llm** is a lightweight Python library for working with LLMs. A simple, chainable interface for building AI workflows—without the complexity of heavier frameworks. One class, one import, all providers.

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-21

### Added
- **New model categories**: `optimal` (balanced best+fast) and `codex` (code-focused models)
- **Deterministic model selection**: Use `best0`, `fast1`, `cheap2` etc. to select exact rank
- **Auto reasoning effort**: Rankings now include `reasoning_effort` metadata, auto-applied when selecting category models
- **Vision support**: Auto-switches to vision-capable model when images are attached
- **`last_json()`**: Parse last response as JSON (for use after `chat_json()`)
- **`models_with_vision()`**: List models with vision support

### Changed
- Default model selection now uses `optimal` category instead of best/fast intersection
- `res_json()` now strips markdown code fences before parsing
- Improved open source model detection (llama, deepseek, qwen, mistral, kimi, etc.)

---

## [0.2.2] - 2026-01-21

### Added
- **Image support**: Attach images to prompts via `.user(prompt, image="path")` or with multiple images as a list
  - Supports local files (automatically converted to base64)
  - Supports URLs (passed directly to vision-capable models)
- **Cost tracking**: Track token usage and costs across completions
  - `last_cost()` - cost of most recent completion
  - `total_cost()` - sum of all completion costs in session
  - `all_costs()` - full array of cost objects with token breakdowns
  - Uses litellm's model_cost database (98% coverage)
  - Warns when search is enabled (search costs vary by provider and aren't captured)

---

## [0.2.1] - 2026-01-15

### Fixed
- Claude web search now works correctly (server-handled tool calls no longer trigger client execution)
- Groq search uses correct `browser_search` tool format
- Grok/Groq reasoning parameter compatibility

### Changed
- README caveat addition

---

## [0.2.0] - 2026-01-02

### Breaking Changes
- Renamed `.add_followup()` method to `.queue()` for clearer intent

### Added
- Comprehensive docstrings across all public methods and the `LLM` class
- Public method `get_models_for_category(category_str)` to retrieve ranked models for a given category
- Bigger test suite to tests/test_llm.py

### Changed
- Removed hard-coded `max_tokens` default, now respects model/provider defaults unless explicitly set

### Fixed
- Removed broken or unavailable models from the static rankings list

---

## [0.1.3] - 2025-12-19

### Added
- **Model categories**: Pass `"best"`, `"cheap"`, `"fast"`, or `"open"` as the `model` parameter to auto-select from top-ranked models in that category
  ```python
  LLM(model="fast")   # Selects a fast model
  LLM(model="cheap")  # Selects a budget-friendly model
  LLM(model="best")   # Selects a top-performing model
  LLM(model="open")   # Selects an open-source model
  ```
- Static model rankings bundled with the package (`rankings/static_rankings_2025-12-19.json`)
- Smarter default model selection: when no model is specified, selects from models that rank high in both "best" and "fast" categories

### Changed
- Moved ranking generation scripts to `scripts/` folder (not distributed with package)

## [0.1.2] - 2024

### Added
- Initial public release
- Core `LLM` class with chat, result, and JSON modes
- Tool/function calling support
- Web search integration
- Reasoning model support
- Multi-provider support via litellm (OpenAI, Anthropic, Gemini, etc.)
- Fuzzy model name matching with `rapidfuzz`
- Save/load LLM state to JSON
- Export conversations to Markdown

