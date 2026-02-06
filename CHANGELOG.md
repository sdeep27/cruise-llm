# Changelog

**cruise-llm** is a lightweight Python library for working with LLMs. A simple, chainable interface for building AI workflows.

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-02-06

### Added
- **`.compact()`**: Summarize older messages to manage long conversations
  - Keeps last 10 messages, summarizes the rest into a structured summary appended to system prompt
  - Iterative: subsequent compactions merge into existing summary rather than regenerating
  - Optional `model=` override for the summarization LLM
- **`auto_compact`**: Automatic compaction when conversations get long
  - `LLM(auto_compact=30)` (default) — compacts when messages reach 30
  - `LLM(auto_compact=0)` to disable

### Changed
- Updated model rankings (2026-02-06)

---

## [0.6.0] - 2026-02-06

### Added
- **Audio support**: `.user(audio="file.mp3")` sends audio natively to models
  - Supports WAV, MP3, FLAC, OGG, M4A, AAC, OPUS, WebM formats
  - `prompt` is now optional in `.user()` — audio can be the entire input
  - Auto-switches to audio-capable model when current model doesn't support it
  - Multiple audio files: `.user("Compare", audio=["a.wav", "b.wav"])`
  - Combined with images: `.user("Describe", image="photo.jpg", audio="clip.wav")`
  - URL audio downloaded and base64-encoded automatically
- **`.transcribe()`**: Standalone transcription utility via Whisper
  - `LLM().transcribe("recording.wav")` — tries whisper-1, then groq/whisper variants
  - Supports single files or lists, local paths or URLs
- **`models_with_audio_input()`**: Discover audio-capable models
- **`evaluate()`**: Pairwise LLM output comparison and ranking
  - `evaluate(results, prompts, metrics)` for ranking multiple outputs
  - `LLM.evaluate_last()` for scoring single responses with absolute metrics
  - Auto-generated metrics when none provided
  - Position swap for bias mitigation (default on)
  - Bradley-Terry sampling for >5 items
- **`require_audio()` generator tool**: `generate()` can now flag audio capability

### Changed
- `get_models_for_category()` accepts `audio=True` filter
- Model auto-switch now normalizes reasoning_effort for cross-provider compatibility

---

## [0.5.0] - 2026-02-02

### Added
- **`generate()` method**: Create configured LLM instances from natural language descriptions
  - `LLM().generate("A DCF analyst that takes a stock ticker")` returns a ready-to-use LLM
  - Uses tool-calling internally to configure system prompt, inputs, reasoning, search, etc.
  - Generator LLM configuration (model, reasoning) influences quality of generated instance
- **Optional template variables**: `{var?}` syntax for optional inputs that default to empty string
  - `llm.user("Analyze {ticker} {context?}")` - context becomes "" if not provided
  - `get_template_vars(split=True)` returns `{'required': set(), 'optional': set()}`
- **Positional argument for `run()`**: When exactly one required variable, pass it directly
  - `dcf.run("TSLA")` instead of `dcf.run(ticker="TSLA")`
- **Simple numeric model selection**: Numbers 1-N zip optimal and best rankings
  - `LLM(model=1)` = top optimal, `LLM(model=2)` = top best, `LLM(model=3)` = second optimal, etc.

### Changed
- Model rank suffixes are now 1-indexed: `best1` (not `best0`) selects the top model
- Default model selection changed from `optimal` to `optimal1` (deterministic top optimal)

---

## [0.4.0] - 2026-01-27

### Added
- **LLM as function pattern**: New `run()` and `run_json()` methods with template interpolation
  - Define prompts with `{placeholders}`, then call `llm.run(var="value")`
  - `get_template_vars()` returns all placeholder names in the LLM
- **JSON enforcement**: Auto-fix malformed JSON using a fast LLM
  - `result_json()`, `run_json()`, and `last_json()` now have `enforce=True` by default
  - Falls back to LLM-based repair when `json.loads()` fails

### Changed
- Improved `_strip_markdown_json()` to handle edge cases with markdown code fences

---

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

