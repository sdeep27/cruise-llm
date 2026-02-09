# cruise-llm

Quickly build and reuse LLM workflows/agents with a clean, composable API — inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s chainability and [litellm](https://github.com/BerriAI/litellm)'s model flexibility.

```python
from cruise_llm import LLM
LLM().user("Explain quantum computing").chat(stream=True)
```

---

### LLM as Function

Define reusable LLM functions with `{placeholders}`, call with `.run()`. Each call is isolated — history resets automatically, so you can call the same template thousands of times without context bleeding.

```python
# Define with {placeholders}, call with .run()
sentiment = LLM().sys("Classify sentiment").user("Text: {text}")
sentiment.run("I love this product!")  # positional arg when 1 required var
sentiment.run(text="This is terrible")  # or use kwargs

# Optional variables with {var?} syntax
analyzer = LLM().user("Analyze {ticker} focusing on {aspect?}")
analyzer.run(ticker="TSLA")                        # aspect becomes ""
analyzer.run(ticker="TSLA", aspect="growth")       # aspect = "growth"

# JSON output
extractor = LLM().sys("Extract entities as JSON").user("{text}")
entities = extractor.run_json("Apple announced new MacBooks")
```

---

### Batch Processing

Run the same template across many inputs concurrently with `batch_run()`:

```python
classifier = LLM().sys("Classify sentiment as positive/negative").user("Text: {text}")
results = classifier.batch_run(
    [{"text": "Love it!"}, {"text": "Terrible experience"}, {"text": "It's okay"}],
    concurrency=10,
)
# results: ["positive", "negative", ...]

# JSON batch output
extractor = LLM().sys("Extract entities as JSON with key 'entities'").user("{text}")
entities = extractor.batch_run_json(
    [{"text": "Apple launched iPhone"}, {"text": "Google acquired DeepMind"}]
)
# entities: [{"entities": ["Apple", "iPhone"]}, {"entities": ["Google", "DeepMind"]}]

# Graceful error handling for production workloads
results = classifier.batch_run(inputs, return_errors=True)
# Failed calls return {"error": "...", "input": {...}} instead of raising
```

---

### Multi-turn Prompt Queues

Build complex micro-workflows by queuing prompts that the model will execute sequentially.

```python
# Automatic multi-step processing
news_processor = (
    LLM(model="fast")
    .user(f"Process this article: {raw_text}")
    .queue("Summarize the key points into 3 bullet points for an executive.")
    .queue("Translate those points into Spanish.")
    .queue("Format the Spanish summary as a Slack message with emojis.")
    .chat()
)

# Create reusable bot templates
def style_refiner(style):
    return LLM().sys(f"Rewrite in a {style} tone").queue("Make it half the length")

casual = style_refiner("casual")
formal = style_refiner("formal")

casual.user("We need to discuss Q3 deliverables").res()
formal.user("hey wanna grab coffee and chat about the project?").res()
```

---

### Easy Tool Calling for Fast Agent Building

Simply define functions, no schema necessary:

```python
def search_docs(query: str):
    """Search internal documentation."""
    return f"Found: '{query}' appears in onboarding.md and api-reference.md"

def create_ticket(title: str, priority: str):
    """Create a support ticket."""
    return f"Created ticket #{hash(title) % 1000}: {title} [{priority}]"

def send_slack(channel: str, message: str):
    """Send a Slack message."""
    return f"Sent to #{channel}: {message[:50]}..."

support_agent = (
    LLM()
    .sys("You are a support agent")
    .tools(fns=[search_docs, create_ticket, send_slack])
)

support_agent.user("User can't log in. Check docs, create a P1 ticket, and alert #incidents").chat()
```

---

### Image & Audio Support

Attach images and audio to prompts — auto-switches to a capable model if needed:

```python
# Images
LLM().user("What's in this image?", image="photo.jpg").chat()
LLM().user("Compare these", image=["before.png", "after.png"]).chat()

# Audio
LLM().user(audio="meeting.mp3").chat()                          # audio as the prompt
LLM().user("What language is this?", audio="clip.wav").chat()    # audio + text
LLM().user("Compare these", audio=["clip1.wav", "clip2.wav"]).chat()

# Combined
LLM().user("Describe the scene", image="photo.jpg", audio="narration.mp3").chat()

# URLs work for both
LLM().user("Describe", image="https://example.com/img.jpg").chat()
LLM().user("Summarize", audio="https://example.com/podcast.mp3").chat()

# Standalone transcription (uses Whisper)
text = LLM().transcribe("recording.wav")
```

---

### Evaluate & Compare Outputs

Rank multiple LLM outputs with pairwise comparison, or score a single response:

```python
from cruise_llm import evaluate

# Compare outputs from different models
outputs = [model.run(text=article) for model in models]
result = evaluate(results=outputs)
print(result["rankings"])  # [2, 0, 1] = third output was best
print(result["scores"])    # {0: 0.35, 1: 0.15, 2: 0.50}

# Custom metrics
result = evaluate(
    results=outputs,
    metrics={"How interesting is it?": "1-10", "How easy to understand?": "1-10"},
    weights={"How interesting is it?": 0.3, "How easy to understand?": 0.7}
)

# Score a single response
llm = LLM().user("Explain quantum computing").chat()
score = llm.evaluate_last(metrics={"How clear?": "1-10"})
print(score["score"])  # 0.82
```

---

### Flexible Conversations

`.chat()` accumulates history for multi-turn conversations. For batch/single-turn work, use `.run()` instead — it resets history automatically after each call.

```python
chat1 = (
    LLM(model="fast")
    .sys("You are a bitcoin analyst")
    .user("What is proof of work?").chat()
    .user("Steel man the case for bitcoin mining").chat()
    .user("Now steel man the case against").chat()
)

# Replay history with more intelligent yet expensive config
chat2 = chat1.run_history(model="best", reasoning=True, reasoning_effort="high")

# Save chat histories to analyze offline or load later
chat1.save_llm("chats/bitcoin_analysis_fast_model.json")
chat2.save_llm("chats/bitcoin_analysis_best_model.json")
```

---

### Context Compaction

Long conversations auto-compact to stay within context limits:

```python
llm = LLM(auto_compact=30)  # compacts at 30 messages (default)

# Or compact manually at any time
llm.compact()

# Disable auto-compact
LLM(auto_compact=0)
```

---

### Model Discovery & A/B Testing

Pick specific models or get up-to-date top-10 from category:

```python
LLM(model="gpt-5.2")
LLM(model="best")     # top intelligence rankings
LLM(model="fast")     # optimized for speed
LLM(model="cheap")
LLM(model="open")     # open-source models
LLM(model="optimal")  # balanced best+fast (default)
LLM(model="codex")

# Simple numeric selection (zips optimal and best)
LLM(model=1)          # top optimal (default)
LLM(model=2)          # top best
LLM(model=3)          # second optimal

# Deterministic selection by rank (1-indexed)
LLM(model="best1")    # top model in best category
LLM(model="fast3")    # 3rd fastest model

# Discover and filter what's available
LLM().get_models("claude")
LLM().models_with_vision()
LLM().models_with_audio_input()
LLM().models_with_search()
```

---

### Generate LLMs from Descriptions

Create configured LLM instances from natural language:

```python
# Generate a specialized LLM
summarizer = LLM().generate("Text summarizer that outputs 3 bullet points")
result = summarizer.run(text="Long article here...")

# Use a powerful model as the generator for better results
analyst = LLM(model="best", reasoning=True).generate(
    "A senior financial analyst for DCF valuations"
)
result = analyst.run(ticker="NVDA")

# Generated LLMs can be saved and reused
analyst.save_llm("agents/dcf_analyst.json")
```

---

### Cost Tracking

Track token usage and costs across your session:

```python
llm = LLM(model="best")
llm.user("Explain quantum computing").chat()
llm.user("Summarize in one sentence").chat()

print(f"Last call: ${llm.last_cost():.6f}")
print(f"Session total: ${llm.total_cost():.6f}")
print(f"Breakdown: {llm.all_costs()}")

# Full cost report with per-model breakdown
report = llm.cost_report()
# {"total_cost": 0.003, "num_calls": 2, "avg_cost": 0.0015, "by_model": {...}}
```

---

### Save, Load, Export

```python
# Save an agent config
researcher = LLM("claude-sonnet-4-5").tools(search=True)
researcher.save_llm("agents/researcher.json")

# Load
r = LLM.load_llm("agents/researcher.json")
r.user(f"What happened in tech {todays_date}?").chat()

# Export conversation to markdown
r.to_md(f"tech_briefing/{todays_date}.md")
```

---

### Install

```bash
pip install cruise-llm
```

Your access to models is based on your API keys from the various providers—keys are available for free from most providers. Create a local `.env` file in your project root with at least one API key. Use litellm-specific variable names:

```env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
```
*Caveat:* Search, reasoning, and model categories/rankings (best, cheap, fast, open, etc.) has only been tested with the above listed providers.  Calling other providers (perplexity, huggingface etc.) is still available with explicit litellm model strings but may require different search/reasoning setup.
