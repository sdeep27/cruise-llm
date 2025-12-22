# 🏎️ cruise-llm

**The fastest way to build, chain, and reuse LLM agents and flows.**

```python
from cruise_llm import LLM

LLM().user("Explain quantum computing").chat(stream=True)
```
---

## Chaining

LLM instances that are designed to have minimal verbosity and maximum flexability:

```python
rapper_llm = (LLM()
.sys("You are a rapper")
.user("Give me 2 bars about Python").chat()
.user("Now make it about Rust").chat()
)
```

Build a long conversation, then replay it with a different model in one line:

```python
chat1 = (
    LLM(model="fast")
    .sys("You are a bitcoin analyst")
    .user("What is proof of work?").chat()
    .user("Steel man the case for bitcoin mining").chat()
    .user("Now steel man the case against").chat()
)

# Replay history with a new config
chat2 = chat1.run_history(model="best", reasoning=True, reasoning_effort="high")

# Save chat histories and configurations
chat1.save_llm("chats/bitcoin_analysis_fast_model.json")
chat2.save_llm("chats/bitcoin_analysis_best_model.json")
```

## Reusable Pipelines

```python
sharpener = (
    LLM()
    .sys("You are an ad copywriter.")
    .add_followup("Make it punchier.")
)

sharpener.user("Tagline for toothpaste").res()  # automatically runs "Make it punchier" on the response
sharpener.user("Tagline for coffee").res()      # pipeline variable is reusable with no context bleed
```

## Tool Calling

Pass functions directly without writing schema

```python
def get_weather(city):
    """Get weather for a city."""
    return f"Weather in {city}: 72°F, sunny"

def get_time(timezone):
    """Get current time in a timezone."""
    return f"Current time in {timezone}: 3:00 PM"

(
    LLM()
    .tools(fns=[get_weather, get_time])
    .user("Time and weather in Tokyo?")
    .chat()
)
```

## Any Model

Pick specific model, or by randomization of the best from certain categories. 

```python
LLM(model="gpt-5-2")
LLM(model="best")   # top-tier reasoning
LLM(model="fast")   # lowest latency
LLM(model="cheap")  # budget-friendly
LLM(model="open")   # open-source models

# Discover what's available
LLM().get_models("claude")
```

## Search & Reasoning

Enable with a flag.

```python
# Web search
LLM().tools(search=True).user("Latest news on SpaceX").chat()

# Extended thinking
(
    LLM()
    .tools(reasoning=True, reasoning_effort="high")
    .user("Analyze this problem...")
    .chat()
)
```

## Save, Load, Export

Persist your agents. Export conversations.

```python
# Save an agent config
researcher = (
    LLM("claude-sonnet-4")
    .tools(search=True)
)
researcher.save_llm("agents/researcher.json")

# Load and use later
r = LLM.load_llm("agents/researcher.json")
r.user("What happened in tech today?").chat()

# Export conversation to markdown
llm.to_md("conversations/session.md")
```

---

## Install

```bash
pip install cruise-llm
```

**Critical step:** Create a `.env` file in your project root with at least one API key. Create a free key with any of these providers. Use litellm specific variable names for each model:

```env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
XAI_API_KEY=xai-...
```

---

*Enjoy cruising for yourself.*
