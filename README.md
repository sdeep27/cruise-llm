# 🏎️ cruise-llm

Quickly build and reuse LLM workflows/agents with a clean, composable API — inspired by [scikit-learn](https://github.com/scikit-learn/scikit-learn)'s chainability and [litellm](https://github.com/BerriAI/litellm)'s model flexibility.

```python
from cruise_llm import LLM
LLM().user("Explain quantum computing").chat(stream=True)
```

---

## ⛓️ Multi-turn Prompt Queues

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

## 🔧 Easy Tool Calling for Fast Agent Building

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

## 🖼️ Image Support

Attach images to prompts - auto-switches to a vision-capable model if needed:

```python
# Single image
LLM().user("What's in this image?", image="photo.jpg").chat()

# Multiple images
LLM().user("Compare these", image=["before.png", "after.png"]).chat()

# URL
LLM().user("Describe this", image="https://example.com/image.jpg").chat()
```

---

## 🔄 Flexible conversations

Chat instances with swappable models and minimal verbosity:

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

## 🔀 Model Discovery & A/B Testing

Pick specific models or get up-to-date top-10 from category:

```python
LLM(model="gpt-5.2")
LLM(model="best")     # top intelligence rankings
LLM(model="fast")     # optimized for speed
LLM(model="cheap")    
LLM(model="open")     # open-source models
LLM(model="optimal")  # balanced best+fast (default)
LLM(model="codex")    

# Deterministic selection by rank
LLM(model="best0")    # top model in best category
LLM(model="fast2")    # 3rd fastest model

# Discover and filter what's available
LLM().get_models("claude")
LLM().models_with_vision()
LLM().models_with_search()
```

---



## 💰 Cost Tracking

Track token usage and costs across your session:

```python
llm = LLM(model="best")
llm.user("Explain quantum computing").chat()
llm.user("Summarize in one sentence").chat()

print(f"Last call: ${llm.last_cost():.6f}")
print(f"Session total: ${llm.total_cost():.6f}")
print(f"Breakdown: {llm.all_costs()}")
```

---

## 💾 Save, Load, Export

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

## 📦 Install

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
