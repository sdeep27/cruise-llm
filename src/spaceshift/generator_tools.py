class GeneratorToolkit:
    """Tools for configuring an LLM instance via tool calls."""

    def __init__(self, target_llm):
        self.target = target_llm
        self.is_done = False
        self.configuration_summary = ""
        self._vision_required = False
        self._audio_required = False

    def get_tools(self) -> list:
        """Return list of tool functions for the generator LLM."""
        return [
            self.set_specific_model,
            self.set_temperature,
            self.enable_reasoning,
            self.enable_search,
            self.require_vision,
            self.require_audio,
            self.set_system_prompt,
            self.set_input,
            self.done,
        ]

    def set_specific_model(self, litellm_model_str: str) -> str:
        """
        Set a specific model by name. Only use when the user explicitly requests a particular model.
        The model name will be fuzzy matched against available models.

        Args:
            litellm_model_str: The model name (e.g., "gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash")

        Returns:
            Confirmation message with the matched model
        """
        import rapidfuzz
        available = self.target.available_models
        match = rapidfuzz.process.extractOne(litellm_model_str, available, scorer=rapidfuzz.fuzz.WRatio)
        if match:
            self.target.model = match[0]
            return f"Model set to: {self.target.model}"
        return f"Could not find model matching: {litellm_model_str}"

    def set_temperature(self, temperature: float) -> str:
        """
        Set the sampling temperature for response generation.

        Args:
            temperature: Value between 0.0 and 2.0. Lower = more deterministic, higher = more creative.
                - 0.0-0.3: Factual, consistent responses
                - 0.4-0.7: Balanced creativity
                - 0.8-1.2: More creative/varied
                - 1.3-2.0: Highly random/experimental

        Returns:
            Confirmation message
        """
        self.target.temperature = max(0.0, min(2.0, temperature))
        return f"Temperature set to: {self.target.temperature}"


    def enable_reasoning(self, effort: str) -> str:
        """
        Enable extended thinking/reasoning capabilities.

        Args:
            effort: Reasoning effort level - "low", "medium", or "high".
                - low: Quick reasoning, suitable for simple logic
                - medium: Balanced reasoning depth (recommended)
                - high: Deep reasoning for complex problems

        Returns:
            Confirmation message
        """
        valid_efforts = ["low", "medium", "high"]
        if not effort or effort.lower() not in valid_efforts:
            effort = "medium"
        self.target.reasoning_enabled = True
        self.target.reasoning_effort = effort.lower()
        return f"Reasoning enabled with effort: {effort}"

    def enable_search(self, context_size: str) -> str:
        """
        Enable web search capabilities for real-time information.

        Args:
            context_size: How much search context to include - "short", "medium", or "long".
                - short: Brief excerpts, faster
                - medium: Balanced context (recommended)
                - long: More comprehensive search results

        Returns:
            Confirmation message
        """
        valid_sizes = ["short", "medium", "long"]
        if not context_size or context_size.lower() not in valid_sizes:
            context_size = "medium"
        self.target.search_enabled = True
        self.target.search_context_size = context_size.lower()
        return f"Search enabled with context size: {context_size}"

    def require_vision(self) -> str:
        """
        Flag that this LLM will need to process images.
        The system will ensure a vision-capable model is selected.

        Returns:
            Confirmation message
        """
        self._vision_required = True
        return "Vision capability flagged as required"

    def require_audio(self) -> str:
        """
        Flag that this LLM will need to process audio input.
        The system will ensure an audio-capable model is selected.

        Returns:
            Confirmation message
        """
        self._audio_required = True
        return "Audio input capability flagged as required"

    def set_system_prompt(self, prompt: str) -> str:
        """
        Set the main system prompt defining the LLM's role and behavior.

        Args:
            prompt: The system instructions. Should include:
                - Role/persona definition
                - Expertise areas
                - Response style guidelines
                - Any constraints or requirements

        Returns:
            Confirmation message
        """
        self.target.sys(prompt, append=False)
        return f"System prompt set ({len(prompt)} chars)"

    def set_input(self, template: str) -> str:
        """
        Define the input format with placeholder variables.

        Args:
            template: Input template with placeholders.
                - {var} = required variable
                - {var?} = optional variable (becomes empty string if not provided)
                Example: "Analyze {ticker} {context?}" -> .run(ticker="AAPL") or .run(ticker="AAPL", context="focus on growth")

        Returns:
            Confirmation message
        """
        self.target.user(template)
        vars_info = self.target.get_template_vars(split=True)
        return f"Input set - required: {vars_info['required']}, optional: {vars_info['optional']}"

    def done(self, summary: str) -> str:
        """
        Signal that configuration is complete.

        Args:
            summary: Brief summary of what was configured

        Returns:
            Completion message
        """
        self.is_done = True
        self.configuration_summary = summary or ""
        return "Configuration complete"


GENERATOR_SYSTEM_PROMPT = """You are an LLM configuration assistant. Your job is to configure an LLM instance based on user descriptions using the provided tools.

## Your Workflow

1. **Understand the task**: Parse what the user wants the LLM to do
2. **Set the model**: Only use set_specific_model if the user explicitly names a model (e.g., "use gpt-4o", "use claude sonnet"). Otherwise leave as default.
3. **Set the system prompt**: Write a detailed system prompt defining the role, expertise, and behavior. The system prompt MUST specify the expected JSON output keys/structure, since all LLMs return JSON via .run().
4. **Set the input**: Define input format with {variables} for dynamic inputs
5. **Configure capabilities**: Enable reasoning/search if needed, err on the side of them being needed
6. **Finalize**: Call done() with a brief summary

**Important**: All configured LLMs return JSON dicts via .run(). The system prompt should specify the expected output keys and structure (e.g., "Return JSON with keys 'sentiment' and 'confidence'").

**Enable reasoning with parameter "high" when:**
- Task would improve with logic or multi-step analysis
- Multiple factors to consider in user prompt

**Enable search when:**
- Research or fact-checking required
- Domain is rapidly changing (in time frames fresher than 1 year)
- Be generous with enabling search

## System Prompt Best Practices

Write system prompts that include:
- Perspective
- Specific expertise or focus areas if needed
- Expected JSON output keys/structure
- Be more declarative rather than imperative in your prompting - theres no need for steps.  We go on the side of open and minimal system prompts.

Avoid writing system prompts that:
- Too restrictive and constraining (models are very good now and dont need them)
- Too verbose

## Input Best Practices

- Use descriptive variable names: {ticker}, {topic}, {code}, {document}
- Required variables: {var} - user must provide
- Optional variables: {var?} - becomes empty string if not provided
- Usually just 1 required variable, optional vars for extra context
- Match variables to what the user will likely provide

## Example

For "A DCF analyst that takes a stock ticker and current date":
1. enable_reasoning("high")
2. set_system_prompt with detailed analyst role, DCF methodology, JSON output format (e.g., keys: valuation, assumptions, risks)
3. set_input("{ticker}")
4. set_input("{current_date}")
5. done("DCF analyst configured for stock valuation")

Always call done() when finished configuring."""
