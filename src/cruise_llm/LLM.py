import litellm
from litellm import completion, responses
import json
import base64
import mimetypes
from dotenv import load_dotenv
from function_schema import get_function_schema
import logging
import rapidfuzz
import random
from pathlib import Path

_RANKINGS_PATH = Path(__file__).parent / "rankings" / "static_rankings_2026-01-21.json"
with open(_RANKINGS_PATH, "r") as f:
    _raw_rankings = json.load(f)

def _parse_rankings(raw):
    """Convert new dict format to model list, preserving reasoning_effort metadata."""
    parsed = {}
    for category, items in raw.items():
        if isinstance(items, list) and items and isinstance(items[0], dict):
            parsed[category] = items
        else:
            parsed[category] = [{"model": m} for m in items]
    return parsed

model_rankings = _parse_rankings(_raw_rankings)

def _get_model_name(entry):
    """Extract model name from ranking entry (dict or string)."""
    return entry["model"] if isinstance(entry, dict) else entry

def _get_reasoning_effort(entry):
    """Extract reasoning_effort from ranking entry if present."""
    if isinstance(entry, dict):
        return entry.get("reasoning_effort")
    return None

load_dotenv()
litellm.drop_params = True

class LLM:
    """
    A chainable, stateful wrapper around LiteLLM for building composable agents and workflows.

    This class manages a highly flexible conversation history with interchangeable models. 
    The API is meant for method chaining with a concise syntax, supporting quick prototyping. 
    It supports dynamic model aliasing (e.g., "best", "fast") with fuzzy model matching, 
    saving and loading preset workflows and prompt queues, 
    and easy tool integration without manual schema definition.
    """
    def __init__(self, model=None, temperature=None, stream=False, v=True, debug=False, max_tokens=None, search=False, reasoning=False, search_context_size="medium", reasoning_effort="medium",sub_closest_model=True):
        """
        Initialize the LLM client.

        Args (tends to match OpenAI/litellm completion API spec):
            model (str, optional): The model name (e.g., "gpt-4o", "claude-3-5-sonnet") or a category alias
                ("best", "fast", "cheap", "open", "optimal", "codex"). Supports deterministic selection
                with numeric suffix (e.g., "best0", "cheap1", "optimal2"). Defaults to None which uses "optimal".
            temperature (float, optional): Sampling temperature.
            stream (bool): If True, streams output to stdout.
            v (bool): Verbosity flag. If True, prints prompts, tool calls, and responses to stdout.
            debug (bool): If True, enables verbose LiteLLM logging.
            max_tokens (int, optional): Max tokens for generation.
            search (bool): If True, enables web search capabilities for all messages. Can instead be enabled on a per message basis, using .tools method
            reasoning (bool): If True, enables reasoning capabilities for all messages. Can instead be enabled on a per message basis, using .tools method
            search_context_size (str): Context size for search ("short", "medium", "long"). Defaults to "medium".
            reasoning_effort (str): Effort level for reasoning models ("low", "medium", "high").
            sub_closest_model (bool):  Defaults to True. Attempts to fuzzy match the model name if the exact name isn't found (e.g., "gpt4" -> "gpt-4").

        Raises:
            ValueError: If no API keys are found in the environment.
        """
        self.chat_msgs = []
        self.logs = []
        self.response_metadatas = []
        self.costs = []
        self.prompt_queue = []
        self.prompt_queue_remaining = 0
        self.last_chunk_metadata = None
        self.available_models = self.get_models()
        if not self.available_models:
            raise ValueError("Make sure you have at least one API key configured. Create a .env file in your project and add a variable line: (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)")
        self.sub_closest_model = sub_closest_model
        self.model = self._check_model(model)
        self.temperature = temperature
        self.stream = stream
        self.v = v #verbosity
        self.available_tools = None
        self.schemas = None
        self.tool_choice = "auto"
        self.parallel_tool_calls = True
        self.fn_map = None
        self.search_enabled = search
        self.search_context_size = search_context_size
        if not getattr(self, 'reasoning_enabled', None):
            self.reasoning_enabled = reasoning
        if not getattr(self, 'reasoning_effort', None):
            self.reasoning_effort = reasoning_effort
        if self.search_enabled:
            if not self._has_search(self.model):
                self._update_model_to_search()
        if self.reasoning_enabled:
            if not self._has_reasoning(self.model):
                self._update_model_to_reasoning()
        self.reasoning_contents = []
        self.search_annotations = []
        self.max_tokens = max_tokens
        if debug == True:
            self.turn_on_debug()
        else:
            self.turn_off_debug()
    
    def _resolve_args(self, **kwargs):
        """Merges chat, chat_json, res, res_json kwargs with instance init defaults."""
        instance_defaults   = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": kwargs.get("stream", self.stream),
            "v": kwargs.get("v", self.v),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        return {**instance_defaults, **kwargs}

    def _logger_fn(self, model_call_dict):
        self.logs.append(model_call_dict)

    def _process_image(self, image_source):
        if image_source.startswith(('http://', 'https://')):
            return {"type": "image_url", "image_url": {"url": image_source}}
        mime_type, _ = mimetypes.guess_type(image_source)
        mime_type = mime_type or "image/png"
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}

    def _strip_markdown_json(self, text):
        """Strip markdown code block wrapper from JSON if present."""
        if not isinstance(text, str):
            return text
        text = text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first line (```json or ```)
            if lines[-1].strip() == '```':
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = '\n'.join(lines).strip()
        return text

    def _enforce_json(self, text):
        """Use LLM to fix malformed JSON. Returns parsed dict or {} on failure."""
        enforcer = LLM(model="fast", stream=False, v=False) \
            .sys('You are a JSON enforcer. The user will provide text that should be valid JSON but may have issues. Return ONLY valid JSON that can be parsed by json.loads. Fix any syntax errors, missing brackets, or malformed structures. Output nothing except the corrected JSON.') \
            .user('{text}')
        result = enforcer.run_json(text=text, enforce=False)  # enforce=False to avoid recursion
        return result

    def _track_cost(self, response, model):
        usage = getattr(response, 'usage', None)
        if not usage:
            return
        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage, 'completion_tokens', 0) or 0
        try:
            total_cost = litellm.completion_cost(completion_response=response, model=model)
        except:
            total_cost = 0
        self.costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
        })

    def tools(self, fns = [], search=False, search_context_size=None, reasoning=False, reasoning_effort=None):
        """
        Register tools (functions) or enable capabilities like Web Search or Reasoning.
        Automatically generates JSON schemas from the provided Python functions.

        Args:
            fns (list): A list of Python callable functions. Type hints and docstrings 
                are recommended on the functions for accurate schema generation.
            search (bool): Enable web search. If the current model doesn't support it, 
                attempts to switch to a supported model.
            search_context_size (str, optional): "short", "medium", "long".
            reasoning (bool): Enable reasoning. If current model doesn't support it,
                attempts to switch to a supported model.
            reasoning_effort (str, optional): "low", "medium", "high".

        Returns:
            self: For chaining.
        """
        schemas = [get_function_schema(fn) for fn in fns]
        self.schemas = schemas
        self.fn_map = {schema['name']: fn for schema, fn in zip(schemas, fns)}
        tools = []
        for fn in fns:
            tool = {
                "type": "function",
                "function": get_function_schema(fn)
            }
            tool["function"]["strict"] = True
            tool["function"]["parameters"]["additionalProperties"] = False
            tools.append(tool)
        self.available_tools = tools
        if search:
            if not self._has_search(self.model):
                self._update_model_to_search()
            self.search_enabled = True
            self.temperature = None # openAI search model does not accept temperature
            if search_context_size:
                self.search_context_size = search_context_size
        else:
            self.search_enabled = False
            if search_context_size:
                self.search_context_size = None
        if reasoning:
            if not self._has_reasoning(self.model):
                self._update_model_to_reasoning()
            self.reasoning_enabled = True
            self.temperature = None # Anthropic doesnt want temperature when theres reasoning 
            if reasoning_effort:
                self.reasoning_effort = reasoning_effort
        else:
            self.reasoning_enabled = False
            if reasoning_effort:
                self.reasoning_effort = None
        return self

    def chat(self, **kwargs):
        """
        Run the LLM prediction based on current history, appending the response to the internal log.
        Generally follows a .user update, e.g. LLM().user("hi").chat()
        This is a stateful call; it updates `self.chat_msgs`.

        Args:
            **kwargs: Overrides for run-specific settings (temperature, model, etc.).

        Returns:
            self: For chaining (e.g., `.chat().user("Next question")`).
        """
        self._run_prediction(**kwargs)
        return self

    c = ch = chat

    def chat_json(self, **kwargs):
        """
        Same as `chat()`, but enforces JSON mode on the model response.
        
        Returns:
            self: For chaining.
        """
        self._run_prediction(jsn_mode=True, **kwargs)
        return self
    
    cjson = c_json = ch_json = chat_json

    def result(self, **kwargs):
        """
        Run the prediction, return the response text, and reset the chat history 
        (preserving the System prompt).

        Useful for single-turn tasks where you don't want history functionality 
        cluttering the context window.

        Args:
            **kwargs: Overrides for run-specific settings.

        Returns:
            str: The assistant's response content.
        """
        self._run_prediction(**kwargs)
        last_res = self.last()
        self._reset_msgs()
        return last_res
    
    r = res = result

    def result_json(self, enforce=True, **kwargs):
        """
        Run the prediction in JSON mode, parse the result, and reset chat history.
        (preserving the System prompt).

        Args:
            enforce (bool): If True (default), uses an LLM to fix malformed JSON on parse failure.
            **kwargs: Overrides for run-specific settings.

        Returns:
            dict: The parsed JSON response. Returns empty dict on parsing error.
        """
        self._run_prediction(jsn_mode=True, **kwargs)
        last_res = self._strip_markdown_json(self.last())
        self._reset_msgs()
        try:
            return json.loads(last_res)
        except (json.JSONDecodeError, TypeError):
            if enforce:
                return self._enforce_json(last_res)
            print(f"!! Error parsing JSON: {last_res}")
            return {}

    rjson = res_json = result_json

    def _interpolate_templates(self, **kwargs):
        """
        Interpolate {placeholder} template variables in all chat messages.
        
        Args:
            **kwargs: Template variables to interpolate (e.g., ticker="TSLA")
        
        Returns:
            self: For chaining.
        """
        import re
        for msg in self.chat_msgs:
            content = msg.get('content', '')
            if isinstance(content, str):
                msg['content'] = content.format(**kwargs)
            elif isinstance(content, list):  # Handle vision messages with content arrays
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        item['text'] = item['text'].format(**kwargs)
        # Also interpolate queued prompts
        self.prompt_queue = [p.format(**kwargs) for p in self.prompt_queue]
        return self

    def get_template_vars(self):
        """
        Return set of placeholder variable names found in all prompts.
        
        Useful for introspecting a loaded LLM to see what inputs it expects.
        
        Returns:
            set: Variable names (e.g., {'ticker', 'timeframe'})
        """
        import re
        pattern = re.compile(r'\{(\w+)\}')
        vars_found = set()
        for msg in self.chat_msgs:
            content = msg.get('content', '')
            if isinstance(content, str):
                vars_found.update(pattern.findall(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        vars_found.update(pattern.findall(item.get('text', '')))
        # Check prompt queue too
        for prompt in self.prompt_queue:
            vars_found.update(pattern.findall(prompt))
        return vars_found

    def run(self, **kwargs):
        """
        Execute the LLM with template variable interpolation.
        
        Interpolates {key} placeholders in all chat_msgs with provided kwargs,
        runs the prediction, and returns the response directly.
        
        This enables "LLM as function" usage:
            dcf = LLM().sys("Analyze {ticker}").user("Provide DCF valuation")
            result = dcf.run(ticker="TSLA")
        
        Args:
            **kwargs: Template variables to interpolate (e.g., ticker="TSLA").
                      Any unknown kwargs are passed to the prediction (model, temperature, etc.)
        
        Returns:
            str: The assistant's response content.
        """
        # Separate template vars from prediction kwargs
        template_vars = self.get_template_vars()
        interp_kwargs = {k: v for k, v in kwargs.items() if k in template_vars}
        pred_kwargs = {k: v for k, v in kwargs.items() if k not in template_vars}
        
        # Validate all template vars are provided
        missing = template_vars - set(interp_kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        
        self._interpolate_templates(**interp_kwargs)
        self._run_prediction(**pred_kwargs)
        last_res = self.last()
        self._reset_msgs()
        return last_res

    def run_json(self, enforce=True, **kwargs):
        """
        Execute the LLM with template interpolation, returning parsed JSON.
        
        Same as run() but enforces JSON mode and parses the response.
        
        Args:
            enforce (bool): If True (default), uses an LLM to fix malformed JSON on parse failure.
            **kwargs: Template variables and/or prediction kwargs.
        
        Returns:
            dict: The parsed JSON response. Returns empty dict on parsing error.
        """
        # Separate template vars from prediction kwargs
        template_vars = self.get_template_vars()
        interp_kwargs = {k: v for k, v in kwargs.items() if k in template_vars}
        pred_kwargs = {k: v for k, v in kwargs.items() if k not in template_vars}
        
        # Validate all template vars are provided
        missing = template_vars - set(interp_kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        
        self._interpolate_templates(**interp_kwargs)
        self._run_prediction(jsn_mode=True, **pred_kwargs)
        last_res = self._strip_markdown_json(self.last())
        self._reset_msgs()
        try:
            return json.loads(last_res)
        except (json.JSONDecodeError, TypeError):
            if enforce:
                return self._enforce_json(last_res)
            print(f"!! Error parsing JSON: {last_res}")
            return {}

    def last_json(self, enforce=True):
        """
        Parse the last assistant response as JSON, stripping markdown fences if present.

        Args:
            enforce (bool): If True (default), uses an LLM to fix malformed JSON on parse failure.

        Returns:
            dict: The parsed JSON response. Returns empty dict on parsing error.
        """
        last_res = self._strip_markdown_json(self.last())
        try:
            return json.loads(last_res)
        except (json.JSONDecodeError, TypeError):
            if enforce:
                return self._enforce_json(last_res)
            print(f"!! Error parsing JSON: {last_res}")
            return {}

    def _check_model(self, inputted_model):
        if not self.sub_closest_model:
            return inputted_model
        avail_models = self.get_models()
        if inputted_model in avail_models:
            return inputted_model

        category_result = self._handle_model_category(inputted_model)
        if category_result:
            return category_result

        def closest_match(inputted_model, choices):
            return rapidfuzz.process.extractOne(inputted_model, choices, scorer=rapidfuzz.fuzz.WRatio)[0]
        print(f"{inputted_model} not a valid model name.")
        new_model = closest_match(inputted_model, avail_models)
        print(f"Substituting {new_model}")
        return new_model
    
    def _handle_model_category(self, category_str):
        """Handle category aliases like 'best', 'fast', 'optimal', 'codex'.

        Also supports deterministic selection with numeric suffix:
        - best0, best1, cheap2 -> select exact rank in category
        - best, fast, optimal -> random selection from top N

        Returns None if not a valid category.
        """
        valid_categories = ['best', 'cheap', 'fast', 'open', 'optimal', 'codex', 'reasoning', 'search']

        # Default: use optimal category
        if category_str is None:
            category_str = "optimal"

        # Check for deterministic selection (e.g., best0, cheap1, optimal2)
        import re
        match = re.match(r'^([a-z]+)(\d+)$', str(category_str).lower())
        if match:
            base_category = match.group(1)
            rank_index = int(match.group(2))
            if base_category in valid_categories:
                entries = model_rankings.get(base_category, [])
                if rank_index < len(entries):
                    entry = entries[rank_index]
                    model_name = _get_model_name(entry)
                    # Set reasoning effort from rankings if present
                    effort = _get_reasoning_effort(entry)
                    if effort and not getattr(self, 'reasoning_effort', None):
                        self.reasoning_effort = effort
                        self.reasoning_enabled = True
                    return model_name
                else:
                    raise ValueError(f"Rank {rank_index} not available for {base_category} (max: {len(entries)-1})")
            return None

        # Random selection from category
        if category_str.lower() in valid_categories:
            candidates = self.get_models_for_category(category_str.lower())
            if not candidates:
                raise ValueError(f"No models available for category: {category_str}")
            selected = random.choice(candidates)
            # Find the entry to get reasoning_effort
            for entry in model_rankings.get(category_str.lower(), []):
                if _get_model_name(entry) == selected:
                    effort = _get_reasoning_effort(entry)
                    if effort and not getattr(self, 'reasoning_effort', None):
                        self.reasoning_effort = effort
                        self.reasoning_enabled = True
                    break
            return selected

        return None

    def get_models_for_category(self, category_str, search=False, reasoning=False, vision=False):
        """
        Retrieve the list of models associated with a specific alias category.
        Optionally filter by capability (search, reasoning, vision).

        Categories are defined in the static rankings file.

        Args:
            category_str (str): One of "best", "fast", "cheap", "open", "optimal", "codex".
            search (bool): If True, only return models that support web search.
            reasoning (bool): If True, only return models that support reasoning.
            vision (bool): If True, only return models that support vision.

        Returns:
            list: A list of model names (strings) sorted by rank for that category.
        """
        entries = model_rankings.get(category_str, [])
        
        # If filtering by capability, get all models first, filter, then apply limit
        if search or reasoning or vision:
            models = [_get_model_name(entry) for entry in entries]
            if search:
                models = [m for m in models if self._has_search(m)]
            if reasoning:
                models = [m for m in models if self._has_reasoning(m)]
            if vision:
                models = [m for m in models if self._has_vision(m)]
            return models[:10]
        
        # Default behavior: apply category-specific limits
        total = len(entries)
        category_limits = {
            "best": min(10, total),
            "cheap": min(10, total),
            "fast": min(10, total),
            "open": min(5, total),  
            "optimal": min(10, total),
            "codex": min(5, total), 
        }
        top_n = category_limits.get(category_str, min(10, total))

        return [_get_model_name(entry) for entry in entries[:top_n]]

    get_models_category = get_models_for_category

    def _run_xai_search(self, args, jsn_mode=False):
        """Handle xAI search using the Responses API (Agent Tools API).
        xAI deprecated Live Search Dec 2025 - web_search tool only works via Responses API.
        """
        # Convert chat messages to Responses API input format
        input_messages = []
        for msg in self.chat_msgs:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'system':
                input_messages.append({"role": "developer", "content": content})
            elif role in ('user', 'assistant'):
                input_messages.append({"role": role, "content": content})

        resp_args = {
            "model": args['model'],
            "input": input_messages,
            "tools": [{"type": "web_search"}],
        }
        if args['temperature'] is not None:
            resp_args["temperature"] = args['temperature']
        if args['max_tokens'] is not None:
            resp_args["max_output_tokens"] = args['max_tokens']

        if args['v']:
            print(f"Requesting {args['model']} (Responses API with web_search)")

        resp = responses(**resp_args)
        self.response_metadatas.append(resp)

        # Extract text from the Responses API output
        output_text = ""
        for output_item in resp.output:
            if hasattr(output_item, 'content'):
                for content_item in output_item.content:
                    if hasattr(content_item, 'text'):
                        output_text += content_item.text

        self.asst(output_text, merge=False)
        self._track_cost_responses(resp, args['model'])
        if args['v']:
            actual_model = resp.model or args['model']
            print(f"ASSISTANT ({actual_model}):")
            print(f"{output_text}\n")

        # Handle prompt queue
        if self.prompt_queue and self.prompt_queue_remaining > 0:
            prompt_queue_index = len(self.prompt_queue) - self.prompt_queue_remaining
            self.user(self.prompt_queue[prompt_queue_index])
            self.prompt_queue_remaining -= 1
            self._run_xai_search(args, jsn_mode)

    def _track_cost_responses(self, resp, model):
        """Track costs from Responses API format."""
        usage = getattr(resp, 'usage', None)
        if not usage:
            return
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        total_cost = getattr(usage, 'cost_in_usd_ticks', 0)
        if total_cost:
            total_cost = total_cost / 1e9  # Convert from ticks to USD
        self.costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
        })

    def _run_prediction(self, jsn_mode=False, **kwargs):
        args = self._resolve_args(**kwargs)
        args['model'] = self._check_model(args['model'])
        model_lower = args['model'].lower()

        # xAI with search requires Responses API (Agent Tools API)
        if self.search_enabled and model_lower.startswith('xai/'):
            return self._run_xai_search(args, jsn_mode)

        chat_args = {
            "model": args['model'],
            "temperature": args['temperature'],
            "messages": self.chat_msgs,
            "logger_fn": self._logger_fn,
            "max_tokens": args['max_tokens'],
            "num_retries": 2,
            "stream": args['stream'],
        }
        if self.available_tools:
            chat_args["tools"] = self.available_tools
            chat_args["tool_choice"] = self.tool_choice
            chat_args["parallel_tool_calls"] = self.parallel_tool_calls

        if self.reasoning_enabled:
            if 'grok' in model_lower and 'reasoning' in model_lower:
                pass
            elif model_lower.startswith('groq/'):
                pass
            elif self.reasoning_effort == 'default':
                pass
            else:
                chat_args['reasoning_effort'] = self.reasoning_effort
        else:
            if "gemini-3" in self.model:
                chat_args['reasoning_effort'] = "minimal"
        if self.search_enabled:
            if model_lower.startswith('groq/'):
                chat_args["tools"] = [{"type": "browser_search"}]
                chat_args["tool_choice"] = "required"
            else:
                chat_args["web_search_options"] = {
                    "search_context_size": self.search_context_size
                }

        if jsn_mode:
            chat_args["response_format"] = {"type": "json_object"}
        if args['v']: print(f"Requesting {args['model']}")
        comp = completion(**chat_args)
        ## saving metadata
        self.response_metadatas.append(comp)

        if args['stream']:
            printed_header = False
            for chunk in comp:
                if not printed_header and args['v']:
                    actual_model = chunk.model or args['model']
                    print(f"ASSISTANT ({actual_model}):")
                    printed_header = True
                chunk_content = chunk.choices[0].delta.content or ""
                self.asst(chunk_content, merge=True)
                if args['v']: print(chunk_content, end="")
            self.last_chunk_metadata = chunk
            self._track_cost(chunk, args['model'])
            if args['v']: print("\n")
        else:
            self._save_reasoning_trace(self.response_metadatas[-1])
            self._save_search_trace(self.response_metadatas[-1])
            asst_msg = comp.choices[0].message
            if args['v']:
                actual_model = comp.model or args['model']
                print(f"ASSISTANT ({actual_model}):")
            # Tool loop
            while self._requests_tool(asst_msg):
                self._execute_tools(asst_msg)
                comp = completion(**chat_args)
                self.response_metadatas.append(comp)
                self._track_cost(comp, args['model'])
                asst_msg = comp.choices[0].message
            # Final text response
            self.asst(asst_msg.content, merge=False)
            self._track_cost(comp, args['model'])
            if args['v']: print(f"{asst_msg.content}\n")
        
        if self.prompt_queue and self.prompt_queue_remaining > 0:
            prompt_queue_index = len(self.prompt_queue) - self.prompt_queue_remaining
            self.user(self.prompt_queue[prompt_queue_index])
            self.prompt_queue_remaining -= 1
            self._run_prediction(jsn_mode, **kwargs)

    def _execute_tools(self, asst_msg):
        """Execute all tool calls, append results to history"""
        self.chat_msgs.append(asst_msg.to_dict())

        for tool_call in asst_msg.tool_calls:
            name = tool_call.function.name
            if name == 'web_search':
                if self.v: print(f'Skipping server-handled tool: {name}')
                continue
            arguments_str = tool_call.function.arguments
            args = json.loads(arguments_str) if arguments_str and arguments_str.strip() not in ("", "null", "{}") else None
            if self.v: print(f'Found Tool {name} {args}.')
            result = self.fn_map[name](**args) if args else self.fn_map[name]()
            if self.v: print('Executed Tool. Result:', result,"\n")

            self.chat_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
   
    def _requests_tool(self, asst_msg):
        if asst_msg.tool_calls:
            if self.v: print('Received Tool Call Request.\n')
            return True
        else:
            return False
    def _requests_tool_streaming(self):
        "Currently tool calls are not supported for streaming responses. To be added."
        pass
    
    def user(self, prompt, image=None):
        """
        Add a User message to the history.

        Args:
            prompt (str): The message content.
            image (str or list, optional): Path(s) to image file(s) or URL(s) to attach.
                If provided, will check if current model supports vision and switch if needed.

        Returns:
            self: For chaining.
            Generally is followed by an inference call -> .chat, .result etc. or a preset .asst message
        """
        if image is None:
            content = prompt
        else:
            # Check if current model supports vision, switch if needed
            if not self._has_vision(self.model):
                self._update_model_to_vision()
            images = [image] if isinstance(image, str) else image
            content = [{"type": "text", "text": prompt}]
            for img in images:
                content.append(self._process_image(img))
        user_msg_obj = {"role": "user", "content": content}
        self.chat_msgs.append(user_msg_obj)
        if self.v:
            print(f"USER:")
            print(f"{prompt}\n")
            if image:
                img_count = 1 if isinstance(image, str) else len(image)
                print(f"[{img_count} image(s) attached]\n")
        return self
    
    u = usr = user

    def asst(self, prompt_response, merge=False):
        """
        Manually append an Assistant message to the conversation history.

        This is primarily used for **Few-Shot Prompting** (In-Context Learning), 
        where you provide examples of "User -> Assistant" pairs to teach the 
        model how to behave before asking your real question. 

        It can also be used to manually restore conversation history from a 
        previous session.

        Args:
            prompt_response (str): The full content of the assistant's message.
            merge (bool): If True, appends this text to the *immediately preceding* assistant message. Used internally for stitching stream chunks.

        Returns:
            self: For chaining.
        """
        last_msg = self.chat_msgs[-1]
        if merge and last_msg["role"] == 'assistant':  
            last_msg['content'] += prompt_response
        else:
            self.chat_msgs.append({"role": "assistant", "content": prompt_response})
        return self
    
    a = asssistant = asst

    def sys(self, prompt, append=True):
        """
        Add or update the System message.

        If a system message exists, this appends to it (unless `append=False`). 
        If none exists, it inserts one at the start of the history.

        Args:
            prompt (str): The system instructions.
            append (bool): If True, appends to existing system prompt. 
                           If False, overwrites it. Defaults to True.

        Returns:
            self: For chaining. Generally followed by a .user prompt or .tools tool enabling
        """
        if not len(self.chat_msgs):
            self.chat_msgs.append({"role": "system", "content": prompt})
        else:
            first_msg = self.chat_msgs[0]
            if first_msg['role'] == 'system':
                if append:
                    first_msg['content'] = first_msg['content'] + ('\n' if first_msg['content'] else '') + prompt
                else:
                    first_msg['content'] = prompt
            else:
                self.chat_msgs.insert(0, {"role": "system", "content": prompt}) 
        if self.v: 
            print(f"SYSTEM MSG:")
            print(f"{prompt}\n")
        return self

    system = s = sys

    def last(self):
        """
        Retrieve the content of the most recent Assistant message.

        Returns:
            str: The text content of the last response, or None if no assistant
            message is found.
        """
        for msg in reversed(self.chat_msgs):
            if msg['role'] == 'assistant':
                return msg['content']

    msg = last_msg = last

    def last_cost(self, warn=True):
        """Return the cost of the most recent completion, or None if no costs tracked."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Cost may not include search-specific charges.")
        return self.costs[-1]["total_cost"] if self.costs else None

    def total_cost(self, warn=True):
        """Return the sum of all completion costs in this session."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Cost may not include search-specific charges.")
        return sum(c["total_cost"] for c in self.costs)

    def all_costs(self, warn=True):
        """Return the full array of cost objects for all completions."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Costs may not include search-specific charges.")
        return self.costs

    def _reset_msgs(self, keep_sys=True):
        if keep_sys:
            self.chat_msgs = [i for i in self.chat_msgs if i['role'] == 'system']
        else:
            self.chat_msgs = []
        self.prompt_queue_remaining = len(self.prompt_queue)

    def queue(self, prompt):
        """
        Queue a user message to be sent automatically after the next assistant response.
        Useful for defining a multi-turn conversation script in advance.

        Args:
            prompt (str): The follow-up message to send.

        Returns:
            self: For chaining.
        """
        self.prompt_queue.append(prompt)
        self.prompt_queue_remaining += 1
        return self

    followup = then = queue 

    def fwd(self, fwd_llm, instructions=''):
        """
        Forward the last response from this LLM to another LLM instance.

        Args:
            fwd_llm (LLM): The target LLM instance to receive the context.
            instructions (str, optional): Additional instructions to append to the forwarded context.

        Returns:
            self: The *target* LLM instance (fwd_llm), after the chat call.
        """
        last_res = self.last()
        return fwd_llm.user(last_res+'\n'+instructions).chat()
    
    def turn_on_debug(self):
        """
        Enable verbose debug logging for the underlying LiteLLM library.

        This will print raw API payloads, full request/response objects, and 
        connection details to the console. Useful for troubleshooting API key 
        issues or unexpected model behavior.

        Do not use in production as API key details can leak. 
        """
        self._set_litellm_level(logging.DEBUG)

    def turn_off_debug(self):
        self._set_litellm_level(logging.WARNING)

    def _set_litellm_level(self, level):
        for name in ["LiteLLM", "LiteLLM Router", "LiteLLM Proxy"]:
            logging.getLogger(name).setLevel(level)

    def _save_reasoning_trace(self, metadata):
        try:
            choices = getattr(metadata, 'choices', None)
            if choices is None or callable(choices):
                return
            if len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    content = getattr(message, 'reasoning_content', None)
                    if content:
                        self.reasoning_contents.append(content)
        except (TypeError, IndexError, AttributeError):
            pass

    def _save_search_trace(self, metadata):
        try:
            choices = getattr(metadata, 'choices', None)
            if choices is None or callable(choices):
                return
            if len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    annotations = getattr(message, 'annotations', None)
                    if annotations:
                        self.search_annotations.append(annotations)
        except (TypeError, IndexError, AttributeError):
            pass
    
    def _has_search(self, model):
        return litellm.supports_web_search(model=model) == True

    def _has_reasoning(self, model):
        return litellm.supports_reasoning(model=model) == True

    def _has_vision(self, model):
        return litellm.supports_vision(model=model) == True

    def _update_model_to_reasoning(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_reasoning(model):
                    self.model = model
                    effort = _get_reasoning_effort(entry)
                    if effort:
                        self.reasoning_effort = effort
                    print(f'Updated model for reasoning: {model}')
                    return

    def _update_model_to_search(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_search(model):
                    self.model = model
                    print(f'Updated model for search: {model}')
                    return

    def _update_model_to_vision(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_vision(model):
                    self.model = model
                    print(f'Updated model for vision: {model}')
                    return

    def get_models(self, model_str=None, text_model=True):
        """
        List available models, optionally filtered by name.

        Args:
            model_str (str, optional): Substring to filter models (e.g., "claude").
            text_model (bool): Default True, filters for models that support chat/text generation.

        Returns:
            list: A list of model name strings.
        """
        if model_str:
            models = [i for i in litellm.get_valid_models() if model_str in i]
        else:
            models = litellm.get_valid_models()
        if text_model:
            models = [model for model in models if litellm.model_cost.get(model, {}).get('mode') in ['chat', 'responses']]
        return models

    def models_with_search(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_web_search(model=model) == True]

    def models_with_reasoning(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_reasoning(model=model) == True]

    def models_with_vision(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_vision(model=model) == True]
    
    def to_md(self, filename):
        """Export the chat history to a Markdown file."""
        lines = []
        for msg in self.chat_msgs:
            if not msg.get("role") or msg.get("role") not in ["user", "assistant"]:
                continue
            role = msg.get("role").capitalize()
            content = msg.get("content", "")
            
            # Add role as a header
            lines.append(f"## {role}\n")
            lines.append(content)
            lines.append("\n---\n")
        md_output = "\n".join(lines)
        with open(filename, "w") as f:
            f.write(md_output)       

    def save_llm(self, filepath):
        """
        Serialize the full state of the LLM (config, history, tools) to a JSON file.

        Args:
            filepath (str): Path to the output JSON file.

        Returns:
            self: For chaining.
        """
        state = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "v": self.v,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "search_enabled": self.search_enabled,
            "search_context_size": self.search_context_size,
            "reasoning_enabled": self.reasoning_enabled,
            "reasoning_effort": self.reasoning_effort,
            "schemas": self.schemas,
            "chat_msgs": self.chat_msgs,
            "prompt_queue": self.prompt_queue,
            "prompt_queue_remaining": self.prompt_queue_remaining,
            "reasoning_contents": self.reasoning_contents,
            "search_annotations": self.search_annotations,
            "costs": self.costs,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved instance {filepath}!")
        return self

    def run_history(self, **kwargs):
        """
        Re-run the entire conversation history of this instance using a new LLM configuration.

        Useful for "upgrading" a conversation (e.g., switching from a fast model to a reasoning model)
        or A/B testing the same context on different models.

        Args:
            **kwargs: Arguments for the new LLM instance (model, temperature, etc.).

        Returns:
            LLM: The new LLM instance after executing the history.
        """
        new_llm = LLM(**kwargs)
        first_user_filled = False
        for chat_msg in self.chat_msgs:
            if chat_msg['role'] == 'sys':
                new_llm.sys(chat_msg['content'])
            elif chat_msg['role'] == 'user':
                if not first_user_filled:
                    new_llm.user(chat_msg['content'])
                    first_user_filled = True
                new_llm.queue(chat_msg['content'])
        return new_llm.chat()

    @classmethod
    def load_llm(cls, filepath):
        """
        Reconstruct an LLM instance from a saved JSON state file. Run on the LLM class rather than an LLM instance.

        Args:
            filepath (str): Path to the source JSON file.

        Returns:
            LLM: A fully restored instance.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        llm = cls()
        if "model" in state:
            llm.model = state["model"]
        if "temperature" in state:
            llm.temperature = state["temperature"]
        if "stream" in state:
            llm.stream = state["stream"]
        if "v" in state:
            llm.v = state["v"]
        if "max_tokens" in state:
            llm.max_tokens = state["max_tokens"]
        if "tool_choice" in state:
            llm.tool_choice = state["tool_choice"]
        if "parallel_tool_calls" in state:
            llm.parallel_tool_calls = state["parallel_tool_calls"]
        if "search_enabled" in state:
            llm.search_enabled = state["search_enabled"]
        if "reasoning_enabled" in state:
            llm.reasoning_enabled = state["reasoning_enabled"]
        if "reasoning_effort" in state:
            llm.reasoning_effort = state["reasoning_effort"]
        if "chat_msgs" in state:
            llm.chat_msgs = state["chat_msgs"]
        if "prompt_queue" in state:
            llm.prompt_queue = state["prompt_queue"]
        if "prompt_queue_remaining" in state:
            llm.prompt_queue_remaining = state["prompt_queue_remaining"]
        if "schemas" in state:
            llm.schemas = state["schemas"]
        if "reasoning_contents" in state:
            llm.reasoning_contents = state["reasoning_contents"]
        if "search_annotations" in state:
            llm.search_annotations = state["search_annotations"]
        if "costs" in state:
            llm.costs = state["costs"]
        print(f"Loaded instance {filepath}!")
        return llm
    