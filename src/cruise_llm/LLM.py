import litellm
from litellm import completion
import json
from dotenv import load_dotenv
from function_schema import get_function_schema
import logging
import rapidfuzz
import random
from pathlib import Path

_RANKINGS_PATH = Path(__file__).parent / "rankings" / "static_rankings_2025-12-19.json"
with open(_RANKINGS_PATH, "r") as f:
    model_rankings = json.load(f)

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
                ("best", "fast", "cheap", "open"). Defaults to None (logic handles fallback).
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
        self.reasoning_enabled = reasoning
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

    def result_json(self, **kwargs):
        """
        Run the prediction in JSON mode, parse the result, and reset chat history.
        (preserving the System prompt).

        Returns:
            dict: The parsed JSON response. Returns empty dict on parsing error.
        """
        self._run_prediction(jsn_mode=True, **kwargs)
        last_res = self.last()
        self._reset_msgs()
        try:
            return json.loads(last_res)
        except (json.JSONDecodeError, TypeError):
            print(f"!! Error parsing JSON: {last_res}")
            return {}

    rjson = res_json = result_json

    def _check_model(self,inputted_model):
        if not self.sub_closest_model:
            return inputted_model
        avail_models = self.get_models()
        if inputted_model in avail_models:
            return inputted_model
        if inputted_model is None or inputted_model in ['best','cheap','fast','open','reasoning','search']:
            return self._handle_model_category(inputted_model)
        def closest_match(inputted_model, choices):
            return rapidfuzz.process.extractOne(inputted_model,choices,scorer=rapidfuzz.fuzz.WRatio)[0]
        print(f"{inputted_model} not a valid model name.")
        new_model = closest_match(inputted_model, avail_models)
        print(f"Substituting {new_model}")
        return new_model
    
    def _handle_model_category(self, category_str):
        if category_str is None:
            top_fast_models_limit = 30
            fast_subset = set(model_rankings["fast"][:top_fast_models_limit])
            best_fast_intersection = [m for m in model_rankings["best"] if m in fast_subset]
            if best_fast_intersection:
                return best_fast_intersection[0]
            category_str = "best"
        candidates = self.get_models_for_category(category_str)
        if not candidates:
            raise ValueError(f"No models available for category: {category_str}")
        return random.choice(candidates)

    def get_models_for_category(self, category_str):
        """
        Retrieve the list of models associated with a specific alias category.

        Categories are defined in the static rankings file.

        Args:
            category_str (str): One of "best", "fast", "cheap", "open".

        Returns:
            list: A list of model names (strings) sorted by rank for that category.
        """
        category_limits = {
            "best": 15,
            "cheap": 5,
            "fast": 10,
            "open": 10,
        }
        top_n = category_limits.get(category_str, 10)
        return model_rankings.get(category_str, [])[:top_n]

    get_models_category = get_models_for_category

    def _run_prediction(self, jsn_mode=False, **kwargs):
        args = self._resolve_args(**kwargs)
        args['model'] = self._check_model(args['model'])
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
            model_lower = args['model'].lower()
            if 'grok' in model_lower and 'reasoning' in model_lower:
                pass
            elif model_lower.startswith('groq/'):
                chat_args['reasoning_effort'] = 'default'
            else:
                chat_args['reasoning_effort'] = self.reasoning_effort
        else:
            if "gemini-3" in self.model:
                chat_args['reasoning_effort'] = "minimal"
        if self.search_enabled:
            model_lower = args['model'].lower()
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
                asst_msg = comp.choices[0].message
            # Final text response
            self.asst(asst_msg.content, merge=False)
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
    
    def user(self, prompt):
        """
        Add a User message to the history. 

        Args:
            prompt (str): The message content.

        Returns:
            self: For chaining.
            Generally is followed by an inference call -> .chat, .result etc. or a preset .asst message
        """
        user_msg_obj = {"role": "user", "content": prompt}
        self.chat_msgs.append(user_msg_obj)
        if self.v:
            print(f"USER:")
            print(f"{prompt}\n")
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
    
    def _has_search(self,model):
        return litellm.supports_web_search(model=model) == True

    def _has_reasoning(self,model):
        return litellm.supports_reasoning(model=model) == True

    def _update_model_to_reasoning(self):
        for model in model_rankings["best"]:
            if self._has_reasoning(model):
                self.model = model
                print(f'Updated model for reasoning: {model}')
                break

    def _update_model_to_search(self):
        for model in model_rankings["best"]:
            if self._has_search(model):
                self.model = model
                print(f'Updated model for search: {model}')
                break

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

    def models_with_search(self,model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_web_search(model=model) == True]

    def models_with_reasoning(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_reasoning(model=model) == True]
    
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
        print(f"Loaded instance {filepath}!")
        return llm
    