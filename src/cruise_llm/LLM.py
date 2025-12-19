import litellm
from litellm import completion
import json
from dotenv import load_dotenv
from typing import Optional, Union, Dict, Any
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
    def __init__(self, model=None, temperature=None, stream=False, v=True, debug=False, max_tokens=24000, search=False, reasoning=False, search_context_size="medium", reasoning_effort="medium",sub_closest_model=True):
        self.chat_msgs = []
        self.logs = []
        self.response_metadatas = []
        self.followups = []
        self.available_followups = 0
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
        self.reasoning_contents = []
        self.search_annotations = []
        self.max_tokens = max_tokens
        if debug == True:
            self.turn_on_debug()
        else:
            self.turn_off_debug()
    
    def _resolve_args(self, **kwargs):
        """Merges chat,run,runjson kwargs with instance defaults."""
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
        """Stateful: makes call, keeps history, returns self for chaining"""
        self._run_prediction(**kwargs)
        return self

    c = ch = chat

    def chat_json(self, **kwargs):
        self._run_prediction(jsn_mode=True, **kwargs)
        return self
    
    cjson = c_json = ch_json = chat_json

    def result(self, **kwargs) -> str:
        """Runs prompt, returns res string, resets chat history, keeps system prompt"""
        self._run_prediction(**kwargs)
        last_res = self.last()
        self._reset_msgs()
        return last_res
    
    r = res = result

    def result_json(self, **kwargs) -> Dict:
        """Runs prompt with JSON mode, returns dict, resets chat history, keeps system prompt"""
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
    
    def _handle_model_category(self,category_str):
        if category_str is None:
            best_fast_intersection = [i for i in model_rankings["best"] if i in model_rankings["fast"][:30]]
            return best_fast_intersection[0]
        if category_str == "best":
            top_n = 15
            return random.choice(model_rankings["best"][:top_n])
        elif category_str == "cheap":
            top_n = 5
            return random.choice(model_rankings["cheap"][:top_n])
        elif category_str == "fast":
            top_n = 10
            return random.choice(model_rankings["fast"][:top_n])
        elif category_str == "open":
            top_n = 10
            return random.choice(model_rankings["open"][:top_n])


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
            chat_args['reasoning_effort'] = self.reasoning_effort
        else:
            if "gemini-3" in self.model:
                chat_args['reasoning_effort'] = "minimal"
        if self.search_enabled:
            chat_args["web_search_options"] = {
                 "search_context_size": self.search_context_size
            }

        if jsn_mode:
            chat_args["response_format"] = {"type": "json_object"}
        if self.v:
            print(f"Requesting {args['model']}")
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
        
        if self.followups and self.available_followups > 0:
            followup_index = len(self.followups) - self.available_followups
            self.user(self.followups[followup_index])
            self.available_followups -= 1
            self._run_prediction(jsn_mode, **kwargs)

    def _execute_tools(self, asst_msg):
        """Execute all tool calls, append results to history"""
        self.chat_msgs.append(asst_msg.to_dict())
        
        for tool_call in asst_msg.tool_calls:
            name = tool_call.function.name
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
        pass
    
    def user(self, msg_str):
        user_msg_obj = {"role": "user", "content": msg_str}
        self.chat_msgs.append(user_msg_obj)
        if self.v:
            print(f"USER:")
            print(f"{msg_str}\n")
        return self
    
    u = usr = user

    def asst(self, msg_str, merge=False):
        last_msg = self.chat_msgs[-1]
        if merge and last_msg["role"] == 'assistant':  
            last_msg['content'] += msg_str
        else:
            self.chat_msgs.append({"role": "assistant", "content": msg_str})
        return self
    
    a = asssistant = asst

    def sys(self, new_str, append=True):
        if not len(self.chat_msgs):
            self.chat_msgs.append({"role": "system", "content": new_str})
        else:
            first_msg = self.chat_msgs[0]
            if first_msg['role'] == 'system':
                if append:
                    first_msg['content'] = first_msg['content'] + ('\n' if first_msg['content'] else '') + new_str
                else:
                    first_msg['content'] = new_str
            else:
                self.chat_msgs.insert(0, {"role": "system", "content": new_str}) 
        if self.v: 
            print(f"SYSTEM MSG:")
            print(f"{new_str}\n")
        return self

    system = s = sys

    def last(self):
        for msg in reversed(self.chat_msgs):
            if msg['role'] == 'assistant':
                return msg['content']

    msg = last_msg = last

    def _reset_msgs(self, keep_sys=True):
        if keep_sys:
            self.chat_msgs = [i for i in self.chat_msgs if i['role'] == 'system']
        else:
            self.chat_msgs = []
        self.available_followups = len(self.followups)


    def add_followup(self, prompt):
        self.followups.append(prompt)
        self.available_followups += 1
        return self

    def fwd(self, fwd_llm, instructions=''):
        last_res = self.last()
        return fwd_llm.user(last_res+'\n'+instructions).chat()
    
    def turn_on_debug(self):
        self._set_litellm_level(logging.DEBUG)

    def turn_off_debug(self):
        self._set_litellm_level(logging.WARNING)

    def _set_litellm_level(self, level):
        for name in ["LiteLLM", "LiteLLM Router", "LiteLLM Proxy"]:
            logging.getLogger(name).setLevel(level)

    def _save_reasoning_trace(self, metadata):
        if (hasattr(metadata, 'choices') and 
            hasattr(metadata.choices[0], 'message') and 
            hasattr(metadata.choices[0].message, 'reasoning_content')):
            self.reasoning_contents.append(metadata.choices[0].message.reasoning_content)

    def _save_search_trace(self, metadata):
        if (hasattr(metadata, 'choices') and 
            hasattr(metadata.choices[0], 'message') and 
            hasattr(metadata.choices[0].message, 'annotations')):            
            self.search_annotations.append(metadata.choices[0].message.annotations)
    
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
        """Save LLM state to JSON file"""
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
            "followups": self.followups,
            "available_followups": self.available_followups, 
            "reasoning_contents": self.reasoning_contents,
            "search_annotations": self.search_annotations,          
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved instance {filepath}!")
        return self

    def run_history(self, **kwargs):
        new_llm = LLM(**kwargs)
        first_user_filled = False
        for chat_msg in self.chat_msgs:
            if chat_msg['role'] == 'sys':
                new_llm.sys(chat_msg['content'])
            elif chat_msg['role'] == 'user':
                if not first_user_filled:
                    new_llm.user(chat_msg['content'])
                    first_user_filled = True
                new_llm.add_followup(chat_msg['content'])
        return new_llm.chat()

        

    @classmethod
    def load_llm(cls, filepath):
        """Load LLM state from JSON file, returns new instance"""
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
        if "followups" in state:
            llm.followups = state["followups"]
        if "available_followups" in state:
            llm.available_followups = state["available_followups"]
        if "schemas" in state:
            llm.schemas = state["schemas"]
        if "reasoning_contents" in state:
            llm.reasoning_contents = state["reasoning_contents"]
        if "search_annotations" in state:
            llm.search_annotations = state["search_annotations"]
        print(f"Loaded instance {filepath}!")
        return llm
    