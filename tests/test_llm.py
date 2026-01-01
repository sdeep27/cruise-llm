"""
Tests for cruise_llm LLM class.
Extracted from test_notebook.ipynb
"""
import os
import sys
import tempfile

import pytest

# Add src to path for local development
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from cruise_llm import LLM


class TestImportAndInit:
    """Test that LLM class can be imported and initialized."""

    def test_import(self):
        """Test that LLM class can be imported."""
        assert LLM is not None

    def test_init_with_model(self):
        """Test LLM can be instantiated with explicit model."""
        llm = LLM()
        assert llm is not None


class TestModelDiscovery:
    """Test get_models(), models_with_search(), and models_with_reasoning() methods."""

    def test_get_models_returns_list(self):
        """Test get_models() returns a non-empty list."""
        llm = LLM(v=False)
        all_models = llm.get_models()
        assert isinstance(all_models, list), "get_models() should return a list"
        assert len(all_models) > 0, "Should have at least one model available"

    def test_get_models_with_filter(self):
        """Test get_models() with model_str filter."""
        llm = LLM(v=False)
        gpt_models = llm.get_models(model_str="gpt")
        assert all("gpt" in m for m in gpt_models), "All filtered models should contain 'gpt'"

    def test_models_with_search(self):
        """Test models_with_search() returns models."""
        llm = LLM(v=False)
        search_models = llm.models_with_search()
        assert isinstance(search_models, list), "Should return a list"

    def test_models_with_reasoning(self):
        """Test models_with_reasoning() returns models."""
        llm = LLM(v=False)
        reasoning_models = llm.models_with_reasoning()
        assert isinstance(reasoning_models, list), "Should return a list"


class TestConversationalMode:
    """Test that chat() maintains conversation history."""

    def test_chat_maintains_history(self):
        """Test conversational mode maintains message history."""
        llm = LLM(stream=False, v=False, max_tokens=150)

        llm.sys('You are helpful. Keep responses under 50 words.')
        llm.user('What is 2+2?').chat()

        assert len(llm.chat_msgs) >= 3, "Should have system, user, and assistant messages"
        assert llm.chat_msgs[0]['role'] == 'system'
        assert llm.chat_msgs[1]['role'] == 'user'
        assert llm.chat_msgs[2]['role'] == 'assistant'

    def test_followup_maintains_context(self):
        """Test follow-up maintains context."""
        llm = LLM(stream=False, v=False, max_tokens=150)
        llm.sys('You are helpful. Keep responses under 50 words.')
        llm.user('What is 2+2?').chat()
        llm.user('What is that number times 10?').chat()

        assert len(llm.chat_msgs) >= 5, "Should have 5 messages after follow-up"

    def test_last_returns_response(self):
        """Test last() returns assistant response."""
        llm = LLM(stream=False, v=False, max_tokens=150)
        llm.sys('You are helpful.')
        llm.user('Say hello').chat()

        last_response = llm.last()
        assert last_response is not None, "last() should return a response"
        assert isinstance(last_response, str), "last() should return a string"


class TestPipelineMode:
    """Test that res() resets history but keeps system prompt."""

    def test_res_resets_history(self):
        """Test pipeline mode resets history after call."""
        pipeline = LLM(v=False, max_tokens=100)
        pipeline.sys('You are a helpful assistant. Keep responses brief.')

        result1 = pipeline.user('Say hello').res()
        assert isinstance(result1, str), "res() should return a string"
        assert len(pipeline.chat_msgs) == 1, f"History should only have system msg, got {len(pipeline.chat_msgs)}"
        assert pipeline.chat_msgs[0]['role'] == 'system', "Should keep system prompt"

    def test_res_no_history_bleed(self):
        """Test second res() has no history from first."""
        pipeline = LLM(v=False, max_tokens=100)
        pipeline.sys('You are a helpful assistant. Keep responses brief.')

        pipeline.user('Say hello').res()
        result2 = pipeline.user('Say goodbye').res()

        assert isinstance(result2, str), "Second res() should also return a string"
        assert len(pipeline.chat_msgs) == 1, "History should still only have system msg"


class TestJsonOutput:
    """Test res_json() returns parsed JSON."""

    def test_res_json_returns_dict(self):
        """Test JSON output mode returns a dict."""
        llm = LLM(v=False, max_tokens=200)
        llm.sys('Extract entities from text. Return JSON with key "entities" containing a list of entity strings.')

        result = llm.user('Apple announced a new iPhone in Cupertino').res_json()

        assert isinstance(result, dict), f"res_json() should return dict, got {type(result)}"
        assert 'entities' in result, f"Result should have 'entities' key, got: {result}"
        assert isinstance(result['entities'], list), "'entities' should be a list"


class TestMultiFollowupChains:
    """Test queue() for automatic follow-up prompts."""

    def test_queue_processes_followups(self):
        """Test multi-followup chains execute properly."""
        llm = LLM(v=False, max_tokens=150)
        llm.queue('Make it shorter.')
        llm.queue('Now make it one sentence.')

        result = llm.user('Explain photosynthesis').res()

        assert llm.prompt_queue_remaining == 2, "prompt_queue_remaining should reset after res()"
        assert len(llm.prompt_queue) == 2, "prompt_queue list should be preserved"
        assert isinstance(result, str), "Should return a string result"


class TestSystemPromptHandling:
    """Test sys() append and replace behavior."""

    def test_sys_initial(self):
        """Test initial system prompt."""
        llm = LLM(v=False)
        llm.sys('You are helpful.')

        assert len(llm.chat_msgs) == 1
        assert llm.chat_msgs[0]['content'] == 'You are helpful.'

    def test_sys_append(self):
        """Test appending to system prompt (default behavior)."""
        llm = LLM(v=False)
        llm.sys('You are helpful.')
        llm.sys('Be concise.')

        assert 'You are helpful.' in llm.chat_msgs[0]['content']
        assert 'Be concise.' in llm.chat_msgs[0]['content']

    def test_sys_replace(self):
        """Test replacing system prompt."""
        llm = LLM(v=False)
        llm.sys('You are helpful.')
        llm.sys('New system prompt.', append=False)

        assert llm.chat_msgs[0]['content'] == 'New system prompt.'


class TestParallelToolCalling:
    """Test tools with parallel function calls."""

    def test_parallel_tools_registered(self):
        """Test tools are registered correctly."""
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 72°F, sunny"

        def get_time(timezone: str) -> str:
            """Get current time in a timezone."""
            return f"Current time in {timezone}: 3:00 PM"

        llm = LLM(v=False, max_tokens=300)
        llm.tools(fns=[get_weather, get_time])

        assert llm.available_tools is not None, "Tools should be registered"
        assert len(llm.available_tools) == 2, "Should have 2 tools"
        assert llm.fn_map is not None, "Function map should be set"

    def test_parallel_tools_execute(self):
        """Test parallel tool calls execute."""
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: 72°F, sunny"

        def get_time(timezone: str) -> str:
            """Get current time in a timezone."""
            return f"Current time in {timezone}: 3:00 PM"

        llm = LLM(v=False, max_tokens=300)
        llm.tools(fns=[get_weather, get_time])
        llm.user("What's the weather in Paris and the time in EST?").chat()

        response = llm.last()
        assert response is not None, "Should have a response"


class TestSequentialToolCalling:
    """Test tools that chain results from one to another."""

    def test_sequential_tools_execute(self):
        """Test sequential tool calling works."""
        def get_public_ip() -> dict:
            """Returns the caller's current public IP address."""
            return {"ip": "203.0.113.42"}

        def geoip_lookup(ip: str) -> dict:
            """Given an IP address, returns geolocation info (city and country)."""
            return {"city": "Bangkok", "country": "Thailand"}

        def get_current_temperature(city: str, country: str) -> dict:
            """Given a city and country, fetches the CURRENT temperature in Celsius."""
            return {"temperature_c": 21.5, "conditions": "clear"}

        llm = LLM(v=False, max_tokens=400)
        llm.tools(fns=[get_public_ip, geoip_lookup, get_current_temperature])

        llm.user(
            "Get my public IP address. Then look up its geolocation. "
            "Then fetch the current temperature there. Report the final result."
        ).chat()

        response = llm.last()
        assert response is not None, "Should have a response"


class TestStreamingMode:
    """Test stream=True works correctly."""

    def test_streaming_produces_response(self):
        """Test streaming mode produces a response."""
        llm = LLM(stream=True, v=False, max_tokens=100)
        llm.user('Count from 1 to 5').chat()

        response = llm.last()
        assert response is not None, "Streaming should produce a response"
        assert len(response) > 0, "Response should not be empty"

    def test_streaming_captures_metadata(self):
        """Test streaming captures chunk metadata."""
        llm = LLM(stream=True, v=False, max_tokens=100)
        llm.user('Count from 1 to 5').chat()

        assert llm.last_chunk_metadata is not None, "Should capture last chunk metadata"


class TestSaveAndLoad:
    """Test save_llm() and load_llm() persistence."""

    def test_save_and_load_preserves_state(self):
        """Test save and load preserves LLM state."""
        llm = LLM(v=False, max_tokens=100, stream=False)
        llm.sys('You are a test assistant.')
        llm.user('Hello').chat()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            llm.save_llm(temp_path)
            assert os.path.exists(temp_path), "Save file should exist"

            loaded = LLM.load_llm(temp_path)
            assert loaded.max_tokens == 100, "max_tokens should be preserved"
            assert loaded.stream == False, "stream setting should be preserved"
            assert len(loaded.chat_msgs) == len(llm.chat_msgs), "chat history should be preserved"
            assert loaded.chat_msgs[0]['content'] == 'You are a test assistant.', "system prompt should be preserved"
        finally:
            os.unlink(temp_path)


class TestMarkdownExport:
    """Test to_md() export functionality."""

    def test_to_md_exports_conversation(self):
        """Test markdown export contains conversation."""
        llm = LLM(v=False, max_tokens=100)
        llm.sys('Be brief.')
        llm.user('What is AI?').chat()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name

        try:
            llm.to_md(temp_path)
            assert os.path.exists(temp_path), "Markdown file should exist"

            with open(temp_path, 'r') as f:
                md_content = f.read()

            assert '## User' in md_content, "Should have User header"
            assert '## Assistant' in md_content, "Should have Assistant header"
            assert 'What is AI?' in md_content, "Should contain user message"
        finally:
            os.unlink(temp_path)


class TestMethodAliases:
    """Test that method aliases work correctly."""

    def test_sys_aliases(self):
        """Test sys aliases."""
        llm = LLM(v=False)
        assert llm.sys == llm.s == llm.system, "sys aliases should be equal"

    def test_user_aliases(self):
        """Test user aliases."""
        llm = LLM(v=False)
        assert llm.user == llm.u == llm.usr, "user aliases should be equal"

    def test_asst_aliases(self):
        """Test asst aliases."""
        llm = LLM(v=False)
        assert llm.asst == llm.a == llm.asssistant, "asst aliases should be equal"

    def test_chat_aliases(self):
        """Test chat aliases."""
        llm = LLM(v=False)
        assert llm.chat == llm.c == llm.ch, "chat aliases should be equal"

    def test_result_aliases(self):
        """Test result aliases."""
        llm = LLM(v=False)
        assert llm.result == llm.r == llm.res, "result aliases should be equal"

    def test_result_json_aliases(self):
        """Test result_json aliases."""
        llm = LLM(v=False)
        assert llm.result_json == llm.rjson == llm.res_json, "result_json aliases should be equal"

    def test_last_aliases(self):
        """Test last aliases."""
        llm = LLM(v=False)
        assert llm.last == llm.msg == llm.last_msg, "last aliases should be equal"


class TestMethodChaining:
    """Test fluent API method chaining."""

    def test_chaining_returns_llm(self):
        """Test method chaining returns LLM instance."""
        llm = LLM(v=False, max_tokens=100)
        result = llm.sys('Be brief.').user('Say hi').chat().user('Say bye').chat()

        assert isinstance(result, LLM), "Chaining should return LLM instance"
        assert len(llm.chat_msgs) >= 4, "Should have multiple messages from chain"


class TestResponseMetadata:
    """Test that response metadata is captured."""

    def test_metadata_captured(self):
        """Test response metadata is captured."""
        llm = LLM(v=False, max_tokens=50, stream=False)
        llm.user('Say hello').chat()

        assert len(llm.response_metadatas) > 0, "Should capture response metadata"
        assert llm.response_metadatas[-1] is not None, "Metadata should not be None"

    def test_logs_captured(self):
        """Test logs are captured."""
        llm = LLM(v=False, max_tokens=50, stream=False)
        llm.user('Say hello').chat()

        assert len(llm.logs) > 0, "Should capture logs"


class TestModelSelection:
    """Test specifying different models."""

    def test_default_model(self):
        """Test default model is set."""
        llm = LLM(v=False)
        assert llm.model is not None, "Should have a default model"

    def test_custom_model(self):
        """Test model can be set via constructor."""
        llm = LLM(v=False)
        available_models = llm.get_models()
        if len(available_models) > 1:
            other_model = available_models[1]
            custom_llm = LLM(model=other_model, v=False)
            assert custom_llm.model == other_model, "Model should be set via constructor"


class TestToolsConfiguration:
    """Test tools() method with search and reasoning flags."""

    def test_search_enabled(self):
        """Test search flag enables search."""
        llm = LLM(v=False)
        llm.tools(search=True, search_context_size='high')

        assert llm.search_enabled == True, "Search should be enabled"
        assert llm.search_context_size == 'high', "Search context size should be set"

    def test_reasoning_enabled(self):
        """Test reasoning flag enables reasoning."""
        llm = LLM(v=False)
        llm.tools(reasoning=True, reasoning_effort='high')

        assert llm.reasoning_enabled == True, "Reasoning should be enabled"
        assert llm.reasoning_effort == 'high', "Reasoning effort should be set"


class TestForward:
    """Test fwd() method for piping output to another LLM."""

    def test_fwd_pipes_output(self):
        """Test forward pipes output to another LLM."""
        llm1 = LLM(v=False, max_tokens=100)
        llm1.sys('You generate single words.')
        llm1.user('Give me a word for happiness').chat()

        llm2 = LLM(v=False, max_tokens=100)
        llm2.sys('You define words briefly.')

        result = llm1.fwd(llm2, instructions='Define this word in 10 words or less.')

        assert isinstance(result, LLM), "fwd() should return the target LLM"
        assert len(llm2.chat_msgs) > 1, "Target LLM should have messages"
