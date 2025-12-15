from cruise_llm import LLM

def test_import():
    """Test that LLM class can be imported"""
    assert LLM is not None

def test_init_with_model():
    """Test LLM can be instantiated with explicit model"""
    llm = LLM()
    assert llm is not None