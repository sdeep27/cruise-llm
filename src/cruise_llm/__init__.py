from .LLM import LLM
from .evaluate import pairwise_evaluate
from .subprompt import subprompt


def to_md(text, path=None):
    """Write text to a Markdown file. If no path given, returns the text unchanged."""
    if path is None:
        return text
    if not path.endswith(".md"):
        path += ".md"
    with open(path, "w") as f:
        f.write(str(text))
    return path


__all__ = ["LLM", "pairwise_evaluate", "subprompt", "to_md"]
