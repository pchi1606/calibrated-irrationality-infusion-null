"""Minimal parser shim extracted from code/pilot.py.

The full pilot.py drives LLM generation (requires llama.cpp + model
weights). The analysis pipeline needs only the output parser, which is
reproduced here verbatim so the reproduction package has no heavy
dependencies.
"""
import re

NUM_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")


def num_parse(text):
    m = NUM_RE.search(text.replace(",", ""))
    return float(m.group(1)) if m else None
