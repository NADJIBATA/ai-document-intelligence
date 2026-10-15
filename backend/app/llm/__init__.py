"""Language model providers behind one small interface (``LLMProvider``).

Providers are plain HTTP clients (no vendor SDK): OpenAI-compatible servers (OpenAI,
vLLM, Groq, LM Studio...), Ollama, and a deterministic fake for tests and demos.
"""
