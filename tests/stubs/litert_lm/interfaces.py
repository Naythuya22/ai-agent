"""Sampler configs used by core/llm.py."""
class SamplerConfig:
    def __init__(self, **kw):
        self.kw = kw

    def __repr__(self):
        return f"SamplerConfig({self.kw})"


class RepetitionPenaltyConfig:
    def __init__(self, **kw):
        self.kw = kw


class NoRepeatNgramConfig:
    def __init__(self, **kw):
        self.kw = kw
