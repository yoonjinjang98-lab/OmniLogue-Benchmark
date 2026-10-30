"""Original implementation of the "Learning to Know Myself" (Li et al., AAAI
2023) coarse-to-fine persona-aware training framework -- see
plugins/methods/ltkm/method.py's module docstring for the paper summary,
why this is original code rather than a vendored repo (no official code
was ever released), and what "paa-level" means here: real, structured
training/inference code, not yet run end-to-end against real DNLI/ConvAI2
data (that would need downloading DNLI, ConvAI2, and real GPU time -- see
the module docstring for exactly what's smoke-tested vs. not).
"""
