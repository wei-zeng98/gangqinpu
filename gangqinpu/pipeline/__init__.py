"""Gangqinpu construction pipeline: one module per processing stage.

The command-line steps live in ``scripts/``; heavy dependencies (Partitura,
SyMuPe, Transkun, CLAP) are imported only by the modules that need them.
"""
