"""
Serendipity Generator — Component of the Idea Automation Engine (IAE)
======================================================================
Generates unexpected, creative trading ideas by combining signals from
disparate domains, applying random creative mutations, and drawing
structural analogies between scientific fields and trading strategy.

Public API
----------
    from serendipity import SerendipityGenerator, AnalogyEngine, StrategyMutator

Typical usage::

    from serendipity import SerendipityGenerator

    gen   = SerendipityGenerator(db_path="idea_engine.db")
    ideas = gen.generate_wild_ideas(n=5)
    for idea in ideas:
        print(idea.description)
        print("  Experiment:", idea.suggested_experiment)
"""

from .generator      import SerendipityGenerator, WildIdea
from .analogy_engine import AnalogyEngine, Analogy
from .mutation_engine import StrategyMutator, MutatedStrategy

__all__ = [
    "SerendipityGenerator",
    "WildIdea",
    "AnalogyEngine",
    "Analogy",
    "StrategyMutator",
    "MutatedStrategy",
]

__version__ = "0.1.0"
