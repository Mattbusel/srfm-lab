"""
genealogy — Evolutionary lineage tracking for the Idea Automation Engine.

Tracks parent/child relationships, mutation history, and performance arcs
for every genome produced by the evolutionary engine.

Public API
----------
::

    from genealogy.tree       import GenealogyTree
    from genealogy.visualizer import GenealogyVisualizer
    from genealogy.tracker    import EvolutionTracker

Example
-------
::

    tree = GenealogyTree(db_path="idea_engine.db")
    tree.add_genome(
        genome_id=42,
        parent_ids=[17, 23],
        mutation_ops=["crossover", "mutate_threshold"],
        params={"fast_period": 12, "slow_period": 48},
        fitness=1.84,
        island="alpha",
        generation=5,
    )
    lineage = tree.get_lineage(42)
    best    = tree.best_branch()
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("idea-engine-genealogy")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "GenealogyTree",
    "GenealogyVisualizer",
    "EvolutionTracker",
]
