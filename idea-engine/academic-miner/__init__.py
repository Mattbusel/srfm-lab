"""
Academic Miner — Component of the Idea Automation Engine (IAE)
===============================================================
Mines arXiv, SSRN, and a local paper corpus for ideas applicable
to the BH (Black Hole) physics-inspired trading strategy.

Public API
----------
    from academic_miner import ArXivMiner, SSRNMiner, IdeaExtractor, LocalCorpusMiner

Typical usage::

    from academic_miner import ArXivMiner, IdeaExtractor

    miner   = ArXivMiner(db_path="idea_engine.db")
    papers  = miner.search("momentum trading", max_results=10)
    miner.store_papers(papers)

    extractor = IdeaExtractor(db_path="idea_engine.db")
    for p in papers:
        candidates = extractor.extract_hypothesis(p.abstract)
        extractor.store_candidates(candidates, source_paper_id=p.db_id)
"""

from .arxiv_miner     import ArXivMiner
from .ssrn_miner      import SSRNMiner
from .idea_extractor  import IdeaExtractor, IdeaCandidate, HypothesisTemplate
from .local_corpus    import LocalCorpusMiner

__all__ = [
    "ArXivMiner",
    "SSRNMiner",
    "IdeaExtractor",
    "IdeaCandidate",
    "HypothesisTemplate",
    "LocalCorpusMiner",
]

__version__ = "0.1.0"
