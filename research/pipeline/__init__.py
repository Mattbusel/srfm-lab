# research/pipeline/__init__.py
# SRFM -- Signal Research and Factor Management pipeline
from .signal_research_pipeline import SignalResearchPipeline, SignalUniverse, ResearchResult
from .factor_construction import FactorBuilder, FactorCombiner, FactorNeutralizer
from .cross_sectional_study import CrossSectionalStudy, QuintileResult, FamaMacBethResult
