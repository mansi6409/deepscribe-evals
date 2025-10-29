"""
Pydantic data models for DeepScribe Evals Suite
"""
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceSpan(BaseModel):
    """Evidence from transcript supporting or contradicting a claim"""
    text: str
    section: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class Finding(BaseModel):
    """A single evaluation finding (missing, hallucinated, contradicted, etc.)"""
    type: Literal["missing", "hallucinated", "contradicted", "inaccurate", "unsupported"]
    claim_or_entity: str
    section: str
    evidence_span: Optional[str] = None  # "NO_EVIDENCE" or actual span
    severity: Literal["critical", "major", "minor"] = "minor"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    detected_by_tier: int = Field(ge=1, le=3)


class SectionCoverage(BaseModel):
    """SOAP section presence and completeness"""
    subjective: bool = False
    objective: bool = False
    assessment: bool = False
    plan: bool = False
    
    @property
    def num_present(self) -> int:
        return sum([self.subjective, self.objective, self.assessment, self.plan])
    
    @property
    def gap_penalty(self) -> float:
        """Penalty for missing sections (0-1, higher is worse)"""
        return (4 - self.num_present) / 4


class Metrics(BaseModel):
    """Quantitative evaluation metrics"""
    missing_rate_critical: float = Field(ge=0.0, le=1.0, default=0.0)
    hallucination_rate_critical: float = Field(ge=0.0, le=1.0, default=0.0)
    contradicted_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    unsupported_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    composite: float = Field(ge=0.0, le=1.0, default=1.0)
    
    tier1_score: Optional[float] = None
    tier2_score: Optional[float] = None
    tier3_score: Optional[float] = None
    
    runtime_seconds: float = 0.0
    cost_usd: float = 0.0


class Meta(BaseModel):
    """Metadata about the evaluation run"""
    model_config = {"protected_namespaces": ()}  # Allow model_ prefix
    
    model_version: str = "v1.0.0"
    data_hash: Optional[str] = None
    eval_mode: Literal["fast", "standard", "thorough"] = "standard"
    timestamp: datetime = Field(default_factory=datetime.now)
    tiers_executed: List[int] = Field(default_factory=list)


class EvalInput(BaseModel):
    """Input for a single evaluation case"""
    case_id: str
    transcript: str
    generated_note: str
    ground_truth_note: Optional[str] = None


class EvalOutput(BaseModel):
    """Complete evaluation output for a single case"""
    case_id: str
    
    # Findings
    missing_critical: List[Finding] = Field(default_factory=list)
    hallucinated: List[Finding] = Field(default_factory=list)
    contradicted: List[Finding] = Field(default_factory=list)
    inaccurate: List[Finding] = Field(default_factory=list)
    unsupported: List[Finding] = Field(default_factory=list)
    
    # Section analysis
    section_coverage: SectionCoverage = Field(default_factory=SectionCoverage)
    
    # Metrics
    metrics: Metrics = Field(default_factory=Metrics)
    
    # Metadata
    meta: Meta = Field(default_factory=Meta)
    
    def all_findings(self) -> List[Finding]:
        """Get all findings across all categories"""
        return (
            self.missing_critical + 
            self.hallucinated + 
            self.contradicted + 
            self.inaccurate + 
            self.unsupported
        )
    
    def critical_findings(self) -> List[Finding]:
        """Get only critical severity findings"""
        return [f for f in self.all_findings() if f.severity == "critical"]


class BatchEvalResults(BaseModel):
    """Results for a batch of evaluations"""
    results: List[EvalOutput] = Field(default_factory=list)
    
    # Aggregate statistics
    total_cases: int = 0
    mean_composite: float = 0.0
    mean_missing_rate: float = 0.0
    mean_hallucination_rate: float = 0.0
    mean_contradicted_rate: float = 0.0
    
    total_runtime_seconds: float = 0.0
    total_cost_usd: float = 0.0
    
    # Top issues
    most_common_missing: List[str] = Field(default_factory=list)
    most_common_hallucinated: List[str] = Field(default_factory=list)
    
    def compute_statistics(self):
        """Compute aggregate statistics from individual results"""
        if not self.results:
            return
        
        self.total_cases = len(self.results)
        self.mean_composite = sum(r.metrics.composite for r in self.results) / self.total_cases
        self.mean_missing_rate = sum(r.metrics.missing_rate_critical for r in self.results) / self.total_cases
        self.mean_hallucination_rate = sum(r.metrics.hallucination_rate_critical for r in self.results) / self.total_cases
        self.mean_contradicted_rate = sum(r.metrics.contradicted_rate for r in self.results) / self.total_cases
        
        self.total_runtime_seconds = sum(r.metrics.runtime_seconds for r in self.results)
        self.total_cost_usd = sum(r.metrics.cost_usd for r in self.results)
        
        # Find most common issues
        from collections import Counter
        missing_entities = [f.claim_or_entity for r in self.results for f in r.missing_critical]
        halluc_entities = [f.claim_or_entity for r in self.results for f in r.hallucinated]
        
        self.most_common_missing = [item for item, _ in Counter(missing_entities).most_common(10)]
        self.most_common_hallucinated = [item for item, _ in Counter(halluc_entities).most_common(10)]

