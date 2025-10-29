"""
Tier 1: Deterministic Evaluation
"""
import time
from typing import List
from src.models import EvalInput, EvalOutput, Finding, SectionCoverage, Metrics, Meta
from src.evaluators.tier1.entity_extractor import MedicalEntityExtractor
from src.evaluators.tier1.semantic_checker import SemanticChecker
from src.utils.text_processing import sentence_split, detect_soap_sections
from src.utils.config import config


class Tier1Evaluator:
    """Fast deterministic checks"""
    
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        self.semantic_checker = SemanticChecker(
            model_name=config.get('tier1.embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        self.threshold = config.get('tier1.thresholds.unsupported_similarity', 0.72)
    
    def evaluate(self, eval_input: EvalInput, mode: str = "standard") -> EvalOutput:
        """
        Run Tier 1 evaluation
        Returns EvalOutput with findings and scores
        """
        start_time = time.time()
        
        # Initialize output
        output = EvalOutput(case_id=eval_input.case_id)
        output.meta.eval_mode = mode
        output.meta.tiers_executed = [1]
        
        # 1. Entity extraction and comparison
        missing, hallucinated = self._check_entities(eval_input)
        output.missing_critical = missing
        output.hallucinated = hallucinated
        
        # 2. Section structure check
        output.section_coverage = self._check_sections(eval_input.generated_note)
        
        # 3. Semantic support check
        unsupported = self._check_semantic_support(eval_input)
        output.unsupported = unsupported
        
        # 4. Range/sanity checks
        # (Could add vital range checks here if needed, skipping for now for the sake of assessment)
        
        # 5. Compute Tier 1 score
        tier1_score = self._compute_tier1_score(output)
        output.metrics.tier1_score = tier1_score
        
        # 6. Compute rates
        output.metrics = self._compute_metrics(output, eval_input)
        
        # Runtime
        output.metrics.runtime_seconds = time.time() - start_time
        output.metrics.cost_usd = 0.0  # Free
        
        return output
    
    def _check_entities(self, eval_input: EvalInput) -> tuple:
        """
        Check for missing and hallucinated entities
        Returns (missing_findings, hallucinated_findings)
        """
        missing_findings = []
        hallucinated_findings = []
        
        # Extract entities
        tx_entities = self.entity_extractor.extract_entities(eval_input.transcript)
        note_entities = self.entity_extractor.extract_entities(eval_input.generated_note)
        
        # Compare
        missing_set, halluc_set = self.entity_extractor.compare_entities(
            eval_input.transcript, 
            eval_input.generated_note
        )
        
        # Create findings for missing entities
        for entity in missing_set:
            # Determine if critical
            is_critical = any(
                self.entity_extractor.is_critical_entity(e['text'], e['label']) 
                for e in tx_entities if e['text'].lower() == entity
            )
            
            if is_critical or len(missing_findings) < 5:  # Limit non-critical
                missing_findings.append(Finding(
                    type="missing",
                    claim_or_entity=entity,
                    section="Unknown",
                    evidence_span=entity,  # Found in transcript
                    severity="critical" if is_critical else "major",
                    confidence=config.get('tier1.confidence.missing_entity', 0.8),
                    detected_by_tier=1
                ))
        
        # Create findings for hallucinated entities
        for entity in halluc_set:
            # Check if it's a paraphrase or truly hallucinated
            is_critical = any(
                self.entity_extractor.is_critical_entity(e['text'], e['label']) 
                for e in note_entities if e['text'].lower() == entity
            )
            
            if is_critical or len(hallucinated_findings) < 5:
                hallucinated_findings.append(Finding(
                    type="hallucinated",
                    claim_or_entity=entity,
                    section="Unknown",
                    evidence_span="NO_EVIDENCE",
                    severity="critical" if is_critical else "major",
                    confidence=config.get('tier1.confidence.hallucinated_entity', 0.7),
                    detected_by_tier=1
                ))
        
        return missing_findings, hallucinated_findings
    
    def _check_sections(self, note: str) -> SectionCoverage:
        """Check for SOAP section presence"""
        sections = detect_soap_sections(note)
        
        return SectionCoverage(
            subjective=sections['subjective'],
            objective=sections['objective'],
            assessment=sections['assessment'],
            plan=sections['plan']
        )
    
    def _check_semantic_support(self, eval_input: EvalInput) -> List[Finding]:
        """Check if note claims are supported by transcript"""
        unsupported_findings = []
        
        # Split into sentences
        note_sentences = sentence_split(eval_input.generated_note)
        tx_sentences = sentence_split(eval_input.transcript)
        
        # Check support
        unsupported = self.semantic_checker.check_support(
            note_sentences, 
            tx_sentences,
            threshold=self.threshold
        )
        
        # Convert to findings (limit to top 10)
        for note_sent, best_match, similarity in unsupported[:10]:
            unsupported_findings.append(Finding(
                type="unsupported",
                claim_or_entity=note_sent,
                section="Unknown",
                evidence_span=best_match if best_match != "NO_EVIDENCE" else "NO_EVIDENCE",
                severity="major" if similarity < config.get('tier1.thresholds.severity_major_threshold', 0.5) else "minor",
                confidence=1.0 - similarity,
                detected_by_tier=1
            ))
        
        return unsupported_findings
    
    def _compute_tier1_score(self, output: EvalOutput) -> float:
        """
        Compute Tier 1 risk score (higher = more problems)
        Returns value in [0, 1] where 1 = perfect, 0 = many issues
        """
        # Count critical issues
        num_missing = len([f for f in output.missing_critical if f.severity == "critical"])
        num_halluc = len([f for f in output.hallucinated if f.severity == "critical"])
        num_unsupported = len([f for f in output.unsupported if f.severity in ["critical", "major"]])
        
        # Section gap penalty
        section_gap = output.section_coverage.gap_penalty
        
        # Compute score (inverse of problems)
        weights = config.get('tier1.internal_weights', {})
        w_missing = weights.get('missing', 0.4)
        w_halluc = weights.get('hallucinated', 0.4)
        w_unsupported = weights.get('unsupported', 0.1)
        w_section = weights.get('section_gap', 0.1)
        normalization = weights.get('normalization', 10.0)
        
        problems = (
            num_missing * w_missing +
            num_halluc * w_halluc +
            num_unsupported * w_unsupported +
            section_gap * w_section
        )
        
        # Normalize to [0, 1], where 1 = no problems
        score = max(0.0, 1.0 - (problems / normalization))
        
        return score
    
    def _compute_metrics(self, output: EvalOutput, eval_input: EvalInput) -> Metrics:
        """Compute final metrics"""
        metrics = output.metrics
        
        # Total entities extracted from transcript
        tx_entities = self.entity_extractor.extract_entities(eval_input.transcript)
        total_entities = len(tx_entities)
        
        if total_entities > 0:
            critical_missing = len([f for f in output.missing_critical if f.severity == "critical"])
            metrics.missing_rate_critical = critical_missing / max(total_entities, 1)
        
        # Hallucination rate
        note_entities = self.entity_extractor.extract_entities(eval_input.generated_note)
        total_note_entities = len(note_entities)
        
        if total_note_entities > 0:
            critical_halluc = len([f for f in output.hallucinated if f.severity == "critical"])
            metrics.hallucination_rate_critical = critical_halluc / max(total_note_entities, 1)
        
        # Unsupported rate
        note_sentences = sentence_split(eval_input.generated_note)
        if note_sentences:
            metrics.unsupported_rate = len(output.unsupported) / len(note_sentences)
        
        # Contradicted rate (placeholder for Tier 2)
        metrics.contradicted_rate = 0.0
        
        # Composite score
        weights = config.get('scoring.weights', {})
        w1 = weights.get('w1_missing_critical', 0.35)
        w2 = weights.get('w2_hallucinated_critical', 0.35)
        w3 = weights.get('w3_contradicted', 0.20)
        w4 = weights.get('w4_unsafe_llm_flags', 0.05)
        w5 = weights.get('w5_section_gap_penalty', 0.05)
        
        metrics.composite = 1.0 - (
            w1 * metrics.missing_rate_critical +
            w2 * metrics.hallucination_rate_critical +
            w3 * metrics.contradicted_rate +
            w5 * output.section_coverage.gap_penalty
        )
        
        metrics.composite = max(0.0, min(1.0, metrics.composite))
        
        return metrics

