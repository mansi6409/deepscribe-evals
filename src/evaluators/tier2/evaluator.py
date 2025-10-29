"""
Tier 2: NLI + Retrieval-based Validation
"""
import time
from typing import List
from src.models import EvalOutput, Finding
from src.utils.text_processing import sentence_split
from src.utils.config import config


class Tier2Evaluator:
    """NLI-based contradiction detection"""

    def __init__(self):
        self.nli_model = None
        self.tokenizer = None
        self._load_nli_model()
        self.contradiction_threshold = config.get('tier2.thresholds.contradiction_confidence', 0.8)

    def _load_nli_model(self):
        """Lazy load NLI model"""
        try:
            from transformers import pipeline
            self.nli_model = pipeline(
            "text-classification",
            model=config.get('tier2.nli_model', 'facebook/bart-large-mnli'),
            device=-1 # CPU
            )
            print(f" Loaded NLI model")
        except Exception as e:
            print(f"Warning: Could not load NLI model: {e}")
            self.nli_model = None

    def evaluate(self, eval_output: EvalOutput, transcript: str, note: str) -> EvalOutput:
        """
        Run Tier 2 evaluation (conditional on Tier 1 results)
        Adds contradiction findings to existing EvalOutput
        """
        start_time = time.time()

        # Check if we should run Tier 2
        tier1_score = eval_output.metrics.tier1_score or 1.0
        gate1 = config.get('tier2.thresholds.trigger_score', 0.25)

        if tier1_score > (1.0 - gate1): # Convert to risk score
            # Tier 1 was good, skip Tier 2
            return eval_output

        # Run NLI checks
        if self.nli_model is not None:
            contradictions = self._check_contradictions(transcript, note)
        eval_output.contradicted.extend(contradictions)

        # Update metrics
        eval_output.metrics.contradicted_rate = len(eval_output.contradicted) / max(
        len(sentence_split(note)), 1
        )

        # Recompute composite
        eval_output.metrics = self._update_metrics(eval_output)

        # Update tier 2 score
        eval_output.metrics.tier2_score = 1.0 - eval_output.metrics.contradicted_rate
        eval_output.meta.tiers_executed.append(2)

        # Add runtime
        eval_output.metrics.runtime_seconds += time.time() - start_time

        return eval_output

    def _check_contradictions(self, transcript: str, note: str) -> List[Finding]:
        """Use NLI to detect contradictions"""
        contradictions = []

        note_sentences = sentence_split(note)
        tx_sentences = sentence_split(transcript)

        # Check each note sentence against transcript
        for note_sent in note_sentences[:20]: # Limit for speed
            if len(note_sent.strip()) < 15:
                continue

        # Find potential contradictions
        for tx_sent in tx_sentences:
            if len(tx_sent.strip()) < 10:
                continue

            # Run NLI
            result = self._predict_nli(tx_sent, note_sent)

            if result['label'] == 'contradiction' and result['score'] > self.contradiction_threshold:
                contradictions.append(Finding(
                    type="contradicted",
                    claim_or_entity=note_sent,
                    section="Unknown",
                    evidence_span=tx_sent,
                    severity="critical",
                    confidence=result['score'],
                    detected_by_tier=2
                    ))
                break # Found contradiction, move to next note sentence

        return contradictions

    def _predict_nli(self, premise: str, hypothesis: str) -> dict:
        """
        Predict NLI relationship
        Returns {label: 'entailment'|'contradiction'|'neutral', score: float}
        """
        if self.nli_model is None:
            return {'label': 'neutral', 'score': 0.5}

        try:
            # Format for NLI model
            input_text = f"{premise} [SEP] {hypothesis}"
            result = self.nli_model(input_text)[0]

            return {
            'label': result['label'].lower(),
            'score': result['score']
            }
        except:
            return {'label': 'neutral', 'score': 0.5}

    def _update_metrics(self, eval_output: EvalOutput):
        """Recompute metrics with contradiction rate"""
        metrics = eval_output.metrics

        weights = config.get('scoring.weights', {})
        w1 = weights.get('w1_missing_critical', 0.35)
        w2 = weights.get('w2_hallucinated_critical', 0.35)
        w3 = weights.get('w3_contradicted', 0.20)
        w5 = weights.get('w5_section_gap_penalty', 0.05)

        metrics.composite = 1.0 - (
        w1 * metrics.missing_rate_critical +
        w2 * metrics.hallucination_rate_critical +
        w3 * metrics.contradicted_rate +
        w5 * eval_output.section_coverage.gap_penalty
        )

        metrics.composite = max(0.0, min(1.0, metrics.composite))

        return metrics

