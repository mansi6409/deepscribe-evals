"""
Tier 3: LLM-as-a-Judge using Gemini
"""
import time
import json
from typing import List, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import EvalOutput, Finding
from src.utils.config import config


class LLMJudge:
    """LLM-based evaluation using Gemini"""
    
    def __init__(self):
        self.api_key = config.gemini_api_key
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            config.get('tier3.primary.model', 'gemini-1.5-flash')
        )
        
        self.temperature = config.get('tier3.primary.temperature', 0.1)
        self.max_tokens = config.get('tier3.primary.max_tokens', 2000)
    
    def evaluate(self, eval_output: EvalOutput, transcript: str, note: str, 
                 ground_truth: Optional[str] = None) -> EvalOutput:
        """
        Run Tier 3 evaluation (LLM judge)
        Only runs if Tier 2 detected significant issues
        """
        start_time = time.time()
        
        # Check if we should run Tier 3
        tier2_score = eval_output.metrics.tier2_score
        gate2 = config.get('tier3.thresholds.trigger_score', 0.15)
        
        if tier2_score is not None and tier2_score > (1.0 - gate2):
            # Tier 2 was good, skip Tier 3
            return eval_output
        
        # Run LLM evaluation
        llm_findings = self._run_llm_evaluation(transcript, note, ground_truth)
        
        # Add findings to output
        for finding in llm_findings:
            if finding.type == "missing":
                eval_output.missing_critical.append(finding)
            elif finding.type == "hallucinated":
                eval_output.hallucinated.append(finding)
            elif finding.type == "contradicted":
                eval_output.contradicted.append(finding)
            elif finding.type == "inaccurate":
                eval_output.inaccurate.append(finding)
        
        # Update metrics
        eval_output.metrics.tier3_score = self._compute_tier3_score(llm_findings)
        eval_output.meta.tiers_executed.append(3)
        
        # Recompute composite with LLM findings
        eval_output.metrics = self._update_metrics(eval_output, llm_findings)
        
        # Add runtime and cost
        eval_output.metrics.runtime_seconds += time.time() - start_time
        eval_output.metrics.cost_usd += 0.0  # Free tier
        
        return eval_output
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _run_llm_evaluation(self, transcript: str, note: str, 
                           ground_truth: Optional[str] = None) -> List[Finding]:
        """Call Gemini to evaluate note"""
        
        prompt = self._build_prompt(transcript, note, ground_truth)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            findings_data = json.loads(response_text)
            
            # Convert to Finding objects
            findings = []
            for item in findings_data.get('findings', []):
                findings.append(Finding(
                    type=item.get('type', 'inaccurate'),
                    claim_or_entity=item.get('claim', ''),
                    section=item.get('section', 'Unknown'),
                    evidence_span=item.get('evidence_span', 'NO_EVIDENCE'),
                    severity=item.get('severity', 'major'),
                    confidence=0.9,  # High confidence for LLM
                    detected_by_tier=3
                ))
            
            return findings
            
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return []
    
    def _build_prompt(self, transcript: str, note: str, ground_truth: Optional[str]) -> str:
        """Build structured prompt for LLM"""
        
        prompt = f"""You are a strict medical QA assistant evaluating AI-generated SOAP notes against clinical transcripts.

TASK: Identify errors in three categories:
1. MISSING: Critical clinical facts in transcript omitted from note (e.g., new symptoms, medication changes, abnormal vitals)
2. HALLUCINATED: Facts in note not supported by transcript
3. INACCURATE: Clinically incorrect, contradicted, or dangerous (e.g., wrong dosage, contradictory diagnosis)

CRITICAL SECTIONS: Pay extra attention to Allergies, Medications, Vitals, Assessment, Plan.

OUTPUT FORMAT: JSON object with array of findings:
{{
  "findings": [
    {{
      "type": "missing" | "hallucinated" | "inaccurate",
      "claim": "the specific claim or entity",
      "section": "Subjective" | "Objective" | "Assessment" | "Plan",
      "evidence_span": "VERBATIM quote from transcript, or NO_EVIDENCE",
      "severity": "critical" | "major" | "minor"
    }}
  ]
}}

ONLY flag genuine errors. Normal clinical summarization (paraphrasing, reordering) is acceptable.

=== TRANSCRIPT ===
{transcript}

=== GENERATED NOTE ===
{note}
"""
        
        if ground_truth:
            prompt += f"\n=== GROUND TRUTH NOTE (reference) ===\n{ground_truth}\n"
        
        prompt += "\nOutput your evaluation as JSON:"
        
        return prompt
    
    def _compute_tier3_score(self, findings: List[Finding]) -> float:
        """Compute score from LLM findings"""
        if not findings:
            return 1.0
        
        critical_count = len([f for f in findings if f.severity == "critical"])
        major_count = len([f for f in findings if f.severity == "major"])
        
        # Penalize more for critical findings
        penalty = (critical_count * 0.2) + (major_count * 0.1)
        score = max(0.0, 1.0 - penalty)
        
        return score
    
    def _update_metrics(self, eval_output: EvalOutput, llm_findings: List[Finding]):
        """Update metrics with LLM findings"""
        metrics = eval_output.metrics
        
        # Count unsafe LLM flags
        unsafe_count = len([f for f in llm_findings if f.severity == "critical"])
        
        weights = config.get('scoring.weights', {})
        w1 = weights.get('w1_missing_critical', 0.35)
        w2 = weights.get('w2_hallucinated_critical', 0.35)
        w3 = weights.get('w3_contradicted', 0.20)
        w4 = weights.get('w4_unsafe_llm_flags', 0.05)
        w5 = weights.get('w5_section_gap_penalty', 0.05)
        
        llm_penalty = min(1.0, unsafe_count / 5.0)  # Normalize
        
        metrics.composite = 1.0 - (
            w1 * metrics.missing_rate_critical +
            w2 * metrics.hallucination_rate_critical +
            w3 * metrics.contradicted_rate +
            w4 * llm_penalty +
            w5 * eval_output.section_coverage.gap_penalty
        )
        
        metrics.composite = max(0.0, min(1.0, metrics.composite))
        
        return metrics

