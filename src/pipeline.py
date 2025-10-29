"""
Main evaluation pipeline orchestrator
"""
from typing import List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.models import EvalInput, EvalOutput, BatchEvalResults
from src.evaluators.tier1.evaluator import Tier1Evaluator
from src.evaluators.tier2.evaluator import Tier2Evaluator
from src.evaluators.tier3.llm_judge import LLMJudge
from src.utils.config import config


class EvaluationPipeline:
    """
    Main pipeline for cascading evaluation
    Orchestrates Tier 1 → Tier 2 → Tier 3 flow
    """
    
    def __init__(self, mode: str = "standard"):
        """
        Initialize pipeline
        
        Args:
            mode: 'fast' (Tier 1 only), 'standard' (Tier 1+2), 'thorough' (all tiers)
        """
        self.mode = mode
        self.enabled_tiers = config.get(f'modes.{mode}.enabled_tiers', [1])
        
        # Initialize evaluators
        print(f"Initializing evaluation pipeline in '{mode}' mode...")
        self.tier1 = Tier1Evaluator()
        
        self.tier2 = None
        if 2 in self.enabled_tiers:
            self.tier2 = Tier2Evaluator()
        
        self.tier3 = None
        if 3 in self.enabled_tiers:
            try:
                self.tier3 = LLMJudge()
            except Exception as e:
                print(f"Warning: Could not initialize Tier 3 (LLM): {e}")
                self.tier3 = None
    
    def evaluate_single(self, eval_input: EvalInput) -> EvalOutput:
        """
        Evaluate a single case through the cascade
        
        Args:
            eval_input: Input case with transcript and note
            
        Returns:
            EvalOutput with findings and metrics
        """
        # Tier 1: Always run
        output = self.tier1.evaluate(eval_input, mode=self.mode)
        
        # Tier 2: Conditional (if Tier 1 found issues)
        if self.tier2 is not None and 2 in self.enabled_tiers:
            output = self.tier2.evaluate(output, eval_input.transcript, eval_input.generated_note)
        
        # Tier 3: Conditional (if Tier 2 found issues)
        if self.tier3 is not None and 3 in self.enabled_tiers:
            output = self.tier3.evaluate(
                output, 
                eval_input.transcript, 
                eval_input.generated_note,
                eval_input.ground_truth_note
            )
        
        return output
    
    def evaluate_batch(self, eval_inputs: List[EvalInput], 
                      parallel: bool = True,
                      max_workers: int = 4) -> BatchEvalResults:
        """
        Evaluate a batch of cases
        
        Args:
            eval_inputs: List of input cases
            parallel: Whether to run Tier 1 in parallel
            max_workers: Number of parallel workers
            
        Returns:
            BatchEvalResults with all outputs and aggregate stats
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Running {self.mode.upper()} evaluation on {len(eval_inputs)} cases")
        print(f"Enabled tiers: {self.enabled_tiers}")
        print(f"{'='*60}\n")
        
        results = []
        
        # Evaluate all cases
        if parallel and self.mode == "fast":
            # Parallel only for fast mode (Tier 1 only, no API calls)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_single, inp): inp 
                    for inp in eval_inputs
                }
                
                for future in tqdm(as_completed(futures), total=len(eval_inputs), desc="Evaluating"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Error evaluating case: {e}")
        else:
            # Sequential (for modes with API calls)
            for inp in tqdm(eval_inputs, desc="Evaluating"):
                try:
                    result = self.evaluate_single(inp)
                    results.append(result)
                except Exception as e:
                    print(f"Error evaluating case {inp.case_id}: {e}")
        
        # Create batch results
        batch_results = BatchEvalResults(results=results)
        batch_results.compute_statistics()
        
        total_time = time.time() - start_time
        
        # Print summary
        self._print_summary(batch_results, total_time)
        
        return batch_results
    
    def _print_summary(self, batch_results: BatchEvalResults, total_time: float):
        """Print evaluation summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total cases: {batch_results.total_cases}")
        print(f"Total time: {total_time:.2f}s ({total_time/batch_results.total_cases:.2f}s per case)")
        print(f"Total cost: ${batch_results.total_cost_usd:.4f}")
        print(f"\n--- Aggregate Metrics ---")
        print(f"Mean composite score: {batch_results.mean_composite:.3f}")
        print(f"Mean missing rate: {batch_results.mean_missing_rate:.3f}")
        print(f"Mean hallucination rate: {batch_results.mean_hallucination_rate:.3f}")
        print(f"Mean contradiction rate: {batch_results.mean_contradicted_rate:.3f}")
        
        if batch_results.most_common_missing:
            print(f"\n--- Most Common Missing Entities ---")
            for i, entity in enumerate(batch_results.most_common_missing[:5], 1):
                print(f"{i}. {entity}")
        
        if batch_results.most_common_hallucinated:
            print(f"\n--- Most Common Hallucinated Entities ---")
            for i, entity in enumerate(batch_results.most_common_hallucinated[:5], 1):
                print(f"{i}. {entity}")
        
        print(f"{'='*60}\n")

