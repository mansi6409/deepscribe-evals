#!/usr/bin/env python3
"""
DeepScribe Evals Suite - CLI Entry Point

Usage:
 python run_eval.py --mode fast --num-cases 10
 python run_eval.py --mode standard --num-cases 100
 python run_eval.py --mode thorough --num-cases 20
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data_loader import DataLoader
from src.pipeline import EvaluationPipeline
from src.utils.config import config


def main():
    parser = argparse.ArgumentParser(description='DeepScribe SOAP Note Evaluation Suite')

    parser.add_argument(
    '--mode',
    type=str,
    choices=['fast', 'standard', 'thorough'],
    default='standard',
    help='Evaluation mode (fast=Tier1, standard=Tier1+2, thorough=all)'
    )

    parser.add_argument(
    '--num-cases',
    type=int,
    default=100,
    help='Number of cases to evaluate'
    )

    parser.add_argument(
    '--dataset',
    type=str,
    default='omi-health/medical-dialogue-to-soap-summary',
    help='Hugging Face dataset name'
    )

    parser.add_argument(
    '--output-dir',
    type=str,
    default='results',
    help='Output directory for results'
    )

    parser.add_argument(
    '--add-synthetic',
    action='store_true',
    help='Add synthetic error cases for validation'
    )

    parser.add_argument(
    '--parallel',
    action='store_true',
    help='Use parallel processing (fast mode only)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"\n{'='*60}")
    print(f"DEEPSCRIBE EVALS SUITE")
    print(f"{'='*60}\n")

    loader = DataLoader(
    dataset_name=args.dataset,
    sample_size=args.num_cases
    )

    eval_inputs = loader.load_dataset()
    
    if not eval_inputs:
        print("No data loaded. Exiting.")
        return

    # Add synthetic error cases if requested
    if args.add_synthetic:
        print("\nGenerating synthetic error cases for validation...")
        from src.data_loader import SyntheticErrorGenerator
        generator = SyntheticErrorGenerator()

        # Take first 5 cases and generate variants
        synthetic_inputs = []
        for inp in eval_inputs[:5]:
            variants = generator.generate_error_variants(inp)
            synthetic_inputs.extend(variants.values())

        eval_inputs.extend(synthetic_inputs)
        print(f" Added {len(synthetic_inputs)} synthetic error cases")

    # Initialize pipeline
    pipeline = EvaluationPipeline(mode=args.mode)

    # Run evaluation
    batch_results = pipeline.evaluate_batch(
    eval_inputs,
    parallel=args.parallel
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"eval_results_{args.mode}_{timestamp}.json"

    # Convert to dict for JSON serialization
    results_dict = {
    'metadata': {
    'mode': args.mode,
    'num_cases': len(eval_inputs),
    'dataset': args.dataset,
    'timestamp': timestamp
    },
    'aggregate': {
    'total_cases': batch_results.total_cases,
    'mean_composite': batch_results.mean_composite,
    'mean_missing_rate': batch_results.mean_missing_rate,
    'mean_hallucination_rate': batch_results.mean_hallucination_rate,
    'mean_contradicted_rate': batch_results.mean_contradicted_rate,
    'total_runtime_seconds': batch_results.total_runtime_seconds,
    'total_cost_usd': batch_results.total_cost_usd,
    'most_common_missing': batch_results.most_common_missing,
    'most_common_hallucinated': batch_results.most_common_hallucinated
    },
    'cases': [result.model_dump() for result in batch_results.results]
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f" Results saved to: {output_file}")

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Cases evaluated: {batch_results.total_cases}")
    print(f"Mean composite score: {batch_results.mean_composite:.3f} (higher is better)")
    print(f"Runtime: {batch_results.total_runtime_seconds:.2f}s")
    print(f"Cost: ${batch_results.total_cost_usd:.4f}")

    # Distribution of scores
    scores = [r.metrics.composite for r in batch_results.results]
    print(f"\nScore distribution:")
    print(f" Min: {min(scores):.3f}")
    print(f" Max: {max(scores):.3f}")
    print(f" Median: {sorted(scores)[len(scores)//2]:.3f}")

    print(f"\n{'='*60}")
    print(" Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

