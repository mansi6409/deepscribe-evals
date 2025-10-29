"""
Data loading and synthetic perturbation generation
"""
from typing import List, Dict, Optional
import random
import re
from pathlib import Path

# Make datasets import optional
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except Exception as e:
    print(f"Note: datasets library not available ({e}). Will use synthetic data.")
    DATASETS_AVAILABLE = False
    load_dataset = None

try:
    from tqdm import tqdm
except:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable

from src.models import EvalInput
from src.utils.text_processing import extract_soap_sections


class DataLoader:
    """Load and prepare evaluation datasets from Hugging Face or synthetic sources.

    This class handles loading medical dialogue-to-SOAP datasets, with automatic
    fallback to synthetic data generation if the datasets library is unavailable
    or if there are loading errors. Supports sampling for efficient evaluation.

    Attributes:
    dataset_name (str): Name of the Hugging Face dataset to load
    sample_size (int): Maximum number of cases to load
    cache_dir (Path): Directory for caching processed data
    """

    def __init__(self, dataset_name: str = "omi-health/medical-dialogue-to-soap-summary", 
    sample_size: int = 100,
    cache_dir: str = "data/processed"):
        """Initialize the data loader.

        Args:
        dataset_name (str): Hugging Face dataset identifier. Defaults to 
        omi-health/medical-dialogue-to-soap-summary
        sample_size (int): Number of cases to sample from dataset. Defaults to 100
        cache_dir (str): Directory path for caching processed data. Defaults to data/processed
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def load_dataset(self) -> List[EvalInput]:
        """Load dataset from Hugging Face or generate synthetic data as fallback.

        Attempts to load the specified dataset from Hugging Face. If the datasets
        library is unavailable or loading fails, automatically falls back to
        generating synthetic test cases.

        Returns:
        List[EvalInput]: List of evaluation input cases, each containing:
        - case_id: Unique identifier
        - transcript: Patient-doctor dialogue
        - generated_note: SOAP note to evaluate
        - ground_truth_note: Reference note (if available)

        Raises:
        None: Errors are caught and trigger fallback to synthetic data

        Note:
        If loading succeeds, randomly samples up to sample_size cases.
        """
        # Check if datasets library is available
        if not DATASETS_AVAILABLE:
            print("Datasets library not available. Using synthetic data...")
            return self._generate_synthetic_data()

        print(f"Loading dataset: {self.dataset_name}")

        try:
        # Load from Hugging Face
            dataset = load_dataset(self.dataset_name, split="train")

        # Sample if needed
            if len(dataset) > self.sample_size:
                indices = random.sample(range(len(dataset)), self.sample_size)
                dataset = dataset.select(indices)
            
            # Convert to EvalInput format
            eval_inputs = []
            for idx, item in enumerate(tqdm(dataset, desc="Processing data")):
            # Try different field names (datasets vary)
                transcript = self._extract_field(item, ['dialogue', 'transcript', 'conversation', 'text'])
                note = self._extract_field(item, ['soap', 'note', 'summary', 'soap_note'])
                
                if transcript and note:
                    eval_inputs.append(EvalInput(
                    case_id=f"case_{idx:04d}",
                    transcript=transcript,
                    generated_note=note,
                    ground_truth_note=note # For now, use same as ground truth
                    ))
            
            print(f" Loaded {len(eval_inputs)} cases")
            return eval_inputs

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data generation...")
            return self._generate_synthetic_data()

    def _extract_field(self, item: dict, possible_keys: List[str]) -> Optional[str]:
        """Try to extract field from dataset item using multiple possible key names.

        Different datasets use different field names for the same concept
        (e.g., 'dialogue', 'transcript', 'conversation' all refer to the same thing).
        This method tries each possible key in order.

        Args:
        item (dict): Dataset item to extract from
        possible_keys (List[str]): List of potential field names to try

        Returns:
        Optional[str]: Field value if found, None otherwise
        """
        for key in possible_keys:
            if key in item and item[key]:
                return str(item[key])
        return None

    def _generate_synthetic_data(self) -> List[EvalInput]:
        """Generate synthetic medical dialogue-to-SOAP cases for testing.

        Creates realistic synthetic cases covering common scenarios:
        - Chest pain evaluation with cardiac workup
        - Routine medication refill for chronic disease

        These cases are used when real data is unavailable or for
        testing the evaluation pipeline without external dependencies.

        Returns:
        List[EvalInput]: List of synthetic evaluation cases
        """
        synthetic_cases = [
            {
            'case_id': 'synthetic_001',
            'transcript': """
            Doctor: Hello, what brings you in today?
            Patient: I've been having chest pain for the last two days.
            Doctor: Can you describe the pain?
            Patient: It's a sharp pain in the center of my chest, gets worse when I breathe deeply.
            Doctor: Any shortness of breath?
            Patient: Yes, a little.
            Doctor: Any history of heart problems?
            Patient: My father had a heart attack at age 55.
            Doctor: Let me check your vitals. Blood pressure is 140/90, heart rate 88, temperature 98.6F.
            Doctor: I'm going to prescribe you aspirin 81mg daily and refer you to cardiology.
            Patient: Okay, thank you.
            """,
            'generated_note': """
            SUBJECTIVE:
            Patient presents with chest pain for 2 days. Sharp, central chest pain, worse with deep breathing. 
            Associated shortness of breath. Family history of MI (father at age 55).

            OBJECTIVE:
            Vitals: BP 140/90, HR 88, Temp 98.6F

            ASSESSMENT:
            Chest pain, likely musculoskeletal vs cardiac etiology. Family history significant for CAD.

            PLAN:
            1. Aspirin 81mg daily
            2. Cardiology referral
            3. Follow up in 1 week
            """
            },
            {
            'case_id': 'synthetic_002',
            'transcript': """
            Doctor: How can I help you?
            Patient: I need a refill on my diabetes medication.
            Doctor: What medication are you taking?
            Patient: Metformin 500mg twice a day.
            Doctor: How's your blood sugar control?
            Patient: Pretty good, usually around 120-130.
            Doctor: Any side effects from the medication?
            Patient: No, feeling fine.
            Doctor: Great, I'll send the refill to your pharmacy.
            """,
            'generated_note': """
            SUBJECTIVE:
            Patient here for medication refill. Taking Metformin 500mg BID for diabetes.
            Reports good blood sugar control (120-130). No side effects.

            OBJECTIVE:
            Patient appears well.

            ASSESSMENT:
            Type 2 Diabetes Mellitus, well-controlled on current regimen.

            PLAN:
            1. Continue Metformin 500mg BID
            2. Refill sent to pharmacy
            3. Follow up in 3 months
            """
            }
        ]

        eval_inputs = []
        for case in synthetic_cases:
            eval_inputs.append(EvalInput(
            case_id=case['case_id'],
            transcript=case['transcript'],
            generated_note=case['generated_note'],
            ground_truth_note=case['generated_note']
            ))

        print(f" Generated {len(eval_inputs)} synthetic cases")
        return eval_inputs


class SyntheticErrorGenerator:
    """Generate perturbed SOAP notes with known errors for validation.

    This class creates synthetic error variants by intentionally introducing
    specific types of errors into clean SOAP notes. Used for:
    - Validating that the evaluation system detects known issues
    - Testing edge cases and error handling
    - Calibrating detection thresholds

    Each method introduces a specific error type that the evaluation
    system should catch.
    """
    
    @staticmethod
    def inject_missing_medication(note: str, transcript: str) -> str:
        """Remove medication mentions from note to create missing critical info error.

        Args:
        note (str): Original SOAP note
        transcript (str): Original transcript (unused but kept for consistency)

        Returns:
        str: Modified note with medication mentions removed
        """
        # Simple pattern to remove medication lines
        note = re.sub(r'.*(?:aspirin|metformin|lisinopril|atorvastatin).*\n?', '', note, flags=re.IGNORECASE)
        return note

    @staticmethod
    def inject_hallucinated_medication(note: str) -> str:
        """Add medication that was never mentioned in the transcript.

        Injects a randomly selected medication into the note's Plan section,
        creating a hallucination that should be detected by the evaluator.

        Args:
        note (str): Original SOAP note

        Returns:
        str: Modified note with hallucinated medication added
        """
        halluc_meds = [
        "Lisinopril 10mg daily",
        "Atorvastatin 20mg at bedtime",
        "Gabapentin 300mg TID"
        ]

        # Add to plan section
        halluc_med = random.choice(halluc_meds)
        if "PLAN:" in note:
            note = note.replace("PLAN:", f"PLAN:\n{halluc_med}\n")
        else:
            note += f"\n\nPLAN:\n{halluc_med}"

        return note
    
    @staticmethod
    def inject_wrong_dosage(note: str) -> str:
        """Modify medication dosage to an incorrect value.

        Multiplies the first dosage found by 10, creating a dangerous
        dosage error (e.g., 50mg -> 500mg) that should be flagged.

        Args:
        note (str): Original SOAP note

        Returns:
        str: Modified note with incorrect dosage
        """
        # Change dosages
        note = re.sub(r'(\d+)\s*mg', lambda m: f"{int(m.group(1)) * 10}mg", note, count=1)
        return note

    @staticmethod
    def inject_contradicted_fact(note: str, transcript: str) -> str:
        """Add information that contradicts the transcript.

        If the transcript mentions a symptom (e.g., "chest pain"),
        modifies the note to claim the patient denies that symptom,
        creating a direct contradiction.

        Args:
        note (str): Original SOAP note
        transcript (str): Original transcript

        Returns:
        str: Modified note with contradictory statement
        """
        # If transcript says "chest pain", note says "no chest pain"
        if "chest pain" in transcript.lower() and "chest pain" in note.lower():
            note = note.replace("chest pain", "denies chest pain")

        return note
    
    @staticmethod
    def remove_section(note: str, section: str = "Assessment") -> str:
        """Remove an entire SOAP section from the note.

        Tests the section coverage detection by removing a required
        section (defaults to Assessment).

        Args:
        note (str): Original SOAP note
        section (str): Section name to remove. Defaults to "Assessment"

        Returns:
        str: Modified note with section removed
        """
        pattern = rf'{section.upper()}:.*?(?=\n[A-Z]+:|$)'
        note = re.sub(pattern, '', note, flags=re.DOTALL)
        return note
    
    def generate_error_variants(self, eval_input: EvalInput) -> Dict[str, EvalInput]:
        """Generate multiple perturbed variants of a single evaluation case.

        Creates a suite of test cases from one clean case, each with a different
        type of intentional error. Useful for:
        - Validating detection capabilities
        - Testing that each error type is caught
        - Ensuring false positive rate is acceptable

        Args:
        eval_input (EvalInput): Original clean evaluation case

        Returns:
        Dict[str, EvalInput]: Dictionary mapping error type names to perturbed cases:
        - 'missing_med': Medication removed from note
        - 'halluc_med': Fake medication added to note
        - 'wrong_dosage': Dosage multiplied by 10
        - 'contradicted': Contradictory statement added
        """
        variants = {}

        # Missing medication
        variants['missing_med'] = EvalInput(
        case_id=f"{eval_input.case_id}_missing_med",
        transcript=eval_input.transcript,
        generated_note=self.inject_missing_medication(eval_input.generated_note, eval_input.transcript),
        ground_truth_note=eval_input.ground_truth_note
        )

        # Hallucinated medication
        variants['halluc_med'] = EvalInput(
        case_id=f"{eval_input.case_id}_halluc_med",
        transcript=eval_input.transcript,
        generated_note=self.inject_hallucinated_medication(eval_input.generated_note),
        ground_truth_note=eval_input.ground_truth_note
        )

        # Wrong dosage
        variants['wrong_dosage'] = EvalInput(
        case_id=f"{eval_input.case_id}_wrong_dosage",
        transcript=eval_input.transcript,
        generated_note=self.inject_wrong_dosage(eval_input.generated_note),
        ground_truth_note=eval_input.ground_truth_note
        )

        # Contradicted fact
        variants['contradicted'] = EvalInput(
        case_id=f"{eval_input.case_id}_contradicted",
        transcript=eval_input.transcript,
        generated_note=self.inject_contradicted_fact(eval_input.generated_note, eval_input.transcript),
        ground_truth_note=eval_input.ground_truth_note
        )

        return variants

