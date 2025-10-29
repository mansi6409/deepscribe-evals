"""
Medical entity extraction using scispaCy
"""
from typing import List, Set, Dict, Tuple
import re


class MedicalEntityExtractor:
    """Extract medical entities from clinical text using scispaCy or rule-based fallback.
    
    This class provides robust medical entity extraction with automatic graceful
    degradation. If the scispaCy medical NER model is available, uses it for
    high-quality extraction. Otherwise, falls back to rule-based patterns for
    common medical entities.
    
    Handles:
    - Medications (drugs, dosages)
    - Diagnoses and symptoms
    - Vital signs (BP, HR, temperature)
    - Clinical concepts
    
    Attributes:
        nlp: scispaCy language model (None if not available)
    """
    
    def __init__(self):
        """Initialize entity extractor with lazy model loading."""
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Attempt to load scispaCy model, fall back to rule-based if unavailable.
        
        Tries to load the en_core_sci_md model. If loading fails (model not
        installed, import errors, etc.), silently continues with rule-based
        extraction as fallback.
        """
        try:
            import spacy
            self.nlp = spacy.load("en_core_sci_md")
        except:
            print("Warning: scispaCy model not found. Using rule-based extraction as fallback.")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract all medical entities from text.
        
        Uses scispaCy if available, otherwise falls back to rule-based patterns.
        Entities include medications, diagnoses, symptoms, and vital signs.
        
        Args:
            text (str): Clinical text to extract entities from
            
        Returns:
            List[Dict]: List of entity dictionaries, each containing:
                - text (str): Entity text span
                - label (str): Entity type (MEDICATION, DIAGNOSIS, VITAL_BP, etc.)
                - start (int): Character offset of entity start
                - end (int): Character offset of entity end
                
        Example:
            >>> extractor = MedicalEntityExtractor()
            >>> extractor.extract_entities("Patient takes aspirin 81mg daily")
            [{'text': 'aspirin 81mg', 'label': 'MEDICATION', 'start': 14, 'end': 26}]
        """
        entities = []
        
        if self.nlp is not None:
            # Use scispaCy
            doc = self.nlp(text)
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        else:
            # Fallback to rule-based
            entities.extend(self._extract_medications(text))
            entities.extend(self._extract_diagnoses(text))
            entities.extend(self._extract_vitals(text))
        
        return entities
    
    def _extract_medications(self, text: str) -> List[Dict]:
        """Extract medications using regex patterns (fallback method).
        
        Detects common medication names and medication + dosage patterns.
        Covers frequently prescribed medications and generic dosage formats.
        
        Args:
            text (str): Text to extract medications from
            
        Returns:
            List[Dict]: List of medication entity dictionaries
        """
        entities = []
        
        # Common medication patterns
        med_patterns = [
            r'\b(aspirin|metformin|lisinopril|atorvastatin|amlodipine|gabapentin|insulin|ibuprofen|acetaminophen)\b',
            r'\b([A-Z][a-z]+)\s+\d+\s*mg\b',  # Generic pattern: "Medication 100 mg"
        ]
        
        for pattern in med_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(0),
                    'label': 'MEDICATION',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def _extract_diagnoses(self, text: str) -> List[Dict]:
        """Extract diagnoses and symptoms using regex patterns (fallback method).
        
        Detects common diagnoses, chronic conditions, and acute symptoms.
        
        Args:
            text (str): Text to extract diagnoses from
            
        Returns:
            List[Dict]: List of diagnosis entity dictionaries
        """
        entities = []
        
        # Common diagnoses
        dx_patterns = [
            r'\b(diabetes|hypertension|hyperlipidemia|asthma|copd|chf|mi|cad|stroke|cancer)\b',
            r'\b(chest pain|headache|fever|cough|nausea|vomiting|diarrhea)\b'
        ]
        
        for pattern in dx_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'text': match.group(0),
                    'label': 'DIAGNOSIS',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def _extract_vitals(self, text: str) -> List[Dict]:
        """Extract vital signs using regex patterns (fallback method).
        
        Detects blood pressure, temperature, and heart rate measurements.
        
        Args:
            text (str): Text to extract vitals from
            
        Returns:
            List[Dict]: List of vital sign entity dictionaries
        """
        entities = []
        
        # Blood pressure
        bp_pattern = r'\b(\d{2,3})/(\d{2,3})\s*(?:mmHg)?\b'
        for match in re.finditer(bp_pattern, text):
            entities.append({
                'text': match.group(0),
                'label': 'VITAL_BP',
                'start': match.start(),
                'end': match.end()
            })
        
        # Temperature
        temp_pattern = r'\b(\d{2,3}(?:\.\d)?)\s*(?:°F|F|degrees)\b'
        for match in re.finditer(temp_pattern, text):
            entities.append({
                'text': match.group(0),
                'label': 'VITAL_TEMP',
                'start': match.start(),
                'end': match.end()
            })
        
        # Heart rate
        hr_pattern = r'\b(?:hr|heart rate|pulse)[:\s]*(\d{2,3})\b'
        for match in re.finditer(hr_pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(0),
                'label': 'VITAL_HR',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def get_entity_sets(self, text: str) -> Dict[str, Set[str]]:
        """Group extracted entities by type into sets.
        
        Extracts all entities and organizes them into a dictionary where
        keys are entity labels and values are sets of unique entity texts
        (lowercased for consistency).
        
        Args:
            text (str): Text to extract and group entities from
            
        Returns:
            Dict[str, Set[str]]: Dictionary mapping entity labels to sets of entity texts
            
        Example:
            >>> extractor.get_entity_sets("Patient has diabetes and takes metformin")
            {'DIAGNOSIS': {'diabetes'}, 'MEDICATION': {'metformin'}}
        """
        entities = self.extract_entities(text)
        
        entity_sets = {}
        for ent in entities:
            label = ent['label']
            if label not in entity_sets:
                entity_sets[label] = set()
            entity_sets[label].add(ent['text'].lower())
        
        return entity_sets
    
    def compare_entities(self, transcript_text: str, note_text: str) -> Tuple[Set[str], Set[str]]:
        """Compare entities between transcript and note to find discrepancies.
        
        Extracts entities from both texts and performs set comparison to identify:
        - Missing entities: Present in transcript but absent from note
        - Hallucinated entities: Present in note but not in transcript
        
        Args:
            transcript_text (str): Original patient-doctor dialogue
            note_text (str): Generated SOAP note
            
        Returns:
            Tuple[Set[str], Set[str]]: Two sets:
                - Set 1: Entity texts missing from note
                - Set 2: Entity texts hallucinated in note
                
        Example:
            >>> missing, halluc = extractor.compare_entities(
            ...     "Patient takes aspirin",
            ...     "Patient takes metformin"
            ... )
            >>> missing
            {'aspirin'}
            >>> halluc
            {'metformin'}
        """
        tx_entities = self.get_entity_sets(transcript_text)
        note_entities = self.get_entity_sets(note_text)
        
        # Flatten all entities
        tx_all = set()
        note_all = set()
        
        for entity_set in tx_entities.values():
            tx_all.update(entity_set)
        
        for entity_set in note_entities.values():
            note_all.update(entity_set)
        
        missing = tx_all - note_all
        hallucinated = note_all - tx_all
        
        return missing, hallucinated
    
    def is_critical_entity(self, entity_text: str, entity_label: str) -> bool:
        """Determine if an entity is safety-critical and requires special attention.
        
        Critical entities include:
        - Medications (all)
        - Allergies (all)
        - Abnormal vitals (blood pressure, temperature)
        - Red flag symptoms (chest pain, stroke symptoms, bleeding, etc.)
        
        Missing or hallucinating critical entities can have serious clinical
        consequences and are weighted more heavily in evaluation.
        
        Args:
            entity_text (str): The text of the entity
            entity_label (str): The entity type label
            
        Returns:
            bool: True if entity is safety-critical, False otherwise
        """
        critical_labels = ['MEDICATION', 'ALLERGY', 'VITAL_BP', 'VITAL_TEMP']
        
        if entity_label in critical_labels:
            return True
        
        # Check for red flag symptoms
        red_flags = ['chest pain', 'shortness of breath', 'stroke', 'bleeding', 'unconscious']
        entity_lower = entity_text.lower()
        
        for flag in red_flags:
            if flag in entity_lower:
                return True
        
        return False

