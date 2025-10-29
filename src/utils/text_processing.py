"""
Text normalization and processing utilities
"""
import re
from typing import List


def normalize_text(text: str) -> str:
    """Normalize text for consistent comparison and matching.
    
    Applies standard text normalization steps:
    - Converts to lowercase for case-insensitive comparison
    - Collapses multiple whitespace characters to single spaces
    - Removes special punctuation (keeps alphanumerics and periods)
    - Strips leading/trailing whitespace
    
    Args:
        text (str): Raw text to normalize
        
    Returns:
        str: Normalized text suitable for comparison
        
    Example:
        >>> normalize_text("Patient   has CHEST-PAIN!!")
        'patient has chestpain'
    """
    # Lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize punctuation
    text = re.sub(r'[^\w\s\.]', '', text)
    
    return text.strip()


def sentence_split(text: str) -> List[str]:
    """Split text into sentences using basic punctuation rules.
    
    Uses periods, exclamation marks, and question marks as delimiters.
    Handles whitespace and newlines appropriately.
    
    Args:
        text (str): Text to split into sentences
        
    Returns:
        List[str]: List of sentence strings (whitespace trimmed, empty strings removed)
        
    Note:
        This is a simple splitter that may not handle all edge cases
        (e.g., abbreviations like "Dr.", decimals, etc.). Sufficient for
        medical notes which typically have clear sentence boundaries.
    """
    # Split on period, exclamation, question mark followed by space/newline
    sentences = re.split(r'[.!?]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_soap_sections(note: str) -> dict:
    """Extract content from each SOAP section of a clinical note.
    
    Uses regex patterns to identify and extract the four standard SOAP sections:
    - Subjective: Patient's reported symptoms and history
    - Objective: Physical exam findings and measurements
    - Assessment: Clinical impression and diagnosis
    - Plan: Treatment plan and follow-up
    
    Handles various header formats (e.g., "Subjective:", "HPI:", "Chief Complaint:")
    and extracts the content until the next section header or end of note.
    
    Args:
        note (str): Clinical note text (may or may not follow SOAP format)
        
    Returns:
        dict: Dictionary with keys ['subjective', 'objective', 'assessment', 'plan'],
              each mapping to the extracted content string (empty string if section not found)
              
    Example:
        >>> note = "Subjective: Chest pain\\nObjective: BP 120/80\\n..."
        >>> sections = extract_soap_sections(note)
        >>> sections['subjective']
        'Chest pain'
    """
    sections = {
        'subjective': '',
        'objective': '',
        'assessment': '',
        'plan': ''
    }
    
    # Patterns for section headers
    patterns = {
        'subjective': r'(?:^|\n)(?:subjective|chief complaint|cc|hpi|history)[:\s]*\n?(.*?)(?=\n(?:objective|physical|assessment|plan)|$)',
        'objective': r'(?:^|\n)(?:objective|physical exam|pe|vitals)[:\s]*\n?(.*?)(?=\n(?:assessment|plan)|$)',
        'assessment': r'(?:^|\n)(?:assessment|impression|diagnosis)[:\s]*\n?(.*?)(?=\n(?:plan)|$)',
        'plan': r'(?:^|\n)(?:plan|recommendations|treatment)[:\s]*\n?(.*?)$'
    }
    
    note_lower = note.lower()
    
    for section, pattern in patterns.items():
        match = re.search(pattern, note_lower, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections


def detect_soap_sections(note: str) -> dict:
    """Detect whether each SOAP section is present in a note.
    
    Returns boolean flags indicating which sections exist, without
    extracting the actual content. Useful for checking note completeness.
    
    Args:
        note (str): Clinical note text
        
    Returns:
        dict: Dictionary with keys ['subjective', 'objective', 'assessment', 'plan'],
              each mapping to a boolean indicating if that section was found
              
    Example:
        >>> note = "Subjective: Test\\nPlan: Follow up"
        >>> detect_soap_sections(note)
        {'subjective': True, 'objective': False, 'assessment': False, 'plan': True}
    """
    sections = extract_soap_sections(note)
    return {
        'subjective': bool(sections['subjective']),
        'objective': bool(sections['objective']),
        'assessment': bool(sections['assessment']),
        'plan': bool(sections['plan'])
    }


def extract_vitals(text: str) -> List[dict]:
    """Extract vital signs measurements from clinical text using regex patterns.
    
    Detects and extracts common vital signs:
    - Blood pressure (e.g., "120/80 mmHg")
    - Temperature (e.g., "98.6°F", "98.6 degrees")
    - Heart rate (e.g., "HR: 72", "pulse 88 bpm")
    - Respiratory rate (e.g., "RR: 16 breaths/min")
    
    Args:
        text (str): Clinical text to extract vitals from
        
    Returns:
        List[dict]: List of vital sign dictionaries, each containing:
            - type (str): Vital sign type (e.g., 'blood_pressure', 'temperature')
            - value (str): Measured value
            - unit (str): Unit of measurement
            
    Example:
        >>> text = "BP 120/80, HR 72, Temp 98.6F"
        >>> extract_vitals(text)
        [{'type': 'blood_pressure', 'value': '120/80', 'unit': 'mmHg'},
         {'type': 'heart_rate', 'value': '72', 'unit': 'bpm'},
         {'type': 'temperature', 'value': '98.6', 'unit': 'F'}]
    """
    vitals = []
    
    # Blood pressure
    bp_pattern = r'\b(\d{2,3})/(\d{2,3})\s*(?:mmHg)?\b'
    for match in re.finditer(bp_pattern, text):
        vitals.append({
            'type': 'blood_pressure',
            'value': f"{match.group(1)}/{match.group(2)}",
            'unit': 'mmHg'
        })
    
    # Temperature
    temp_pattern = r'\b(\d{2,3}(?:\.\d)?)\s*(?:°F|F|degrees)\b'
    for match in re.finditer(temp_pattern, text):
        vitals.append({
            'type': 'temperature',
            'value': match.group(1),
            'unit': 'F'
        })
    
    # Heart rate
    hr_pattern = r'\b(?:hr|heart rate|pulse)[:\s]*(\d{2,3})\s*(?:bpm)?\b'
    for match in re.finditer(hr_pattern, text, re.IGNORECASE):
        vitals.append({
            'type': 'heart_rate',
            'value': match.group(1),
            'unit': 'bpm'
        })
    
    # Respiratory rate
    rr_pattern = r'\b(?:rr|respiratory rate)[:\s]*(\d{1,2})\s*(?:breaths?/min)?\b'
    for match in re.finditer(rr_pattern, text, re.IGNORECASE):
        vitals.append({
            'type': 'respiratory_rate',
            'value': match.group(1),
            'unit': 'breaths/min'
        })
    
    return vitals


def is_negated(text: str, entity_span: tuple) -> bool:
    """Check if an entity mention is negated in the surrounding context.
    
    Looks for negation words (no, not, denies, etc.) in the text preceding
    the entity. Simple rule-based approach suitable for clinical text where
    negations are typically explicit and close to the entity.
    
    Args:
        text (str): Full text containing the entity
        entity_span (tuple): (start_index, end_index) of entity in text
        
    Returns:
        bool: True if entity appears to be negated, False otherwise
        
    Example:
        >>> text = "Patient denies chest pain"
        >>> span = (15, 25)  # "chest pain"
        >>> is_negated(text, span)
        True
        
    Note:
        Checks approximately 5 words (50 characters) before the entity.
        May produce false positives in complex sentences with multiple clauses.
    """
    start, end = entity_span
    # Look at 5 words before entity
    context_start = max(0, start - 50)
    context = text[context_start:start].lower()
    
    negation_words = ['no', 'not', 'denies', 'denied', 'without', 'negative for', 'absent']
    
    for neg_word in negation_words:
        if neg_word in context:
            return True
    
    return False

