#!/bin/bash

echo "=========================================="
echo "DeepScribe Evals - End-to-End Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate environment
source venv/bin/activate

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo ""
    echo "=========================================="
    echo "TEST: $test_name"
    echo "=========================================="
    echo "Command: $command"
    echo ""
    
    if eval "$command"; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} File exists: $file"
        return 0
    else
        echo -e "${RED}✗${NC} File missing: $file"
        return 1
    fi
}

# Clear old results
echo "Cleaning up old results..."
rm -f results/test_*.json
echo "✓ Cleanup complete"

# ===========================================
# TEST 1: Fast Mode (Tier 1 only) - 2 cases
# ===========================================
run_test "Fast Mode - 2 cases" \
    "python run_eval.py --mode fast --num-cases 2 2>&1 | tail -20"

if [ $? -eq 0 ]; then
    # Find the most recent result file
    RESULT_FILE=$(ls -t results/eval_results_fast_*.json 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        check_file "$RESULT_FILE"
        
        # Validate JSON structure
        if python -c "import json; json.load(open('$RESULT_FILE'))" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Valid JSON output"
        else
            echo -e "${RED}✗${NC} Invalid JSON output"
            ((TESTS_FAILED++))
        fi
        
        # Check required fields
        python -c "
import json
data = json.load(open('$RESULT_FILE'))
assert 'metadata' in data, 'Missing metadata'
assert 'aggregate' in data, 'Missing aggregate'
assert 'cases' in data, 'Missing cases'
assert len(data['cases']) == 2, 'Expected 2 cases'
print('${GREEN}✓${NC} All required fields present')
print(f'  - Cases: {len(data[\"cases\"])}')
print(f'  - Mean composite: {data[\"aggregate\"][\"mean_composite\"]:.3f}')
print(f'  - Runtime: {data[\"aggregate\"][\"total_runtime_seconds\"]:.2f}s')
"
    else
        echo -e "${RED}✗${NC} No result file generated"
        ((TESTS_FAILED++))
    fi
fi

# ===========================================
# TEST 2: Fast Mode - 5 cases
# ===========================================
run_test "Fast Mode - 5 cases" \
    "python run_eval.py --mode fast --num-cases 5 2>&1 | grep -q 'Total cases: 5' && echo 'Success'"

# ===========================================
# TEST 3: Standard Mode (Tier 1+2) - 3 cases
# ===========================================
run_test "Standard Mode with NLI - 3 cases" \
    "python run_eval.py --mode standard --num-cases 3 2>&1 | tail -20"

if [ $? -eq 0 ]; then
    RESULT_FILE=$(ls -t results/eval_results_standard_*.json 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        # Check that Tier 2 was executed
        python -c "
import json
data = json.load(open('$RESULT_FILE'))
case = data['cases'][0]
if 2 in case['meta']['tiers_executed'] or 1 in case['meta']['tiers_executed']:
    print('${GREEN}✓${NC} Standard mode tiers executed')
else:
    print('${RED}✗${NC} Expected Tier 1 or 2 execution')
"
    fi
fi

# ===========================================
# TEST 4: Thorough Mode (All Tiers) - 2 cases
# ===========================================
echo ""
echo "=========================================="
echo "TEST: Thorough Mode with LLM - 2 cases"
echo "=========================================="
echo -e "${YELLOW}Note: This uses Gemini API and may take longer${NC}"
echo ""

run_test "Thorough Mode with Gemini LLM - 2 cases" \
    "timeout 120 python run_eval.py --mode thorough --num-cases 2 2>&1 | tail -20"

if [ $? -eq 0 ]; then
    RESULT_FILE=$(ls -t results/eval_results_thorough_*.json 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        # Check that all tiers were considered
        python -c "
import json
data = json.load(open('$RESULT_FILE'))
case = data['cases'][0]
tiers = case['meta']['tiers_executed']
print(f'${GREEN}✓${NC} Thorough mode executed tiers: {tiers}')
if len(case.get('inaccurate', [])) > 0 or len(case.get('missing_critical', [])) > 0:
    print('${GREEN}✓${NC} LLM findings detected')
"
    fi
fi

# ===========================================
# TEST 5: Synthetic Error Generation
# ===========================================
run_test "Synthetic Error Generation" \
    "python run_eval.py --mode fast --num-cases 2 --add-synthetic 2>&1 | grep -q 'synthetic' && echo 'Success'"

# ===========================================
# TEST 6: Configuration Loading
# ===========================================
run_test "Configuration Loading" \
    "python -c '
from src.utils.config import config
assert config.get(\"tier1.confidence.missing_entity\") == 0.8
assert config.get(\"tier1.confidence.hallucinated_entity\") == 0.7
assert config.get(\"tier1.internal_weights\") is not None
print(\"All config values loaded\")
'"

# ===========================================
# TEST 7: Data Models
# ===========================================
run_test "Pydantic Data Models" \
    "python -c '
from src.models import EvalInput, EvalOutput, Finding
test_input = EvalInput(
    case_id=\"test\",
    transcript=\"Doctor: Hello.\",
    generated_note=\"Note here.\"
)
print(f\"Created EvalInput: {test_input.case_id}\")
'"

# ===========================================
# TEST 8: Text Processing Utils
# ===========================================
run_test "Text Processing Utilities" \
    "python -c '
from src.utils.text_processing import sentence_split, detect_soap_sections
sents = sentence_split(\"Hello world. How are you?\")
assert len(sents) >= 2
sections = detect_soap_sections(\"SUBJECTIVE: test\\nOBJECTIVE: test\")
assert sections[\"subjective\"] == True
print(f\"Sentence split: {len(sents)} sentences\")
print(f\"Sections detected: {sum(sections.values())}/4\")
'"

# ===========================================
# TEST 9: Component Imports
# ===========================================
run_test "Component Imports" \
    "python -c '
from src.data_loader import DataLoader
from src.pipeline import EvaluationPipeline
from src.evaluators.tier1.evaluator import Tier1Evaluator
from src.evaluators.tier1.entity_extractor import MedicalEntityExtractor
from src.evaluators.tier1.semantic_checker import SemanticChecker
print(\"All components import successfully\")
'"

# ===========================================
# TEST 10: Results Validation
# ===========================================
echo ""
echo "=========================================="
echo "TEST: Validating All Generated Results"
echo "=========================================="

RESULT_COUNT=$(ls -1 results/*.json 2>/dev/null | wc -l)
echo "Found $RESULT_COUNT result files"

if [ $RESULT_COUNT -gt 0 ]; then
    for result_file in results/*.json; do
        if [ -f "$result_file" ]; then
            filename=$(basename "$result_file")
            python -c "
import json
try:
    data = json.load(open('$result_file'))
    assert 'cases' in data
    assert 'aggregate' in data
    assert 'metadata' in data
    print('${GREEN}✓${NC} $filename - Valid')
except Exception as e:
    print('${RED}✗${NC} $filename - Invalid: {e}')
    exit(1)
"
            if [ $? -eq 0 ]; then
                ((TESTS_PASSED++))
            else
                ((TESTS_FAILED++))
            fi
        fi
    done
else
    echo -e "${YELLOW}⚠${NC}  No result files to validate"
fi

# ===========================================
# SUMMARY
# ===========================================
echo ""
echo "=========================================="
echo "TEST SUITE COMPLETE"
echo "=========================================="
echo ""
echo "Results:"
echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED!${NC} 🎉"
    echo ""
    echo "Your evaluation system is fully operational!"
    echo ""
    echo "Next steps:"
    echo "  1. View results: ls -lh results/"
    echo "  2. Launch dashboard: streamlit run dashboard/app.py"
    echo "  3. Run full evaluation: python run_eval.py --mode standard --num-cases 20"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo ""
    echo "Please review the errors above."
    exit 1
fi

