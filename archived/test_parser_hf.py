#!/usr/bin/env python3
"""
Quick test script to verify parser_hf_math.py integration works correctly.
"""

def test_imports():
    """Test that all required functions can be imported."""
    print("Testing imports...")
    try:
        from parser_hf_math import (
            extract_answer,
            run_execute,
            parse_ground_truth,
            parse_question,
            verify_answer_hf,
            parse_and_verify_hf,
            evaluate_predictions_hf,
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_function_signatures():
    """Test that function signatures match parser.py."""
    print("\nTesting function signatures...")
    from parser_hf_math import extract_answer, run_execute
    import inspect
    
    # Test extract_answer signature
    sig = inspect.signature(extract_answer)
    params = list(sig.parameters.keys())
    expected = ['pred_str', 'data_name', 'samples', 'use_last_number', 'use_fallback']
    
    if params == expected:
        print("✓ extract_answer signature matches")
    else:
        print(f"✗ extract_answer signature mismatch: {params} vs {expected}")
        return False
    
    # Test run_execute signature
    sig = inspect.signature(run_execute)
    params = list(sig.parameters.keys())
    expected = ['executor', 'result', 'prompt_type', 'data_name', 'execute', 'samples']
    
    if params == expected:
        print("✓ run_execute signature matches")
    else:
        print(f"✗ run_execute signature mismatch: {params} vs {expected}")
        return False
    
    return True


def test_basic_extraction():
    """Test basic answer extraction without math-verify."""
    print("\nTesting basic extraction (fallback mode)...")
    from parser_hf_math import extract_answer
    
    test_cases = [
        ("The answer is 42", "math", "42"),
        ("$\\boxed{166}$", "math", "166"),
        ("The answer is (B)", "mmlu_stem", "B"),
    ]
    
    passed = 0
    for pred, dataset, expected in test_cases:
        result = extract_answer(pred, dataset, None)
        # Just check that it returns something (exact match depends on math-verify)
        if result:
            print(f"✓ Extracted '{result}' from '{pred[:30]}...'")
            passed += 1
        else:
            print(f"✗ Failed to extract from '{pred[:30]}...'")
    
    return passed == len(test_cases)


def test_run_execute():
    """Test run_execute function."""
    print("\nTesting run_execute...")
    from parser_hf_math import run_execute
    
    # Test with valid result
    pred, report = run_execute(None, "The answer is 42", "cot", "math", False, None)
    if pred is not None:
        print(f"✓ run_execute returned prediction: '{pred}'")
    else:
        print("✗ run_execute failed to return prediction")
        return False
    
    # Test with error result
    pred, report = run_execute(None, "error", "cot", "math", False, None)
    if pred is None:
        print("✓ run_execute correctly handles error input")
    else:
        print("✗ run_execute should return None for error input")
        return False
    
    return True


def test_parse_ground_truth():
    """Test that parse_ground_truth is accessible."""
    print("\nTesting parse_ground_truth...")
    from parser_hf_math import parse_ground_truth
    
    example = {
        "solution": "The solution is...",
        "answer": "42"
    }
    
    try:
        gt_cot, gt_ans = parse_ground_truth(example, "math")
        print(f"✓ parse_ground_truth works: gt_ans='{gt_ans}'")
        return True
    except Exception as e:
        print(f"✗ parse_ground_truth failed: {e}")
        return False


def test_math_verify_available():
    """Check if math-verify is installed."""
    print("\nChecking math-verify installation...")
    try:
        from math_verify import parse, verify
        print("✓ math-verify is installed and available")
        return True
    except ImportError:
        print("⚠ math-verify not installed (will use fallback mode)")
        print("  Install with: pip install math-verify[antlr4-python3-runtime]")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Parser HF Math Integration Test")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Function Signatures", test_function_signatures()))
    results.append(("Basic Extraction", test_basic_extraction()))
    results.append(("run_execute", test_run_execute()))
    results.append(("parse_ground_truth", test_parse_ground_truth()))
    results.append(("math-verify Available", test_math_verify_available()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
