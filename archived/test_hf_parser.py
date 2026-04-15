import unittest
from parser_hf_math_v2 import extract_answer, verify_answer

class TestParserHFMathV2(unittest.TestCase):
    def test_extract_answer(self):
        # Test 1: Simple number
        self.assertTrue(extract_answer("The answer is 42"))

        # Test 2: LaTeX
        self.assertTrue(extract_answer("The answer is $\\frac{1}{2}$"))
        
        # Test 3: Boxed
        self.assertTrue(extract_answer("The answer is \\boxed{10}"))

    def test_verify_answer(self):
        # Test 1: Exact match
        self.assertTrue(verify_answer("42", "42"))
        
        # Test 2: Equivalent expressions
        self.assertTrue(verify_answer("1/2", "0.5"))
        
        # Test 3: LaTeX equivalence
        self.assertTrue(verify_answer("\\frac{1}{2}", "0.5"))
        
        # Test 4: Mismatch
        self.assertFalse(verify_answer("42", "43"))

if __name__ == '__main__':
    unittest.main()
