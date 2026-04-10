from math_verify import parse, verify

def extract_answer(pred_str: str, data_name: str = None) -> str:
    """
    Extracts the answer from the prediction string using math_verify.
    
    Args:
        pred_str: The prediction string.
        data_name: Optional data name, kept for compatibility but not strictly used by math_verify
                   which handles formats agnostically.
                   
    Returns:
        The extracted answer as a string, or empty string if failed.
    """
    try:
        # math_verify.parse returns a list of MathExpression objects or similar structure depending on version
        # But based on doc: parse(gold, extraction_config=[...]) -> MathExpression
        # We want to extract the "answer" part. 
        # Actually math_verify.parse returns a value that can be passed to verify.
        # If we need the string representation:
        parsed = parse(pred_str)
        # The parsed object might be a list or a single object. 
        # If it's a list candidates, we usually want the last one or the one that is most likely the answer.
        # However, math_verify.parse typically extracts all candidates.
        # Let's inspect what parse returns by default or if we need to be more specific.
        # For now, let's return the string representation of the parsed object.
        return str(parsed)
    except Exception:
        return ""

def verify_answer(pred_str: str, gold_str: str) -> bool:
    """
    Verifies if the prediction matches the gold answer using math_verify.
    
    Args:
        pred_str: The prediction string.
        gold_str: The gold answer string.
        
    Returns:
        True if they match, False otherwise.
    """
    try:
        if not pred_str or not gold_str:
            return False
            
        # math_verify.verify(gold, pred) 
        # Note: doc says verify(gold, answer)
        return verify(gold_str, pred_str)
    except Exception:
        return False
