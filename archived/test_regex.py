
import re

def test_regex():
    p = "The final answer is <atok> 42 </atok>"
    regex = r"(The|final|answer|is|atok>|</|<atok|\n)"
    print(f"Original: '{p}'")
    modified = re.sub(regex, "", p)
    print(f"Modified: '{modified}'")
    
    p2 = "<atok>42</atok>"
    print(f"Original: '{p2}'")
    modified2 = re.sub(regex, "", p2)
    print(f"Modified: '{modified2}'")

    p3 = "The final answer is <atok>42</atok>"
    print(f"Original: '{p3}'")
    modified3 = re.sub(regex, "", p3)
    print(f"Modified: '{modified3}'")

if __name__ == "__main__":
    test_regex()
