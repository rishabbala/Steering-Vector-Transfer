
import re

def debug_regex():
    p = "<atok>42</atok>"
    regex = r"(The|final|answer|is|atok>|</|<atok|\n)"
    print(f"Regex: {regex}")
    print(f"String: {p}")
    
    matches = list(re.finditer(regex, p))
    print(f"Found {len(matches)} matches:")
    for m in matches:
        print(f"  Match: '{m.group()}' at {m.span()}")
        
    subbed = re.sub(regex, "", p)
    print(f"Subbed: '{subbed}'")

if __name__ == "__main__":
    debug_regex()
