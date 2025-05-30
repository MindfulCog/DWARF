"""
Utility script to find and replace problematic Unicode characters in all Python files.
"""
import os
import re

def fix_unicode_in_file(filepath):
    """Replace Unicode characters with ASCII alternatives in the given file."""
    # Character replacements
    replacements = {
        '~': '~',     # Approximately equal to
        'x': 'x',     # Multiplication sign
        '^2': '^2',    # Superscript 2
        '^3': '^3',    # Superscript 3
        ' degrees': ' degrees', # Degree symbol
        '*': '*',     # Middle dot
        '->': '->',    # Right arrow
        '<-': '<-',    # Left arrow
        '+/-': '+/-',   # Plus-minus sign
        'integral': 'integral', # Integral
        'pi': 'pi',    # Pi
        'alpha': 'alpha', # Alpha
        'beta': 'beta',  # Beta
        'gamma': 'gamma', # Gamma
        'lambda': 'lambda' # Lambda
    }
    
    try:
        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any replacements were made
        original_content = content
        replaced = False
        
        # Replace each problematic character
        for unicode_char, ascii_replacement in replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, ascii_replacement)
                replaced = True
        
        # Write back if changes were made
        if replaced:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode characters in {filepath}")
            return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return False

def find_and_fix_unicode():
    """Find all Python files in the current directory tree and fix Unicode characters."""
    files_fixed = 0
    files_checked = 0
    
    # Walk through directory tree
    for root, dirs, files in os.walk('.'):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                files_checked += 1
                if fix_unicode_in_file(filepath):
                    files_fixed += 1
    
    print(f"Checked {files_checked} Python files")
    print(f"Fixed Unicode characters in {files_fixed} files")

if __name__ == "__main__":
    print("Finding and replacing problematic Unicode characters in Python files...")
    find_and_fix_unicode()
    print("Done!")