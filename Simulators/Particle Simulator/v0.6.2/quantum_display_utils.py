"""
Utility functions for displaying quantum information with console-safe characters.
"""

def safe_text(text):
    """Convert special unicode characters to ASCII-compatible alternatives."""
    return text.replace('~', '~').replace('x', 'x').replace('^2', '^2').replace('^3', '^3')

def format_quantum_info(state, use_ascii=True):
    """
    Format quantum state information in a display-friendly way.
    
    Args:
        state: Quantum state dictionary
        use_ascii: Whether to use ASCII-compatible characters only
        
    Returns:
        str: Formatted quantum state information
    """
    if not state:
        return "No quantum state detected"
        
    n = state.get('n', '?')
    l = state.get('l', '?')
    j = state.get('j', '?')
    orbital_type = state.get('orbital_type', 'unknown')
    energy = state.get('energy', 0.0)
    
    if use_ascii:
        return f"n~{n}, l~{l}, j~{j}, {orbital_type}, E~{energy:.6f} eV"
    else:
        return f"n~{n}, l~{l}, j~{j}, {orbital_type}, E~{energy:.6f} eV"