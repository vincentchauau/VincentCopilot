"""
Vincent Copilot Jupyter Notebook Extension
Adds code completion and AI assistance to Jupyter notebooks
"""

import requests
from IPython.core.magic import register_line_magic, register_cell_magic
from IPython.display import display, Markdown, Code

API_URL = "http://localhost:5000"

@register_line_magic
def vincent(line):
    """
    Vincent Copilot magic command
    Usage: %vincent explain <code>
           %vincent fix <code>
           %vincent optimize <code>
    """
    parts = line.split(maxsplit=1)
    if len(parts) < 2:
        return "Usage: %vincent <command> <code>"
    
    command, code = parts
    
    try:
        if command == "explain":
            response = requests.post(
                f"{API_URL}/explain",
                json={"code": code, "language": "python"}
            )
            data = response.json()
            display(Markdown(f"**Explanation:**\n\n{data.get('explanation', 'No explanation')}"))
        
        elif command == "fix":
            response = requests.post(
                f"{API_URL}/fix-bug",
                json={"code": code, "language": "python"}
            )
            data = response.json()
            display(Markdown(f"**Analysis:**\n\n{data.get('analysis', '')}"))
            if data.get('fixed_code'):
                display(Code(data['fixed_code'], language='python'))
        
        elif command == "optimize":
            response = requests.post(
                f"{API_URL}/refactor",
                json={"code": code, "language": "python", "instruction": "optimize performance"}
            )
            data = response.json()
            display(Code(data.get('refactored_code', ''), language='python'))
        
        else:
            return f"Unknown command: {command}. Available: explain, fix, optimize"
    
    except Exception as e:
        return f"Error: {str(e)}"

@register_cell_magic
def vincent_complete(line, cell):
    """
    Complete code in a cell
    Usage: %%vincent_complete
           <partial code>
    """
    try:
        response = requests.post(
            f"{API_URL}/complete",
            json={"prompt": cell, "max_new_tokens": 200}
        )
        data = response.json()
        completion = data.get('completion', '')
        
        # Extract only the new part
        if completion.startswith(cell):
            completion = completion[len(cell):]
        
        display(Markdown("**Completion:**"))
        display(Code(cell + completion, language='python'))
        
        return completion
    except Exception as e:
        return f"Error: {str(e)}"

@register_line_magic
def vincent_tests(line):
    """
    Generate tests for code
    Usage: %vincent_tests <code>
    """
    try:
        response = requests.post(
            f"{API_URL}/generate-tests",
            json={"code": line, "language": "python", "test_framework": "pytest"}
        )
        data = response.json()
        test_code = data.get('test_code', '')
        
        display(Markdown("**Generated Tests:**"))
        display(Code(test_code, language='python'))
        
        return test_code
    except Exception as e:
        return f"Error: {str(e)}"

def load_ipython_extension(ipython):
    """Load the extension in IPython/Jupyter"""
    print("Vincent Copilot Jupyter extension loaded!")
    print("Available commands:")
    print("  %vincent explain <code>      - Explain code")
    print("  %vincent fix <code>          - Fix bugs")
    print("  %vincent optimize <code>     - Optimize code")
    print("  %vincent_tests <code>        - Generate tests")
    print("  %%vincent_complete           - Complete code in cell")

# Usage in Jupyter:
# %load_ext vincent_notebook
