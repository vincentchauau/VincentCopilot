#!/usr/bin/env python3
"""
Vincent Copilot CLI - Code generation and suggestions
Usage: vincent "task description"
"""

import sys
import requests
import platform

API_URL = "http://localhost:5000"

def suggest_command(task):
    """Get code suggestion from Vincent Copilot API"""
    os_type = platform.system().lower()
    if os_type == "darwin":
        os_type = "macos"
    
    try:
        response = requests.post(
            f"{API_URL}/terminal-suggest",
            json={"task": task, "os": os_type},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("command", "No suggestion available")
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Vincent Copilot API. Is the server running?"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("Usage: vincent \"task description\"")
        print("Example: vincent \"find all python files\"")
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    print(f"Task: {task}")
    print(f"OS: {platform.system()}")
    print("\nSuggested code completion:")
    print(suggest_command(task))

if __name__ == "__main__":
    main()
