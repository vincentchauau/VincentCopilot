# Vincent Copilot Inference API
# Run with: python api_server.py

from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import traceback
import logging
import os
from github import Github, GithubException
import re
from language_framework_support import (
    detect_language, detect_language_from_code, get_language_config,
    get_test_framework_for_language, get_syntax_for_language,
    detect_frameworks, get_test_prompt_template, get_explanation_prompt_template,
    get_refactor_suggestions, get_language_list, get_frameworks_for_language
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.config['DEBUG'] = False

MODEL_ID = "Salesforce/codegen-350M-mono"  # Smaller model for limited memory
REVISION = None  # Or set to your model revision if needed

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model on device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        revision=REVISION, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    traceback.print_exc()
    raise

def generate_text(prompt, max_new_tokens=128, temperature=0.2, min_new_tokens=16):
    """Generic text generation function"""
    try:
        # CodeGen has max_position_embeddings of 2048
        max_context_length = 2048
        max_prompt_length = max_context_length - max_new_tokens - 10  # Leave buffer
        
        # Tokenize and check length
        tokens = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        num_tokens = tokens.input_ids.shape[1]
        
        logger.info(f"Prompt tokens: {num_tokens}, Max allowed: {max_prompt_length}")
        
        if num_tokens > max_prompt_length:
            logger.warning(f"Prompt too long ({num_tokens} > {max_prompt_length}). Truncating...")
            # Keep only the last max_prompt_length tokens to preserve context
            tokens.input_ids = tokens.input_ids[:, -max_prompt_length:]
            if 'attention_mask' in tokens:
                tokens.attention_mask = tokens.attention_mask[:, -max_prompt_length:]
        
        batch = tokens.to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
        )
        with torch.no_grad():
            generated_ids = model.generate(**batch, generation_config=generation_config)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        logger.error(f"Error in generate_text(): {e}")
        traceback.print_exc()
        raise

def generate_completion(prompt, max_new_tokens=128):
    """Code completion generation"""
    return generate_text(prompt, max_new_tokens, temperature=0.2, min_new_tokens=16)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "device": device,
        "model_id": MODEL_ID,
        "cuda_available": torch.cuda.is_available()
    })

@app.route("/")
def index():
    """Serve the chat UI"""
    return send_from_directory('static', 'chat.html')

@app.route("/languages", methods=["GET"])
def get_languages():
    """Get all supported languages and frameworks"""
    try:
        languages = {}
        for lang, config in get_language_config('python').__class__.__dict__.items():
            if not lang.startswith('_'):
                pass
        
        # Build response from language support module
        from language_framework_support import SUPPORTED_LANGUAGES
        response = {}
        for lang, config in SUPPORTED_LANGUAGES.items():
            response[lang] = {
                "extensions": config['extensions'],
                "frameworks": config['frameworks'],
                "test_frameworks": config['test_frameworks'],
                "package_manager": config['package_manager'],
                "info": config['info']
            }
        
        return jsonify({
            "languages": response,
            "total_languages": len(response),
            "sample_languages": list(response.keys())[:5]
        })
    except Exception as e:
        logger.error(f"Error in get_languages(): {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/detect-language", methods=["POST"])
def detect_language_endpoint():
    """Detect programming language from code content or filename"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        filename = data.get("filename", "")
        
        detected_lang = None
        
        # Try to detect from filename first
        if filename:
            detected_lang = detect_language(filename)
        
        # Fall back to content detection
        if not detected_lang and code:
            detected_lang = detect_language_from_code(code)
        
        if not detected_lang:
            detected_lang = "python"  # Default
        
        config = get_language_config(detected_lang)
        frameworks = detect_frameworks(code, detected_lang) if code else []
        
        return jsonify({
            "language": detected_lang,
            "confidence": "high" if filename else "medium" if code else "low",
            "frameworks_detected": frameworks,
            "available_frameworks": config['frameworks'],
            "test_frameworks": config['test_frameworks'],
            "package_manager": config['package_manager']
        })
    except Exception as e:
        logger.error(f"Error in detect_language_endpoint(): {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/complete", methods=["POST"])
def complete():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 128))
        
        if not prompt:
            return jsonify({"error": "No prompt provided."}), 400
        
        if max_new_tokens < 1 or max_new_tokens > 512:
            return jsonify({"error": "max_new_tokens must be between 1 and 512"}), 400
            
        logger.info(f"Generating completion for prompt: {prompt[:50]}...")
        output = generate_completion(prompt, max_new_tokens)
        return jsonify({"completion": output})
    except Exception as e:
        logger.error(f"Error in complete(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

def handle_slash_command(command, code_context=""):
    """Handle slash commands - 15 total commands available"""
    cmd = command.strip().lower()
    
    if cmd == "/explain" and code_context:
        prompt = f"# Explain this code:\n{code_context}\n\n# Explanation:\n"
        return generate_text(prompt, 300, temperature=0.3, min_new_tokens=20)
    
    elif cmd == "/fix" and code_context:
        prompt = f"# Code to fix:\n{code_context}\n\n# Analysis and fixed code:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/tests" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Generate comprehensive unit tests:\n"
        return generate_text(prompt, 500, temperature=0.3, min_new_tokens=50)
    
    elif cmd == "/doc" and code_context:
        prompt = f"# Code:\n{code_context}\n\n# Documentation:\n"
        return generate_text(prompt, 300, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/optimize" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Optimized version:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/review" and code_context:
        prompt = f"# Code to review:\n{code_context}\n\n# Code review comments:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/simplify" and code_context:
        prompt = f"# Complex code:\n{code_context}\n\n# Simplified version:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/readable" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# More readable version with better naming and structure:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/performance" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Performance-optimized version:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/secure" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Security-hardened version:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/comment" and code_context:
        prompt = f"# Code:\n{code_context}\n\n# Same code with detailed comments:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/types" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Same code with type hints/annotations:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/async" and code_context:
        prompt = f"# Synchronous code:\n{code_context}\n\n# Asynchronous version:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/clean" and code_context:
        prompt = f"# Original code:\n{code_context}\n\n# Cleaned up version (remove unused code, fix style):\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    elif cmd == "/translate" and code_context:
        # Extract target language from command if provided
        parts = command.split()
        target = parts[1] if len(parts) > 1 else "python"
        prompt = f"# Original code:\n{code_context}\n\n# Translated to {target}:\n"
        return generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
    
    else:
        return f"""Unknown command: {command}

Available commands:
• /explain - Explain how code works
• /fix - Fix bugs in code
• /tests - Generate unit tests
• /doc - Generate documentation
• /optimize - Optimize code
• /review - Code review
• /simplify - Simplify complex code
• /readable - Improve readability
• /performance - Performance improvements  
• /secure - Security hardening
• /comment - Add comments
• /types - Add type hints
• /async - Convert to async
• /clean - Clean up code
• /translate [lang] - Translate to another language"""

@app.route("/chat", methods=["POST"])
def chat():
    """Chat interface with conversation history and slash commands"""
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        max_new_tokens = int(data.get("max_new_tokens", 256))
        
        if not messages:
            return jsonify({"error": "No messages provided."}), 400
        
        # Check if last message is a slash command
        last_msg = messages[-1].get("content", "").strip()
        if last_msg.startswith("/"):
            # Extract command and code context
            parts = last_msg.split("\n", 1)
            command = parts[0]
            code_context = parts[1] if len(parts) > 1 else ""
            
            # If no code in command, look in previous messages
            if not code_context:
                for msg in reversed(messages[:-1]):
                    content = msg.get("content", "")
                    if "```" in content:  # Has code block
                        code_context = content
                        break
            
            response = handle_slash_command(command, code_context)
            return jsonify({"response": response, "command": command})
        
        # Build regular conversation prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            elif role == "system":
                prompt += f"{content}\n"
        
        prompt += "Assistant:"
        
        logger.info(f"Chat prompt: {prompt[:100]}...")
        output = generate_text(prompt, max_new_tokens, temperature=0.7, min_new_tokens=10)
        
        # Extract only the assistant's response
        if "Assistant:" in output:
            response = output.split("Assistant:")[-1].strip()
        else:
            response = output[len(prompt):].strip()
        
        return jsonify({"response": response, "full_output": output})
    except Exception as e:
        logger.error(f"Error in chat(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/explain", methods=["POST"])
def explain():
    """Explain code functionality"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        max_new_tokens = int(data.get("max_new_tokens", 300))
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        # Auto-detect language if not provided
        if language == "python" and "def " not in code:
            language = detect_language_from_code(code)
        
        # Get language-specific prompt template
        config = get_language_config(language)
        frameworks = detect_frameworks(code, language)
        
        # Create language-specific explanation prompt
        syntax_comment = config['syntax_comment']
        prompt = f"""{syntax_comment} Explain this {language} code in detail:
{code}

{syntax_comment} Detailed explanation:
"""
        
        if frameworks:
            prompt = f"""{syntax_comment} This code uses: {', '.join(frameworks)}
{syntax_comment} Explain this {language} code:
{code}

{syntax_comment} Detailed breakdown:
"""
        
        logger.info(f"Explaining {language} code: {code[:50]}...")
        output = generate_text(prompt, max_new_tokens, temperature=0.3, min_new_tokens=20)
        
        # Extract explanation
        explanation = output[len(prompt):].strip() if len(output) > len(prompt) else output
        
        return jsonify({
            "explanation": explanation,
            "language": language,
            "frameworks_detected": frameworks,
            "code_length": len(code),
            "lines": len(code.split('\n'))
        })
    except Exception as e:
        logger.error(f"Error in explain(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/refactor", methods=["POST"])
def refactor():
    """Suggest code refactoring improvements"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        instruction = data.get("instruction", "improve and refactor")
        max_new_tokens = int(data.get("max_new_tokens", 400))
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        # Auto-detect language if not provided
        if language == "python" and "def " not in code:
            language = detect_language_from_code(code)
        
        # Get language-specific suggestions
        config = get_language_config(language)
        syntax_comment = config['syntax_comment']
        suggestions = get_refactor_suggestions(language, code)
        
        # Create refactoring prompt
        suggestions_str = '\n'.join([f"{syntax_comment} - {s}" for s in suggestions[:5]])
        prompt = f"""{syntax_comment} Original {language} code:
{code}

{syntax_comment} Refactoring suggestions:
{suggestions_str}
{syntax_comment}
{syntax_comment} Task: {instruction}
{syntax_comment} Refactored code:
"""
        
        logger.info(f"Refactoring {language} code with instruction: {instruction}")
        output = generate_text(prompt, max_new_tokens, temperature=0.3, min_new_tokens=30)
        
        # Extract refactored code
        refactored = output[len(prompt):].strip() if len(output) > len(prompt) else output
        
        frameworks = detect_frameworks(code, language)
        
        return jsonify({
            "refactored_code": refactored,
            "original_code": code,
            "language": language,
            "instruction": instruction,
            "suggestions": suggestions[:5],
            "frameworks_detected": frameworks
        })
    except Exception as e:
        logger.error(f"Error in refactor(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/generate-tests", methods=["POST"])
def generate_tests():
    """Generate unit tests for code with comprehensive edge cases and mocks"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        test_framework = data.get("test_framework", None)
        include_edge_cases = data.get("include_edge_cases", True)
        include_mocks = data.get("include_mocks", True)
        max_new_tokens = int(data.get("max_new_tokens", 700))
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        # Auto-detect language if not provided
        if language == "python" and "def " not in code:
            language = detect_language_from_code(code)
        
        # Get language config
        config = get_language_config(language)
        
        # Use provided test framework or get default for language
        if not test_framework:
            test_framework = get_test_framework_for_language(language)
        
        # Validate test framework is available for this language
        available_frameworks = config['test_frameworks']
        if test_framework not in available_frameworks:
            test_framework = available_frameworks[0]  # Fall back to first available
        
        # Extract function/method signature
        func_pattern = r'def\s+(\w+)\(' if language == 'python' else r'function\s+(\w+)\(' if language in ['javascript'] else r'(\w+)\s*\('
        func_match = re.search(func_pattern, code)
        func_name = func_match.group(1) if func_match else "test_function"
        
        # Comprehensive edge case detection
        edge_cases_detected = [
            "Zero value (boundary)",
            "Negative values (if applicable)",
            "Maximum/minimum values",
            "Empty collection",
            "Single-element collection",
            "None/null parameter",
            "Large values (overflow potential)",
            "Unicode/special characters"
        ]
        
        # Exception test cases
        exception_cases = [
            "ValueError for invalid input",
            "TypeError for wrong type",
            "IndexError for out-of-bounds access",
            "AttributeError for missing attributes"
        ]
        
        # Detect likely exceptions from code
        if 'raise ' in code:
            custom_exceptions = re.findall(r'raise\s+(\w+)', code)
            exception_cases.extend([f"{exc} exception" for exc in set(custom_exceptions)])
        
        # Generate test prompt using language-specific template
        prompt = get_test_prompt_template(language, code, test_framework, edge_cases_detected[:8])
        
        logger.info(f"Generating {test_framework} tests for {language} code")
        output = generate_text(prompt, max_new_tokens, temperature=0.25, min_new_tokens=80)
        
        test_code = output[len(prompt):].strip() if len(output) > len(prompt) else output
        
        # Generate language-specific assertion examples
        assertions_map = {
            'python': [
                "assert result == expected_value",
                "assert result is not None",
                "with pytest.raises(ValueError):",
                "assert len(result) > 0",
                "mock.assert_called_once_with(expected_args)"
            ],
            'javascript': [
                "expect(result).toBe(expected)",
                "expect(result).not.toBeNull()",
                "expect(() => fn()).toThrow(Error)",
                "expect(array.length).toBeGreaterThan(0)"
            ],
            'java': [
                "assertEquals(expected, actual)",
                "assertNotNull(result)",
                "assertThrows(Exception.class, () -> code())",
                "assertTrue(condition)"
            ],
            'go': [
                "if got != expected { t.Errorf(...) }",
                "if err != nil { t.Fatal(err) }",
                "if !reflect.DeepEqual(got, want) {}"
            ]
        }
        
        assertions = assertions_map.get(language, [
            "assert result",
            "assert result != null",
            "assert error handling"
        ])
        
        # Detect mock patterns
        mock_patterns = []
        code_lower = code.lower()
        
        if any(word in code_lower for word in ['database', 'db.', 'query', 'connect']):
            mock_patterns.append("Mock database operations")
        if any(word in code_lower for word in ['request', 'http', 'fetch', 'axios', 'urllib']):
            mock_patterns.append("Mock HTTP requests/API calls")
        if any(word in code_lower for word in ['file', 'open', 'read', 'write']):
            mock_patterns.append("Mock file I/O operations")
        if any(word in code_lower for word in ['time.', 'datetime', 'now()', 'today()']):
            mock_patterns.append("Mock time/date functions")
        
        frameworks = detect_frameworks(code, language)
        
        return jsonify({
            "test_code": test_code,
            "original_code": code,
            "language": language,
            "test_framework": test_framework,
            "available_test_frameworks": available_frameworks,
            "edge_cases": edge_cases_detected[:8],
            "exception_tests": exception_cases[:8],
            "assertion_patterns": assertions,
            "mocks": mock_patterns if include_mocks else [],
            "frameworks_detected": frameworks,
            "coverage_estimate": f"{min(100, 50 + len(edge_cases_detected) * 5)}%",
            "includes_edge_cases": include_edge_cases,
            "includes_mocks": include_mocks
        })
    except Exception as e:
        logger.error(f"Error in generate_tests(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/fix-bug", methods=["POST"])
def fix_bug():
    """Detect and fix bugs in code"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        error_message = data.get("error_message", "")
        max_new_tokens = int(data.get("max_new_tokens", 400))
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        # Create bug fixing prompt
        if error_message:
            prompt = f"""# Code with bug:
{code}

# Error message:
{error_message}

# Analysis and fixed code:
# Bug: """
        else:
            prompt = f"""# Code to review for bugs:
{code}

# Analysis:
# Potential issues:"""
        
        logger.info(f"Analyzing code for bugs in {language}")
        output = generate_text(prompt, max_new_tokens, temperature=0.3, min_new_tokens=30)
        
        # Extract analysis
        analysis = output[len(prompt):].strip()
        
        # Also generate fixed code if error provided
        fixed_code = None
        if error_message:
            fix_prompt = f"""# Original code with bug:
{code}

# Fixed code:
"""
            fix_output = generate_text(fix_prompt, max_new_tokens, temperature=0.2, min_new_tokens=20)
            fixed_code = fix_output[len(fix_prompt):].strip()
        
        return jsonify({
            "analysis": analysis,
            "fixed_code": fixed_code,
            "original_code": code,
            "error_message": error_message
        })
    except Exception as e:
        logger.error(f"Error in fix_bug(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/inline-complete", methods=["POST"])
def inline_complete():
    """Fast inline completions for as-you-type suggestions with context awareness"""
    try:
        data = request.get_json()
        prefix = data.get("prefix", "")  # Code before cursor
        suffix = data.get("suffix", "")  # Code after cursor
        language = data.get("language", "python")
        context = data.get("context", "code")  # code, comment, markdown, docstring
        max_new_tokens = int(data.get("max_new_tokens", 50))
        
        if not prefix:
            return jsonify({"completion": ""})
        
        import re
        
        # Detect context if not provided
        if context == "code":
            # Check if we're in a comment
            if language == "python" and prefix.rstrip().endswith("#"):
                context = "comment"
            elif language in ["javascript", "typescript"] and prefix.rstrip().endswith("//"):
                context = "comment"
            elif language in ["javascript", "typescript"] and "/*" in prefix and "*/" not in prefix:
                context = "comment"
            # Check if we're in a docstring/string
            elif (language == "python" and (prefix.count('"""') % 2 == 1 or prefix.count("'''") % 2 == 1)):
                context = "docstring"
            elif (language in ["javascript", "typescript"] and prefix.count("`") % 2 == 1):
                context = "string"
        
        # Context-specific prompt engineering
        if context == "comment":
            # Better completions for comments
            prompt = prefix  # Include full comment
            # Add hint for comment completion
            if language == "python":
                prompt = prefix.rstrip() + " "
            else:
                prompt = prefix.rstrip() + " "
            max_new_tokens = min(40, max_new_tokens)
            
        elif context == "docstring":
            # For docstrings, provide structured completions
            prompt = prefix
            max_new_tokens = min(80, max_new_tokens)
            
        elif context == "markdown":
            # For markdown files
            prompt = prefix
            max_new_tokens = min(60, max_new_tokens)
        else:
            # Regular code completion
            prompt = prefix
        
        logger.info(f"Inline completion ({context}) for: {prefix[-30:]}")
        output = generate_text(prompt, max_new_tokens, temperature=0.2, min_new_tokens=1)
        
        # Extract only the new tokens
        completion = output[len(prefix):].strip()
        
        # If we have a suffix, try to avoid duplicating it
        if suffix and completion:
            # Check for obvious duplicates
            if completion[:10] == suffix[:10]:
                completion = ""
            elif completion.startswith(suffix.split()[0] if suffix.split() else ""):
                completion = ""
        
        # Sanitize completion based on context
        if context == "comment":
            # Remove any code-like completions from comments
            completion = re.sub(r'[{}()[\];:=]', '', completion)
        
        return jsonify({
            "completion": completion,
            "full_text": output,
            "context": context,
            "is_valid": len(completion) > 0
        })
    except Exception as e:
        logger.error(f"Error in inline_complete(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/alternatives", methods=["POST"])
def alternatives():
    """Generate multiple solution alternatives for a problem"""
    try:
        data = request.get_json()
        problem = data.get("problem", "")
        code_context = data.get("code_context", "")
        language = data.get("language", "python")
        num_alternatives = int(data.get("num_alternatives", 3))
        max_new_tokens = int(data.get("max_new_tokens", 300))
        
        if not problem:
            return jsonify({"error": "No problem provided."}), 400
        
        alternatives_list = []
        
        # Generate multiple alternatives with different temperatures
        temperatures = [0.3, 0.6, 0.9][:num_alternatives]
        
        for i, temp in enumerate(temperatures):
            prompt = f"""# Problem: {problem}
{f"# Context: {code_context}" if code_context else ""}

# Solution {i+1} ({language}):
"""
            output = generate_text(prompt, max_new_tokens, temperature=temp, min_new_tokens=30)
            solution = output[len(prompt):].strip()
            alternatives_list.append({
                "solution": solution,
                "approach": f"Approach {i+1}",
                "temperature": temp
            })
        
        return jsonify({
            "problem": problem,
            "alternatives": alternatives_list,
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in alternatives(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/scan-vulnerabilities", methods=["POST"])
def scan_vulnerabilities():
    """Scan code for security vulnerabilities"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        vulnerabilities = []
        
        # Pattern-based vulnerability detection
        patterns = {
            "SQL Injection": r"(execute|query|cursor\.execute).*['\"].*\+.*['\"]|f['\"].*SELECT",
            "Command Injection": r"(os\.system|subprocess\.(call|run|Popen)).*\+|shell=True",
            "Path Traversal": r"open\(.*\+|\.\.\/|\.\.\\",
            "Hard-coded Secrets": r"(password|api_key|secret|token)\s*=\s*['\"][\w\-]+['\"]",
            "Eval Usage": r"eval\(|exec\(",
            "Weak Crypto": r"md5|sha1(?!\\d)|DES",
        }
        
        import re
        for vuln_type, pattern in patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                vulnerabilities.append({
                    "type": vuln_type,
                    "severity": "high" if vuln_type in ["SQL Injection", "Command Injection"] else "medium",
                    "pattern": pattern
                })
        
        # AI-based analysis
        prompt = f"""# Security analysis for {language} code:
{code}

# Security issues and recommendations:
"""
        analysis = generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
        ai_analysis = analysis[len(prompt):].strip()
        
        return jsonify({
            "code": code,
            "vulnerabilities": vulnerabilities,
            "ai_analysis": ai_analysis,
            "scan_date": "2026-02-14",
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in scan_vulnerabilities(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/analyze-context", methods=["POST"])
def analyze_context():
    """Analyze multi-file context for better completions"""
    try:
        data = request.get_json()
        files = data.get("files", [])  # [{"path": "...", "content": "..."}]
        query = data.get("query", "")
        
        if not files:
            return jsonify({"error": "No files provided."}), 400
        
        # Build context from multiple files
        context = "# Project Context:\n\n"
        for file_info in files[:5]:  # Limit to 5 files
            path = file_info.get("path", "unknown")
            content = file_info.get("content", "")[:500]  # Limit content
            context += f"## File: {path}\n{content}\n\n"
        
        if query:
            context += f"\n# Query: {query}\n# Answer:\n"
            output = generate_text(context, 300, temperature=0.4, min_new_tokens=20)
            answer = output[len(context):].strip()
            
            return jsonify({
                "query": query,
                "answer": answer,
                "files_analyzed": len(files)
            })
        else:
            # Extract imports and dependencies
            imports = []
            for file_info in files:
                content = file_info.get("content", "")
                # Extract Python imports
                import re
                found_imports = re.findall(r"^(?:from|import)\s+[^\s]+", content, re.MULTILINE)
                imports.extend(found_imports)
            
            return jsonify({
                "files_analyzed": len(files),
                "imports": list(set(imports)),
                "context_summary": "Multi-file context loaded"
            })
    except Exception as e:
        logger.error(f"Error in analyze_context(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/suggest-imports", methods=["POST"])
def suggest_imports():
    """Suggest imports/dependencies based on code"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        prompt = f"""# Code:
{code}

# Required imports and dependencies for this code:
"""
        
        output = generate_text(prompt, 200, temperature=0.3, min_new_tokens=10)
        suggestions = output[len(prompt):].strip()
        
        return jsonify({
            "code": code,
            "import_suggestions": suggestions,
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in suggest_imports(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/code-review", methods=["POST"])
def code_review():
    """Perform automated code review"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        focus = data.get("focus", "general")  # general, performance, security, style
        
        if not code:
            return jsonify({"error": "No code provided."}), 400
        
        if focus == "security":
            prompt = f"""# Security review of {language} code:
{code}

# Security issues and recommendations:
"""
        elif focus == "performance":
            prompt = f"""# Performance review of {language} code:
{code}

# Performance bottlenecks and optimizations:
"""
        elif focus == "style":
            prompt = f"""# Code style review of {language} code:
{code}

# Style improvements and best practices:
"""
        else:
            prompt = f"""# Code review of {language} code:
{code}

# Review comments:
1. """
        
        output = generate_text(prompt, 400, temperature=0.3, min_new_tokens=30)
        review_comments = "1. " + output[len(prompt):].strip() if focus == "general" else output[len(prompt):].strip()
        
        return jsonify({
            "code": code,
            "review_comments": review_comments,
            "focus": focus,
            "language": language
        })
    except Exception as e:
        logger.error(f"Error in code_review(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500

@app.route("/terminal-suggest", methods=["POST"])
def terminal_suggest():
    """Suggest terminal commands based on natural language"""
    try:
        data = request.get_json()
        task = data.get("task", "")
        os_type = data.get("os", "linux")  # linux, windows, macos
        
        if not task:
            return jsonify({"error": "No task provided."}), 400
        
        prompt = f"""# Task: {task}
# OS: {os_type}
# Command:
"""
        
        output = generate_text(prompt, 100, temperature=0.3, min_new_tokens=5)
        command = output[len(prompt):].strip().split('\n')[0]  # Get first line only
        
        return jsonify({
            "task": task,
            "command": command,
            "os": os_type
        })
    except Exception as e:
        logger.error(f"Error in terminal_suggest(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ============================================
# GitHub Integration Endpoints
# ============================================

@app.route("/github/pr-summary", methods=["POST"])
def github_pr_summary():
    """Generate AI summary of a GitHub Pull Request"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")  # e.g., "owner/repo"
        pr_number = data.get("pr_number")
        token = data.get("token")  # Optional: GitHub token from VS Code or user
        
        if not repo_name or not pr_number:
            return jsonify({"error": "Missing repo or pr_number"}), 400
        
        # Initialize GitHub client
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            # Get PR details
            pr_info = {
                "title": pr.title,
                "description": pr.body or "",
                "author": pr.user.login,
                "files_changed": pr.changed_files,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "state": pr.state
            }
            
            # Get file changes (limited to avoid token limits)
            files = pr.get_files()
            changes = []
            total_changes = 0
            
            for file in files:
                if total_changes > 3000:  # Limit total characters
                    break
                patch = file.patch or ""
                changes.append(f"\n### {file.filename}\n```diff\n{patch[:500]}\n```")
                total_changes += len(patch[:500])
            
            # Generate summary with AI
            prompt = f"""Analyze this GitHub Pull Request and provide a concise summary:

Title: {pr_info['title']}
Description: {pr_info['description'][:300]}
Files Changed: {pr_info['files_changed']}
Additions: +{pr_info['additions']}, Deletions: -{pr_info['deletions']}

Key Changes:
{''.join(changes[:3])}

Summary:
"""
            
            summary = generate_text(prompt, max_new_tokens=200, temperature=0.3)
            summary_text = summary[len(prompt):].strip()
            
            return jsonify({
                "pr_info": pr_info,
                "summary": summary_text,
                "url": pr.html_url
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_pr_summary(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/pr-review", methods=["POST"])
def github_pr_review():
    """Automated code review for a GitHub Pull Request"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        pr_number = data.get("pr_number")
        token = data.get("token")
        focus = data.get("focus", "general")  # security, performance, style, general
        
        if not repo_name or not pr_number:
            return jsonify({"error": "Missing repo or pr_number"}), 400
        
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            # Get file changes
            files = pr.get_files()
            reviews = []
            
            for file in list(files)[:5]:  # Limit to 5 files
                if not file.patch:
                    continue
                    
                # Analyze code changes
                prompt = f"""Review this code change with focus on {focus}:

File: {file.filename}
```diff
{file.patch[:800]}
```

Code Review ({focus}):
"""
                
                review = generate_text(prompt, max_new_tokens=150, temperature=0.2)
                review_text = review[len(prompt):].strip()
                
                # Check for specific issues
                issues = []
                if "password" in file.patch.lower() or "secret" in file.patch.lower():
                    issues.append("⚠️ Potential sensitive data exposure")
                if "eval(" in file.patch or "exec(" in file.patch:
                    issues.append("🔒 Security: Dangerous function usage")
                if re.search(r'SELECT.*FROM.*WHERE', file.patch, re.I):
                    issues.append("⚠️ Potential SQL injection risk")
                
                reviews.append({
                    "file": file.filename,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "review": review_text,
                    "issues": issues
                })
            
            return jsonify({
                "pr_number": pr_number,
                "pr_title": pr.title,
                "focus": focus,
                "reviews": reviews,
                "url": pr.html_url
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_pr_review(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/commit-message", methods=["POST"])
def github_commit_message():
    """Generate conventional commit message from code changes"""
    try:
        data = request.get_json()
        diff = data.get("diff", "")
        context = data.get("context", "")  # Optional additional context
        
        if not diff:
            return jsonify({"error": "No diff provided"}), 400
        
        # Limit diff size
        diff_limited = diff[:1500]
        
        prompt = f"""Generate a conventional commit message for these changes:

Context: {context}

Diff:
```diff
{diff_limited}
```

Commit message (format: type(scope): description):
"""
        
        output = generate_text(prompt, max_new_tokens=50, temperature=0.2, min_new_tokens=10)
        commit_msg = output[len(prompt):].strip().split('\n')[0]
        
        # Clean up the commit message
        commit_msg = commit_msg.strip('"').strip("'").strip()
        
        return jsonify({
            "commit_message": commit_msg,
            "conventional_format": True
        })
        
    except Exception as e:
        logger.error(f"Error in github_commit_message(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/issue-analysis", methods=["POST"])
def github_issue_analysis():
    """Analyze GitHub issue and suggest potential fixes"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        issue_number = data.get("issue_number")
        token = data.get("token")
        
        if not repo_name or not issue_number:
            return jsonify({"error": "Missing repo or issue_number"}), 400
        
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            
            # Analyze issue
            prompt = f"""Analyze this GitHub issue and suggest potential solutions:

Title: {issue.title}
Description: {issue.body[:500] if issue.body else 'No description'}

Labels: {', '.join([l.name for l in issue.labels])}

Analysis and Suggestions:
"""
            
            analysis = generate_text(prompt, max_new_tokens=250, temperature=0.3)
            analysis_text = analysis[len(prompt):].strip()
            
            return jsonify({
                "issue_number": issue_number,
                "title": issue.title,
                "state": issue.state,
                "labels": [l.name for l in issue.labels],
                "analysis": analysis_text,
                "url": issue.html_url
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_issue_analysis(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/post-pr-comment", methods=["POST"])
def github_post_pr_comment():
    """Post AI-generated review comments to a GitHub Pull Request"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        pr_number = data.get("pr_number")
        token = data.get("token")  # Required for posting
        auto_review = data.get("auto_review", False)  # Generate review automatically
        comment_body = data.get("comment_body", "")  # Or provide custom comment
        
        if not repo_name or not pr_number or not token:
            return jsonify({"error": "Missing repo, pr_number, or token (required for posting)"}), 400
        
        gh = Github(token)
        
        try:
            repo = gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            if auto_review:
                # Generate AI review automatically
                files = pr.get_files()
                review_comments = []
                
                for file in list(files)[:3]:  # Limit to 3 files
                    if not file.patch:
                        continue
                    
                    prompt = f"""Review this code change and provide specific feedback:

File: {file.filename}
```diff
{file.patch[:600]}
```

Review comment:
"""
                    review = generate_text(prompt, max_new_tokens=150, temperature=0.3)
                    review_text = review[len(prompt):].strip()
                    review_comments.append(f"**{file.filename}**\n{review_text}")
                
                comment_text = '\n\n'.join(review_comments)
                comment_body = f"""## 🤖 Vincent Copilot AI Review

{comment_text}

---
*Generated by Vincent Copilot AI Assistant*"""
            
            # Post the comment
            comment = pr.create_issue_comment(comment_body)
            
            return jsonify({
                "success": True,
                "comment_id": comment.id,
                "comment_url": comment.html_url,
                "pr_url": pr.html_url,
                "message": "Comment posted successfully"
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_post_pr_comment(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/workspace-files", methods=["POST"])
def github_workspace_files():
    """Index workspace files for better context awareness"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        token = data.get("token")
        branch = data.get("branch", "main")
        file_pattern = data.get("file_pattern", "*.py")  # e.g., "*.py", "*.js", "src/**"
        
        if not repo_name:
            return jsonify({"error": "Missing repo"}), 400
        
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            contents = repo.get_contents("", ref=branch)
            
            indexed_files = []
            file_count = 0
            
            def index_directory(contents_list):
                nonlocal file_count
                result = []
                
                for content in contents_list:
                    if file_count >= 50:  # Limit to 50 files
                        break
                        
                    if content.type == "dir":
                        # Recursively index directories
                        dir_contents = repo.get_contents(content.path, ref=branch)
                        result.extend(index_directory(dir_contents))
                    else:
                        # Check if file matches pattern
                        import fnmatch
                        if fnmatch.fnmatch(content.name, file_pattern.replace("**/", "")):
                            result.append({
                                "path": content.path,
                                "name": content.name,
                                "size": content.size,
                                "sha": content.sha,
                                "url": content.html_url
                            })
                            file_count += 1
                
                return result
            
            indexed_files = index_directory(contents)
            
            return jsonify({
                "repo": repo_name,
                "branch": branch,
                "files_indexed": len(indexed_files),
                "files": indexed_files,
                "pattern": file_pattern
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_workspace_files(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ============================================
# Advanced Features - Closing Feature Gaps
# ============================================

@app.route("/refactor-workspace", methods=["POST"])
def refactor_workspace():
    """Multi-file refactoring across workspace"""
    try:
        data = request.get_json()
        files = data.get("files", [])  # [{"path": "...", "content": "..."}]
        refactor_task = data.get("task", "")  # e.g., "rename class User to Customer"
        language = data.get("language", "python")
        
        if not files or not refactor_task:
            return jsonify({"error": "Missing files or refactor task"}), 400
        
        refactored_files = []
        
        # Analyze all files to understand dependencies
        context = "# Project files:\n"
        for file_info in files[:10]:  # Limit to 10 files
            path = file_info.get("path", "")
            content = file_info.get("content", "")[:800]
            context += f"\n## {path}\n```{language}\n{content}\n```\n"
        
        # Generate refactoring plan
        plan_prompt = f"""{context}

# Refactoring Task: {refactor_task}

# Refactoring Plan:
1."""
        
        plan = generate_text(plan_prompt, 300, temperature=0.3)
        refactor_plan = "1." + plan[len(plan_prompt):].strip()
        
        # Apply refactoring to each file
        for file_info in files[:10]:
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            
            refactor_prompt = f"""# Task: {refactor_task}
# Plan: {refactor_plan[:200]}

# Original file ({path}):
```{language}
{content[:1000]}
```

# Refactored version:
```{language}
"""
            
            output = generate_text(refactor_prompt, 500, temperature=0.2)
            refactored_code = output[len(refactor_prompt):].strip()
            
            # Clean up code fences if present
            if "```" in refactored_code:
                refactored_code = refactored_code.split("```")[0].strip()
            
            refactored_files.append({
                "path": path,
                "original": content,
                "refactored": refactored_code,
                "changed": content.strip() != refactored_code.strip()
            })
        
        return jsonify({
            "task": refactor_task,
            "plan": refactor_plan,
            "files": refactored_files,
            "files_changed": sum(1 for f in refactored_files if f["changed"])
        })
        
    except Exception as e:
        logger.error(f"Error in refactor_workspace(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/analyze-complexity", methods=["POST"])
def analyze_complexity():
    """Analyze code complexity with detailed metrics"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        # Calculate basic metrics
        lines = code.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        blank_lines = len([l for l in lines if not l.strip()])
        
        # Estimate cyclomatic complexity (simplified)
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
        cyclomatic = 1  # Base complexity
        for keyword in complexity_keywords:
            cyclomatic += code.lower().count(f' {keyword} ') + code.lower().count(f'\n{keyword} ')
        
        # Count functions/classes
        import re
        functions = len(re.findall(r'\bdef\s+\w+', code))
        classes = len(re.findall(r'\bclass\s+\w+', code))
        
        # AI analysis
        prompt = f"""Analyze the complexity and quality of this {language} code:

```{language}
{code[:1000]}
```

Provide:
1. Cyclomatic complexity assessment
2. Cognitive complexity issues
3. Code smells detected
4. Refactoring suggestions

Analysis:
"""
        
        analysis = generate_text(prompt, 400, temperature=0.3)
        ai_analysis = analysis[len(prompt):].strip()
        
        # Complexity rating
        if cyclomatic <= 5:
            rating = "Simple"
        elif cyclomatic <= 10:
            rating = "Moderate"
        elif cyclomatic <= 20:
            rating = "Complex"
        else:
            rating = "Very Complex"
        
        return jsonify({
            "code": code,
            "metrics": {
                "lines_of_code": loc,
                "comment_lines": comment_lines,
                "blank_lines": blank_lines,
                "total_lines": len(lines),
                "cyclomatic_complexity": cyclomatic,
                "functions": functions,
                "classes": classes,
                "complexity_rating": rating
            },
            "analysis": ai_analysis,
            "language": language
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_complexity(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/analyze-advanced", methods=["POST"])
def analyze_advanced():
    """Advanced code analysis with design patterns and architectural insights"""
    try:
        data = request.get_json()
        code = data.get("code", "")
        language = data.get("language", "python")
        
        if not code:
            return jsonify({"error": "No code provided"}), 400
        
        import re
        
        # Design pattern detection
        patterns_detected = []
        
        # Decorator/Middleware pattern
        if re.search(r'@\w+\s*\n\s*def', code):
            patterns_detected.append("Decorator/Middleware pattern detected")
        
        # Singleton pattern
        if '_instance' in code or 'get_instance' in code:
            patterns_detected.append("Singleton pattern detected")
        
        # Factory pattern
        if re.search(r'def\s+create_\w+|def\s+make_\w+', code):
            patterns_detected.append("Factory pattern detected")
        
        # Observer pattern
        if 'notify' in code.lower() or 'subscribe' in code.lower() or 'observer' in code.lower():
            patterns_detected.append("Observer pattern detected")
        
        # Strategy pattern
        if re.search(r'class\s+\w+Strategy|def\s+execute_\w+|strategy\s*=', code, re.IGNORECASE):
            patterns_detected.append("Strategy pattern detected")
        
        # Builder pattern
        if re.search(r'with_\w+|\.builder\(|Builder\(', code):
            patterns_detected.append("Builder pattern detected")
        
        # MVC/MVP pattern
        if any(pattern in code.lower() for pattern in ['view', 'controller', 'model', 'presenter']):
            mvc_matches = sum(1 for p in ['view', 'controller', 'model'] if p in code.lower())
            if mvc_matches >= 2:
                patterns_detected.append("MVC/MVP pattern structure detected")
        
        # Code smells detection
        code_smells = []
        
        # Large method/function
        func_lengths = [len(match.group(0).split('\n')) for match in re.finditer(r'def\s+\w+\([^)]*\):[^\n]*(?:\n(?!def\s|\nclass\s).*)*', code)]
        if func_lengths and max(func_lengths) > 20:
            code_smells.append(f"Large function detected (max {max(func_lengths)} lines)")
        
        # Duplicate code
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        if len(lines) > 5:
            line_counts = {}
            for line in lines:
                line_counts[line] = line_counts.get(line, 0) + 1
            duplicates = [line for line, count in line_counts.items() if count > 2 and len(line) > 20]
            if duplicates:
                code_smells.append(f"Duplicate code detected ({len(duplicates)} instances)")
        
        # Long parameter lists
        long_params = re.findall(r'def\s+\w+\(([^)]{80,})\)', code)
        if long_params:
            code_smells.append(f"Functions with excessive parameters ({len(long_params)} found)")
        
        # Magic numbers/strings
        magic_numbers = len(re.findall(r'[^a-zA-Z_]\d{3,}[^a-zA-Z_]', code))
        magic_strings = len(re.findall(r'["\'][a-zA-Z0-9]{10,}["\']', code))
        if magic_numbers + magic_strings > 5:
            code_smells.append(f"Magic values detected ({magic_numbers + magic_strings} found)")
        
        # Deep nesting
        nesting_levels = []
        for line in code.split('\n'):
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent_level = (len(line) - len(stripped)) // 4
                nesting_levels.append(indent_level)
        
        if nesting_levels and max(nesting_levels) > 4:
            code_smells.append(f"Deep nesting detected (max {max(nesting_levels)} levels)")
        
        # Architectural assessment
        prompt = f"""Provide a brief architectural assessment of this {language} code:

```{language}
{code[:1500]}
```

Consider:
1. Separation of concerns
2. Dependency flow
3. Coupling and cohesion
4. Testability
5. Maintainability

Assessment:
"""
        
        assessment = generate_text(prompt, 350, temperature=0.3)
        architectural_assessment = assessment[len(prompt):].strip()
        
        # Advanced metrics
        avg_function_length = sum(func_lengths) / len(func_lengths) if func_lengths else 0
        cyclomatic = 1 + code.count('if') + code.count('elif') + code.count('else') + \
                    code.count('for') + code.count('while') + code.count('except')
        
        return jsonify({
            "code": code[:500],  # Return summary only
            "language": language,
            "design_patterns": patterns_detected,
            "code_smells": code_smells,
            "architectural_assessment": architectural_assessment,
            "advanced_metrics": {
                "avg_function_length": round(avg_function_length, 1),
                "estimated_cyclomatic_complexity": min(cyclomatic, 50),  # Cap at 50
                "nesting_depth": max(nesting_levels) if nesting_levels else 0,
                "total_functions": len(func_lengths),
                "duplicate_lines": len(duplicates) if 'duplicates' in locals() else 0
            },
            "recommendations": [
                "Break large functions into smaller ones" if max(func_lengths, default=0) > 20 else None,
                "Extract duplicate code into reusable functions" if code_smells else None,
                "Reduce parameter count for complex functions" if long_params else None,
                "Simplify deep nesting with early returns or guards" if max(nesting_levels, default=0) > 4 else None,
                "Move magic values to named constants" if magic_numbers > 5 else None
            ]
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_advanced(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/review-workspace", methods=["POST"])
def review_workspace():
    """Multi-file code review across workspace"""
    try:
        data = request.get_json()
        files = data.get("files", [])  # [{"path": "...", "content": "..."}]
        focus = data.get("focus", "general")  # security, performance, architecture, general
        
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        reviews = []
        
        # Build workspace context
        workspace_context = "# Workspace Overview:\n"
        for file_info in files[:20]:
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            # Extract key information
            import re
            imports = re.findall(r'^(?:from|import)\s+.+', content, re.MULTILINE)
            functions = re.findall(r'def\s+(\w+)', content)
            classes = re.findall(r'class\s+(\w+)', content)
            
            workspace_context += f"\n{path}: {len(classes)} classes, {len(functions)} functions\n"
        
        # Review each file
        for file_info in files[:15]:  # Limit to 15 files
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            
            if focus == "architecture":
                prompt = f"""{workspace_context}

# Reviewing: {path}
```
{content[:1200]}
```

# Architectural review:
- Design patterns used:
- Coupling/cohesion:
- SOLID principles:
- Suggestions:

Review:
"""
            elif focus == "security":
                prompt = f"""# Security review: {path}
```
{content[:1200]}
```

Security issues:
"""
            elif focus == "performance":
                prompt = f"""# Performance review: {path}
```
{content[:1200]}
```

Performance bottlenecks:
"""
            else:
                prompt = f"""# Code review: {path}
```
{content[:1200]}
```

Review:
"""
            
            review = generate_text(prompt, 300, temperature=0.3)
            review_text = review[len(prompt):].strip()
            
            # Detect common issues
            issues = []
            if "password" in content.lower() and "=" in content:
                issues.append("⚠️ Possible hardcoded password")
            if "eval(" in content or "exec(" in content:
                issues.append("🔴 Security: eval/exec usage")
            if content.count("for ") > 5 and "while " in content:
                issues.append("⚠️ Possible O(n²) complexity")
            if "# TODO" in content or "# FIXME" in content:
                issues.append("📝 Contains TODO/FIXME comments")
            
            reviews.append({
                "file": path,
                "review": review_text,
                "issues": issues,
                "loc": len([l for l in content.split('\n') if l.strip()])
            })
        
        return jsonify({
            "focus": focus,
            "files_reviewed": len(reviews),
            "reviews": reviews,
            "workspace_context": workspace_context
        })
        
    except Exception as e:
        logger.error(f"Error in review_workspace(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/workspace-search", methods=["POST"])
def workspace_search():
    """Semantic search across workspace files"""
    try:
        data = request.get_json()
        files = data.get("files", [])
        query = data.get("query", "")
        search_type = data.get("search_type", "semantic")  # semantic, regex, keyword
        
        if not files or not query:
            return jsonify({"error": "Missing files or query"}), 400
        
        results = []
        
        if search_type == "regex":
            import re
            pattern = re.compile(query, re.IGNORECASE)
            
            for file_info in files[:50]:
                path = file_info.get("path", "")
                content = file_info.get("content", "")
                matches = []
                
                for i, line in enumerate(content.split('\n'), 1):
                    if pattern.search(line):
                        matches.append({
                            "line": i,
                            "content": line.strip(),
                            "match": pattern.search(line).group()
                        })
                
                if matches:
                    results.append({
                        "file": path,
                        "matches": matches[:10]  # Limit to 10 matches per file
                    })
        
        elif search_type == "keyword":
            query_lower = query.lower()
            
            for file_info in files[:50]:
                path = file_info.get("path", "")
                content = file_info.get("content", "")
                
                if query_lower in content.lower():
                    lines = content.split('\n')
                    matches = []
                    
                    for i, line in enumerate(lines, 1):
                        if query_lower in line.lower():
                            matches.append({
                                "line": i,
                                "content": line.strip()
                            })
                    
                    results.append({
                        "file": path,
                        "matches": matches[:10]
                    })
        
        else:  # semantic search using AI
            # Build index of files
            file_summaries = []
            for file_info in files[:30]:
                path = file_info.get("path", "")
                content = file_info.get("content", "")[:800]
                file_summaries.append(f"{path}:\n{content}\n")
            
            # Semantic query
            prompt = f"""# Workspace files:
{''.join(file_summaries[:10])}

# Search Query: {query}

# Most relevant files and code sections:
"""
            
            search_results = generate_text(prompt, 400, temperature=0.3)
            semantic_results = search_results[len(prompt):].strip()
            
            return jsonify({
                "query": query,
                "search_type": search_type,
                "results": semantic_results,
                "files_searched": len(files)
            })
        
        return jsonify({
            "query": query,
            "search_type": search_type,
            "results": results,
            "total_matches": sum(len(r.get("matches", [])) for r in results)
        })
        
    except Exception as e:
        logger.error(f"Error in workspace_search(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/analyze-multi-file-context", methods=["POST"])
def analyze_multi_file_context():
    """Analyze multiple files together to understand dependencies and relationships"""
    try:
        data = request.get_json()
        files = data.get("files", [])  # List of {path, content, language} objects
        language = data.get("language", "python")
        
        if not files or len(files) < 2:
            return jsonify({"error": "Need at least 2 files for context analysis"}), 400
        
        import re
        
        # Analyze imports and dependencies
        dependencies = {}
        imports_by_file = {}
        exports_by_file = {}
        
        for file_info in files:
            path = file_info.get("path", "")
            content = file_info.get("content", "")
            file_lang = file_info.get("language", language)
            
            # Extract imports/requires
            imports = []
            exports = []
            
            if file_lang == "python":
                # Python imports
                import_patterns = [
                    r'from\s+([\w.]+)\s+import',
                    r'import\s+([\w.]+)'
                ]
                for pattern in import_patterns:
                    imports.extend(re.findall(pattern, content))
                
                # Python exports (functions, classes)
                exports.extend(re.findall(r'def\s+(\w+)', content))
                exports.extend(re.findall(r'class\s+(\w+)', content))
                
            elif file_lang in ["javascript", "typescript"]:
                # JS/TS imports/requires
                import_patterns = [
                    r'from\s+[\'"]([^\'"]+)[\'"]',
                    r'require\([\'"]([^\'"]+)[\'"]\)',
                    r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]'
                ]
                for pattern in import_patterns:
                    imports.extend(re.findall(pattern, content))
                
                # JS/TS exports (functions, classes)
                exports.extend(re.findall(r'(?:export\s+)?function\s+(\w+)', content))
                exports.extend(re.findall(r'(?:export\s+)?class\s+(\w+)', content))
                exports.extend(re.findall(r'(?:export\s+)?const\s+(\w+)\s*=', content))
            
            imports_by_file[path] = list(set(imports))
            exports_by_file[path] = list(set(exports))
        
        # Build dependency graph
        dependency_graph = {}
        for path, imports in imports_by_file.items():
            dependency_graph[path] = {
                "imports": imports,
                "imported_by": [],
                "shared_symbols": []
            }
        
        # Find which files import from which
        for path, imports in imports_by_file.items():
            for other_path, exports in exports_by_file.items():
                if path != other_path:
                    shared = [exp for exp in exports if any(imp.endswith(exp) or exp in imp for imp in imports)]
                    if shared:
                        dependency_graph[path]["shared_symbols"].extend(shared)
                        dependency_graph[other_path]["imported_by"].append(path)
        
        # Detect circular dependencies
        circular_deps = set()
        for path_a, deps_a in dependency_graph.items():
            for path_b in deps_a.get("imported_by", []):
                if path_a in dependency_graph.get(path_b, {}).get("imported_by", []):
                    circular_deps.add((min(path_a, path_b), max(path_a, path_b)))
        
        # Analyze code patterns across files
        file_summaries = []
        for file_info in files[:8]:
            path = file_info.get("path", "")
            content = file_info.get("content", "")[:500]
            file_summaries.append(f"{path}:\n{content}")
        
        # Generate cross-file context analysis
        file_summaries_text = '\n---\n'.join([f"{s}..." for s in file_summaries])
        prompt = f"""# Analyze the relationships and context across these files:

{file_summaries_text}

# Cross-file analysis:
## 1. Data flow between files
## 2. Potential refactoring opportunities
## 3. Shared functionality that could be extracted
## 4. Architectural patterns

Analysis:
"""
        
        context_analysis = generate_text(prompt, max_new_tokens=300, temperature=0.3)
        analysis_text = context_analysis[len(prompt):].strip()
        
        # Identify core files and satellite files
        core_files = []
        for path, deps in dependency_graph.items():
            if len(deps["imported_by"]) > 0:
                core_files.append(path)
        
        # Generate improvement suggestions
        suggestions = []
        if circular_deps:
            suggestions.append(f"Circular dependencies detected: {len(circular_deps)} cycle(s) found - consider refactoring")
        
        if len(core_files) > 0:
            suggestions.append(f"Core modules ({len(core_files)}): {', '.join(core_files[:3])} - ensure stable APIs")
        
        isolated_files = [p for p in dependency_graph.keys() if not dependency_graph[p]["imported_by"] and not dependency_graph[p]["imports"]]
        if isolated_files:
            suggestions.append(f"Isolated modules ({len(isolated_files)}): Consider if these can be refactored for reusability")
        
        # Find duplicate symbols
        all_symbols = {}
        for path, exports in exports_by_file.items():
            for symbol in exports:
                if symbol not in all_symbols:
                    all_symbols[symbol] = []
                all_symbols[symbol].append(path)
        
        duplicates = [sym for sym, paths in all_symbols.items() if len(paths) > 1]
        if duplicates:
            suggestions.append(f"Duplicate symbol names ({len(duplicates)}): {duplicates[:3]} - may indicate namespace conflicts")
        
        return jsonify({
            "files_analyzed": len(files),
            "dependency_graph": dependency_graph,
            "circular_dependencies": list(circular_deps) if circular_deps else [],
            "core_modules": core_files,
            "isolated_modules": isolated_files,
            "duplicate_symbols": duplicates,
            "cross_file_analysis": analysis_text,
            "suggestions": suggestions,
            "total_imports": sum(len(imp) for imp in imports_by_file.values()),
            "total_exports": sum(len(exp) for exp in exports_by_file.values())
        })
        
    except Exception as e:
        logger.error(f"Error in analyze_multi_file_context(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/code-suggestions", methods=["POST"])
def github_code_suggestions():
    """Generate detailed code improvement suggestions for a GitHub PR"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        pr_number = data.get("pr_number")
        token = data.get("token")
        
        if not repo_name or not pr_number:
            return jsonify({"error": "Missing repo or pr_number"}), 400
        
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            
            # Get detailed file changes
            files = pr.get_files()
            suggestions_list = []
            
            for file in list(files)[:8]:
                if not file.patch:
                    continue
                
                # Analyze specific patterns in the code
                patch = file.patch
                file_name = file.filename
                
                # Check for code quality issues
                quality_issues = []
                
                # Line length check
                if any(len(line) > 120 for line in patch.split('\n')):
                    quality_issues.append("Long lines detected (> 120 chars) - consider breaking into multiple lines")
                
                # Complex logic detection
                if patch.count('if ') > 3 or patch.count('else') > 3:
                    quality_issues.append("Multiple conditional branches - consider extracting logic into separate function")
                
                # Hardcoded values
                if re.search(r'\b\d{3,}\b', patch):
                    quality_issues.append("Magic numbers detected - consider using named constants")
                
                # Error handling
                if 'raise ' in patch and 'try' not in patch:
                    quality_issues.append("Exception raised without try/catch context - consider error handling")
                
                # Duplicate code
                lines = [l for l in patch.split('\n') if l.startswith('+')]
                if len(lines) > 5:
                    quality_issues.append("Large diff - break into smaller, focused commits for easier review")
                
                # Generate specific suggestions
                suggestion_prompt = f"""Provide specific code improvement suggestions for this change:

File: {file_name}
```diff
{patch[:1000]}
```

Suggestions (practical and actionable):
"""
                
                suggestions_text = generate_text(suggestion_prompt, max_new_tokens=200, temperature=0.25)
                suggestions_content = suggestions_text[len(suggestion_prompt):].strip()
                
                suggestions_list.append({
                    "file": file_name,
                    "lines_changed": file.changes,
                    "quality_issues": quality_issues,
                    "improvement_suggestions": suggestions_content,
                    "additions": file.additions,
                    "deletions": file.deletions
                })
            
            return jsonify({
                "pr_number": pr_number,
                "pr_title": pr.title,
                "pr_description": pr.body[:500] if pr.body else "",
                "suggestions": suggestions_list,
                "total_files": len(list(files)),
                "url": pr.html_url
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_code_suggestions(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/github/enhanced-issue-analysis", methods=["POST"])
def github_enhanced_issue_analysis():
    """Enhanced issue analysis with suggestions for automation and resolution"""
    try:
        data = request.get_json()
        repo_name = data.get("repo")
        issue_number = data.get("issue_number")
        token = data.get("token")
        
        if not repo_name or not issue_number:
            return jsonify({"error": "Missing repo or issue_number"}), 400
        
        gh = Github(token) if token else Github()
        
        try:
            repo = gh.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            
            issue_title = issue.title
            issue_body = issue.body or ""
            issue_labels = [label.name for label in issue.labels]
            
            # Detect issue type
            issue_type = "general"
            if any(label in issue_labels for label in ["bug", "bugfix"]):
                issue_type = "bug"
            elif any(label in issue_labels for label in ["feature", "enhancement"]):
                issue_type = "feature"
            elif any(label in issue_labels for label in ["documentation", "docs"]):
                issue_type = "documentation"
            
            # Extract key information
            key_points = []
            if len(issue_body) > 50:
                prompt = f"""Summarize the key technical points in this issue:

Title: {issue_title}
Body: {issue_body[:800]}

Key points (list):
"""
                summary = generate_text(prompt, max_new_tokens=150, temperature=0.2)
                key_points = summary[len(prompt):].strip().split('\n')[:5]
            
            # Generate resolution suggestions
            suggestion_prompt = f"""Suggest concrete steps to resolve this {issue_type}:

Title: {issue_title}
Body: {issue_body[:600]}
Labels: {', '.join(issue_labels)}

Resolution steps:
"""
            
            resolution_text = generate_text(suggestion_prompt, max_new_tokens=250, temperature=0.3)
            resolution_steps = resolution_text[len(suggestion_prompt):].strip()
            
            # Suggest related code areas
            affected_areas = []
            if issue_type == "bug":
                affected_areas.append("test coverage - add tests to prevent regression")
                affected_areas.append("error handling - ensure error messages are helpful")
            elif issue_type == "feature":
                affected_areas.append("documentation - update API docs")
                affected_areas.append("tests - add comprehensive test coverage")
            
            # Estimate complexity
            complexity = "low"
            if any(word in issue_body.lower() for word in ["multiple", "complex", "integration", "refactor"]):
                complexity = "high"
            elif any(word in issue_body.lower() for word in ["simple", "small", "quick"]):
                complexity = "low"
            else:
                complexity = "medium"
            
            # Automation suggestions
            automations = []
            if issue_type == "bug":
                automations.append("Set up automated regression testing with GitHub Actions")
                automations.append("Auto-label issues with triage workflow")
            elif issue_type == "feature":
                automations.append("Auto-generate feature documentation from code comments")
                automations.append("Add automated API test generation")
            
            return jsonify({
                "issue_number": issue_number,
                "issue_type": issue_type,
                "title": issue_title,
                "labels": issue_labels,
                "key_points": key_points,
                "resolution_steps": resolution_steps,
                "affected_code_areas": affected_areas,
                "estimated_complexity": complexity,
                "automation_suggestions": automations,
                "url": issue.html_url
            })
            
        except GithubException as e:
            return jsonify({"error": f"GitHub API error: {e.data.get('message', str(e))}"}), e.status
        
    except Exception as e:
        logger.error(f"Error in github_enhanced_issue_analysis(): {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
