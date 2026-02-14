# Language and Framework Support Configuration
# Comprehensive language and framework support for Vincent Copilot

import re
from typing import Dict, List, Tuple, Optional

# Supported languages mapping with file extensions
SUPPORTED_LANGUAGES = {
    # Python
    "python": {
        "extensions": [".py"],
        "syntax_comment": "#",
        "block_comment": ('"""', '"""'),
        "frameworks": ["Django", "Flask", "FastAPI", "Celery", "SQLAlchemy", "NumPy", "Pandas", "PyTorch", "TensorFlow"],
        "test_frameworks": ["pytest", "unittest", "nose2", "hypothesis"],
        "package_manager": "pip",
        "info": "Python 3.8+"
    },
    
    # JavaScript
    "javascript": {
        "extensions": [".js", ".jsx"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["React", "Vue", "Angular", "Express.js", "Next.js", "Node.js", "Nuxt", "Svelte"],
        "test_frameworks": ["jest", "mocha", "vitest", "Vitest"],
        "package_manager": "npm/yarn",
        "info": "ES6+"
    },
    
    # TypeScript
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["React", "Vue", "Angular", "Express", "Next.js", "NestJS", "Svelte", "Deno"],
        "test_frameworks": ["jest", "vitest", "mocha", "Jasmine"],
        "package_manager": "npm/yarn",
        "info": "TypeScript 4.5+"
    },
    
    # Java
    "java": {
        "extensions": [".java"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Spring Boot", "Spring Framework", "Hibernate", "Quarkus", "Micronaut", "Grails"],
        "test_frameworks": ["JUnit5", "TestNG", "Mockito", "AssertJ"],
        "package_manager": "Maven/Gradle",
        "info": "Java 11+"
    },
    
    # C++
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Boost", "Qt", "OpenGL", "Unreal Engine"],
        "test_frameworks": ["GoogleTest", "Catch2", "CppUnit"],
        "package_manager": "conan/vcpkg",
        "info": "C++17+"
    },
    
    # Go
    "go": {
        "extensions": [".go"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Gin", "Echo", "Fiber", "GORM", "Beego"],
        "test_frameworks": ["testing", "Testify", "GoConvey"],
        "package_manager": "go get",
        "info": "Go 1.18+"
    },
    
    # Rust
    "rust": {
        "extensions": [".rs"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Tokio", "Actix-web", "Rocket", "Diesel", "Serde"],
        "test_frameworks": ["Criterion", "Proptest", "Assertion libraries"],
        "package_manager": "Cargo",
        "info": "Rust 1.60+"
    },
    
    # C#
    "csharp": {
        "extensions": [".cs"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["ASP.NET", ".NET Core", "Entity Framework", "Unity", "WPF"],
        "test_frameworks": ["xUnit", "NUnit", "Moq"],
        "package_manager": "NuGet",
        "info": ".NET 6+"
    },
    
    # PHP
    "php": {
        "extensions": [".php"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Laravel", "Symfony", "CodeIgniter", "Yii", "WordPress"],
        "test_frameworks": ["PHPUnit", "Pest", "Behat"],
        "package_manager": "Composer",
        "info": "PHP 8.0+"
    },
    
    # Ruby
    "ruby": {
        "extensions": [".rb"],
        "syntax_comment": "#",
        "block_comment": ("=begin", "=end"),
        "frameworks": ["Rails", "Sinatra", "Hanami", "Padrino"],
        "test_frameworks": ["RSpec", "Minitest", "Test::Unit"],
        "package_manager": "Bundler",
        "info": "Ruby 2.7+"
    },
    
    # Kotlin
    "kotlin": {
        "extensions": [".kt", ".kts"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Ktor", "Spring Boot", "Android", "Exposed"],
        "test_frameworks": ["JUnit", "Spek", "Kotest"],
        "package_manager": "Maven/Gradle",
        "info": "Kotlin 1.6+"
    },
    
    # Swift
    "swift": {
        "extensions": [".swift"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Combine", "SwiftUI", "ARKit", "CoreData"],
        "test_frameworks": ["XCTest", "Quick/Nimble"],
        "package_manager": "Swift Package Manager",
        "info": "Swift 5.5+"
    },
    
    # Scala
    "scala": {
        "extensions": [".scala"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Play Framework", "Akka", "Spark", "Scalatra"],
        "test_frameworks": ["ScalaTest", "Specs2", "Scalacheck"],
        "package_manager": "sbt",
        "info": "Scala 3.0+"
    },
    
    # R
    "r": {
        "extensions": [".r", ".R"],
        "syntax_comment": "#",
        "block_comment": ("#", "#"),
        "frameworks": ["Shiny", "ggplot2", "dplyr", "data.table"],
        "test_frameworks": ["testthat", "RUnit", "vdiffr"],
        "package_manager": "CRAN/renv",
        "info": "R 4.0+"
    },
    
    # SQL
    "sql": {
        "extensions": [".sql"],
        "syntax_comment": "--",
        "block_comment": ("/*", "*/"),
        "frameworks": ["T-SQL", "PL/pgSQL", "MySQL", "PostgreSQL"],
        "test_frameworks": ["SQLTest", "TSQLUnit"],
        "package_manager": "N/A",
        "info": "SQL standard"
    },
    
    # Shell/Bash
    "bash": {
        "extensions": [".sh", ".bash"],
        "syntax_comment": "#",
        "block_comment": (":", ":"),
        "frameworks": ["GNU Coreutils", "BusyBox"],
        "test_frameworks": ["Bash Automated Testing System", "shUnit2"],
        "package_manager": "apt/brew/yum",
        "info": "Bash 4.0+"
    },
    
    # Dart
    "dart": {
        "extensions": [".dart"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Flutter", "Aqueduct", "Angel", "Shelf"],
        "test_frameworks": ["test", "mockito", "flutter_test"],
        "package_manager": "pub",
        "info": "Dart 2.17+"
    },
    
    # Elixir
    "elixir": {
        "extensions": [".ex", ".exs"],
        "syntax_comment": "#",
        "block_comment": ("@doc \"\"\"", "\"\"\""),
        "frameworks": ["Phoenix", "Nerves", "Ecto", "Plug"],
        "test_frameworks": ["ExUnit", "Wallaby", "Mox"],
        "package_manager": "Mix",
        "info": "Elixir 1.12+"
    },
    
    # Perl
    "perl": {
        "extensions": [".pl", ".pm"],
        "syntax_comment": "#",
        "block_comment": ("=pod", "=cut"),
        "frameworks": ["Catalyst", "Mojolicious", "Dancer2", "CGI"],
        "test_frameworks": ["Test::More", "Test::Simple", "Test::Most"],
        "package_manager": "cpan/cpanm",
        "info": "Perl 5.30+"
    },
    
    # Lua
    "lua": {
        "extensions": [".lua"],
        "syntax_comment": "--",
        "block_comment": ("--[[", "]]"),
        "frameworks": ["Love2D", "Lapis", "OpenResty", "Corona"],
        "test_frameworks": ["busted", "LuaUnit", "Telescope"],
        "package_manager": "LuaRocks",
        "info": "Lua 5.4+"
    },
    
    # Objective-C
    "objective-c": {
        "extensions": [".m", ".mm", ".h"],
        "syntax_comment": "//",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Cocoa", "UIKit", "CoreData", "Foundation"],
        "test_frameworks": ["XCTest", "OCMock", "Specta"],
        "package_manager": "CocoaPods",
        "info": "Objective-C 2.0+"
    },
    
    # YAML
    "yaml": {
        "extensions": [".yml", ".yaml"],
        "syntax_comment": "#",
        "block_comment": ("#", "#"),
        "frameworks": ["Ansible", "Kubernetes", "Docker Compose", "CI/CD"],
        "test_frameworks": ["yamllint", "YAML validator"],
        "package_manager": "N/A",
        "info": "YAML 1.2"
    },
    
    # HTML/CSS
    "html": {
        "extensions": [".html", ".htm"],
        "syntax_comment": "<!--",
        "block_comment": ("<!--", "-->"),
        "frameworks": ["Bootstrap", "Tailwind CSS", "Bulma", "HTML5"],
        "test_frameworks": ["HTML validator", "axe-core"],
        "package_manager": "npm/cdn",
        "info": "HTML5"
    },
    
    "css": {
        "extensions": [".css", ".scss", ".sass", ".less"],
        "syntax_comment": "/*",
        "block_comment": ("/*", "*/"),
        "frameworks": ["Bootstrap", "Tailwind", "SASS", "LESS", "PostCSS"],
        "test_frameworks": ["Stylelint", "CSS validator"],
        "package_manager": "npm",
        "info": "CSS3"
    },
    
    # Haskell
    "haskell": {
        "extensions": [".hs", ".lhs"],
        "syntax_comment": "--",
        "block_comment": ("{-", "-}"),
        "frameworks": ["Yesod", "Snap", "Servant", "Reflex"],
        "test_frameworks": ["HUnit", "QuickCheck", "Hspec"],
        "package_manager": "Cabal/Stack",
        "info": "GHC 9.0+"
    },
}

def detect_language(filename: str) -> Optional[str]:
    """Detect language from file extension"""
    ext = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    
    for lang, config in SUPPORTED_LANGUAGES.items():
        if ext in config['extensions']:
            return lang
    
    return None

def detect_language_from_code(code: str) -> Optional[str]:
    """Detect language from code content (shebang and patterns)"""
    lines = code.split('\n')[:5]
    
    # Check shebang
    if lines and lines[0].startswith('#!'):
        shebang = lines[0].lower()
        if 'python' in shebang:
            return 'python'
        elif 'ruby' in shebang:
            return 'ruby'
        elif 'bash' in shebang or 'sh' in shebang:
            return 'bash'
    
    # Pattern matching
    if 'import ' in code and 'def ' in code:
        return 'python'
    elif 'function ' in code or 'const ' in code or 'let ' in code:
        return 'javascript'
    elif 'class ' in code and 'public ' in code:
        return 'java'
    elif 'fn ' in code and 'let ' in code:
        return 'rust'
    elif 'def ' in code and code.count(' do ') > 0:
        return 'ruby'
    
    return 'python'  # Default to Python

def get_language_config(language: str) -> Dict:
    """Get configuration for a specific language"""
    lang_lower = language.lower()
    return SUPPORTED_LANGUAGES.get(lang_lower, SUPPORTED_LANGUAGES['python'])

def get_test_framework_for_language(language: str) -> str:
    """Get primary test framework for language"""
    config = get_language_config(language)
    frameworks = config.get('test_frameworks', [])
    return frameworks[0] if frameworks else 'generic'

def get_syntax_for_language(language: str) -> Tuple[str, Tuple[str, str]]:
    """Get comment syntax for a language"""
    config = get_language_config(language)
    return config['syntax_comment'], config['block_comment']

def get_framework_info(language: str, framework: str) -> Optional[Dict]:
    """Get information about a specific framework"""
    config = get_language_config(language)
    frameworks = config.get('frameworks', [])
    
    # Framework-specific patterns
    framework_patterns = {
        # Python
        'Django': {
            'patterns': ['views.py', 'models.py', 'urls.py', 'settings.py'],
            'imports': ['django.', 'from django'],
            'structure': 'MVT (Model-View-Template)'
        },
        'Flask': {
            'patterns': ['@app.route', 'Blueprint', 'app.run()'],
            'imports': ['from flask', 'flask.'],
            'structure': 'Minimal routing framework'
        },
        'FastAPI': {
            'patterns': ['@app.get', '@app.post', 'FastAPI()', 'Depends'],
            'imports': ['from fastapi', 'fastapi.'],
            'structure': 'Async API framework'
        },
        # JavaScript/TypeScript
        'React': {
            'patterns': ['useState', 'useEffect', 'Component', 'JSX'],
            'imports': ['react', 'from react'],
            'structure': 'Component-based UI library'
        },
        'Vue': {
            'patterns': ['<template>', '<script>', '<style>', 'v-if'],
            'imports': ['vue', 'from vue'],
            'structure': 'Progressive framework'
        },
        'Angular': {
            'patterns': ['@Component', '@Injectable', 'ngOnInit', 'Observable'],
            'imports': ['@angular', 'from @angular'],
            'structure': 'Full-featured framework'
        },
        'Express.js': {
            'patterns': ['app.get', 'app.post', 'router.', 'middleware'],
            'imports': ['express', 'from express'],
            'structure': 'Web framework'
        },
        # Java
        'Spring Boot': {
            'patterns': ['@SpringBootApplication', '@Controller', '@Service', '@Repository'],
            'imports': ['org.springframework', 'spring-boot'],
            'structure': 'Enterprise framework'
        },
        # Go
        'Gin': {
            'patterns': ['gin.Engine', 'gin.GET', 'gin.POST'],
            'imports': ['github.com/gin-gonic/gin'],
            'structure': 'Web framework'
        },
    }
    
    if framework in framework_patterns:
        return framework_patterns[framework]
    
    return {'patterns': [], 'imports': [], 'structure': framework}

def detect_frameworks(code: str, language: str) -> List[str]:
    """Detect which frameworks are being used in the code"""
    detected = []
    config = get_language_config(language)
    frameworks = config.get('frameworks', [])
    
    code_lower = code.lower()
    
    for framework in frameworks:
        framework_info = get_framework_info(language, framework)
        
        # Check imports
        for import_pattern in framework_info.get('imports', []):
            if import_pattern.lower() in code_lower:
                detected.append(framework)
                break
        
        # Check patterns
        for pattern in framework_info.get('patterns', []):
            if pattern in code:
                if framework not in detected:
                    detected.append(framework)
                break
    
    return detected

def get_language_list() -> Dict[str, str]:
    """Get list of all supported languages with descriptions"""
    result = {}
    for lang, config in SUPPORTED_LANGUAGES.items():
        result[lang] = f"{config['info']} - {len(config['frameworks'])} frameworks"
    return result

def get_frameworks_for_language(language: str) -> List[str]:
    """Get list of supported frameworks for a language"""
    config = get_language_config(language)
    return config.get('frameworks', [])

def get_test_prompt_template(language: str, code: str, test_framework: str, edge_cases: List[str]) -> str:
    """Generate language-specific test generation prompt"""
    config = get_language_config(language)
    syntax_comment, block_comment = config['syntax_comment'], config['block_comment']
    
    edge_cases_str = '\n'.join([f"{syntax_comment} - {ec}" for ec in edge_cases[:8]])
    
    lang_lower = language.lower()
    
    if lang_lower == "python":
        return f"""{block_comment[0]} Code to test:
{code}

{block_comment[1]}

{syntax_comment} Generate comprehensive tests using {test_framework}:
{syntax_comment} Edge cases:
{edge_cases_str}
{syntax_comment}
{syntax_comment} Test patterns: arrange-act-assert
import {test_framework}
"""
    
    elif lang_lower in ["javascript", "typescript"]:
        return f"""// Code to test:
{code}

// Generate comprehensive tests using {test_framework}:
// Edge cases:
{edge_cases_str}
//
// Test patterns: arrange-act-assert

import {{ describe, it, expect }} from '{test_framework}';
"""
    
    elif lang_lower == "java":
        return f"""/*
Code to test:
{code}
*/

// Generate comprehensive tests using {test_framework}:
// Edge cases:
{edge_cases_str}
//
// Test patterns: arrange-act-assert

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
"""
    
    elif lang_lower == "go":
        return f"""/*
Code to test:
{code}
*/

// Generate comprehensive tests using {test_framework}:
// Edge cases:
{edge_cases_str}
//
// Test patterns: table-driven tests

package main_test

import (
	"testing"
)
"""
    
    elif lang_lower == "rust":
        return f"""/*
Code to test:
{code}
*/

// Generate comprehensive tests using {test_framework}:
// Edge cases:
{edge_cases_str}
//
// Test patterns: test modules

#[cfg(test)]
mod tests {{
	use super::*;

	#[test]
	fn test_() {{
"""
    
    else:
        return f"""{syntax_comment} Code to test:
{syntax_comment} {code[:100]}
{syntax_comment}
{syntax_comment} Generate comprehensive tests using {test_framework}:
{syntax_comment} Edge cases: {', '.join(edge_cases[:3])}
"""

def get_explanation_prompt_template(language: str, code: str) -> str:
    """Generate language-specific code explanation prompt"""
    config = get_language_config(language)
    syntax_comment = config['syntax_comment']
    
    lang_lower = language.lower()
    
    if lang_lower == "python":
        return f"""{syntax_comment} Explain this {language} code in detail:
{code}

{syntax_comment} Break down:
{syntax_comment} 1. Purpose: 
{syntax_comment} 2. Key functions/classes:
{syntax_comment} 3. Data structures used:
{syntax_comment} 4. Algorithm complexity:
{syntax_comment} 5. Edge cases handled:
"""
    
    elif lang_lower in ["javascript", "typescript"]:
        return f"""// Explain this {language} code in detail:
{code}

// Break down:
// 1. Purpose: 
// 2. Key functions/classes:
// 3. Data structures used:
// 4. Algorithm complexity:
// 5. Edge cases handled:
"""
    
    else:
        return f"""{syntax_comment} Explain this {language} code:
{code}

{syntax_comment} Break down the main components and logic
"""

def get_refactor_suggestions(language: str, code: str) -> List[str]:
    """Get language-specific refactoring suggestions"""
    suggestions = {
        'python': [
            "Use list comprehensions instead of loops",
            "Apply type hints for better code clarity",
            "Use f-strings for string formatting",
            "Break large functions into smaller ones",
            "Use decorators for cross-cutting concerns",
            "Use context managers for resource management"
        ],
        'javascript': [
            "Use const/let instead of var",
            "Use arrow functions for concise syntax",
            "Use destructuring for cleaner code",
            "Use async/await instead of promises",
            "Use template literals for string interpolation",
            "Use optional chaining and nullish coalescing"
        ],
        'java': [
            "Use records for immutable data classes",
            "Use var keyword for local variable type inference",
            "Use sealed classes for restricted hierarchies",
            "Use text blocks for multi-line strings",
            "Use streams for functional operations",
            "Use modules for better code organization"
        ],
        'go': [
            "Use interfaces for code abstraction",
            "Use goroutines for concurrent operations",
            "Use channels for inter-goroutine communication",
            "Defer cleanup operations with defer",
            "Use table-driven tests for comprehensive testing",
            "Use context for request cancellation"
        ]
    }
    
    lang_lower = language.lower()
    return suggestions.get(lang_lower, [
        "Break large functions into smaller ones",
        "Improve variable naming for clarity",
        "Apply DRY principle to reduce duplication",
        "Consider design patterns for reusability"
    ])
