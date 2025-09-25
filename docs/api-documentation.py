#!/usr/bin/env python3
"""
API Documentation Generator for Python for Semiconductors Learning Series

This module automatically generates comprehensive API documentation for all
pipeline classes, assessment systems, and interactive widgets.

Usage:
    python api-documentation.py generate --output docs/api --format html
    python api-documentation.py serve --port 8080
"""

import argparse
import inspect
import json
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import sys
import os
import importlib.util
from datetime import datetime

# Add modules to path
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class ClassDocumentation:
    """Documentation for a single class."""
    name: str
    module: str
    docstring: str
    methods: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    inheritance: List[str]
    examples: List[str]

@dataclass
class FunctionDocumentation:
    """Documentation for a function."""
    name: str
    module: str
    signature: str
    docstring: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, str]
    examples: List[str]

@dataclass
class ModuleDocumentation:
    """Documentation for a complete module."""
    name: str
    path: str
    docstring: str
    classes: List[ClassDocumentation]
    functions: List[FunctionDocumentation]
    constants: List[Dict[str, Any]]

class APIDocumentationGenerator:
    """Generates comprehensive API documentation."""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modules_docs: List[ModuleDocumentation] = []
    
    def discover_modules(self) -> List[Path]:
        """Discover all Python modules in the source directory."""
        python_files = []
        
        # Find all Python files
        for pattern in ['**/*.py', '**/test_*.py']:
            python_files.extend(self.source_dir.glob(pattern))
        
        # Filter out __pycache__ and other irrelevant files
        filtered_files = []
        for file in python_files:
            if '__pycache__' not in str(file) and file.name != '__init__.py':
                filtered_files.append(file)
        
        return filtered_files
    
    def analyze_module(self, module_path: Path) -> Optional[ModuleDocumentation]:
        """Analyze a single Python module."""
        try:
            # Read and parse the module
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree) or ""
            
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(
                module_path.stem, module_path
            )
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Warning: Could not execute module {module_path}: {e}")
                # Continue with AST analysis only
                return self._analyze_ast_only(module_path, tree, module_docstring)
            
            # Analyze classes and functions
            classes_docs = []
            functions_docs = []
            constants = []
            
            # Get all classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module.__name__:  # Only classes defined in this module
                    class_doc = self._analyze_class(obj)
                    classes_docs.append(class_doc)
            
            # Get all functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if obj.__module__ == module.__name__:  # Only functions defined in this module
                    func_doc = self._analyze_function(obj)
                    functions_docs.append(func_doc)
            
            # Get constants (simple values at module level)
            for name in dir(module):
                if not name.startswith('_') and name.isupper():
                    value = getattr(module, name)
                    if isinstance(value, (int, float, str, bool)):
                        constants.append({
                            'name': name,
                            'value': repr(value),
                            'type': type(value).__name__
                        })
            
            return ModuleDocumentation(
                name=module_path.stem,
                path=str(module_path.relative_to(self.source_dir)),
                docstring=module_docstring,
                classes=classes_docs,
                functions=functions_docs,
                constants=constants
            )
            
        except Exception as e:
            print(f"Error analyzing module {module_path}: {e}")
            return None
    
    def _analyze_ast_only(self, module_path: Path, tree: ast.AST, module_docstring: str) -> ModuleDocumentation:
        """Analyze module using only AST when dynamic import fails."""
        classes_docs = []
        functions_docs = []
        constants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ClassDocumentation(
                    name=node.name,
                    module=module_path.stem,
                    docstring=ast.get_docstring(node) or "",
                    methods=[],
                    attributes=[],
                    inheritance=[base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                    examples=[]
                )
                classes_docs.append(class_doc)
            
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                func_doc = FunctionDocumentation(
                    name=node.name,
                    module=module_path.stem,
                    signature=f"{node.name}(...)",
                    docstring=ast.get_docstring(node) or "",
                    parameters=[],
                    returns={'type': 'Unknown', 'description': ''},
                    examples=[]
                )
                functions_docs.append(func_doc)
        
        return ModuleDocumentation(
            name=module_path.stem,
            path=str(module_path.relative_to(self.source_dir)),
            docstring=module_docstring,
            classes=classes_docs,
            functions=functions_docs,
            constants=constants
        )
    
    def _analyze_class(self, cls: Type) -> ClassDocumentation:
        """Analyze a single class."""
        methods = []
        attributes = []
        
        # Get all methods
        for name, method in inspect.getmembers(cls, inspect.ismethod):
            if not name.startswith('_') or name in ['__init__', '__call__']:
                methods.append(self._analyze_method(method))
        
        # Get all functions (unbound methods)
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_') or name in ['__init__', '__call__']:
                methods.append(self._analyze_method(func))
        
        # Get class attributes
        for name in dir(cls):
            if not name.startswith('_') and not callable(getattr(cls, name)):
                attr_value = getattr(cls, name)
                attributes.append({
                    'name': name,
                    'type': type(attr_value).__name__,
                    'value': repr(attr_value) if isinstance(attr_value, (int, float, str, bool)) else '...'
                })
        
        # Get inheritance
        inheritance = [base.__name__ for base in cls.__bases__ if base != object]
        
        return ClassDocumentation(
            name=cls.__name__,
            module=cls.__module__,
            docstring=inspect.getdoc(cls) or "",
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
            examples=self._extract_examples_from_docstring(inspect.getdoc(cls) or "")
        )
    
    def _analyze_method(self, method) -> Dict[str, Any]:
        """Analyze a single method."""
        try:
            signature = inspect.signature(method)
            parameters = []
            
            for param_name, param in signature.parameters.items():
                parameters.append({
                    'name': param_name,
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                })
            
            return {
                'name': method.__name__,
                'signature': str(signature),
                'docstring': inspect.getdoc(method) or "",
                'parameters': parameters,
                'returns': {
                    'type': str(signature.return_annotation) if signature.return_annotation != inspect.Signature.empty else 'Any',
                    'description': ''
                },
                'examples': self._extract_examples_from_docstring(inspect.getdoc(method) or "")
            }
        except Exception as e:
            return {
                'name': getattr(method, '__name__', 'unknown'),
                'signature': 'unknown',
                'docstring': '',
                'parameters': [],
                'returns': {'type': 'Any', 'description': ''},
                'examples': []
            }
    
    def _analyze_function(self, func) -> FunctionDocumentation:
        """Analyze a single function."""
        try:
            signature = inspect.signature(func)
            parameters = []
            
            for param_name, param in signature.parameters.items():
                parameters.append({
                    'name': param_name,
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                    'required': param.default == inspect.Parameter.empty
                })
            
            return FunctionDocumentation(
                name=func.__name__,
                module=func.__module__,
                signature=str(signature),
                docstring=inspect.getdoc(func) or "",
                parameters=parameters,
                returns={
                    'type': str(signature.return_annotation) if signature.return_annotation != inspect.Signature.empty else 'Any',
                    'description': ''
                },
                examples=self._extract_examples_from_docstring(inspect.getdoc(func) or "")
            )
        except Exception as e:
            return FunctionDocumentation(
                name=getattr(func, '__name__', 'unknown'),
                module='unknown',
                signature='unknown',
                docstring='',
                parameters=[],
                returns={'type': 'Any', 'description': ''},
                examples=[]
            )
    
    def _extract_examples_from_docstring(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split('\n')
        
        in_example = False
        current_example = []
        
        for line in lines:
            stripped = line.strip()
            
            if 'Example' in stripped or 'Usage' in stripped or '>>>' in stripped:
                in_example = True
                if current_example:
                    examples.append('\n'.join(current_example))
                    current_example = []
            
            if in_example:
                if stripped.startswith('>>>') or stripped.startswith('...'):
                    current_example.append(line)
                elif stripped and not stripped.startswith('>>>') and current_example:
                    # End of example block
                    if current_example:
                        examples.append('\n'.join(current_example))
                    current_example = []
                    in_example = False
        
        if current_example:
            examples.append('\n'.join(current_example))
        
        return examples
    
    def generate_documentation(self) -> None:
        """Generate complete API documentation."""
        print("🔍 Discovering modules...")
        modules = self.discover_modules()
        print(f"Found {len(modules)} modules")
        
        print("📝 Analyzing modules...")
        for module_path in modules:
            print(f"  Analyzing {module_path.name}...")
            module_doc = self.analyze_module(module_path)
            if module_doc:
                self.modules_docs.append(module_doc)
        
        print(f"✅ Analyzed {len(self.modules_docs)} modules successfully")
        
        # Generate different output formats
        self._generate_json_docs()
        self._generate_html_docs()
        self._generate_markdown_docs()
    
    def _generate_json_docs(self) -> None:
        """Generate JSON API documentation."""
        output_file = self.output_dir / 'api-docs.json'
        
        docs_data = {
            'generated_at': datetime.now().isoformat(),
            'modules': [asdict(module) for module in self.modules_docs]
        }
        
        with open(output_file, 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"📄 Generated JSON docs: {output_file}")
    
    def _generate_html_docs(self) -> None:
        """Generate HTML API documentation."""
        html_dir = self.output_dir / 'html'
        html_dir.mkdir(exist_ok=True)
        
        # Generate index page
        index_html = self._generate_html_index()
        with open(html_dir / 'index.html', 'w') as f:
            f.write(index_html)
        
        # Generate individual module pages
        for module in self.modules_docs:
            module_html = self._generate_html_module(module)
            with open(html_dir / f'{module.name}.html', 'w') as f:
                f.write(module_html)
        
        print(f"🌐 Generated HTML docs: {html_dir}")
    
    def _generate_html_index(self) -> str:
        """Generate HTML index page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Python for Semiconductors - API Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #2196F3; color: white; padding: 20px; border-radius: 5px; }
        .module-list { margin-top: 30px; }
        .module-item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .module-item h3 { margin-top: 0; }
        .stats { display: flex; gap: 20px; margin: 10px 0; }
        .stat { background: #f5f5f5; padding: 5px 10px; border-radius: 3px; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Python for Semiconductors - API Documentation</h1>
        <p>Comprehensive API reference for the ML learning series</p>
    </div>
    
    <div class="module-list">
        <h2>Available Modules</h2>
"""
        
        for module in self.modules_docs:
            html += f"""
        <div class="module-item">
            <h3><a href="{module.name}.html">{module.name}</a></h3>
            <p>{module.docstring[:200]}{'...' if len(module.docstring) > 200 else ''}</p>
            <div class="stats">
                <span class="stat">Classes: {len(module.classes)}</span>
                <span class="stat">Functions: {len(module.functions)}</span>
                <span class="stat">Constants: {len(module.constants)}</span>
            </div>
        </div>
"""
        
        html += """
    </div>
    
    <div style="margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px; font-size: 0.9em; color: #666;">
        Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_html_module(self, module: ModuleDocumentation) -> str:
        """Generate HTML for a single module."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{module.name} - API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .class-item, .function-item {{ border: 1px solid #ddd; margin: 15px 0; padding: 15px; border-radius: 5px; }}
        .method {{ margin-left: 20px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        .signature {{ font-family: monospace; background: #f0f0f0; padding: 5px; border-radius: 3px; }}
        .docstring {{ margin: 10px 0; line-height: 1.6; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{module.name}</h1>
        <p>Module: {module.path}</p>
    </div>
    
    <div class="section">
        <h2>Module Description</h2>
        <div class="docstring">{module.docstring.replace('\n', '<br>')}</div>
    </div>
"""
        
        if module.constants:
            html += """
    <div class="section">
        <h2>Constants</h2>
"""
            for const in module.constants:
                html += f"""
        <div><code>{const['name']}</code> = {const['value']} <em>({const['type']})</em></div>
"""
            html += "</div>"
        
        if module.classes:
            html += """
    <div class="section">
        <h2>Classes</h2>
"""
            for cls in module.classes:
                html += f"""
        <div class="class-item">
            <h3>{cls.name}</h3>
            <div class="docstring">{cls.docstring.replace('\n', '<br>')}</div>
"""
                if cls.inheritance:
                    html += f"<p><strong>Inherits from:</strong> {', '.join(cls.inheritance)}</p>"
                
                if cls.methods:
                    html += "<h4>Methods</h4>"
                    for method in cls.methods:
                        html += f"""
            <div class="method">
                <strong>{method['name']}</strong>
                <div class="signature">{method['signature']}</div>
                <div class="docstring">{method['docstring'].replace('\n', '<br>')}</div>
            </div>
"""
                
                html += "</div>"
            
            html += "</div>"
        
        if module.functions:
            html += """
    <div class="section">
        <h2>Functions</h2>
"""
            for func in module.functions:
                html += f"""
        <div class="function-item">
            <h3>{func.name}</h3>
            <div class="signature">{func.signature}</div>
            <div class="docstring">{func.docstring.replace('\n', '<br>')}</div>
"""
                if func.examples:
                    html += "<h4>Examples</h4>"
                    for example in func.examples:
                        html += f"<pre>{example}</pre>"
                
                html += "</div>"
            
            html += "</div>"
        
        html += """
    <div style="margin-top: 40px;">
        <a href="index.html">← Back to API Index</a>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_markdown_docs(self) -> None:
        """Generate Markdown API documentation."""
        md_dir = self.output_dir / 'markdown'
        md_dir.mkdir(exist_ok=True)
        
        # Generate index
        index_md = "# Python for Semiconductors - API Documentation\n\n"
        index_md += "## Available Modules\n\n"
        
        for module in self.modules_docs:
            index_md += f"- [{module.name}]({module.name}.md) - {module.docstring.split('.')[0] if module.docstring else 'No description'}\n"
        
        with open(md_dir / 'README.md', 'w') as f:
            f.write(index_md)
        
        # Generate individual module docs
        for module in self.modules_docs:
            module_md = self._generate_markdown_module(module)
            with open(md_dir / f'{module.name}.md', 'w') as f:
                f.write(module_md)
        
        print(f"📝 Generated Markdown docs: {md_dir}")
    
    def _generate_markdown_module(self, module: ModuleDocumentation) -> str:
        """Generate Markdown for a single module."""
        md = f"# {module.name}\n\n"
        md += f"**Module:** `{module.path}`\n\n"
        
        if module.docstring:
            md += f"{module.docstring}\n\n"
        
        if module.constants:
            md += "## Constants\n\n"
            for const in module.constants:
                md += f"- `{const['name']}` = `{const['value']}` _{const['type']}_\n"
            md += "\n"
        
        if module.classes:
            md += "## Classes\n\n"
            for cls in module.classes:
                md += f"### {cls.name}\n\n"
                if cls.docstring:
                    md += f"{cls.docstring}\n\n"
                
                if cls.inheritance:
                    md += f"**Inherits from:** {', '.join(cls.inheritance)}\n\n"
                
                if cls.methods:
                    md += "#### Methods\n\n"
                    for method in cls.methods:
                        md += f"##### {method['name']}\n\n"
                        md += f"```python\n{method['signature']}\n```\n\n"
                        if method['docstring']:
                            md += f"{method['docstring']}\n\n"
        
        if module.functions:
            md += "## Functions\n\n"
            for func in module.functions:
                md += f"### {func.name}\n\n"
                md += f"```python\n{func.signature}\n```\n\n"
                if func.docstring:
                    md += f"{func.docstring}\n\n"
        
        return md

def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="API Documentation Generator for Python for Semiconductors"
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate API documentation')
    generate_parser.add_argument('--source', default='../modules', help='Source directory to analyze')
    generate_parser.add_argument('--output', default='./api-docs', help='Output directory')
    generate_parser.add_argument('--format', choices=['all', 'html', 'markdown', 'json'], 
                                default='all', help='Output format')
    generate_parser.set_defaults(func=action_generate)
    
    return parser

def action_generate(args) -> None:
    """Handle generate command."""
    try:
        source_dir = Path(args.source).resolve()
        output_dir = Path(args.output).resolve()
        
        if not source_dir.exists():
            print(f"❌ Source directory not found: {source_dir}")
            return
        
        print(f"🚀 Generating API documentation...")
        print(f"   Source: {source_dir}")
        print(f"   Output: {output_dir}")
        
        generator = APIDocumentationGenerator(source_dir, output_dir)
        generator.generate_documentation()
        
        print(f"✅ Documentation generated successfully!")
        print(f"   View HTML docs: {output_dir}/html/index.html")
        print(f"   View Markdown docs: {output_dir}/markdown/README.md")
        
    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()