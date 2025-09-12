"""Production Documentation & Reproducibility Pipeline Script for Module 10.3

Provides a CLI to manage documentation generation, dataset path validation, 
and reproducibility practices for semiconductor ML projects.

Features:
- Convert Jupyter notebooks to Markdown/HTML documentation
- Validate dataset path resolution across modules
- Export reproducible environment specifications
- Generate project documentation with mkdocs
- JSON error reporting with comprehensive CLI help

Example usage:
    python 10.3-documentation-reproducibility-pipeline.py generate-docs --input notebooks/ --output docs/
    python 10.3-documentation-reproducibility-pipeline.py validate-paths --modules-dir ../../
    python 10.3-documentation-reproducibility-pipeline.py export-env --output environment.yml --format conda
"""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import tempfile
import shutil

# Check for optional dependencies
HAS_JUPYTER = True
HAS_MKDOCS = True
HAS_NBCONVERT = True

try:
    import nbformat
    from nbconvert import MarkdownExporter, HTMLExporter
except ImportError:
    HAS_NBCONVERT = False

try:
    import yaml
except ImportError:
    yaml = None

RANDOM_SEED = 42

@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    input_dir: Path
    output_dir: Path
    format: str = "markdown"  # markdown, html, both
    include_code: bool = True
    include_outputs: bool = True
    template: Optional[str] = None

@dataclass
class PathValidationResult:
    """Results from dataset path validation."""
    module_path: Path
    notebook_files: List[Path]
    valid_paths: List[str]
    invalid_paths: List[str]
    warnings: List[str]

@dataclass
class EnvironmentExport:
    """Environment export configuration."""
    format: str  # conda, pip, requirements
    output_path: Path
    include_versions: bool = True
    exclude_dev: bool = False

class DocumentationReproducibilityPipeline:
    """Pipeline for documentation generation and reproducibility management."""
    
    def __init__(self, config: Optional[DocumentationConfig] = None):
        self.config = config
        self.results = {}
    
    def generate_documentation(self, 
                             input_dir: Union[str, Path], 
                             output_dir: Union[str, Path],
                             format: str = "markdown") -> Dict[str, Any]:
        """Convert notebooks to documentation format.
        
        Args:
            input_dir: Directory containing Jupyter notebooks
            output_dir: Output directory for generated documentation
            format: Output format (markdown, html, both)
            
        Returns:
            Dictionary with conversion results and statistics
        """
        if not HAS_NBCONVERT:
            return {
                "success": False,
                "error": "nbconvert not available. Install with: pip install nbconvert",
                "converted_files": []
            }
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            return {
                "success": False,
                "error": f"Input directory does not exist: {input_path}",
                "converted_files": []
            }
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        notebook_files = list(input_path.glob("**/*.ipynb"))
        converted_files = []
        errors = []
        
        for nb_file in notebook_files:
            try:
                # Read notebook
                with open(nb_file, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Generate relative output path
                rel_path = nb_file.relative_to(input_path)
                
                if format in ["markdown", "both"]:
                    md_exporter = MarkdownExporter()
                    (body, resources) = md_exporter.from_notebook_node(nb)
                    
                    md_output = output_path / rel_path.with_suffix('.md')
                    md_output.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(md_output, 'w', encoding='utf-8') as f:
                        f.write(body)
                    
                    converted_files.append({
                        "source": str(nb_file),
                        "output": str(md_output),
                        "format": "markdown"
                    })
                
                if format in ["html", "both"]:
                    html_exporter = HTMLExporter()
                    (body, resources) = html_exporter.from_notebook_node(nb)
                    
                    html_output = output_path / rel_path.with_suffix('.html')
                    html_output.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(html_output, 'w', encoding='utf-8') as f:
                        f.write(body)
                    
                    converted_files.append({
                        "source": str(nb_file),
                        "output": str(html_output),
                        "format": "html"
                    })
                
            except Exception as e:
                errors.append({
                    "file": str(nb_file),
                    "error": str(e)
                })
        
        return {
            "success": len(errors) == 0,
            "converted_files": converted_files,
            "errors": errors,
            "total_notebooks": len(notebook_files),
            "successful_conversions": len(converted_files)
        }
    
    def validate_dataset_paths(self, modules_dir: Union[str, Path]) -> Dict[str, Any]:
        """Validate dataset path resolution in notebook files.
        
        Args:
            modules_dir: Root directory containing module subdirectories
            
        Returns:
            Dictionary with validation results for each module
        """
        modules_path = Path(modules_dir)
        if not modules_path.exists():
            return {
                "success": False,
                "error": f"Modules directory does not exist: {modules_path}",
                "validation_results": []
            }
        
        validation_results = []
        notebook_files = list(modules_path.glob("**/*.ipynb"))
        
        # Expected dataset path patterns per copilot instructions
        expected_patterns = [
            r"DATA_DIR\s*=\s*Path\(['\"]\.\.\/\.\.\/\.\.\/datasets['\"]",  # Module 2/3 pattern
            r"\.\.\/\.\.\/\.\.\/datasets\/\w+\/",  # Direct relative paths
            r"datasets\/\w+\/\w+\.\w+",  # With subfolder organization
        ]
        
        flat_path_antipatterns = [
            r"datasets\/[^\/]+\.\w+",  # Flat paths like datasets/secom.data
        ]
        
        for nb_file in notebook_files:
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                valid_paths = []
                invalid_paths = []
                warnings = []
                
                # Check for expected patterns
                for pattern in expected_patterns:
                    matches = re.findall(pattern, content)
                    valid_paths.extend(matches)
                
                # Check for antipatterns
                for antipattern in flat_path_antipatterns:
                    matches = re.findall(antipattern, content)
                    if matches:
                        invalid_paths.extend(matches)
                        warnings.append(f"Found flat dataset path pattern: {matches}")
                
                # Additional checks
                if "datasets/" in content and not any(re.search(p, content) for p in expected_patterns):
                    warnings.append("References datasets/ but no standard path resolution pattern found")
                
                module_result = PathValidationResult(
                    module_path=nb_file,
                    notebook_files=[nb_file],
                    valid_paths=valid_paths,
                    invalid_paths=invalid_paths,
                    warnings=warnings
                )
                
                validation_results.append(asdict(module_result))
                
            except Exception as e:
                validation_results.append({
                    "module_path": str(nb_file),
                    "notebook_files": [str(nb_file)],
                    "valid_paths": [],
                    "invalid_paths": [],
                    "warnings": [f"Error reading file: {e}"]
                })
        
        total_warnings = sum(len(r["warnings"]) for r in validation_results)
        total_invalid = sum(len(r["invalid_paths"]) for r in validation_results)
        
        return {
            "success": total_invalid == 0,
            "validation_results": validation_results,
            "summary": {
                "total_notebooks": len(notebook_files),
                "total_warnings": total_warnings,
                "total_invalid_paths": total_invalid
            }
        }
    
    def export_environment(self, 
                          output_path: Union[str, Path],
                          format: str = "conda") -> Dict[str, Any]:
        """Export current environment for reproducibility.
        
        Args:
            output_path: Path for environment file output
            format: Export format (conda, pip, requirements)
            
        Returns:
            Dictionary with export results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "conda":
                # Export conda environment
                result = subprocess.run(
                    ["conda", "env", "export", "--no-builds"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                
                return {
                    "success": True,
                    "output_file": str(output_file),
                    "format": format,
                    "size_bytes": output_file.stat().st_size
                }
                
            elif format in ["pip", "requirements"]:
                # Export pip requirements
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                
                return {
                    "success": True,
                    "output_file": str(output_file),
                    "format": format,
                    "packages_count": len(result.stdout.strip().split('\n'))
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}. Use conda, pip, or requirements"
                }
                
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"Command failed: {e}",
                "stderr": e.stderr if hasattr(e, 'stderr') else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Export failed: {e}"
            }
    
    def setup_mkdocs(self, project_dir: Union[str, Path]) -> Dict[str, Any]:
        """Initialize mkdocs documentation site.
        
        Args:
            project_dir: Directory to set up mkdocs project
            
        Returns:
            Dictionary with setup results
        """
        project_path = Path(project_dir)
        project_path.mkdir(parents=True, exist_ok=True)
        
        mkdocs_yml = project_path / "mkdocs.yml"
        docs_dir = project_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create mkdocs.yml configuration
        mkdocs_config = {
            "site_name": "Python for Semiconductors Documentation",
            "site_description": "ML Learning Series for Semiconductor Engineers",
            "theme": {
                "name": "material",
                "palette": {
                    "primary": "blue",
                    "accent": "light-blue"
                },
                "features": [
                    "navigation.tabs",
                    "navigation.sections",
                    "navigation.expand",
                    "navigation.top",
                    "search.highlight",
                    "search.suggest"
                ]
            },
            "nav": [
                {"Home": "index.md"},
                {"Foundation": [
                    {"Module 1": "foundation/module-1.md"},
                    {"Module 2": "foundation/module-2.md"},
                    {"Module 3": "foundation/module-3.md"}
                ]},
                {"Intermediate": [
                    {"Module 4": "intermediate/module-4.md"}
                ]},
                {"Project Development": [
                    {"Module 10": "project-dev/module-10.md"}
                ]}
            ],
            "markdown_extensions": [
                "codehilite",
                "admonition",
                "toc",
                "pymdownx.details",
                "pymdownx.superfences",
                "pymdownx.tabbed"
            ]
        }
        
        try:
            if yaml:
                with open(mkdocs_yml, 'w') as f:
                    yaml.dump(mkdocs_config, f, default_flow_style=False)
            else:
                # Fallback: write YAML manually
                with open(mkdocs_yml, 'w') as f:
                    f.write("site_name: Python for Semiconductors Documentation\n")
                    f.write("site_description: ML Learning Series for Semiconductor Engineers\n")
                    f.write("theme:\n")
                    f.write("  name: material\n")
                    f.write("nav:\n")
                    f.write("  - Home: index.md\n")
            
            # Create index.md
            index_md = docs_dir / "index.md"
            with open(index_md, 'w') as f:
                f.write("# Python for Semiconductors\n\n")
                f.write("Welcome to the ML Learning Series for Semiconductor Engineers.\n\n")
                f.write("## Modules\n\n")
                f.write("- Foundation (Modules 1-3): Python, statistics, basic ML\n")
                f.write("- Intermediate (Modules 4-5): Ensembles, time series\n")
                f.write("- Advanced (Modules 6-7): Deep learning, computer vision\n")
                f.write("- Cutting-edge (Modules 8-9): Generative AI, MLOps\n")
                f.write("- Project Development (Module 10): Production projects\n")
            
            return {
                "success": True,
                "project_dir": str(project_path),
                "config_file": str(mkdocs_yml),
                "docs_dir": str(docs_dir),
                "files_created": [str(mkdocs_yml), str(index_md)]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"mkdocs setup failed: {e}"
            }
    
    def build_docs(self, project_dir: Union[str, Path]) -> Dict[str, Any]:
        """Build mkdocs documentation site.
        
        Args:
            project_dir: Directory containing mkdocs project
            
        Returns:
            Dictionary with build results
        """
        project_path = Path(project_dir)
        mkdocs_yml = project_path / "mkdocs.yml"
        
        if not mkdocs_yml.exists():
            return {
                "success": False,
                "error": f"mkdocs.yml not found in {project_path}"
            }
        
        try:
            result = subprocess.run(
                ["mkdocs", "build"],
                cwd=project_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            site_dir = project_path / "site"
            return {
                "success": True,
                "site_dir": str(site_dir),
                "stdout": result.stdout,
                "build_time": "completed"
            }
            
        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": f"mkdocs build failed: {e}",
                "stderr": e.stderr if hasattr(e, 'stderr') else None
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "mkdocs command not found. Install with: pip install mkdocs mkdocs-material"
            }

    def save(self, path: Union[str, Path]) -> None:
        """Save pipeline configuration to file."""
        save_path = Path(path)
        with open(save_path, 'w') as f:
            json.dump({
                "config": asdict(self.config) if self.config else None,
                "results": self.results
            }, f, indent=2, default=str)
    
    @staticmethod
    def load(path: Union[str, Path]) -> 'DocumentationReproducibilityPipeline':
        """Load pipeline configuration from file."""
        load_path = Path(path)
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        config = None
        if data.get("config"):
            config_data = data["config"]
            # Convert string paths back to Path objects
            if "input_dir" in config_data:
                config_data["input_dir"] = Path(config_data["input_dir"])
            if "output_dir" in config_data:
                config_data["output_dir"] = Path(config_data["output_dir"])
            config = DocumentationConfig(**config_data)
        
        pipeline = DocumentationReproducibilityPipeline(config)
        pipeline.results = data.get("results", {})
        return pipeline

# -------------------- CLI Implementation -------------------- #

def action_generate_docs(args) -> None:
    """Generate documentation from notebooks."""
    pipeline = DocumentationReproducibilityPipeline()
    
    result = pipeline.generate_documentation(
        input_dir=args.input,
        output_dir=args.output,
        format=args.format
    )
    
    print(json.dumps(result, indent=2))
    if not result["success"]:
        sys.exit(1)

def action_validate_paths(args) -> None:
    """Validate dataset path resolution."""
    pipeline = DocumentationReproducibilityPipeline()
    
    result = pipeline.validate_dataset_paths(modules_dir=args.modules_dir)
    
    print(json.dumps(result, indent=2, default=str))
    if not result["success"]:
        sys.exit(1)

def action_export_env(args) -> None:
    """Export environment for reproducibility."""
    pipeline = DocumentationReproducibilityPipeline()
    
    result = pipeline.export_environment(
        output_path=args.output,
        format=args.format
    )
    
    print(json.dumps(result, indent=2))
    if not result["success"]:
        sys.exit(1)

def action_setup_mkdocs(args) -> None:
    """Set up mkdocs documentation site."""
    pipeline = DocumentationReproducibilityPipeline()
    
    result = pipeline.setup_mkdocs(project_dir=args.project_dir)
    
    print(json.dumps(result, indent=2))
    if not result["success"]:
        sys.exit(1)

def action_build_docs(args) -> None:
    """Build mkdocs documentation site."""
    pipeline = DocumentationReproducibilityPipeline()
    
    result = pipeline.build_docs(project_dir=args.project_dir)
    
    print(json.dumps(result, indent=2))
    if not result["success"]:
        sys.exit(1)

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Module 10.3 Documentation & Reproducibility Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate documentation from notebooks
  python 10.3-documentation-reproducibility-pipeline.py generate-docs \\
    --input ../../foundation/module-3/ --output ./docs/module-3/ --format both

  # Validate dataset paths across all modules
  python 10.3-documentation-reproducibility-pipeline.py validate-paths \\
    --modules-dir ../../

  # Export conda environment
  python 10.3-documentation-reproducibility-pipeline.py export-env \\
    --output environment.yml --format conda

  # Set up mkdocs site
  python 10.3-documentation-reproducibility-pipeline.py setup-mkdocs \\
    --project-dir ./mkdocs-site/

  # Build documentation site
  python 10.3-documentation-reproducibility-pipeline.py build-docs \\
    --project-dir ./mkdocs-site/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # generate-docs subcommand
    p_generate = subparsers.add_parser('generate-docs', 
                                     help='Convert notebooks to documentation')
    p_generate.add_argument('--input', required=True,
                          help='Input directory containing notebooks')
    p_generate.add_argument('--output', required=True,
                          help='Output directory for documentation')
    p_generate.add_argument('--format', choices=['markdown', 'html', 'both'],
                          default='markdown',
                          help='Output format (default: markdown)')
    p_generate.set_defaults(func=action_generate_docs)
    
    # validate-paths subcommand
    p_validate = subparsers.add_parser('validate-paths',
                                     help='Validate dataset path resolution')
    p_validate.add_argument('--modules-dir', required=True,
                          help='Root directory containing modules')
    p_validate.set_defaults(func=action_validate_paths)
    
    # export-env subcommand
    p_export = subparsers.add_parser('export-env',
                                   help='Export environment for reproducibility')
    p_export.add_argument('--output', required=True,
                        help='Output file path')
    p_export.add_argument('--format', choices=['conda', 'pip', 'requirements'],
                        default='conda',
                        help='Export format (default: conda)')
    p_export.set_defaults(func=action_export_env)
    
    # setup-mkdocs subcommand
    p_setup = subparsers.add_parser('setup-mkdocs',
                                  help='Set up mkdocs documentation site')
    p_setup.add_argument('--project-dir', required=True,
                       help='Directory for mkdocs project')
    p_setup.set_defaults(func=action_setup_mkdocs)
    
    # build-docs subcommand
    p_build = subparsers.add_parser('build-docs',
                                  help='Build mkdocs documentation site')
    p_build.add_argument('--project-dir', required=True,
                       help='Directory containing mkdocs project')
    p_build.set_defaults(func=action_build_docs)
    
    return parser

def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "command": args.command if hasattr(args, 'command') else None
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()