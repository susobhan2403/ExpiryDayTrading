#!/usr/bin/env python3
"""
Dependency Audit Tool for Legacy Code Removal

This script performs static analysis to build a dependency graph of the codebase,
detect dynamic imports, and identify dead code candidates that are not reachable
from enhanced engine entry points.

Features:
- AST-based import graph construction
- Dynamic import detection (importlib, __import__, etc.)
- Plugin registry scanning
- Dead node detection relative to entry points
- DOT/JSON export for visualization

Usage:
    python tools/dependency_audit.py [--entry-points file1.py,file2.py] [--output-dir tools]
"""

from __future__ import annotations

import ast
import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

# Typing support
try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]  # Empty for 'import module', filled for 'from module import names'
    is_relative: bool
    source_file: str
    line_number: int
    import_type: str  # 'import', 'from', 'dynamic', 'plugin'

@dataclass
class ModuleNode:
    """Node in the dependency graph."""
    name: str
    file_path: Optional[str]
    imports: List[ImportInfo]
    is_entry_point: bool = False
    is_reachable: bool = False
    coverage_percent: float = 0.0
    
class DependencyAuditor:
    """Main dependency auditor class."""
    
    def __init__(self, project_root: str):
        self.project_root = pathlib.Path(project_root).resolve()
        self.modules: Dict[str, ModuleNode] = {}
        self.file_to_module: Dict[str, str] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.entry_points: Set[str] = set()
        
    def scan_python_files(self) -> List[pathlib.Path]:
        """Find all Python files in the project."""
        python_files = []
        for file_path in self.project_root.rglob("*.py"):
            # Skip __pycache__ and .git directories
            if "__pycache__" in str(file_path) or ".git" in str(file_path):
                continue
            python_files.append(file_path)
        return python_files
    
    def extract_imports_from_file(self, file_path: pathlib.Path) -> List[ImportInfo]:
        """Extract import information from a Python file using AST."""
        imports = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(ImportInfo(
                            module=alias.name,
                            names=[],
                            is_relative=False,
                            source_file=str(file_path),
                            line_number=node.lineno,
                            import_type='import'
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:  # Skip relative imports without module
                        imports.append(ImportInfo(
                            module=node.module,
                            names=[alias.name for alias in node.names],
                            is_relative=node.level > 0,
                            source_file=str(file_path),
                            line_number=node.lineno,
                            import_type='from'
                        ))
                
                # Detect dynamic imports
                elif isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id in ['__import__', 'importlib']):
                        # Basic dynamic import detection
                        if node.args and isinstance(node.args[0], ast.Constant):
                            imports.append(ImportInfo(
                                module=node.args[0].value,
                                names=[],
                                is_relative=False,
                                source_file=str(file_path),
                                line_number=node.lineno,
                                import_type='dynamic'
                            ))
                    
                    # Check for importlib.import_module
                    elif (isinstance(node.func, ast.Attribute) and
                          isinstance(node.func.value, ast.Name) and
                          node.func.value.id == 'importlib' and
                          node.func.attr == 'import_module'):
                        if node.args and isinstance(node.args[0], ast.Constant):
                            imports.append(ImportInfo(
                                module=node.args[0].value,
                                names=[],
                                is_relative=False,
                                source_file=str(file_path),
                                line_number=node.lineno,
                                import_type='dynamic'
                            ))
                
                # Check for plugin patterns (entry_points, etc.)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # Look for module-like strings that might be plugin references
                    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', node.value) and '.' in node.value:
                        imports.append(ImportInfo(
                            module=node.value,
                            names=[],
                            is_relative=False,
                            source_file=str(file_path),
                            line_number=node.lineno,
                            import_type='plugin'
                        ))
        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
        
        return imports
    
    def resolve_module_name(self, file_path: pathlib.Path) -> str:
        """Convert file path to Python module name."""
        rel_path = file_path.relative_to(self.project_root)
        parts = rel_path.with_suffix('').parts
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts)
    
    def build_dependency_graph(self):
        """Build the complete dependency graph."""
        print("🔍 Scanning Python files...")
        python_files = self.scan_python_files()
        print(f"Found {len(python_files)} Python files")
        
        print("📊 Extracting imports...")
        for file_path in python_files:
            module_name = self.resolve_module_name(file_path)
            imports = self.extract_imports_from_file(file_path)
            
            node = ModuleNode(
                name=module_name,
                file_path=str(file_path),
                imports=imports
            )
            
            self.modules[module_name] = node
            self.file_to_module[str(file_path)] = module_name
            
            # Build import graph
            for imp in imports:
                # Resolve relative imports
                if imp.is_relative:
                    target_module = self._resolve_relative_import(module_name, imp.module)
                else:
                    target_module = imp.module
                
                # Only include internal modules in the graph
                if self._is_internal_module(target_module):
                    self.import_graph[module_name].add(target_module)
                    self.reverse_graph[target_module].add(module_name)
        
        print(f"Built graph with {len(self.modules)} modules and {sum(len(deps) for deps in self.import_graph.values())} dependencies")
    
    def _resolve_relative_import(self, source_module: str, relative_module: str) -> str:
        """Resolve relative import to absolute module name."""
        source_parts = source_module.split('.')
        if relative_module:
            return '.'.join(source_parts[:-1] + [relative_module])
        else:
            return '.'.join(source_parts[:-1])
    
    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Consider modules internal if they start with known internal prefixes
        internal_prefixes = ['src', 'tests', 'tools']
        
        # Also check if we have a file for this module
        return (any(module_name.startswith(prefix) for prefix in internal_prefixes) or
                module_name in self.modules)
    
    def detect_entry_points(self, specified_entry_points: Optional[List[str]] = None) -> Set[str]:
        """Detect entry points automatically or use specified ones."""
        entry_points = set()
        
        if specified_entry_points:
            # Use specified entry points
            for entry_file in specified_entry_points:
                entry_path = pathlib.Path(entry_file)
                if not entry_path.is_absolute():
                    entry_path = self.project_root / entry_path
                if entry_path.exists():
                    module_name = self.resolve_module_name(entry_path)
                    entry_points.add(module_name)
        else:
            # Auto-detect entry points
            for module_name, node in self.modules.items():
                if not node.file_path:
                    continue
                    
                file_path = pathlib.Path(node.file_path)
                
                # Check for main entry patterns
                if (file_path.name in ['engine_runner.py', 'main.py', 'app.py'] or
                    file_path.name.endswith('_runner.py') or
                    file_path.name.endswith('_cli.py')):
                    entry_points.add(module_name)
                
                # Check for if __name__ == "__main__" pattern
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'if __name__ == "__main__"' in content:
                            entry_points.add(module_name)
                except (UnicodeDecodeError, FileNotFoundError):
                    pass
        
        self.entry_points = entry_points
        
        # Mark entry points in the graph
        for entry_point in entry_points:
            if entry_point in self.modules:
                self.modules[entry_point].is_entry_point = True
        
        print(f"🎯 Detected entry points: {sorted(entry_points)}")
        return entry_points
    
    def find_reachable_modules(self) -> Set[str]:
        """Find all modules reachable from entry points using BFS."""
        reachable = set()
        queue = deque(self.entry_points)
        
        while queue:
            current = queue.popleft()
            if current in reachable:
                continue
            
            reachable.add(current)
            
            # Add all modules this one imports
            for dependency in self.import_graph.get(current, set()):
                if dependency not in reachable:
                    queue.append(dependency)
        
        # Mark reachable modules
        for module_name in reachable:
            if module_name in self.modules:
                self.modules[module_name].is_reachable = True
        
        print(f"📈 Found {len(reachable)} reachable modules from {len(self.entry_points)} entry points")
        return reachable
    
    def find_dead_modules(self) -> Set[str]:
        """Find modules that are not reachable from any entry point."""
        reachable = self.find_reachable_modules()
        all_modules = set(self.modules.keys())
        dead_modules = all_modules - reachable
        
        print(f"💀 Found {len(dead_modules)} potentially dead modules")
        return dead_modules
    
    def export_dot(self, output_path: str, include_dead: bool = True):
        """Export dependency graph as DOT format for Graphviz."""
        dot_content = ['digraph DependencyGraph {']
        dot_content.append('  rankdir=LR;')
        dot_content.append('  node [shape=box];')
        
        # Add nodes with styling
        for module_name, node in self.modules.items():
            if not include_dead and not node.is_reachable:
                continue
                
            style = []
            color = 'black'
            
            if node.is_entry_point:
                style.append('filled')
                color = 'lightgreen'
            elif not node.is_reachable:
                style.append('filled')
                color = 'lightcoral'
            
            style_str = f'style="{",".join(style)}", color={color}' if style else ''
            dot_content.append(f'  "{module_name}" [{style_str}];')
        
        # Add edges
        for source, targets in self.import_graph.items():
            if not include_dead and not self.modules.get(source, ModuleNode("", "", [])).is_reachable:
                continue
                
            for target in targets:
                if not include_dead and not self.modules.get(target, ModuleNode("", "", [])).is_reachable:
                    continue
                dot_content.append(f'  "{source}" -> "{target}";')
        
        dot_content.append('}')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(dot_content))
        
        print(f"📊 Exported DOT graph to {output_path}")
        
        # Generate PNG if graphviz is available
        if HAS_GRAPHVIZ:
            try:
                graph = graphviz.Source('\n'.join(dot_content))
                png_path = output_path.replace('.dot', '.png')
                graph.render(png_path.replace('.png', ''), format='png', cleanup=True)
                print(f"🖼️  Generated PNG visualization: {png_path}")
            except Exception as e:
                print(f"Warning: Could not generate PNG: {e}")
    
    def export_json(self, output_path: str):
        """Export dependency graph as JSON adjacency list."""
        graph_data = {
            'modules': {},
            'import_graph': {},
            'entry_points': list(self.entry_points)
        }
        
        # Export module information
        for module_name, node in self.modules.items():
            graph_data['modules'][module_name] = {
                'file_path': node.file_path,
                'is_entry_point': node.is_entry_point,
                'is_reachable': node.is_reachable,
                'imports': [asdict(imp) for imp in node.imports]
            }
        
        # Export adjacency list
        for source, targets in self.import_graph.items():
            graph_data['import_graph'][source] = list(targets)
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"📄 Exported JSON graph to {output_path}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        dead_modules = self.find_dead_modules()
        
        report = {
            'summary': {
                'total_modules': len(self.modules),
                'entry_points': len(self.entry_points),
                'reachable_modules': len([m for m in self.modules.values() if m.is_reachable]),
                'dead_modules': len(dead_modules)
            },
            'entry_points': list(self.entry_points),
            'dead_modules': list(dead_modules),
            'module_details': {}
        }
        
        # Add details for each module
        for module_name, node in self.modules.items():
            report['module_details'][module_name] = {
                'file_path': node.file_path,
                'is_entry_point': node.is_entry_point,
                'is_reachable': node.is_reachable,
                'import_count': len(node.imports),
                'dynamic_imports': len([imp for imp in node.imports if imp.import_type == 'dynamic']),
                'plugin_refs': len([imp for imp in node.imports if imp.import_type == 'plugin'])
            }
        
        return report

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Dependency audit tool for legacy code removal",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--entry-points',
        help='Comma-separated list of entry point files (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='tools',
        help='Output directory for generated files (default: tools)'
    )
    
    parser.add_argument(
        '--include-dead',
        action='store_true',
        help='Include dead modules in graph visualization'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize auditor
    auditor = DependencyAuditor(args.project_root)
    
    # Build dependency graph
    auditor.build_dependency_graph()
    
    # Detect entry points
    entry_points_list = None
    if args.entry_points:
        entry_points_list = [ep.strip() for ep in args.entry_points.split(',')]
    auditor.detect_entry_points(entry_points_list)
    
    # Generate outputs
    auditor.export_dot(
        str(output_dir / 'dependency_graph.dot'),
        include_dead=args.include_dead
    )
    auditor.export_json(str(output_dir / 'dependency_graph.json'))
    
    # Generate and save report
    report = auditor.generate_report()
    with open(output_dir / 'dependency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Analysis Summary:")
    print(f"   Total modules: {report['summary']['total_modules']}")
    print(f"   Entry points: {report['summary']['entry_points']}")
    print(f"   Reachable modules: {report['summary']['reachable_modules']}")
    print(f"   Dead modules: {report['summary']['dead_modules']}")
    
    if report['summary']['dead_modules'] > 0:
        print(f"\n💀 Dead modules (candidates for removal):")
        for module in sorted(report['dead_modules']):
            print(f"   - {module}")

if __name__ == "__main__":
    main()