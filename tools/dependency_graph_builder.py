#!/usr/bin/env python3
"""
R2: Repository Dependency Graph Builder

Static analysis: parse imports (AST) across Python packages; capture dynamic imports 
(importlib, __import__), plugin registries, entry_points.
Runtime confirmation: minimal harness to log imports executed by the enhanced engine 
during a representative run.
Output: Graphviz DOT + PNG and JSON adjacency list.
"""

from __future__ import annotations
import ast
import json
import pathlib
import importlib.util
import subprocess
import sys
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict


class DependencyGraphBuilder:
    def __init__(self, repo_root: pathlib.Path):
        self.repo_root = repo_root
        self.static_deps = defaultdict(set)
        self.runtime_deps = defaultdict(set)
        self.dynamic_imports = defaultdict(set)
        self.plugin_registries = defaultdict(set)
        self.all_modules = set()
        
    def build_complete_graph(self) -> Dict[str, Any]:
        """Build complete dependency graph with static and runtime analysis."""
        print("🔗 R2: Building Repository Dependency Graph")
        print("=" * 50)
        
        # 1. Static analysis - AST parsing
        print("📝 1. Static AST Analysis...")
        static_graph = self._build_static_graph()
        
        # 2. Dynamic import detection
        print("🔄 2. Dynamic Import Detection...")
        dynamic_graph = self._detect_dynamic_imports()
        
        # 3. Plugin registry detection
        print("🔌 3. Plugin Registry Detection...")
        plugin_graph = self._detect_plugin_registries()
        
        # 4. Runtime import logging
        print("🏃 4. Runtime Import Logging...")
        runtime_graph = self._run_runtime_analysis()
        
        # 5. Combine all graphs
        print("🔀 5. Combining Graphs...")
        combined_graph = self._combine_graphs(static_graph, dynamic_graph, plugin_graph, runtime_graph)
        
        # 6. Generate outputs
        print("📊 6. Generating Outputs...")
        outputs = self._generate_outputs(combined_graph)
        
        return {
            "static_graph": static_graph,
            "dynamic_graph": dynamic_graph,
            "plugin_graph": plugin_graph,
            "runtime_graph": runtime_graph,
            "combined_graph": combined_graph,
            "outputs": outputs,
            "statistics": self._generate_statistics(combined_graph)
        }
    
    def _build_static_graph(self) -> Dict[str, List[str]]:
        """Build static dependency graph using AST parsing."""
        graph = defaultdict(list)
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                # Convert file path to module name
                rel_path = py_file.relative_to(self.repo_root)
                if rel_path.name == "__init__.py":
                    module_name = str(rel_path.parent).replace(os.sep, ".")
                else:
                    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                
                self.all_modules.add(module_name)
                
                # Parse AST
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                try:
                    tree = ast.parse(content, filename=str(py_file))
                    
                    # Extract imports
                    imports = self._extract_all_imports(tree, module_name)
                    
                    # Add to static dependencies
                    for imp in imports:
                        if self._is_local_module(imp):
                            graph[module_name].append(imp)
                            self.static_deps[module_name].add(imp)
                    
                except SyntaxError as e:
                    print(f"Syntax error in {py_file}: {e}")
                    
            except Exception as e:
                print(f"Error processing {py_file}: {e}")
        
        return dict(graph)
    
    def _extract_all_imports(self, tree: ast.AST, current_module: str) -> List[str]:
        """Extract all import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if node.level > 0:  # Relative import
                        # Handle relative imports
                        if node.level == 1:
                            base_module = ".".join(current_module.split(".")[:-1])
                        else:
                            parts = current_module.split(".")
                            base_module = ".".join(parts[:-(node.level-1)])
                        
                        if node.module:
                            full_module = f"{base_module}.{node.module}"
                        else:
                            full_module = base_module
                        imports.append(full_module)
                    else:
                        imports.append(node.module)
                        
                    # Also add specific imports for completeness
                    for alias in node.names:
                        if alias.name != "*":
                            if node.module:
                                full_name = f"{node.module}.{alias.name}"
                                imports.append(full_name)
        
        return imports
    
    def _detect_dynamic_imports(self) -> Dict[str, List[str]]:
        """Detect dynamic imports using importlib, __import__, etc."""
        graph = defaultdict(list)
        
        dynamic_patterns = [
            r'importlib\.import_module\([\'"]([^\'"]*)[\'"]',
            r'__import__\([\'"]([^\'"]*)[\'"]',
            r'importlib\.util\.spec_from_file_location\([\'"][^\'"]*, [\'"]([^\'"]*)[\'"]',
            r'exec\([\'"]import\s+([^\'"]*)[\'"]',
            r'eval\([\'"]import\s+([^\'"]*)[\'"]'
        ]
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Convert to module name
                rel_path = py_file.relative_to(self.repo_root)
                if rel_path.name == "__init__.py":
                    module_name = str(rel_path.parent).replace(os.sep, ".")
                else:
                    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                
                # Check for dynamic import patterns
                for pattern in dynamic_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if self._is_local_module(match):
                            graph[module_name].append(match)
                            self.dynamic_imports[module_name].add(match)
                            
            except Exception as e:
                print(f"Error detecting dynamic imports in {py_file}: {e}")
        
        return dict(graph)
    
    def _detect_plugin_registries(self) -> Dict[str, List[str]]:
        """Detect plugin registries and entry point patterns."""
        graph = defaultdict(list)
        
        plugin_patterns = [
            r'entry_points\(\)',
            r'pkg_resources\.iter_entry_points',
            r'stevedore\.',
            r'pluggy\.',
            r'register.*plugin',
            r'plugin.*registry'
        ]
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Convert to module name
                rel_path = py_file.relative_to(self.repo_root)
                if rel_path.name == "__init__.py":
                    module_name = str(rel_path.parent).replace(os.sep, ".")
                else:
                    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                
                # Check for plugin patterns
                for pattern in plugin_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Look for potential plugin modules in the same file
                        plugin_refs = re.findall(r'[\'"]([a-zA-Z_][a-zA-Z0-9_.]*)[\'"]', content)
                        for ref in plugin_refs:
                            if self._is_local_module(ref):
                                graph[module_name].append(ref)
                                self.plugin_registries[module_name].add(ref)
                                
            except Exception as e:
                print(f"Error detecting plugins in {py_file}: {e}")
        
        return dict(graph)
    
    def _run_runtime_analysis(self) -> Dict[str, List[str]]:
        """Run runtime import logging to see what's actually imported."""
        graph = defaultdict(list)
        
        # Create runtime import logger
        logger_code = '''
import sys
import os
import json
from pathlib import Path

# Track imports
imported_modules = set()
original_import = __builtins__.__import__

def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
    imported_modules.add(name)
    return original_import(name, globals, locals, fromlist, level)

__builtins__.__import__ = tracking_import

# Import and run enhanced engine components
try:
    print("Running enhanced engine import test...")
    
    # Test main entry points
    import engine_runner
    print("✓ engine_runner imported")
    
    # Test core enhanced engine
    from src.engine_enhanced import EnhancedTradingEngine
    print("✓ EnhancedTradingEngine imported")
    
    # Test orchestrator
    from src.cli.orchestrate import Orchestrator
    print("✓ Orchestrator imported")
    
    # Test other components that might be used
    try:
        from src.config import load_settings
        print("✓ config imported")
    except:
        pass
    
    try:
        from src.provider.kite import KiteProvider
        print("✓ KiteProvider imported")
    except:
        pass
    
    # Try to instantiate enhanced engine to trigger more imports
    try:
        import datetime
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        expiry = IST.localize(datetime.datetime(2024, 1, 25, 15, 30, 0))
        engine = EnhancedTradingEngine("NIFTY", expiry, min_tau_hours=2.0)
        print("✓ EnhancedTradingEngine instantiated")
    except Exception as e:
        print(f"Could not instantiate enhanced engine: {e}")
    
except Exception as e:
    print(f"Error during runtime test: {e}")

# Save results
runtime_imports = {
    "imports": sorted(list(imported_modules)),
    "local_imports": [imp for imp in imported_modules if not imp.startswith(('sys', 'os', 'json', 'datetime', 'pathlib'))]
}

output_file = Path("tools") / "runtime_imports.json"
output_file.parent.mkdir(exist_ok=True)
with open(output_file, "w") as f:
    json.dump(runtime_imports, f, indent=2)

print(f"Runtime imports saved to {output_file}")
print(f"Total imports: {len(imported_modules)}")
print(f"Local imports: {len(runtime_imports['local_imports'])}")
'''
        
        # Write and run the logger
        logger_file = self.repo_root / "tools" / "runtime_logger.py"
        with open(logger_file, "w") as f:
            f.write(logger_code)
        
        try:
            # Run the runtime logger
            result = subprocess.run(
                [sys.executable, str(logger_file)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Runtime logger output:\n{result.stdout}")
            if result.stderr:
                print(f"Runtime logger errors:\n{result.stderr}")
            
            # Load results
            runtime_file = self.repo_root / "tools" / "runtime_imports.json"
            if runtime_file.exists():
                with open(runtime_file) as f:
                    runtime_data = json.load(f)
                
                # Convert to graph format
                local_imports = runtime_data.get("local_imports", [])
                for imp in local_imports:
                    if self._is_local_module(imp):
                        graph["runtime_execution"].append(imp)
                        self.runtime_deps["runtime_execution"].add(imp)
                        
        except subprocess.TimeoutExpired:
            print("Runtime analysis timed out")
        except Exception as e:
            print(f"Error running runtime analysis: {e}")
        
        return dict(graph)
    
    def _is_local_module(self, module_name: str) -> bool:
        """Check if a module is local to this repository."""
        if not module_name:
            return False
        
        # Check if it starts with src (our main package)
        if module_name.startswith("src"):
            return True
        
        # Check if it's a direct module in our repo
        module_file = self.repo_root / f"{module_name.replace('.', os.sep)}.py"
        module_init = self.repo_root / f"{module_name.replace('.', os.sep)}" / "__init__.py"
        
        return module_file.exists() or module_init.exists()
    
    def _combine_graphs(self, static_graph, dynamic_graph, plugin_graph, runtime_graph) -> Dict[str, List[str]]:
        """Combine all dependency graphs."""
        combined = defaultdict(set)
        
        # Add static dependencies
        for module, deps in static_graph.items():
            combined[module].update(deps)
        
        # Add dynamic dependencies
        for module, deps in dynamic_graph.items():
            combined[module].update(deps)
        
        # Add plugin dependencies
        for module, deps in plugin_graph.items():
            combined[module].update(deps)
        
        # Add runtime dependencies
        for module, deps in runtime_graph.items():
            combined[module].update(deps)
        
        # Convert back to dict with lists
        return {module: sorted(list(deps)) for module, deps in combined.items()}
    
    def _generate_outputs(self, combined_graph: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate Graphviz DOT, PNG, and JSON outputs."""
        outputs = {}
        
        # 1. Generate DOT file
        dot_content = self._generate_dot(combined_graph)
        dot_file = self.repo_root / "tools" / "dependency_graph.dot"
        with open(dot_file, "w") as f:
            f.write(dot_content)
        outputs["dot_file"] = str(dot_file)
        
        # 2. Generate PNG (if graphviz available)
        png_file = self.repo_root / "tools" / "dependency_graph.png"
        try:
            subprocess.run(
                ["dot", "-Tpng", str(dot_file), "-o", str(png_file)],
                check=True,
                capture_output=True
            )
            outputs["png_file"] = str(png_file)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: Could not generate PNG (graphviz not available)")
            outputs["png_file"] = None
        
        # 3. Generate JSON adjacency list
        json_file = self.repo_root / "tools" / "dependency_graph.json"
        with open(json_file, "w") as f:
            json.dump(combined_graph, f, indent=2)
        outputs["json_file"] = str(json_file)
        
        return outputs
    
    def _generate_dot(self, graph: Dict[str, List[str]]) -> str:
        """Generate Graphviz DOT format."""
        lines = ["digraph dependency_graph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box, style=rounded];")
        
        # Define node styles based on module type
        enhanced_color = "lightgreen"
        legacy_color = "lightcoral"
        utility_color = "lightblue"
        entry_color = "gold"
        
        # Add nodes with colors
        all_modules = set(graph.keys())
        for deps in graph.values():
            all_modules.update(deps)
        
        for module in sorted(all_modules):
            color = utility_color
            
            if "enhanced" in module.lower():
                color = enhanced_color
            elif "engine" in module.lower() and "enhanced" not in module.lower():
                color = legacy_color
            elif any(entry in module for entry in ["runner", "orchestrate", "main", "cli"]):
                color = entry_color
            
            lines.append(f'  "{module}" [fillcolor={color}, style=filled];')
        
        # Add edges
        for module, deps in graph.items():
            for dep in deps:
                lines.append(f'  "{module}" -> "{dep}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def _generate_statistics(self, combined_graph: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate dependency statistics."""
        all_modules = set(combined_graph.keys())
        for deps in combined_graph.values():
            all_modules.update(deps)
        
        # Calculate in-degree and out-degree
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for module, deps in combined_graph.items():
            out_degree[module] = len(deps)
            for dep in deps:
                in_degree[dep] += 1
        
        # Find modules with no dependencies (roots)
        roots = [module for module in all_modules if in_degree[module] == 0]
        
        # Find modules with no dependents (leaves)
        leaves = [module for module in all_modules if out_degree[module] == 0]
        
        # Calculate strongly connected components (simple cycle detection)
        cycles = self._detect_cycles(combined_graph)
        
        return {
            "total_modules": len(all_modules),
            "total_dependencies": sum(len(deps) for deps in combined_graph.values()),
            "root_modules": sorted(roots),
            "leaf_modules": sorted(leaves),
            "most_dependent": sorted([(module, count) for module, count in in_degree.items()], 
                                   key=lambda x: x[1], reverse=True)[:10],
            "most_dependencies": sorted([(module, count) for module, count in out_degree.items()], 
                                      key=lambda x: x[1], reverse=True)[:10],
            "cycles_detected": len(cycles),
            "cycles": cycles[:5]  # Show first 5 cycles
        }
    
    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Simple cycle detection using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles


def main():
    """Run the dependency graph builder."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    builder = DependencyGraphBuilder(repo_root)
    
    results = builder.build_complete_graph()
    
    # Save complete results
    output_file = repo_root / "tools" / "dependency_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    stats = results["statistics"]
    print(f"\n📊 Dependency Graph Statistics")
    print(f"📦 Total Modules: {stats['total_modules']}")
    print(f"🔗 Total Dependencies: {stats['total_dependencies']}")
    print(f"🌱 Root Modules: {len(stats['root_modules'])}")
    print(f"🍃 Leaf Modules: {len(stats['leaf_modules'])}")
    print(f"🔄 Cycles Detected: {stats['cycles_detected']}")
    
    print(f"\n📋 Top Dependencies:")
    for module, count in stats['most_dependencies'][:3]:
        print(f"  • {module}: {count} dependencies")
    
    print(f"\n📋 Most Depended Upon:")
    for module, count in stats['most_dependent'][:3]:
        print(f"  • {module}: {count} dependents")
    
    outputs = results["outputs"]
    print(f"\n💾 Outputs Generated:")
    print(f"  • DOT: {outputs['dot_file']}")
    print(f"  • PNG: {outputs['png_file'] or 'Not generated (graphviz not available)'}")
    print(f"  • JSON: {outputs['json_file']}")
    print(f"  • Full analysis: {output_file}")
    
    return results


if __name__ == "__main__":
    main()