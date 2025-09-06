#!/usr/bin/env python3
"""
R3: Legacy Code Candidate Identifier

Combine evidence:
• Not reachable from enhanced entry points (graph reachability).
• No static references (no imports / symbol usage / string-based plugin refs).
• 0% runtime import hits from the enhanced run harness.
• 0% test coverage after running the test suite with the enhanced engine enabled.
Handle caveats: reflection, config-string references, dynamic loaders, plugin registries, 
__all__, side-effect modules. Build an "allowlist" for modules that must be retained.
"""

from __future__ import annotations
import ast
import json
import pathlib
import subprocess
import sys
import re
import os
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict


class LegacyCodeIdentifier:
    def __init__(self, repo_root: pathlib.Path):
        self.repo_root = repo_root
        self.evidence_table = {}
        self.allowlist = {}
        self.enhanced_entry_points = set()
        self.reachable_modules = set()
        
    def identify_legacy_candidates(self) -> Dict[str, Any]:
        """Main method to identify legacy code candidates with evidence."""
        print("🕵️ R3: Identifying Legacy Code Candidates")
        print("=" * 50)
        
        # 1. Load previous analysis results
        print("📊 1. Loading Previous Analysis...")
        entry_points = self._load_entry_points()
        dependency_graph = self._load_dependency_graph()
        
        # 2. Determine enhanced entry points
        print("🎯 2. Identifying Enhanced Entry Points...")
        enhanced_entries = self._identify_enhanced_entry_points(entry_points)
        
        # 3. Calculate reachability from enhanced entry points
        print("🗺️ 3. Calculating Graph Reachability...")
        reachability = self._calculate_reachability(dependency_graph, enhanced_entries)
        
        # 4. Static reference analysis
        print("🔍 4. Static Reference Analysis...")
        static_refs = self._analyze_static_references()
        
        # 5. Runtime import analysis
        print("🏃 5. Runtime Import Analysis...")
        runtime_usage = self._analyze_runtime_usage()
        
        # 6. Test coverage analysis
        print("🧪 6. Test Coverage Analysis...")
        test_coverage = self._analyze_test_coverage()
        
        # 7. Build evidence table
        print("📋 7. Building Evidence Table...")
        evidence_table = self._build_evidence_table(
            reachability, static_refs, runtime_usage, test_coverage
        )
        
        # 8. Handle special cases and caveats
        print("⚠️ 8. Handling Special Cases...")
        allowlist = self._build_allowlist(evidence_table)
        
        # 9. Generate final candidates
        print("🎯 9. Generating Final Candidates...")
        candidates = self._generate_candidates(evidence_table, allowlist)
        
        return {
            "enhanced_entry_points": list(enhanced_entries),
            "reachable_modules": list(self.reachable_modules),
            "evidence_table": evidence_table,
            "allowlist": allowlist,
            "legacy_candidates": candidates,
            "statistics": self._generate_statistics(evidence_table, candidates)
        }
    
    def _load_entry_points(self) -> Dict[str, Any]:
        """Load entry point analysis results."""
        entry_file = self.repo_root / "tools" / "enhanced_engine_inventory.json"
        if entry_file.exists():
            with open(entry_file) as f:
                return json.load(f)
        else:
            print("Warning: Entry point inventory not found, running detection...")
            subprocess.run([sys.executable, str(self.repo_root / "tools" / "entry_point_detector.py")])
            if entry_file.exists():
                with open(entry_file) as f:
                    return json.load(f)
        return {}
    
    def _load_dependency_graph(self) -> Dict[str, Any]:
        """Load dependency graph analysis results."""
        dep_file = self.repo_root / "tools" / "dependency_analysis.json"
        if dep_file.exists():
            with open(dep_file) as f:
                return json.load(f)
        else:
            print("Warning: Dependency analysis not found, running builder...")
            subprocess.run([sys.executable, str(self.repo_root / "tools" / "dependency_graph_builder.py")])
            if dep_file.exists():
                with open(dep_file) as f:
                    return json.load(f)
        return {}
    
    def _identify_enhanced_entry_points(self, entry_points: Dict[str, Any]) -> Set[str]:
        """Identify which entry points are for the enhanced engine."""
        enhanced_entries = set()
        
        # From summary
        summary = entry_points.get("summary", {})
        enhanced_files = summary.get("enhanced_engine_files", [])
        enhanced_entries.update(enhanced_files)
        
        # From python main scripts
        for script in entry_points.get("python_main_scripts", []):
            if script.get("enhanced_engine_refs", 0) > 0:
                enhanced_entries.add(script["file"])
        
        # From CLI entry points
        for cli in entry_points.get("cli_entry_points", []):
            if cli.get("enhanced_engine_refs", 0) > 0:
                enhanced_entries.add(cli["file"])
        
        # From service scripts
        for service in entry_points.get("service_scripts", []):
            if service.get("enhanced_engine_refs", 0) > 0:
                enhanced_entries.add(service["file"])
        
        # Convert file paths to module names
        module_entries = set()
        for file_path in enhanced_entries:
            module_name = self._file_to_module(file_path)
            if module_name:
                module_entries.add(module_name)
        
        self.enhanced_entry_points = module_entries
        return module_entries
    
    def _calculate_reachability(self, dependency_graph: Dict[str, Any], entry_points: Set[str]) -> Dict[str, bool]:
        """Calculate which modules are reachable from enhanced entry points."""
        combined_graph = dependency_graph.get("combined_graph", {})
        reachable = set()
        
        # BFS from each entry point
        queue = list(entry_points)
        visited = set(entry_points)
        
        while queue:
            current = queue.pop(0)
            reachable.add(current)
            
            for dependency in combined_graph.get(current, []):
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append(dependency)
        
        self.reachable_modules = reachable
        
        # Create reachability dict for all modules
        all_modules = set(combined_graph.keys())
        for deps in combined_graph.values():
            all_modules.update(deps)
        
        return {module: module in reachable for module in all_modules}
    
    def _analyze_static_references(self) -> Dict[str, Dict[str, Any]]:
        """Analyze static references to each module."""
        references = defaultdict(lambda: {
            "import_count": 0,
            "string_references": 0,
            "symbol_usage": 0,
            "config_references": 0,
            "referencing_files": []
        })
        
        # Scan all Python files for references
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # AST analysis for imports
                try:
                    tree = ast.parse(content)
                    imports = self._extract_imports_with_targets(tree)
                    
                    for target_module in imports:
                        if self._is_local_module(target_module):
                            references[target_module]["import_count"] += 1
                            references[target_module]["referencing_files"].append(str(py_file.relative_to(self.repo_root)))
                            
                except SyntaxError:
                    pass
                
                # String-based references
                for module in self._get_all_modules():
                    # Look for module name in strings
                    module_pattern = re.escape(module.replace(".", r"\."))
                    string_matches = len(re.findall(rf'[\'"].*{module_pattern}.*[\'"]', content))
                    if string_matches > 0:
                        references[module]["string_references"] += string_matches
                    
                    # Look for symbol usage (approximate)
                    if module in content:
                        # More sophisticated symbol detection
                        symbol_matches = len(re.findall(rf'\b{re.escape(module.split(".")[-1])}\b', content))
                        references[module]["symbol_usage"] += symbol_matches
                        
            except Exception as e:
                print(f"Error analyzing static references in {py_file}: {e}")
        
        # Check config files for references
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"]
        for pattern in config_patterns:
            for config_file in self.repo_root.rglob(pattern):
                try:
                    content = config_file.read_text(encoding='utf-8', errors='ignore')
                    
                    for module in self._get_all_modules():
                        if module in content:
                            references[module]["config_references"] += 1
                            
                except Exception as e:
                    print(f"Error analyzing config file {config_file}: {e}")
        
        return dict(references)
    
    def _analyze_runtime_usage(self) -> Dict[str, bool]:
        """Analyze runtime usage from previous runtime analysis."""
        runtime_file = self.repo_root / "tools" / "runtime_imports.json"
        runtime_usage = {}
        
        if runtime_file.exists():
            try:
                with open(runtime_file) as f:
                    runtime_data = json.load(f)
                
                imported_modules = set(runtime_data.get("imports", []))
                local_imports = set(runtime_data.get("local_imports", []))
                
                # Check which local modules were imported
                for module in self._get_all_modules():
                    runtime_usage[module] = module in local_imports or module in imported_modules
                    
            except Exception as e:
                print(f"Error loading runtime usage: {e}")
        
        return runtime_usage
    
    def _analyze_test_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Analyze test coverage for modules."""
        coverage = defaultdict(lambda: {
            "covered": False,
            "coverage_percentage": 0.0,
            "test_files": []
        })
        
        # Look for test files that import each module
        test_dirs = ["tests", "test"]
        
        for test_dir in test_dirs:
            test_path = self.repo_root / test_dir
            if test_path.exists():
                for test_file in test_path.rglob("*.py"):
                    try:
                        content = test_file.read_text(encoding='utf-8', errors='ignore')
                        
                        # Check which modules are imported in tests
                        try:
                            tree = ast.parse(content)
                            imports = self._extract_imports_with_targets(tree)
                            
                            for target_module in imports:
                                if self._is_local_module(target_module):
                                    coverage[target_module]["covered"] = True
                                    coverage[target_module]["test_files"].append(str(test_file.relative_to(self.repo_root)))
                                    
                        except SyntaxError:
                            pass
                            
                    except Exception as e:
                        print(f"Error analyzing test file {test_file}: {e}")
        
        # Try to run actual coverage if pytest-cov is available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--cov=src", "--cov-report=json", "--cov-report=term-missing"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Look for coverage.json
            coverage_file = self.repo_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    cov_data = json.load(f)
                
                for file_path, file_data in cov_data.get("files", {}).items():
                    module_name = self._file_to_module(file_path)
                    if module_name and self._is_local_module(module_name):
                        coverage[module_name]["coverage_percentage"] = file_data.get("summary", {}).get("percent_covered", 0.0)
                        if file_data.get("summary", {}).get("covered_lines", 0) > 0:
                            coverage[module_name]["covered"] = True
                            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Could not run coverage analysis (pytest-cov not available or timeout)")
        except Exception as e:
            print(f"Error running coverage analysis: {e}")
        
        return dict(coverage)
    
    def _build_evidence_table(self, reachability: Dict[str, bool], static_refs: Dict[str, Dict[str, Any]], 
                            runtime_usage: Dict[str, bool], test_coverage: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive evidence table for each module."""
        evidence = {}
        all_modules = self._get_all_modules()
        
        for module in all_modules:
            static_data = static_refs.get(module, {})
            coverage_data = test_coverage.get(module, {})
            
            # Calculate risk level
            risk_score = 0
            
            # Not reachable = high risk
            if not reachability.get(module, False):
                risk_score += 3
            
            # No static references = medium risk
            total_static_refs = (static_data.get("import_count", 0) + 
                               static_data.get("string_references", 0) + 
                               static_data.get("config_references", 0))
            if total_static_refs == 0:
                risk_score += 2
            
            # No runtime usage = medium risk
            if not runtime_usage.get(module, False):
                risk_score += 2
            
            # No test coverage = low risk
            if not coverage_data.get("covered", False):
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                risk_level = "VERY_HIGH"
            elif risk_score >= 4:
                risk_level = "HIGH"
            elif risk_score >= 2:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            evidence[module] = {
                "reachable_from_enhanced": reachability.get(module, False),
                "static_references": total_static_refs,
                "import_count": static_data.get("import_count", 0),
                "string_references": static_data.get("string_references", 0),
                "config_references": static_data.get("config_references", 0),
                "runtime_usage": runtime_usage.get(module, False),
                "test_coverage": coverage_data.get("covered", False),
                "coverage_percentage": coverage_data.get("coverage_percentage", 0.0),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "referencing_files": static_data.get("referencing_files", []),
                "test_files": coverage_data.get("test_files", [])
            }
        
        return evidence
    
    def _build_allowlist(self, evidence_table: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Build allowlist for modules that must be retained despite evidence."""
        allowlist = {}
        
        # Always keep special modules
        special_patterns = [
            r'__init__',
            r'config',
            r'settings',
            r'constants',
            r'utils',
            r'base',
            r'abstract'
        ]
        
        for module in evidence_table:
            # Check special patterns
            for pattern in special_patterns:
                if re.search(pattern, module, re.IGNORECASE):
                    allowlist[module] = f"Special module pattern: {pattern}"
                    break
            
            # Check for side-effect modules (modules that just import for side effects)
            if module.endswith("__init__"):
                allowlist[module] = "Package __init__ module"
            
            # Check for modules with __all__ exports
            module_file = self._module_to_file(module)
            if module_file and module_file.exists():
                try:
                    content = module_file.read_text(encoding='utf-8', errors='ignore')
                    if "__all__" in content:
                        allowlist[module] = "Defines __all__ exports"
                except:
                    pass
            
            # Check for reflection/dynamic usage patterns
            if module in evidence_table:
                referencing_files = evidence_table[module].get("referencing_files", [])
                for ref_file in referencing_files:
                    ref_path = self.repo_root / ref_file
                    if ref_path.exists():
                        try:
                            content = ref_path.read_text(encoding='utf-8', errors='ignore')
                            if any(pattern in content for pattern in ["getattr", "hasattr", "setattr", "importlib"]):
                                allowlist[module] = "Potential reflection/dynamic usage"
                                break
                        except:
                            pass
            
            # Check for plugin registry usage
            if "plugin" in module.lower() or "registry" in module.lower():
                allowlist[module] = "Plugin/registry system"
            
            # Check for configuration string references
            if evidence_table[module].get("config_references", 0) > 0:
                allowlist[module] = "Referenced in configuration files"
        
        return allowlist
    
    def _generate_candidates(self, evidence_table: Dict[str, Dict[str, Any]], allowlist: Dict[str, str]) -> Dict[str, Any]:
        """Generate final legacy code candidates."""
        candidates = {
            "high_confidence": [],
            "medium_confidence": [],
            "low_confidence": [],
            "allowlisted": []
        }
        
        for module, evidence in evidence_table.items():
            if module in allowlist:
                candidates["allowlisted"].append({
                    "module": module,
                    "reason": allowlist[module],
                    "evidence": evidence
                })
                continue
            
            risk_level = evidence["risk_level"]
            
            candidate = {
                "module": module,
                "evidence": evidence,
                "file_path": self._module_to_file_path(module)
            }
            
            if risk_level in ["VERY_HIGH", "HIGH"]:
                candidates["high_confidence"].append(candidate)
            elif risk_level == "MEDIUM":
                candidates["medium_confidence"].append(candidate)
            else:
                candidates["low_confidence"].append(candidate)
        
        # Sort by risk score (highest first)
        for category in ["high_confidence", "medium_confidence", "low_confidence"]:
            candidates[category].sort(key=lambda x: x["evidence"]["risk_score"], reverse=True)
        
        return candidates
    
    def _get_all_modules(self) -> Set[str]:
        """Get all local modules in the repository."""
        modules = set()
        
        for py_file in self.repo_root.rglob("*.py"):
            module_name = self._file_to_module(str(py_file.relative_to(self.repo_root)))
            if module_name and self._is_local_module(module_name):
                modules.add(module_name)
        
        return modules
    
    def _file_to_module(self, file_path: str) -> Optional[str]:
        """Convert file path to module name."""
        if file_path.endswith(".py"):
            if file_path.endswith("__init__.py"):
                module_path = file_path[:-12]  # Remove /__init__.py
            else:
                module_path = file_path[:-3]  # Remove .py
            
            return module_path.replace(os.sep, ".").replace("/", ".")
        
        return None
    
    def _module_to_file(self, module_name: str) -> Optional[pathlib.Path]:
        """Convert module name to file path."""
        module_path = module_name.replace(".", os.sep)
        
        # Try direct .py file
        py_file = self.repo_root / f"{module_path}.py"
        if py_file.exists():
            return py_file
        
        # Try __init__.py in directory
        init_file = self.repo_root / module_path / "__init__.py"
        if init_file.exists():
            return init_file
        
        return None
    
    def _module_to_file_path(self, module_name: str) -> str:
        """Convert module name to relative file path string."""
        file_path = self._module_to_file(module_name)
        if file_path:
            return str(file_path.relative_to(self.repo_root))
        return f"{module_name.replace('.', '/')}.py"
    
    def _is_local_module(self, module_name: str) -> bool:
        """Check if module is local to this repository."""
        if not module_name:
            return False
        
        return (module_name.startswith("src") or 
                module_name.startswith("tests") or
                module_name.startswith("test") or
                self._module_to_file(module_name) is not None)
    
    def _extract_imports_with_targets(self, tree: ast.AST) -> List[str]:
        """Extract import target modules from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    
    def _generate_statistics(self, evidence_table: Dict[str, Dict[str, Any]], candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about the analysis."""
        total_modules = len(evidence_table)
        reachable_count = sum(1 for e in evidence_table.values() if e["reachable_from_enhanced"])
        runtime_used_count = sum(1 for e in evidence_table.values() if e["runtime_usage"])
        test_covered_count = sum(1 for e in evidence_table.values() if e["test_coverage"])
        
        return {
            "total_modules": total_modules,
            "reachable_from_enhanced": reachable_count,
            "runtime_used": runtime_used_count,
            "test_covered": test_covered_count,
            "high_confidence_candidates": len(candidates["high_confidence"]),
            "medium_confidence_candidates": len(candidates["medium_confidence"]),
            "low_confidence_candidates": len(candidates["low_confidence"]),
            "allowlisted_modules": len(candidates["allowlisted"]),
            "removal_potential": {
                "high_confidence": round(len(candidates["high_confidence"]) / total_modules * 100, 1),
                "total_candidates": round((len(candidates["high_confidence"]) + len(candidates["medium_confidence"])) / total_modules * 100, 1)
            }
        }


def main():
    """Run the legacy code identification."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    identifier = LegacyCodeIdentifier(repo_root)
    
    results = identifier.identify_legacy_candidates()
    
    # Save results
    output_file = repo_root / "tools" / "legacy_candidates.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    stats = results["statistics"]
    print(f"\n📊 Legacy Code Analysis Summary")
    print(f"📦 Total Modules: {stats['total_modules']}")
    print(f"🎯 Enhanced Engine Reachable: {stats['reachable_from_enhanced']} ({stats['reachable_from_enhanced']/stats['total_modules']*100:.1f}%)")
    print(f"🏃 Runtime Used: {stats['runtime_used']} ({stats['runtime_used']/stats['total_modules']*100:.1f}%)")
    print(f"🧪 Test Covered: {stats['test_covered']} ({stats['test_covered']/stats['total_modules']*100:.1f}%)")
    
    print(f"\n🗑️ Removal Candidates:")
    print(f"  • High Confidence: {stats['high_confidence_candidates']} ({stats['removal_potential']['high_confidence']}%)")
    print(f"  • Medium Confidence: {stats['medium_confidence_candidates']}")
    print(f"  • Low Confidence: {stats['low_confidence_candidates']}")
    print(f"  • Allowlisted: {stats['allowlisted_modules']}")
    
    candidates = results["legacy_candidates"]
    print(f"\n📋 Top High-Confidence Candidates:")
    for candidate in candidates["high_confidence"][:5]:
        evidence = candidate["evidence"]
        print(f"  • {candidate['module']} (risk: {evidence['risk_level']}, score: {evidence['risk_score']})")
    
    print(f"\n💾 Full analysis saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()