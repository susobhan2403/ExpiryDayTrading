#!/usr/bin/env python3
"""
R1: Enhanced Engine Entry Points & Code Paths Detection

Auto-detects main entry points: runners/CLIs, service start scripts, notebooks, 
UI bindings, and CI workflows that invoke the enhanced engine.
Maps configuration flags/feature toggles that select enhanced vs legacy.
"""

from __future__ import annotations
import ast
import json
import os
import pathlib
import re
import yaml
from typing import Dict, List, Set, Tuple, Any, Optional
import subprocess


class EntryPointDetector:
    def __init__(self, repo_root: pathlib.Path):
        self.repo_root = repo_root
        self.entry_points = {}
        self.feature_flags = {}
        self.config_refs = {}
        
    def detect_all_entry_points(self) -> Dict[str, Any]:
        """Main detection method that runs all entry point detection strategies."""
        print("🔍 R1: Detecting Enhanced Engine Entry Points & Code Paths")
        print("=" * 60)
        
        # 1. Python main entry points
        python_entries = self._detect_python_main_scripts()
        
        # 2. CLI command entry points  
        cli_entries = self._detect_cli_entry_points()
        
        # 3. Service/daemon start scripts
        service_entries = self._detect_service_scripts()
        
        # 4. Notebook entry points
        notebook_entries = self._detect_notebook_entry_points()
        
        # 5. CI workflow entry points
        ci_entries = self._detect_ci_workflows()
        
        # 6. Package entry points (setup.py, pyproject.toml)
        package_entries = self._detect_package_entry_points()
        
        # 7. Configuration-based feature flags
        feature_flags = self._detect_feature_flags()
        
        # 8. Import-based entry detection
        import_entries = self._detect_import_based_entries()
        
        results = {
            "python_main_scripts": python_entries,
            "cli_entry_points": cli_entries, 
            "service_scripts": service_entries,
            "notebook_entry_points": notebook_entries,
            "ci_workflows": ci_entries,
            "package_entry_points": package_entries,
            "feature_flags": feature_flags,
            "import_based_entries": import_entries,
            "summary": self._generate_summary(
                python_entries, cli_entries, service_entries, 
                notebook_entries, ci_entries, package_entries, 
                feature_flags, import_entries
            )
        }
        
        return results
    
    def _detect_python_main_scripts(self) -> List[Dict[str, Any]]:
        """Find Python files with if __name__ == '__main__': patterns."""
        main_scripts = []
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for main entry pattern
                if re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', content):
                    # Parse AST to get more details
                    try:
                        tree = ast.parse(content)
                        imports = self._extract_imports(tree)
                        functions = self._extract_functions(tree)
                        
                        main_scripts.append({
                            "file": str(py_file.relative_to(self.repo_root)),
                            "size_lines": len(content.splitlines()),
                            "imports": imports,
                            "functions": functions,
                            "has_argparse": "argparse" in content,
                            "has_click": "click" in content,
                            "enhanced_engine_refs": self._count_enhanced_refs(content),
                            "legacy_engine_refs": self._count_legacy_refs(content)
                        })
                    except SyntaxError:
                        # Still include files with syntax errors
                        main_scripts.append({
                            "file": str(py_file.relative_to(self.repo_root)),
                            "size_lines": len(content.splitlines()),
                            "parse_error": True,
                            "enhanced_engine_refs": self._count_enhanced_refs(content),
                            "legacy_engine_refs": self._count_legacy_refs(content)
                        })
                        
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
                
        return main_scripts
    
    def _detect_cli_entry_points(self) -> List[Dict[str, Any]]:
        """Detect CLI-style entry points and command patterns."""
        cli_entries = []
        
        # Look for CLI-specific patterns
        cli_patterns = [
            r'@click\.command',
            r'argparse\.ArgumentParser',
            r'parser\.add_argument',
            r'sys\.argv',
            r'CLI|Command Line|commander',
        ]
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                cli_score = 0
                matched_patterns = []
                
                for pattern in cli_patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        cli_score += matches
                        matched_patterns.append(pattern)
                
                if cli_score > 0:
                    cli_entries.append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "cli_score": cli_score,
                        "matched_patterns": matched_patterns,
                        "enhanced_engine_refs": self._count_enhanced_refs(content),
                        "legacy_engine_refs": self._count_legacy_refs(content),
                        "is_executable": os.access(py_file, os.X_OK)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
                
        return sorted(cli_entries, key=lambda x: x['cli_score'], reverse=True)
    
    def _detect_service_scripts(self) -> List[Dict[str, Any]]:
        """Detect service/daemon start scripts."""
        service_entries = []
        
        # Look for service patterns
        service_patterns = [
            r'daemon|service|worker|server',
            r'subprocess\.Popen',
            r'threading\.Thread',
            r'multiprocessing',
            r'signal\.signal',
            r'while\s+True:',
            r'time\.sleep'
        ]
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                service_score = 0
                matched_patterns = []
                
                for pattern in service_patterns:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    if matches > 0:
                        service_score += matches
                        matched_patterns.append(pattern)
                
                if service_score >= 3:  # Threshold for service detection
                    service_entries.append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "service_score": service_score,
                        "matched_patterns": matched_patterns,
                        "enhanced_engine_refs": self._count_enhanced_refs(content),
                        "legacy_engine_refs": self._count_legacy_refs(content)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
                
        return sorted(service_entries, key=lambda x: x['service_score'], reverse=True)
    
    def _detect_notebook_entry_points(self) -> List[Dict[str, Any]]:
        """Detect Jupyter notebook entry points."""
        notebook_entries = []
        
        for nb_file in self.repo_root.rglob("*.ipynb"):
            try:
                with open(nb_file, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Extract code cells
                code_content = ""
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        code_content += '\n'.join(cell.get('source', []))
                
                if code_content:
                    notebook_entries.append({
                        "file": str(nb_file.relative_to(self.repo_root)),
                        "enhanced_engine_refs": self._count_enhanced_refs(code_content),
                        "legacy_engine_refs": self._count_legacy_refs(code_content),
                        "cell_count": len(notebook.get('cells', [])),
                        "has_trading_refs": bool(re.search(r'trading|engine|strategy', code_content, re.IGNORECASE))
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {nb_file}: {e}")
                
        return notebook_entries
    
    def _detect_ci_workflows(self) -> List[Dict[str, Any]]:
        """Detect CI workflow entry points."""
        ci_entries = []
        
        # GitHub Actions
        for workflow_file in self.repo_root.rglob(".github/workflows/*.yml"):
            ci_entries.extend(self._parse_github_workflow(workflow_file))
        
        for workflow_file in self.repo_root.rglob(".github/workflows/*.yaml"):
            ci_entries.extend(self._parse_github_workflow(workflow_file))
        
        # GitLab CI
        gitlab_ci = self.repo_root / ".gitlab-ci.yml"
        if gitlab_ci.exists():
            ci_entries.extend(self._parse_gitlab_ci(gitlab_ci))
        
        # Jenkins
        jenkins_file = self.repo_root / "Jenkinsfile"
        if jenkins_file.exists():
            ci_entries.extend(self._parse_jenkinsfile(jenkins_file))
        
        return ci_entries
    
    def _detect_package_entry_points(self) -> List[Dict[str, Any]]:
        """Detect package-defined entry points."""
        package_entries = []
        
        # setup.py entry points
        setup_py = self.repo_root / "setup.py"
        if setup_py.exists():
            package_entries.extend(self._parse_setup_py(setup_py))
        
        # pyproject.toml entry points
        pyproject_toml = self.repo_root / "pyproject.toml"
        if pyproject_toml.exists():
            package_entries.extend(self._parse_pyproject_toml(pyproject_toml))
        
        # setup.cfg entry points
        setup_cfg = self.repo_root / "setup.cfg"
        if setup_cfg.exists():
            package_entries.extend(self._parse_setup_cfg(setup_cfg))
        
        return package_entries
    
    def _detect_feature_flags(self) -> Dict[str, Any]:
        """Detect configuration flags that toggle enhanced vs legacy."""
        feature_flags = {
            "config_files": [],
            "environment_variables": [],
            "feature_toggles": []
        }
        
        # Look for config files
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg"]
        for pattern in config_patterns:
            for config_file in self.repo_root.rglob(pattern):
                try:
                    content = config_file.read_text(encoding='utf-8', errors='ignore')
                    
                    # Look for enhanced/legacy toggles
                    toggles = []
                    if re.search(r'enhanced|legacy|engine.*mode|version', content, re.IGNORECASE):
                        toggles = re.findall(r'[\'"](enhanced|legacy|engine.*mode|version)[\'"]', content, re.IGNORECASE)
                    
                    if toggles:
                        feature_flags["config_files"].append({
                            "file": str(config_file.relative_to(self.repo_root)),
                            "toggles": toggles
                        })
                        
                except Exception as e:
                    print(f"Warning: Could not process config file {config_file}: {e}")
        
        # Look for environment variables in Python files
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                env_vars = re.findall(r'os\.getenv\([\'"]([^\'"]*)[\'"]\)', content)
                env_vars.extend(re.findall(r'os\.environ\[[\'"]([^\'"]*)[\'"]\]', content))
                
                if env_vars:
                    feature_flags["environment_variables"].append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "env_vars": env_vars
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
        
        return feature_flags
    
    def _detect_import_based_entries(self) -> Dict[str, Any]:
        """Detect entry points based on import patterns."""
        import_entries = {
            "enhanced_engine_importers": [],
            "legacy_engine_importers": [],
            "engine_factories": []
        }
        
        for py_file in self.repo_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Enhanced engine imports
                enhanced_imports = []
                if "engine_enhanced" in content:
                    enhanced_imports.append("engine_enhanced")
                if "EnhancedTradingEngine" in content:
                    enhanced_imports.append("EnhancedTradingEngine")
                
                if enhanced_imports:
                    import_entries["enhanced_engine_importers"].append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "imports": enhanced_imports
                    })
                
                # Legacy engine imports (before removal)
                legacy_imports = []
                if re.search(r'from\s+engine\s+import|import\s+engine', content):
                    legacy_imports.append("engine")
                
                if legacy_imports:
                    import_entries["legacy_engine_importers"].append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "imports": legacy_imports
                    })
                
                # Factory patterns
                if re.search(r'create.*engine|engine.*factory|get.*engine', content, re.IGNORECASE):
                    import_entries["engine_factories"].append({
                        "file": str(py_file.relative_to(self.repo_root)),
                        "patterns": re.findall(r'(create.*engine|engine.*factory|get.*engine)', content, re.IGNORECASE)
                    })
                    
            except Exception as e:
                print(f"Warning: Could not process {py_file}: {e}")
        
        return import_entries
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        """Extract function definitions from AST."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _count_enhanced_refs(self, content: str) -> int:
        """Count references to enhanced engine components."""
        enhanced_patterns = [
            r'engine_enhanced',
            r'EnhancedTradingEngine',
            r'enhanced.*engine',
            r'engine.*enhanced'
        ]
        
        count = 0
        for pattern in enhanced_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        return count
    
    def _count_legacy_refs(self, content: str) -> int:
        """Count references to legacy engine components."""
        legacy_patterns = [
            r'\bengine\.py\b',
            r'from\s+engine\s+import',
            r'import\s+engine\b',
            r'legacy.*engine',
            r'old.*engine'
        ]
        
        count = 0
        for pattern in legacy_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        return count
    
    def _parse_github_workflow(self, workflow_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse GitHub Actions workflow file."""
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow = yaml.safe_load(f)
            
            entries = []
            for job_name, job in workflow.get('jobs', {}).items():
                steps = job.get('steps', [])
                for step in steps:
                    run_cmd = step.get('run', '')
                    if 'python' in run_cmd and ('engine' in run_cmd or 'trading' in run_cmd):
                        entries.append({
                            "file": str(workflow_file.relative_to(self.repo_root)),
                            "job": job_name,
                            "step": step.get('name', 'unnamed'),
                            "command": run_cmd,
                            "type": "github_actions"
                        })
            
            return entries
            
        except Exception as e:
            print(f"Warning: Could not parse GitHub workflow {workflow_file}: {e}")
            return []
    
    def _parse_gitlab_ci(self, ci_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse GitLab CI file."""
        try:
            with open(ci_file, 'r', encoding='utf-8') as f:
                ci_config = yaml.safe_load(f)
            
            entries = []
            for job_name, job in ci_config.items():
                if isinstance(job, dict) and 'script' in job:
                    for script_line in job['script']:
                        if 'python' in script_line and ('engine' in script_line or 'trading' in script_line):
                            entries.append({
                                "file": str(ci_file.relative_to(self.repo_root)),
                                "job": job_name,
                                "command": script_line,
                                "type": "gitlab_ci"
                            })
            
            return entries
            
        except Exception as e:
            print(f"Warning: Could not parse GitLab CI {ci_file}: {e}")
            return []
    
    def _parse_jenkinsfile(self, jenkins_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse Jenkinsfile for entry points."""
        try:
            content = jenkins_file.read_text(encoding='utf-8', errors='ignore')
            
            # Look for sh/bat commands with python
            commands = re.findall(r'sh\s*[\'"]([^\'"]*)[\'"]\s*|bat\s*[\'"]([^\'"]*)[\'"]\s*', content)
            
            entries = []
            for cmd_tuple in commands:
                cmd = cmd_tuple[0] or cmd_tuple[1]
                if 'python' in cmd and ('engine' in cmd or 'trading' in cmd):
                    entries.append({
                        "file": str(jenkins_file.relative_to(self.repo_root)),
                        "command": cmd,
                        "type": "jenkins"
                    })
            
            return entries
            
        except Exception as e:
            print(f"Warning: Could not parse Jenkinsfile {jenkins_file}: {e}")
            return []
    
    def _parse_setup_py(self, setup_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse setup.py for entry points."""
        try:
            content = setup_file.read_text(encoding='utf-8', errors='ignore')
            
            # Look for entry_points parameter
            entry_points_match = re.search(r'entry_points\s*=\s*{([^}]*)}\s*', content, re.DOTALL)
            if entry_points_match:
                return [{
                    "file": str(setup_file.relative_to(self.repo_root)),
                    "entry_points": entry_points_match.group(1),
                    "type": "setup_py"
                }]
            
            return []
            
        except Exception as e:
            print(f"Warning: Could not parse setup.py {setup_file}: {e}")
            return []
    
    def _parse_pyproject_toml(self, toml_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse pyproject.toml for entry points."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print(f"Warning: toml library not available, skipping {toml_file}")
                return []
        
        try:
            with open(toml_file, 'rb') as f:
                config = tomllib.load(f)
            
            entries = []
            
            # Poetry scripts
            poetry_scripts = config.get('tool', {}).get('poetry', {}).get('scripts', {})
            if poetry_scripts:
                entries.append({
                    "file": str(toml_file.relative_to(self.repo_root)),
                    "scripts": poetry_scripts,
                    "type": "poetry_scripts"
                })
            
            # Project entry points
            project_entry_points = config.get('project', {}).get('entry-points', {})
            if project_entry_points:
                entries.append({
                    "file": str(toml_file.relative_to(self.repo_root)),
                    "entry_points": project_entry_points,
                    "type": "project_entry_points"
                })
            
            return entries
            
        except Exception as e:
            print(f"Warning: Could not parse pyproject.toml {toml_file}: {e}")
            return []
    
    def _parse_setup_cfg(self, cfg_file: pathlib.Path) -> List[Dict[str, Any]]:
        """Parse setup.cfg for entry points."""
        try:
            import configparser
            
            config = configparser.ConfigParser()
            config.read(cfg_file)
            
            entries = []
            
            if 'entry_points' in config:
                entries.append({
                    "file": str(cfg_file.relative_to(self.repo_root)),
                    "entry_points": dict(config['entry_points']),
                    "type": "setup_cfg"
                })
            
            return entries
            
        except Exception as e:
            print(f"Warning: Could not parse setup.cfg {cfg_file}: {e}")
            return []
    
    def _generate_summary(self, python_entries, cli_entries, service_entries, 
                         notebook_entries, ci_entries, package_entries, 
                         feature_flags, import_entries) -> Dict[str, Any]:
        """Generate a comprehensive summary of findings."""
        
        # Categorize by enhanced vs legacy
        enhanced_files = set()
        legacy_files = set()
        
        for entry in python_entries:
            if entry.get("enhanced_engine_refs", 0) > 0:
                enhanced_files.add(entry["file"])
            if entry.get("legacy_engine_refs", 0) > 0:
                legacy_files.add(entry["file"])
        
        for entry in cli_entries:
            if entry.get("enhanced_engine_refs", 0) > 0:
                enhanced_files.add(entry["file"])
            if entry.get("legacy_engine_refs", 0) > 0:
                legacy_files.add(entry["file"])
        
        for entry in service_entries:
            if entry.get("enhanced_engine_refs", 0) > 0:
                enhanced_files.add(entry["file"])
            if entry.get("legacy_engine_refs", 0) > 0:
                legacy_files.add(entry["file"])
        
        return {
            "total_entry_points": {
                "python_main": len(python_entries),
                "cli_tools": len(cli_entries),
                "service_scripts": len(service_entries),
                "notebooks": len(notebook_entries),
                "ci_workflows": len(ci_entries),
                "package_defined": len(package_entries)
            },
            "enhanced_engine_files": sorted(list(enhanced_files)),
            "legacy_engine_files": sorted(list(legacy_files)),
            "feature_flag_files": len(feature_flags.get("config_files", [])),
            "top_entry_points": {
                "highest_cli_score": cli_entries[0] if cli_entries else None,
                "highest_service_score": service_entries[0] if service_entries else None,
                "most_enhanced_refs": max(python_entries, 
                                        key=lambda x: x.get("enhanced_engine_refs", 0), 
                                        default=None) if python_entries else None
            }
        }


def main():
    """Run the entry point detection."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    detector = EntryPointDetector(repo_root)
    
    results = detector.detect_all_entry_points()
    
    # Save results
    output_file = repo_root / "tools" / "enhanced_engine_inventory.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    summary = results["summary"]
    print(f"\n📊 Enhanced Engine Inventory Summary")
    print(f"📁 Total Entry Points Found: {sum(summary['total_entry_points'].values())}")
    print(f"🚀 Enhanced Engine Files: {len(summary['enhanced_engine_files'])}")
    print(f"🏚️  Legacy Engine Files: {len(summary['legacy_engine_files'])}")
    print(f"⚙️  Feature Flag Files: {summary['feature_flag_files']}")
    
    print(f"\n📋 Key Entry Points:")
    for file in summary['enhanced_engine_files'][:5]:
        print(f"  • {file}")
    
    print(f"\n💾 Full inventory saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()