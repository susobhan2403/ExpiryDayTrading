#!/usr/bin/env python3
"""
R4: Safety Checks & Removal Simulation

Search for references in:
• Config files (YAML/JSON/ENV), CI workflows, scripts, docs/READMEs, and packaging 
  (pyproject.toml/setup.cfg entry_points).
• UI layer (web/blazor/react) imports or API routes that may hit legacy endpoints.
Propose and run a dry-run removal simulation.
"""

from __future__ import annotations
import json
import pathlib
import re
import subprocess
import sys
import yaml
import os
import shutil
import tempfile
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict


class SafetyChecker:
    def __init__(self, repo_root: pathlib.Path):
        self.repo_root = repo_root
        self.safety_issues = []
        self.external_references = defaultdict(list)
        
    def run_safety_checks(self) -> Dict[str, Any]:
        """Run comprehensive safety checks before removal."""
        print("🛡️ R4: Running Safety Checks & Removal Simulation")
        print("=" * 55)
        
        # 1. Load legacy candidates
        print("📊 1. Loading Legacy Candidates...")
        candidates = self._load_legacy_candidates()
        
        # 2. Check external references
        print("🔍 2. Checking External References...")
        external_refs = self._check_external_references(candidates)
        
        # 3. Check CI/CD workflows
        print("⚙️ 3. Checking CI/CD Workflows...")
        ci_refs = self._check_ci_workflows(candidates)
        
        # 4. Check packaging/entry points
        print("📦 4. Checking Packaging & Entry Points...")
        packaging_refs = self._check_packaging_entry_points(candidates)
        
        # 5. Check documentation
        print("📚 5. Checking Documentation...")
        doc_refs = self._check_documentation(candidates)
        
        # 6. Check UI/API references
        print("🌐 6. Checking UI/API References...")
        ui_refs = self._check_ui_api_references(candidates)
        
        # 7. Run dry-run simulation
        print("🧪 7. Running Dry-Run Simulation...")
        simulation = self._run_removal_simulation(candidates)
        
        # 8. Generate quick fixes
        print("🔧 8. Generating Quick Fixes...")
        quick_fixes = self._generate_quick_fixes(simulation)
        
        return {
            "candidates": candidates,
            "external_references": dict(external_refs),
            "ci_references": ci_refs,
            "packaging_references": packaging_refs,
            "documentation_references": doc_refs,
            "ui_api_references": ui_refs,
            "removal_simulation": simulation,
            "quick_fixes": quick_fixes,
            "safety_summary": self._generate_safety_summary()
        }
    
    def _load_legacy_candidates(self) -> Dict[str, Any]:
        """Load legacy candidate analysis."""
        candidates_file = self.repo_root / "tools" / "legacy_candidates.json"
        if candidates_file.exists():
            with open(candidates_file) as f:
                return json.load(f)
        else:
            print("Warning: Legacy candidates not found, running identification...")
            subprocess.run([sys.executable, str(self.repo_root / "tools" / "legacy_code_identifier.py")])
            if candidates_file.exists():
                with open(candidates_file) as f:
                    return json.load(f)
        return {}
    
    def _check_external_references(self, candidates: Dict[str, Any]) -> Dict[str, List[str]]:
        """Check for external references in config files."""
        external_refs = defaultdict(list)
        
        # Get modules to check
        modules_to_check = set()
        for candidate in candidates.get("legacy_candidates", {}).get("high_confidence", []):
            modules_to_check.add(candidate["module"])
        for candidate in candidates.get("legacy_candidates", {}).get("medium_confidence", []):
            modules_to_check.add(candidate["module"])
        
        # Check config files
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", 
            "*.env", ".env*", "settings.*", "config.*"
        ]
        
        for pattern in config_patterns:
            for config_file in self.repo_root.rglob(pattern):
                if self._should_skip_file(config_file):
                    continue
                    
                try:
                    content = config_file.read_text(encoding='utf-8', errors='ignore')
                    
                    for module in modules_to_check:
                        # Direct module name references
                        if module in content:
                            external_refs[module].append({
                                "file": str(config_file.relative_to(self.repo_root)),
                                "type": "config_file",
                                "context": self._extract_context(content, module)
                            })
                        
                        # Module path references
                        module_path = module.replace(".", "/")
                        if module_path in content:
                            external_refs[module].append({
                                "file": str(config_file.relative_to(self.repo_root)),
                                "type": "config_path_reference",
                                "context": self._extract_context(content, module_path)
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not process config file {config_file}: {e}")
        
        return dict(external_refs)
    
    def _check_ci_workflows(self, candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check CI/CD workflows for references to modules."""
        ci_refs = []
        
        # Get modules to check
        modules_to_check = set()
        for candidate in candidates.get("legacy_candidates", {}).get("high_confidence", []):
            modules_to_check.add(candidate["module"])
        for candidate in candidates.get("legacy_candidates", {}).get("medium_confidence", []):
            modules_to_check.add(candidate["module"])
        
        # GitHub Actions
        github_workflows = self.repo_root / ".github" / "workflows"
        if github_workflows.exists():
            for workflow_file in github_workflows.rglob("*.yml"):
                ci_refs.extend(self._check_workflow_file(workflow_file, modules_to_check, "github_actions"))
            for workflow_file in github_workflows.rglob("*.yaml"):
                ci_refs.extend(self._check_workflow_file(workflow_file, modules_to_check, "github_actions"))
        
        # GitLab CI
        gitlab_ci = self.repo_root / ".gitlab-ci.yml"
        if gitlab_ci.exists():
            ci_refs.extend(self._check_workflow_file(gitlab_ci, modules_to_check, "gitlab_ci"))
        
        # Jenkins
        jenkins_file = self.repo_root / "Jenkinsfile"
        if jenkins_file.exists():
            ci_refs.extend(self._check_workflow_file(jenkins_file, modules_to_check, "jenkins"))
        
        # Azure Pipelines
        azure_pipelines = self.repo_root / ".azure-pipelines.yml"
        if azure_pipelines.exists():
            ci_refs.extend(self._check_workflow_file(azure_pipelines, modules_to_check, "azure_pipelines"))
        
        return ci_refs
    
    def _check_packaging_entry_points(self, candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check packaging files for entry point references."""
        packaging_refs = []
        
        # Get modules to check
        modules_to_check = set()
        for candidate in candidates.get("legacy_candidates", {}).get("high_confidence", []):
            modules_to_check.add(candidate["module"])
        for candidate in candidates.get("legacy_candidates", {}).get("medium_confidence", []):
            modules_to_check.add(candidate["module"])
        
        # setup.py
        setup_py = self.repo_root / "setup.py"
        if setup_py.exists():
            packaging_refs.extend(self._check_packaging_file(setup_py, modules_to_check, "setup_py"))
        
        # pyproject.toml
        pyproject_toml = self.repo_root / "pyproject.toml"
        if pyproject_toml.exists():
            packaging_refs.extend(self._check_packaging_file(pyproject_toml, modules_to_check, "pyproject_toml"))
        
        # setup.cfg
        setup_cfg = self.repo_root / "setup.cfg"
        if setup_cfg.exists():
            packaging_refs.extend(self._check_packaging_file(setup_cfg, modules_to_check, "setup_cfg"))
        
        # requirements files
        for req_file in self.repo_root.rglob("requirements*.txt"):
            packaging_refs.extend(self._check_packaging_file(req_file, modules_to_check, "requirements"))
        
        return packaging_refs
    
    def _check_documentation(self, candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check documentation for references to modules."""
        doc_refs = []
        
        # Get modules to check
        modules_to_check = set()
        for candidate in candidates.get("legacy_candidates", {}).get("high_confidence", []):
            modules_to_check.add(candidate["module"])
        for candidate in candidates.get("legacy_candidates", {}).get("medium_confidence", []):
            modules_to_check.add(candidate["module"])
        
        # Documentation patterns
        doc_patterns = [
            "*.md", "*.rst", "*.txt", "README*", "CHANGELOG*", 
            "docs/**/*", "doc/**/*", "documentation/**/*"
        ]
        
        for pattern in doc_patterns:
            for doc_file in self.repo_root.rglob(pattern):
                if self._should_skip_file(doc_file):
                    continue
                    
                try:
                    content = doc_file.read_text(encoding='utf-8', errors='ignore')
                    
                    for module in modules_to_check:
                        if module in content:
                            doc_refs.append({
                                "file": str(doc_file.relative_to(self.repo_root)),
                                "module": module,
                                "type": "documentation",
                                "context": self._extract_context(content, module)
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not process doc file {doc_file}: {e}")
        
        return doc_refs
    
    def _check_ui_api_references(self, candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check UI/API files for references to modules."""
        ui_refs = []
        
        # Get modules to check
        modules_to_check = set()
        for candidate in candidates.get("legacy_candidates", {}).get("high_confidence", []):
            modules_to_check.add(candidate["module"])
        for candidate in candidates.get("legacy_candidates", {}).get("medium_confidence", []):
            modules_to_check.add(candidate["module"])
        
        # UI/API patterns
        ui_patterns = [
            "*.html", "*.js", "*.ts", "*.jsx", "*.tsx", "*.vue",
            "*.cs", "*.razor", "*.cshtml",  # Blazor
            "*.php", "*.asp", "*.aspx",     # Web
            "templates/**/*", "static/**/*",
            "api/**/*", "routes/**/*", "endpoints/**/*"
        ]
        
        for pattern in ui_patterns:
            for ui_file in self.repo_root.rglob(pattern):
                if self._should_skip_file(ui_file):
                    continue
                    
                try:
                    content = ui_file.read_text(encoding='utf-8', errors='ignore')
                    
                    for module in modules_to_check:
                        # Check for API endpoint references
                        module_url = module.replace(".", "/")
                        if module_url in content or module in content:
                            ui_refs.append({
                                "file": str(ui_file.relative_to(self.repo_root)),
                                "module": module,
                                "type": "ui_api_reference",
                                "context": self._extract_context(content, module)
                            })
                            
                except Exception as e:
                    print(f"Warning: Could not process UI file {ui_file}: {e}")
        
        return ui_refs
    
    def _run_removal_simulation(self, candidates: Dict[str, Any]) -> Dict[str, Any]:
        """Run a dry-run removal simulation."""
        simulation = {
            "files_to_remove": [],
            "import_breaks": [],
            "test_breaks": [],
            "safe_removals": [],
            "risky_removals": []
        }
        
        # Get high confidence candidates
        high_confidence = candidates.get("legacy_candidates", {}).get("high_confidence", [])
        
        # Create temporary copy for simulation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_repo = pathlib.Path(temp_dir) / "simulation"
            shutil.copytree(self.repo_root, temp_repo, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
            
            removal_results = []
            
            for candidate in high_confidence[:10]:  # Limit to first 10 for performance
                module = candidate["module"]
                file_path = candidate.get("file_path", "")
                
                if not file_path:
                    continue
                
                temp_file = temp_repo / file_path
                if not temp_file.exists():
                    continue
                
                # Simulate removal
                backup_content = temp_file.read_text(encoding='utf-8', errors='ignore')
                temp_file.unlink()  # Remove file
                
                # Check for import breaks
                import_breaks = self._check_import_breaks(temp_repo, module)
                
                # Try to run tests
                test_result = self._run_test_simulation(temp_repo)
                
                # Restore file for next iteration
                temp_file.write_text(backup_content, encoding='utf-8')
                
                removal_result = {
                    "module": module,
                    "file_path": file_path,
                    "import_breaks": import_breaks,
                    "test_result": test_result,
                    "safe_to_remove": len(import_breaks) == 0 and test_result.get("success", False)
                }
                
                removal_results.append(removal_result)
                
                if removal_result["safe_to_remove"]:
                    simulation["safe_removals"].append(removal_result)
                else:
                    simulation["risky_removals"].append(removal_result)
                
                simulation["files_to_remove"].append(file_path)
                simulation["import_breaks"].extend(import_breaks)
        
        return simulation
    
    def _check_import_breaks(self, temp_repo: pathlib.Path, removed_module: str) -> List[Dict[str, str]]:
        """Check for import breaks after removing a module."""
        breaks = []
        
        for py_file in temp_repo.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for imports of the removed module
                import_patterns = [
                    rf'from\s+{re.escape(removed_module)}\s+import',
                    rf'import\s+{re.escape(removed_module)}',
                    rf'from\s+{re.escape(removed_module)}\.[\w.]+\s+import',
                ]
                
                for pattern in import_patterns:
                    if re.search(pattern, content):
                        breaks.append({
                            "file": str(py_file.relative_to(temp_repo)),
                            "module": removed_module,
                            "type": "import_break",
                            "pattern": pattern
                        })
                        
            except Exception as e:
                print(f"Warning: Could not check import breaks in {py_file}: {e}")
        
        return breaks
    
    def _run_test_simulation(self, temp_repo: pathlib.Path) -> Dict[str, Any]:
        """Run tests in the simulated environment."""
        try:
            # Quick syntax check
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", "src/engine_enhanced.py"],
                cwd=str(temp_repo),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            syntax_ok = result.returncode == 0
            
            # Try import test
            import_result = subprocess.run(
                [sys.executable, "-c", "from src.engine_enhanced import EnhancedTradingEngine; print('Import OK')"],
                cwd=str(temp_repo),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            import_ok = import_result.returncode == 0
            
            return {
                "success": syntax_ok and import_ok,
                "syntax_check": syntax_ok,
                "import_check": import_ok,
                "output": import_result.stdout,
                "errors": import_result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_quick_fixes(self, simulation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate quick fixes for import breaks."""
        quick_fixes = []
        
        # Group import breaks by file
        breaks_by_file = defaultdict(list)
        for break_info in simulation.get("import_breaks", []):
            breaks_by_file[break_info["file"]].append(break_info)
        
        for file_path, breaks in breaks_by_file.items():
            full_path = self.repo_root / file_path
            if not full_path.exists():
                continue
            
            try:
                content = full_path.read_text(encoding='utf-8', errors='ignore')
                original_content = content
                
                # Remove broken imports
                for break_info in breaks:
                    pattern = break_info["pattern"]
                    # Find and comment out the broken import
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            lines[i] = f"# REMOVED: {line.strip()}"
                    
                    content = '\n'.join(lines)
                
                if content != original_content:
                    quick_fixes.append({
                        "file": file_path,
                        "type": "remove_broken_imports",
                        "original_content": original_content,
                        "fixed_content": content,
                        "description": f"Remove {len(breaks)} broken import(s)"
                    })
                    
            except Exception as e:
                print(f"Warning: Could not generate quick fix for {file_path}: {e}")
        
        return quick_fixes
    
    def _check_workflow_file(self, workflow_file: pathlib.Path, modules: Set[str], workflow_type: str) -> List[Dict[str, Any]]:
        """Check a single workflow file for module references."""
        refs = []
        
        try:
            content = workflow_file.read_text(encoding='utf-8', errors='ignore')
            
            for module in modules:
                if module in content:
                    refs.append({
                        "file": str(workflow_file.relative_to(self.repo_root)),
                        "module": module,
                        "type": workflow_type,
                        "context": self._extract_context(content, module)
                    })
                    
        except Exception as e:
            print(f"Warning: Could not process workflow file {workflow_file}: {e}")
        
        return refs
    
    def _check_packaging_file(self, packaging_file: pathlib.Path, modules: Set[str], packaging_type: str) -> List[Dict[str, Any]]:
        """Check a single packaging file for module references."""
        refs = []
        
        try:
            content = packaging_file.read_text(encoding='utf-8', errors='ignore')
            
            for module in modules:
                if module in content:
                    refs.append({
                        "file": str(packaging_file.relative_to(self.repo_root)),
                        "module": module,
                        "type": packaging_type,
                        "context": self._extract_context(content, module)
                    })
                    
        except Exception as e:
            print(f"Warning: Could not process packaging file {packaging_file}: {e}")
        
        return refs
    
    def _should_skip_file(self, file_path: pathlib.Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            ".git", "__pycache__", ".pytest_cache", "node_modules",
            ".venv", "venv", ".env", "env",
            ".coverage", "coverage.xml", ".tox",
            "*.pyc", "*.pyo", "*.pyd",
            "*.log", "*.tmp"
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _extract_context(self, content: str, search_term: str, context_lines: int = 2) -> List[str]:
        """Extract context lines around search term."""
        lines = content.split('\n')
        context = []
        
        for i, line in enumerate(lines):
            if search_term in line:
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                context.extend(lines[start:end])
                break
        
        return context
    
    def _generate_safety_summary(self) -> Dict[str, Any]:
        """Generate summary of safety check results."""
        return {
            "total_issues": len(self.safety_issues),
            "critical_issues": len([issue for issue in self.safety_issues if issue.get("severity") == "critical"]),
            "warnings": len([issue for issue in self.safety_issues if issue.get("severity") == "warning"]),
            "recommendations": [
                "Review all external references before removal",
                "Update documentation after removal", 
                "Test CI/CD pipelines after changes",
                "Consider gradual removal approach",
                "Maintain rollback capability"
            ]
        }


def generate_removal_simulation_report(safety_results: Dict[str, Any], output_file: pathlib.Path):
    """Generate the REMOVAL_SIMULATION.md report."""
    
    candidates = safety_results["candidates"]
    simulation = safety_results["removal_simulation"]
    
    report_content = f"""# Legacy Code Removal Simulation Report

## Overview

This report provides a comprehensive analysis of legacy code removal candidates and safety checks performed on the ExpiryDayTrading repository.

## Summary Statistics

- **Total Modules Analyzed**: {candidates.get('statistics', {}).get('total_modules', 'N/A')}
- **High Confidence Candidates**: {len(candidates.get('legacy_candidates', {}).get('high_confidence', []))}
- **Medium Confidence Candidates**: {len(candidates.get('legacy_candidates', {}).get('medium_confidence', []))}
- **Safe Removals**: {len(simulation.get('safe_removals', []))}
- **Risky Removals**: {len(simulation.get('risky_removals', []))}

## High Confidence Removal Candidates

The following modules have been identified as safe for removal based on multiple evidence criteria:

"""
    
    # Add high confidence candidates
    high_confidence = candidates.get("legacy_candidates", {}).get("high_confidence", [])
    for i, candidate in enumerate(high_confidence[:20], 1):  # Limit to top 20
        evidence = candidate["evidence"]
        report_content += f"""
### {i}. {candidate['module']}

- **File Path**: `{candidate.get('file_path', 'N/A')}`
- **Risk Level**: {evidence['risk_level']}
- **Risk Score**: {evidence['risk_score']}/8
- **Reachable from Enhanced Engine**: {'✅' if evidence['reachable_from_enhanced'] else '❌'}
- **Runtime Usage**: {'✅' if evidence['runtime_usage'] else '❌'}
- **Test Coverage**: {'✅' if evidence['test_coverage'] else '❌'}
- **Static References**: {evidence['static_references']}

"""
    
    # Add safety check results
    report_content += f"""
## Safety Check Results

### External References
"""
    
    external_refs = safety_results.get("external_references", {})
    if external_refs:
        for module, refs in list(external_refs.items())[:10]:  # Limit to first 10
            report_content += f"\n**{module}**:\n"
            for ref in refs[:3]:  # Limit to first 3 refs per module
                report_content += f"- {ref.get('type', 'unknown')}: `{ref.get('file', 'N/A')}`\n"
    else:
        report_content += "\nNo external references found in config files.\n"
    
    # Add CI/CD references
    ci_refs = safety_results.get("ci_references", [])
    report_content += f"""
### CI/CD References

"""
    if ci_refs:
        for ref in ci_refs[:10]:  # Limit to first 10
            report_content += f"- **{ref['module']}** in `{ref['file']}` ({ref['type']})\n"
    else:
        report_content += "No CI/CD references found.\n"
    
    # Add simulation results
    report_content += f"""
## Removal Simulation Results

### Files Marked for Removal

The following {len(simulation.get('files_to_remove', []))} files have been identified for removal:

"""
    
    for file_path in simulation.get("files_to_remove", [])[:20]:  # Limit to first 20
        report_content += f"- `{file_path}`\n"
    
    # Add import breaks
    import_breaks = simulation.get("import_breaks", [])
    if import_breaks:
        report_content += f"""
### Import Breaks Detected

The following import breaks would occur:

"""
        for break_info in import_breaks[:10]:  # Limit to first 10
            report_content += f"- **{break_info['module']}** in `{break_info['file']}`\n"
    
    # Add quick fixes
    quick_fixes = safety_results.get("quick_fixes", [])
    if quick_fixes:
        report_content += f"""
### Automated Quick Fixes

{len(quick_fixes)} files can be automatically fixed:

"""
        for fix in quick_fixes[:10]:  # Limit to first 10
            report_content += f"- `{fix['file']}`: {fix['description']}\n"
    
    # Add recommendations
    report_content += f"""
## Recommendations

### Immediate Actions
1. **Review High-Confidence Candidates**: The {len(high_confidence)} high-confidence candidates can likely be removed safely
2. **Address Import Breaks**: {len(import_breaks)} import breaks need to be resolved
3. **Update External References**: Review and update any external configuration references

### Removal Strategy
1. **Phase 1**: Remove high-confidence candidates with 0 external references
2. **Phase 2**: Update imports and remove medium-confidence candidates  
3. **Phase 3**: Clean up any remaining references

### Safety Measures
- Maintain git rollback capability for all changes
- Test enhanced engine functionality after each removal phase
- Update documentation to reflect removed modules
- Monitor CI/CD pipelines for failures

## Rollback Procedure

If issues arise, the removal can be rolled back using:

```bash
# Full rollback to current state
git reset --hard HEAD~N  # Where N is number of removal commits

# Partial rollback of specific files
git checkout HEAD~N -- path/to/file
```

## Next Steps

1. Review this simulation report
2. Apply automated quick fixes for import breaks
3. Execute Phase 1 removals (highest confidence)
4. Test enhanced engine functionality
5. Proceed with subsequent phases if tests pass

---

*Generated by Safety Checker & Removal Simulation Tool*
"""
    
    # Write the report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)


def main():
    """Run the safety checks and removal simulation."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    checker = SafetyChecker(repo_root)
    
    results = checker.run_safety_checks()
    
    # Save results
    output_file = repo_root / "tools" / "safety_check_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate removal simulation report
    report_file = repo_root / "REMOVAL_SIMULATION.md"
    generate_removal_simulation_report(results, report_file)
    
    # Print summary
    candidates = results["candidates"]
    simulation = results["removal_simulation"]
    
    print(f"\n🛡️ Safety Check Summary")
    print(f"📊 High Confidence Candidates: {len(candidates.get('legacy_candidates', {}).get('high_confidence', []))}")
    print(f"🔒 Safe Removals: {len(simulation.get('safe_removals', []))}")
    print(f"⚠️  Risky Removals: {len(simulation.get('risky_removals', []))}")
    print(f"💥 Import Breaks: {len(simulation.get('import_breaks', []))}")
    print(f"🔧 Quick Fixes Available: {len(results.get('quick_fixes', []))}")
    
    external_refs = results.get("external_references", {})
    ci_refs = results.get("ci_references", [])
    
    print(f"\n🔍 External References:")
    print(f"  • Config Files: {len(external_refs)} modules referenced")
    print(f"  • CI/CD: {len(ci_refs)} references found")
    
    print(f"\n📋 Reports Generated:")
    print(f"  • Detailed Analysis: {output_file}")
    print(f"  • Removal Simulation: {report_file}")
    
    return results


if __name__ == "__main__":
    main()