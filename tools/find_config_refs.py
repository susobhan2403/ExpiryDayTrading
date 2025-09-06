#!/usr/bin/env python3
"""
Config Reference Finder for Legacy Code Removal

This script searches for references to modules and packages in configuration files,
CI workflows, documentation, and packaging metadata to ensure safe removal.

Features:
- Scan YAML/JSON config files
- Check CI workflow files
- Search documentation and README files
- Examine packaging metadata (pyproject.toml, setup.cfg)
- Find string-based references to modules

Usage:
    python tools/find_config_refs.py --modules engine.py,old_module.py
    python tools/find_config_refs.py --scan-all
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, asdict
import yaml

@dataclass
class Reference:
    """Represents a reference to a module in a config/doc file."""
    file_path: str
    line_number: int
    line_content: str
    reference_type: str  # 'import', 'string', 'entry_point', 'config'
    context: str  # surrounding context

class ConfigReferenceFinder:
    """Finds references to modules in configuration and documentation files."""
    
    def __init__(self, project_root: str):
        self.project_root = pathlib.Path(project_root).resolve()
        self.references: Dict[str, List[Reference]] = {}
        
    def scan_file_for_references(self, file_path: pathlib.Path, module_names: Set[str]) -> List[Reference]:
        """Scan a single file for module references."""
        references = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            return references
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            # Check for each module name
            for module_name in module_names:
                if self._line_contains_module_reference(line, module_name):
                    ref_type = self._determine_reference_type(file_path, line, module_name)
                    context = self._get_context(lines, line_num - 1, 2)
                    
                    references.append(Reference(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line.rstrip('\n'),
                        reference_type=ref_type,
                        context=context
                    ))
        
        return references
    
    def _line_contains_module_reference(self, line: str, module_name: str) -> bool:
        """Check if a line contains a reference to the module."""
        # Remove common file extensions for matching
        module_base = module_name.replace('.py', '').replace('/', '.').replace('\\', '.')
        
        # Various patterns to match
        patterns = [
            # Direct module name
            rf'\b{re.escape(module_name)}\b',
            rf'\b{re.escape(module_base)}\b',
            
            # Import patterns
            rf'import\s+{re.escape(module_base)}',
            rf'from\s+{re.escape(module_base)}',
            
            # String patterns (in quotes)
            rf'["\'].*{re.escape(module_name)}.*["\']',
            rf'["\'].*{re.escape(module_base)}.*["\']',
            
            # Path patterns
            rf'{re.escape(module_name)}',
            
            # Entry point patterns
            rf'{re.escape(module_base)}:',
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    
    def _determine_reference_type(self, file_path: pathlib.Path, line: str, module_name: str) -> str:
        """Determine the type of reference."""
        file_ext = file_path.suffix.lower()
        
        # Check by file type and content
        if file_ext in ['.yml', '.yaml']:
            if 'entry_points' in line or 'console_scripts' in line:
                return 'entry_point'
            return 'config'
        elif file_ext == '.json':
            return 'config'
        elif file_ext in ['.toml', '.cfg', '.ini']:
            if 'entry_points' in line or 'console_scripts' in line:
                return 'entry_point'
            return 'config'
        elif file_ext in ['.md', '.rst', '.txt']:
            return 'documentation'
        elif 'import' in line or 'from' in line:
            return 'import'
        elif '"' in line or "'" in line:
            return 'string'
        else:
            return 'unknown'
    
    def _get_context(self, lines: List[str], center_idx: int, radius: int) -> str:
        """Get surrounding context for a reference."""
        start = max(0, center_idx - radius)
        end = min(len(lines), center_idx + radius + 1)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == center_idx else "    "
            context_lines.append(f"{prefix}{lines[i].rstrip()}")
        
        return '\n'.join(context_lines)
    
    def find_config_files(self) -> List[pathlib.Path]:
        """Find all configuration and documentation files to scan."""
        config_files = []
        
        # Patterns for different file types
        patterns = {
            'config': ['*.yml', '*.yaml', '*.json', '*.toml', '*.cfg', '*.ini'],
            'docs': ['*.md', '*.rst', '*.txt'],
            'ci': ['.github/**/*.yml', '.github/**/*.yaml', '.gitlab-ci.yml', 'Jenkinsfile'],
            'packaging': ['pyproject.toml', 'setup.py', 'setup.cfg', 'requirements*.txt'],
            'scripts': ['*.sh', '*.bat', 'Makefile', 'Dockerfile*']
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                config_files.extend(self.project_root.glob(pattern))
                # Also check in subdirectories
                config_files.extend(self.project_root.rglob(pattern))
        
        # Remove duplicates and filter out unwanted directories
        unique_files = set()
        for file_path in config_files:
            # Skip certain directories
            if any(part in str(file_path) for part in ['.git', '__pycache__', '.venv', 'node_modules']):
                continue
            unique_files.add(file_path)
        
        return sorted(unique_files)
    
    def scan_for_module_references(self, module_names: Set[str]) -> Dict[str, List[Reference]]:
        """Scan all config files for references to specified modules."""
        config_files = self.find_config_files()
        print(f"🔍 Scanning {len(config_files)} config/doc files for module references...")
        
        all_references = {}
        
        for file_path in config_files:
            try:
                references = self.scan_file_for_references(file_path, module_names)
                if references:
                    all_references[str(file_path)] = references
                    print(f"   Found {len(references)} references in {file_path}")
            except Exception as e:
                print(f"   Warning: Could not scan {file_path}: {e}")
        
        self.references = all_references
        return all_references
    
    def scan_entry_points(self) -> Dict[str, Any]:
        """Specifically scan for entry_points in packaging files."""
        entry_points = {}
        
        # Check pyproject.toml
        pyproject_path = self.project_root / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                if sys.version_info >= (3, 11):
                    import tomllib
                    with open(pyproject_path, 'rb') as f:
                        data = tomllib.load(f)
                else:
                    try:
                        import tomli
                        with open(pyproject_path, 'rb') as f:
                            data = tomli.load(f)
                    except ImportError:
                        # Fallback to basic parsing if tomli not available
                        with open(pyproject_path, 'r') as f:
                            content = f.read()
                        print(f"Warning: tomli not available, skipping detailed pyproject.toml parsing")
                        data = {}
                
                if 'project' in data and 'scripts' in data['project']:
                    entry_points['pyproject.toml:project.scripts'] = data['project']['scripts']
                
                if 'project' in data and 'entry-points' in data['project']:
                    entry_points['pyproject.toml:project.entry-points'] = data['project']['entry-points']
                    
            except Exception as e:
                print(f"Warning: Could not parse pyproject.toml: {e}")
        
        # Check setup.cfg
        setup_cfg_path = self.project_root / 'setup.cfg'
        if setup_cfg_path.exists():
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(setup_cfg_path)
                
                if 'entry_points' in config:
                    entry_points['setup.cfg:entry_points'] = dict(config['entry_points'])
                    
            except Exception as e:
                print(f"Warning: Could not parse setup.cfg: {e}")
        
        return entry_points
    
    def generate_report(self, module_names: Set[str]) -> Dict[str, Any]:
        """Generate comprehensive reference report."""
        # Scan for references
        file_references = self.scan_for_module_references(module_names)
        entry_points = self.scan_entry_points()
        
        # Aggregate by module
        module_refs = {}
        for module_name in module_names:
            module_refs[module_name] = {
                'total_references': 0,
                'files_with_references': [],
                'reference_types': {},
                'critical_references': []  # CI, packaging, entry points
            }
        
        # Count references by module
        for file_path, references in file_references.items():
            for ref in references:
                for module_name in module_names:
                    if module_name in ref.line_content:
                        module_data = module_refs[module_name]
                        module_data['total_references'] += 1
                        
                        if file_path not in module_data['files_with_references']:
                            module_data['files_with_references'].append(file_path)
                        
                        ref_type = ref.reference_type
                        module_data['reference_types'][ref_type] = module_data['reference_types'].get(ref_type, 0) + 1
                        
                        # Mark critical references
                        if (ref_type in ['entry_point', 'config'] or 
                            any(keyword in file_path.lower() for keyword in ['.github', 'ci', 'setup', 'pyproject'])):
                            module_data['critical_references'].append({
                                'file': file_path,
                                'line': ref.line_number,
                                'content': ref.line_content,
                                'type': ref_type
                            })
        
        return {
            'summary': {
                'scanned_files': len(self.find_config_files()),
                'files_with_references': len(file_references),
                'modules_with_references': len([m for m, data in module_refs.items() if data['total_references'] > 0])
            },
            'module_references': module_refs,
            'file_references': {fp: [asdict(ref) for ref in refs] for fp, refs in file_references.items()},
            'entry_points': entry_points
        }
    
    def export_report(self, report: Dict[str, Any], output_path: str):
        """Export reference report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Exported reference report to {output_path}")

def extract_module_names_from_file(file_path: str) -> Set[str]:
    """Extract module names from a file (e.g., dependency report)."""
    module_names = set()
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if 'dead_modules' in data:
                    module_names.update(data['dead_modules'])
                elif 'module_details' in data:
                    module_names.update(data['module_details'].keys())
            else:
                # Plain text file, one module per line
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        module_names.add(line)
    except Exception as e:
        print(f"Warning: Could not read modules from {file_path}: {e}")
    
    return module_names

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Find configuration references to modules for safe removal",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--modules',
        help='Comma-separated list of module names to search for'
    )
    
    parser.add_argument(
        '--modules-file',
        help='File containing module names (one per line or JSON with dead_modules)'
    )
    
    parser.add_argument(
        '--scan-all',
        action='store_true',
        help='Scan for common legacy patterns'
    )
    
    parser.add_argument(
        '--output',
        default='tools/config_references.json',
        help='Output file for reference report (default: tools/config_references.json)'
    )
    
    args = parser.parse_args()
    
    # Determine module names to search for
    module_names = set()
    
    if args.modules:
        module_names.update(mod.strip() for mod in args.modules.split(','))
    
    if args.modules_file:
        module_names.update(extract_module_names_from_file(args.modules_file))
    
    if args.scan_all:
        # Add common legacy patterns
        legacy_patterns = [
            'engine.py', 'engine', 'legacy', 'old_', 'deprecated',
            'unused', 'backup', 'temp', 'test_old'
        ]
        module_names.update(legacy_patterns)
    
    if not module_names:
        print("❌ No modules specified. Use --modules, --modules-file, or --scan-all")
        return 1
    
    print(f"🔍 Searching for references to {len(module_names)} modules:")
    for module in sorted(module_names):
        print(f"   - {module}")
    
    # Create output directory
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Run the scan
    finder = ConfigReferenceFinder(args.project_root)
    report = finder.generate_report(module_names)
    finder.export_report(report, str(output_path))
    
    # Print summary
    print(f"\n📊 Reference Scan Summary:")
    print(f"   Scanned files: {report['summary']['scanned_files']}")
    print(f"   Files with references: {report['summary']['files_with_references']}")
    print(f"   Modules with references: {report['summary']['modules_with_references']}")
    
    # Show critical references
    critical_found = False
    for module_name, data in report['module_references'].items():
        if data['critical_references']:
            if not critical_found:
                print(f"\n⚠️  Critical references found:")
                critical_found = True
            print(f"   {module_name}: {len(data['critical_references'])} critical references")
            for ref in data['critical_references'][:3]:  # Show first 3
                print(f"      - {ref['file']}:{ref['line']} ({ref['type']})")
    
    if not critical_found:
        print("\n✅ No critical references found in CI/packaging files")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())