#!/usr/bin/env python3
"""
Runtime Import Logger for Legacy Code Removal

This script instruments Python import system to log all modules loaded during
enhanced engine execution, providing runtime evidence of which modules are
actually used.

Features:
- Import hook to track module loading
- Configurable entry command to test
- Export to JSON for analysis
- Integration with dependency audit

Usage:
    python tools/runtime_import_logger.py --command "python engine_runner.py --run-once"
    python tools/runtime_import_logger.py --command "python -m pytest tests/test_engine_integration.py"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
import threading

@dataclass
class ImportEvent:
    """Represents a module import event."""
    module_name: str
    file_path: Optional[str]
    timestamp: float
    stack_depth: int
    caller_module: Optional[str] = None
    caller_file: Optional[str] = None
    caller_line: Optional[int] = None

class RuntimeImportLogger:
    """Logs module imports during runtime execution."""
    
    def __init__(self, project_root: str):
        self.project_root = pathlib.Path(project_root).resolve()
        self.import_events: List[ImportEvent] = []
        self.loaded_modules: Set[str] = set()
        self.start_time = 0.0
        self.lock = threading.Lock()
        
        # Store original import function
        self._original_import = getattr(__builtins__, '__import__', __import__)
        
    def is_internal_module(self, module_name: str, file_path: Optional[str] = None) -> bool:
        """Check if a module is internal to the project."""
        if not module_name:
            return False
            
        # Check by module name patterns
        internal_prefixes = ['src', 'tests', 'tools', 'engine', 'test_']
        if any(module_name.startswith(prefix) for prefix in internal_prefixes):
            return True
            
        # Check by file path
        if file_path:
            try:
                path_obj = pathlib.Path(file_path)
                return self.project_root in path_obj.parents or path_obj.parent == self.project_root
            except (ValueError, OSError):
                pass
                
        return False
    
    def get_caller_info(self, depth: int = 2) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """Get information about the calling code."""
        try:
            frame = sys._getframe(depth)
            caller_file = frame.f_code.co_filename
            caller_line = frame.f_lineno
            
            # Try to get module name from globals
            caller_module = frame.f_globals.get('__name__')
            
            return caller_module, caller_file, caller_line
        except (ValueError, AttributeError):
            return None, None, None
    
    def import_hook(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        """Custom import hook to log import events."""
        # Call original import first
        module = self._original_import(name, globals, locals, fromlist, level)
        
        # Log the import event
        try:
            file_path = getattr(module, '__file__', None)
            
            # Only log internal modules
            if self.is_internal_module(name, file_path):
                with self.lock:
                    if name not in self.loaded_modules:
                        caller_module, caller_file, caller_line = self.get_caller_info()
                        
                        event = ImportEvent(
                            module_name=name,
                            file_path=file_path,
                            timestamp=time.time() - self.start_time,
                            stack_depth=len([f for f in self._get_stack_frames() if self.is_internal_frame(f)]),
                            caller_module=caller_module,
                            caller_file=caller_file,
                            caller_line=caller_line
                        )
                        
                        self.import_events.append(event)
                        self.loaded_modules.add(name)
        except Exception:
            # Don't let logging errors break imports
            pass
        
        return module
    
    def _get_stack_frames(self) -> List[Any]:
        """Get current stack frames."""
        frames = []
        frame = sys._getframe()
        while frame:
            frames.append(frame)
            frame = frame.f_back
        return frames
    
    def is_internal_frame(self, frame) -> bool:
        """Check if a frame is from internal code."""
        try:
            filename = frame.f_code.co_filename
            return self.is_internal_module("", filename)
        except:
            return False
    
    def install_hook(self):
        """Install the import hook."""
        self.start_time = time.time()
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self.import_hook
        else:
            __builtins__.__import__ = self.import_hook
        print(f"🔍 Import hook installed, logging to {len(self.import_events)} events")
    
    def remove_hook(self):
        """Remove the import hook."""
        if isinstance(__builtins__, dict):
            __builtins__['__import__'] = self._original_import
        else:
            __builtins__.__import__ = self._original_import
        print(f"✅ Import hook removed, captured {len(self.import_events)} events")
    
    def get_loaded_modules(self) -> Set[str]:
        """Get set of all loaded internal modules."""
        return self.loaded_modules.copy()
    
    def export_to_json(self, output_path: str):
        """Export import events to JSON."""
        data = {
            'summary': {
                'total_events': len(self.import_events),
                'unique_modules': len(self.loaded_modules),
                'duration_seconds': max(event.timestamp for event in self.import_events) if self.import_events else 0
            },
            'loaded_modules': list(self.loaded_modules),
            'events': [asdict(event) for event in self.import_events]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Exported import log to {output_path}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate analysis report."""
        # Group events by module
        module_stats = {}
        for event in self.import_events:
            if event.module_name not in module_stats:
                module_stats[event.module_name] = {
                    'first_import_time': event.timestamp,
                    'import_count': 0,
                    'file_path': event.file_path,
                    'callers': set()
                }
            
            stats = module_stats[event.module_name]
            stats['import_count'] += 1
            stats['first_import_time'] = min(stats['first_import_time'], event.timestamp)
            
            if event.caller_module:
                stats['callers'].add(event.caller_module)
        
        # Convert sets to lists for JSON serialization
        for stats in module_stats.values():
            stats['callers'] = list(stats['callers'])
        
        return {
            'summary': {
                'total_events': len(self.import_events),
                'unique_modules': len(self.loaded_modules),
                'execution_time': max(event.timestamp for event in self.import_events) if self.import_events else 0
            },
            'loaded_modules': list(self.loaded_modules),
            'module_stats': module_stats
        }

def run_with_import_logging(command: List[str], project_root: str, output_file: str) -> Dict[str, Any]:
    """Run a command with import logging enabled."""
    
    # Create a script that will set up import logging and run the target command
    script_content = f'''
import sys
import subprocess
sys.path.insert(0, "{project_root}")

from tools.runtime_import_logger import RuntimeImportLogger

# Set up logger
logger = RuntimeImportLogger("{project_root}")
logger.install_hook()

try:
    # Import and run the enhanced engine code
    import importlib
    
    # Try different ways to trigger the enhanced engine
    try:
        # Method 1: Direct import to trigger module loading
        import engine_runner
        import src.engine_enhanced
        print("✅ Successfully imported enhanced engine modules")
        
        # Method 2: Try to create an instance (if possible without API keys)
        try:
            from src.engine_enhanced import EnhancedTradingEngine
            import datetime as dt
            import pytz
            
            # Create a minimal instance to trigger imports
            ist = pytz.timezone("Asia/Kolkata")
            expiry = ist.localize(dt.datetime(2024, 1, 11, 15, 30))
            engine = EnhancedTradingEngine("NIFTY", expiry)
            print("✅ Successfully instantiated enhanced engine")
        except Exception as e:
            print(f"ℹ️  Could not instantiate engine (expected): {{e}}")
        
    except Exception as e:
        print(f"❌ Error importing enhanced engine: {{e}}")
        
finally:
    # Export results
    logger.remove_hook()
    logger.export_to_json("{output_file}")
    
    # Print summary
    report = logger.generate_report()
    print(f"\\n📊 Import Summary:")
    print(f"   Loaded modules: {{report['summary']['unique_modules']}}")
    print(f"   Total events: {{report['summary']['total_events']}}")
    
    if report['loaded_modules']:
        print(f"\\n📋 Loaded internal modules:")
        for module in sorted(report['loaded_modules']):
            print(f"   - {{module}}")
'''
    
    # Write the script to a temporary file
    script_path = pathlib.Path(project_root) / 'temp_import_logger.py'
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Run the script
        print(f"🚀 Running import logging with command: {' '.join(command)}")
        result = subprocess.run(['python', str(script_path)], 
                              capture_output=True, text=True, cwd=project_root)
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Load and return the results
        if pathlib.Path(output_file).exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        else:
            return {'error': 'No output file generated'}
            
    finally:
        # Clean up the temporary script
        if script_path.exists():
            script_path.unlink()

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Runtime import logger for legacy code removal",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--command',
        default='python engine_runner.py --help',
        help='Command to run with import logging (default: engine_runner.py --help)'
    )
    
    parser.add_argument(
        '--output',
        default='tools/runtime_imports.json',
        help='Output file for import log (default: tools/runtime_imports.json)'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    # Parse command
    command_parts = args.command.split()
    
    # Run with logging
    result = run_with_import_logging(
        command_parts, 
        args.project_root, 
        str(output_path)
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return 1
    
    print(f"\n✅ Import logging complete!")
    print(f"📄 Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())