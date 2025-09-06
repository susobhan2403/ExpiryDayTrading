
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
