
import sys
import subprocess
sys.path.insert(0, ".")

from tools.runtime_import_logger import RuntimeImportLogger

# Set up logger
logger = RuntimeImportLogger(".")
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
            print(f"ℹ️  Could not instantiate engine (expected): {e}")
        
    except Exception as e:
        print(f"❌ Error importing enhanced engine: {e}")
        
finally:
    # Export results
    logger.remove_hook()
    logger.export_to_json("tools/runtime_imports.json")
    
    # Print summary
    report = logger.generate_report()
    print(f"\n📊 Import Summary:")
    print(f"   Loaded modules: {report['summary']['unique_modules']}")
    print(f"   Total events: {report['summary']['total_events']}")
    
    if report['loaded_modules']:
        print(f"\n📋 Loaded internal modules:")
        for module in sorted(report['loaded_modules']):
            print(f"   - {module}")
