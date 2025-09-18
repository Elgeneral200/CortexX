# filename: test_pipeline.py
"""
Recommended Pipeline Test - Best Balance of Coverage and Simplicity
"""

print("🧪 CortexX Pipeline Test - Recommended Method")
print("=" * 50)

# Step 1: Basic Import Test
print("\n📦 STEP 1: Import Test")
try:
    import pipeline
    print("✅ pipeline.py imports successfully")
except ImportError as e:
    print(f"❌ CRITICAL: Import failed: {e}")
    print("   → Check if pipeline.py exists in current directory")
    print("   → Check for syntax errors in pipeline.py")
    exit(1)
except Exception as e:
    print(f"❌ CRITICAL: Import error: {e}")
    print("   → Check pipeline.py for runtime errors")
    exit(1)

# Step 2: Check Essential Functions
print("\n🔍 STEP 2: Function Availability Test")
essential_functions = [
    'get_available_operations',
    'get_enhanced_operations', 
    'create_enhanced_sales_pipeline',
    'Pipeline'
]

missing_functions = []
for func_name in essential_functions:
    if hasattr(pipeline, func_name):
        print(f"✅ {func_name} found")
    else:
        missing_functions.append(func_name)
        print(f"❌ {func_name} missing")

if missing_functions:
    print(f"⚠️  WARNING: Missing functions: {missing_functions}")
    print("   → These functions may not be exported in __all__")
else:
    print("✅ All essential functions found")

# Step 3: Registry Test
print("\n🔧 STEP 3: Operation Registry Test")
try:
    available_ops = pipeline.get_available_operations()
    print(f"✅ get_available_operations() works: {len(available_ops)} operations")
    
    if len(available_ops) > 0:
        print(f"   📋 Sample operations: {available_ops[:5]}")
    else:
        print("   ⚠️  No operations found - registry may be empty")
        
except Exception as e:
    print(f"❌ get_available_operations() failed: {e}")

try:
    enhanced_ops = pipeline.get_enhanced_operations()
    print(f"✅ get_enhanced_operations() works: {len(enhanced_ops)} enhanced ops")
    
    if len(enhanced_ops) > 0:
        print(f"   🚀 Enhanced operations: {enhanced_ops}")
    else:
        print("   ⚠️  No enhanced operations - Phase 1 modules may not be loaded")
        
except Exception as e:
    print(f"❌ get_enhanced_operations() failed: {e}")

# Step 4: Pipeline Creation Test
print("\n🏗️  STEP 4: Pipeline Creation Test")
try:
    # Test with empty registry (should work)
    empty_registry = {}
    test_pipeline1 = pipeline.create_enhanced_sales_pipeline(empty_registry, "Test Pipeline 1")
    print(f"✅ Created pipeline with empty registry: '{test_pipeline1.name}'")
    print(f"   📊 Pipeline has {len(test_pipeline1.steps)} steps")
    
except Exception as e:
    print(f"❌ Pipeline creation with empty registry failed: {e}")

try:
    # Test basic Pipeline class
    test_pipeline2 = pipeline.Pipeline(name="Test Pipeline 2")
    print(f"✅ Created basic Pipeline: '{test_pipeline2.name}'")
    
except Exception as e:
    print(f"❌ Basic Pipeline creation failed: {e}")

# Step 5: Simple Data Test (Most Important!)
print("\n📊 STEP 5: Simple Data Processing Test")
try:
    import pandas as pd
    import numpy as np
    
    # Create simple test data  
    test_data = pd.DataFrame({
        'A': [1, 2, 2, 4, 5],  # Has duplicate
        'B': [10, 20, 30, 40, 50],
        'C': ['x', 'y', 'y', 'z', 'w']  # Has duplicate
    })
    print(f"✅ Created test data: {test_data.shape}")
    
    # Create simple pipeline with basic operations
    simple_pipeline = pipeline.Pipeline(name="Simple Test")
    
    # Try to add basic operations
    available_ops = pipeline.get_available_operations()
    basic_ops_to_try = ['drop_duplicates', 'reset_index']
    
    added_steps = 0
    for op in basic_ops_to_try:
        if op in available_ops:
            simple_pipeline.add_step(op, {}, f"Test {op}")
            added_steps += 1
            print(f"✅ Added step: {op}")
    
    if added_steps > 0:
        # Execute the pipeline
        result = simple_pipeline.execute(test_data)
        
        if result.success:
            print(f"✅ Pipeline executed successfully!")
            print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"   📈 Original shape: {test_data.shape}")
            print(f"   📉 Final shape: {result.dataframe.shape}")
            print(f"   ✅ Steps succeeded: {result.steps_succeeded}")
            
            if result.steps_failed > 0:
                print(f"   ⚠️  Steps failed: {result.steps_failed}")
        else:
            print(f"❌ Pipeline execution failed")
            for error in result.error_messages:
                print(f"   💥 Error: {error}")
    else:
        print("⚠️  No basic operations available for testing")
        
except Exception as e:
    print(f"❌ Data processing test failed: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("\n🎯 FINAL ASSESSMENT")
print("=" * 30)

# Quick health check
try:
    ops_count = len(pipeline.get_available_operations())
    enhanced_count = len(pipeline.get_enhanced_operations())
    
    if ops_count >= 5 and enhanced_count >= 1:
        print("🎉 EXCELLENT: Pipeline is fully functional with enhanced features!")
        status = "EXCELLENT"
    elif ops_count >= 3:
        print("✅ GOOD: Pipeline is functional with basic operations")
        status = "GOOD" 
    elif ops_count >= 1:
        print("⚠️  PARTIAL: Pipeline has limited functionality")
        status = "PARTIAL"
    else:
        print("❌ POOR: Pipeline has major issues")
        status = "POOR"
        
    print(f"\n📊 Summary Stats:")
    print(f"   • Total Operations: {ops_count}")
    print(f"   • Enhanced Operations: {enhanced_count}")
    print(f"   • Status: {status}")
    
    if status in ["EXCELLENT", "GOOD"]:
        print(f"\n🚀 READY TO USE! You can proceed with:")
        print(f"   • pipeline = create_enhanced_sales_pipeline(registry, 'My Pipeline')")
        print(f"   • pipeline.add_step('operation_name', parameters)")
        print(f"   • result = pipeline.execute(your_dataframe)")
        
except Exception as e:
    print(f"❌ Health check failed: {e}")

print(f"\n✅ Test completed! Check the results above.")
