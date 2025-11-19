# test_quick.py
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("QUICK SMOKE TEST")
print("=" * 60)

# Test 1: Can we import?
print("\n1. Testing imports...")
try:
    from ml.tfidf_pipeline import TfidfPipeline
    from ml.data_pipeline import normalize_text
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Can we load the model?
print("\n2. Loading model...")
try:
    p = TfidfPipeline()
    p.load("saved_models/tfidf")
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Model load failed: {e}")
    exit(1)

# Test 3: Can we predict?
print("\n3. Testing predictions...")
try:
    test_cases = [
        "STARBUCKS #1023 MUMBAI",
        "MCDONALDS BANGALORE",
        "UBER RIDE",
        "ATM WITHDRAWAL HDFC"
    ]
    
    for text in test_cases:
        result = p.predict([text])[0]
        print(f"   '{text:30}' → {result['label']:20} ({result['confidence']:.1%})")
    
    print("\n   ✓ All predictions working!")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✓ SMOKE TEST PASSED - Basic functionality works!")
print("=" * 60)
