#!/usr/bin/env python
"""
Diagnostic script to test TensorFlow/TFLite loading step by step.
This will help identify where TensorFlow is hanging.
"""
import sys
import os
import time

print("=" * 60)
print("TensorFlow/TFLite Diagnostic Test")
print("=" * 60)

# Step 1: Check Python version
print("\n[1] Python Version:")
print(f"    Python {sys.version}")
print(f"    Executable: {sys.executable}")

# Step 2: Check if TensorFlow or tflite-runtime is installed
print("\n[2] Checking TensorFlow/tflite-runtime installation...")
tf = None
using_tflite_runtime = False

# Try tflite-runtime first (lighter, better for edge devices)
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    print(f"    ✓ tflite-runtime found (recommended for edge devices)")
    using_tflite_runtime = True
    # Create a mock tf namespace for compatibility
    class LiteNamespace:
        Interpreter = TFLiteInterpreter
    class TFNamespace:
        lite = LiteNamespace()
    tf = TFNamespace()
    Interpreter = TFLiteInterpreter
except ImportError:
    # Fall back to full TensorFlow
    try:
        import tensorflow as tf
        print(f"    ✓ TensorFlow imported successfully")
        print(f"    Version: {tf.__version__}")
        print(f"    Location: {tf.__file__}")
        Interpreter = tf.lite.Interpreter
    except ImportError as e:
        print(f"    ✗ Neither TensorFlow nor tflite-runtime found")
        print("    Install with: pip install tensorflow")
        print("    Or (recommended): pip install tflite-runtime")
        sys.exit(1)
    except Exception as e:
        print(f"    ✗ Error importing TensorFlow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Step 3: Check for tensorflow-metal
print("\n[3] Checking for tensorflow-metal (can cause conflicts)...")
try:
    import tensorflow_metal as tfm
    print(f"    ⚠️  tensorflow-metal is installed: {tfm.__file__}")
    print("    This can cause hanging issues on macOS!")
    print("    Try: pip uninstall tensorflow-metal")
except ImportError:
    print("    ✓ tensorflow-metal not installed (good)")

# Step 4: Test basic TensorFlow operations (skip if using tflite-runtime)
if not using_tflite_runtime:
    print("\n[4] Testing basic TensorFlow operations...")
    try:
        print("    Creating a simple tensor...")
        a = tf.constant([1.0, 2.0, 3.0])
        print(f"    ✓ Tensor created: {a}")
        
        print("    Performing simple operation...")
        b = a * 2
        result = b.numpy()
        print(f"    ✓ Operation successful: {result}")
    except Exception as e:
        print(f"    ✗ Error in TensorFlow operations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\n[4] Skipping TensorFlow operations (using tflite-runtime)")

# Step 5: Test TFLite Interpreter creation
print("\n[5] Testing TFLite Interpreter creation...")
try:
    print("    Interpreter class accessible")
    print(f"    ✓ Using: {Interpreter}")
except Exception as e:
    print(f"    ✗ Error accessing TFLite Interpreter: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test loading the actual model file
print("\n[6] Testing model file loading...")
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'keypoint_classifier', 'keypoint_classifier.tflite')

if not os.path.exists(model_path):
    print(f"    ✗ Model file not found at: {model_path}")
    sys.exit(1)

print(f"    Model file found: {model_path}")
print(f"    File size: {os.path.getsize(model_path)} bytes")

try:
    print("    Creating Interpreter with model file...")
    print("    (This is where it might hang...)")
    start_time = time.time()
    
    interpreter = Interpreter(model_path=model_path, num_threads=1)
    
    elapsed = time.time() - start_time
    print(f"    ✓ Interpreter created in {elapsed:.2f} seconds")
    
    print("    Allocating tensors...")
    start_time = time.time()
    interpreter.allocate_tensors()
    elapsed = time.time() - start_time
    print(f"    ✓ Tensors allocated in {elapsed:.2f} seconds")
    
    print("    Getting input/output details...")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"    ✓ Input shape: {input_details[0]['shape']}")
    print(f"    ✓ Output shape: {output_details[0]['shape']}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED! TensorFlow/TFLite is working correctly.")
    print("=" * 60)
    
except Exception as e:
    elapsed = time.time() - start_time if 'start_time' in locals() else 0
    print(f"    ✗ Error after {elapsed:.2f} seconds: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING:")
    print("=" * 60)
    print("1. If it hung at 'Creating Interpreter', try:")
    print("   - pip uninstall tensorflow-metal")
    print("   - pip uninstall tensorflow")
    print("   - pip install tensorflow")
    print("\n2. If it's a C++ runtime issue:")
    print("   - Try: pip install --upgrade tensorflow")
    print("   - Or use: pip install tflite-runtime (lighter alternative)")
    print("\n3. If the model file is corrupted:")
    print("   - Re-download from: https://github.com/kinivi/hand-gesture-recognition-mediapipe")
    sys.exit(1)

