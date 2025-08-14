#!/usr/bin/env python3
"""Test script for the IPC-based program runner."""

from ipc_runner import IPCProgramRunner
from PIL import Image
import numpy as np

def test_basic_execution():
    """Test basic code execution."""
    print("Testing basic execution...")
    
    runner = IPCProgramRunner()
    
    # Test simple calculation
    stdout, stderr, success = runner.execute_code("x = 5 + 3\nprint(f'x = {x}')")
    print(f"Success: {success}")
    print(f"Output: {stdout.strip()}")
    assert success, f"Execution failed: {stderr}"
    assert "x = 8" in stdout
    
    # Test state persistence
    stdout, stderr, success = runner.execute_code("y = x * 2\nprint(f'y = {y}')")
    assert success, f"State persistence failed: {stderr}"
    assert "y = 16" in stdout
    
    print("✓ Basic execution test passed")

def test_image_handling():
    """Test image loading and manipulation."""
    print("\nTesting image handling...")
    
    runner = IPCProgramRunner()
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='red')
    runner.load_image_clue(test_image, 0)
    
    # Test image access
    stdout, stderr, success = runner.execute_code("""
img = image_clue_0
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
""")
    
    assert success, f"Image access failed: {stderr}"
    assert "Image size: (100, 100)" in stdout
    assert "Image mode: RGB" in stdout
    
    # Test image retrieval
    retrieved_image = runner.get_result_image(0)
    assert retrieved_image is not None
    assert retrieved_image.size == (100, 100)
    
    print("✓ Image handling test passed")

def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")
    
    runner = IPCProgramRunner()
    
    # Test syntax error
    stdout, stderr, success = runner.execute_code("x = 5 +")
    assert not success
    assert "SyntaxError" in stderr
    
    # Test runtime error
    stdout, stderr, success = runner.execute_code("x = 1 / 0")
    assert not success
    assert "ZeroDivisionError" in stderr
    
    # Test that state is preserved after errors
    stdout, stderr, success = runner.execute_code("z = 42\nprint(f'z = {z}')")
    assert success
    assert "z = 42" in stdout
    
    print("✓ Error handling test passed")

def test_state_management():
    """Test state clearing and listing."""
    print("\nTesting state management...")
    
    runner = IPCProgramRunner()
    
    # Add some variables
    stdout, stderr, success = runner.execute_code("""
a = 10
b = "hello"
c = [1, 2, 3]
""")
    assert success
    
    # Test state listing
    state = runner.get_state()
    assert 'a' in state
    assert 'b' in state
    assert 'c' in state
    
    # Test state clearing
    runner.clear_state()
    state = runner.get_state()
    
    # Should only have base imports
    user_vars = [k for k in state.keys() if not k.startswith('_') and k not in ['Image', 'np', 'pd', 'wand', 'io']]
    assert len(user_vars) == 0
    
    print("✓ State management test passed")

if __name__ == "__main__":
    try:
        test_basic_execution()
        test_image_handling()
        test_error_handling()
        test_state_management()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()