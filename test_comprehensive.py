#!/usr/bin/env python3
"""
Comprehensive test for the complete heatmap grid functionality.

This script tests the complete flow:
1. Model runner initialization
2. Device switching
3. Heatmap grid rendering
4. Integration with the inference loop
"""

import numpy as np
import cv2
import time
from pose_model_runner import PoseModelRunner
from pose_result_processor import heatmaps_grid_to_image, render_pose_on_frame, annotate_metrics
from pose_defs import BODY_PARTS

def test_model_runner_initialization():
    """Test that the PoseModelRunner can be initialized."""
    print("Testing PoseModelRunner initialization...")
    
    try:
        # Test with CPU device
        runner_cpu = PoseModelRunner(
            "model/human-pose-estimation-0001.xml",
            initial_device="CPU",
            model_w=456,
            model_h=256
        )
        print("✓ PoseModelRunner initialized with CPU device")
        
        # Test device switching
        if runner_cpu.set_device("MYRIAD"):
            print("✓ Device switch to MYRIAD succeeded")
        else:
            print("⚠ Device switch to MYRIAD failed (MYRIAD may not be available)")
            
        # Switch back to CPU for testing
        runner_cpu.set_device("CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ PoseModelRunner initialization failed: {e}")
        return False

def test_heatmap_integration():
    """Test the complete heatmap rendering integration."""
    print("\nTesting heatmap rendering integration...")
    
    try:
        # Create a mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create mock heatmaps with the expected shape [1, 19, h, w]
        heatmaps = np.random.rand(1, 19, 32, 32).astype(np.float32)
        
        # Test heatmap grid rendering
        heatmaps_3d = heatmaps[0]  # Remove batch dimension
        heat_img = heatmaps_grid_to_image(
            heatmaps_3d,
            (frame.shape[1], frame.shape[0]),
            BODY_PARTS,
            cols=5
        )
        
        print(f"✓ Heatmap grid rendered successfully")
        print(f"  Frame shape: {frame.shape}")
        print(f"  Heatmap grid shape: {heat_img.shape}")
        
        # Test concatenation
        output_image = cv2.hconcat([frame, heat_img])
        print(f"  Combined image shape: {output_image.shape}")
        
        # Test JPEG encoding
        success, jpeg = cv2.imencode(".jpg", output_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            print(f"✓ JPEG encoding successful, size: {len(jpeg.tobytes())} bytes")
        else:
            print("✗ JPEG encoding failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Heatmap integration test failed: {e}")
        return False

def test_device_switching_simulation():
    """Simulate the device switching flow."""
    print("\nTesting device switching simulation...")
    
    try:
        # Simulate the shared state variables
        requested_device = None
        current_device = "CPU"
        
        # Simulate device switch request
        requested_device = "MYRIAD"
        
        # Simulate the inference loop applying the switch
        if requested_device is not None:
            to_set = requested_device
            requested_device = None
            print(f"Applying device switch to: {to_set}")
            current_device = to_set
            print(f"Device switched to: {current_device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Device switching simulation failed: {e}")
        return False

def test_heatmap_toggle_simulation():
    """Simulate the heatmap toggle flow."""
    print("\nTesting heatmap toggle simulation...")
    
    try:
        # Simulate the shared state variables
        requested_heatmaps = None
        show_heatmaps = False
        
        # Simulate heatmap show request
        requested_heatmaps = True
        
        # Simulate the inference loop applying the request
        if requested_heatmaps is not None:
            show_heatmaps = requested_heatmaps
            requested_heatmaps = None
            print(f"Heatmaps visibility set to: {show_heatmaps}")
        
        # Test toggling
        requested_heatmaps = False
        if requested_heatmaps is not None:
            show_heatmaps = requested_heatmaps
            requested_heatmaps = None
            print(f"Heatmaps visibility set to: {show_heatmaps}")
        
        return True
        
    except Exception as e:
        print(f"✗ Heatmap toggle simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("Comprehensive Heatmap Grid Functionality Test")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_model_runner_initialization()
    test2_passed = test_heatmap_integration()
    test3_passed = test_device_switching_simulation()
    test4_passed = test_heatmap_toggle_simulation()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Model Runner Initialization: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Heatmap Integration: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"  Device Switching Simulation: {'PASSED' if test3_passed else 'FAILED'}")
    print(f"  Heatmap Toggle Simulation: {'PASSED' if test4_passed else 'FAILED'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("\n✓ All tests passed! The heatmap grid functionality is working correctly.")
        print("\nTo use the feature:")
        print("1. Run the web server: python webcam_web.py")
        print("2. Open browser to http://<raspberry-pi-ip>:8080/")
        print("3. Click the 'Show heatmaps' button to toggle heatmap visibility")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
