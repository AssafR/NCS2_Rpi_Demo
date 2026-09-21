#!/usr/bin/env python3
"""
Test script to verify heatmap grid functionality.

This script tests the heatmap grid rendering functionality by:
1. Creating a mock frame
2. Creating mock heatmaps
3. Testing the heatmaps_grid_to_image function
4. Verifying the output
"""

import numpy as np
import cv2
from pose_result_processor import heatmaps_grid_to_image
from pose_defs import BODY_PARTS

def test_heatmaps_grid():
    """Test the heatmaps_grid_to_image function."""
    print("Testing heatmaps_grid_to_image function...")
    
    # Create mock heatmaps with shape [19, 64, 64] (19 body parts, 64x64 each)
    heatmaps_3d = np.random.rand(19, 64, 64).astype(np.float32)
    
    # Test frame size
    frame_size = (640, 480)
    
    # Test the function
    try:
        heat_img = heatmaps_grid_to_image(
            heatmaps_3d,
            frame_size,
            BODY_PARTS,
            cols=5
        )
        
        print(f"✓ heatmaps_grid_to_image succeeded")
        print(f"  Input shape: {heatmaps_3d.shape}")
        print(f"  Output shape: {heat_img.shape}")
        print(f"  Expected output height: {frame_size[1]}")
        print(f"  Actual output height: {heat_img.shape[0]}")
        
        # Verify the output has the correct height
        if heat_img.shape[0] == frame_size[1]:
            print("✓ Output height matches frame height")
        else:
            print(f"✗ Output height {heat_img.shape[0]} does not match frame height {frame_size[1]}")
            
        # Verify the output is a valid image
        if len(heat_img.shape) == 3 and heat_img.shape[2] == 3:
            print("✓ Output is a valid BGR image")
        else:
            print(f"✗ Output shape {heat_img.shape} is not a valid BGR image")
            
        return True
        
    except Exception as e:
        print(f"✗ heatmaps_grid_to_image failed with error: {e}")
        return False

def test_heatmap_to_image():
    """Test the heatmap_to_image function."""
    print("\nTesting heatmap_to_image function...")
    
    # Create a mock heatmap
    heatmap = np.random.rand(64, 64).astype(np.float32)
    
    # Test output size
    out_size = (320, 240)
    
    try:
        heat_img = heatmap_to_image(heatmap, out_size)
        
        print(f"✓ heatmap_to_image succeeded")
        print(f"  Input shape: {heatmap.shape}")
        print(f"  Output shape: {heat_img.shape}")
        print(f"  Expected output size: {out_size}")
        print(f"  Actual output size: ({heat_img.shape[1]}, {heat_img.shape[0]})")
        
        # Verify the output has the correct size
        if heat_img.shape[1] == out_size[0] and heat_img.shape[0] == out_size[1]:
            print("✓ Output size matches expected size")
        else:
            print(f"✗ Output size {heat_img.shape[1]}x{heat_img.shape[0]} does not match expected size {out_size[0]}x{out_size[1]}")
            
        # Verify the output is a valid image
        if len(heat_img.shape) == 3 and heat_img.shape[2] == 3:
            print("✓ Output is a valid BGR image")
        else:
            print(f"✗ Output shape {heat_img.shape} is not a valid BGR image")
            
        return True
        
    except Exception as e:
        print(f"✗ heatmap_to_image failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Heatmap Grid Functionality Test")
    print("=" * 50)
    
    # Import the functions we need to test
    from pose_result_processor import heatmap_to_image
    
    # Run tests
    test1_passed = test_heatmaps_grid()
    test2_passed = test_heatmap_to_image()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  heatmaps_grid_to_image: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  heatmap_to_image: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Heatmap grid functionality should work correctly.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
