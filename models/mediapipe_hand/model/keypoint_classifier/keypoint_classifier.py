#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KeyPoint Classifier for hand gesture recognition.
Adapted from kinivi/hand-gesture-recognition-mediapipe for use with Qualcomm AI Hub MediaPipe.
"""
import numpy as np
import os

# Try to import tensorflow, fall back to tflite_runtime if available
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        # Create a mock tf.lite namespace for compatibility
        class LiteNamespace:
            Interpreter = Interpreter
        class TFNamespace:
            lite = LiteNamespace()
        tf = TFNamespace()
    except ImportError:
        raise ImportError(
            "TensorFlow or tflite-runtime is required. Install with:\n"
            "  pip install tensorflow\n"
            "  or\n"
            "  pip install tflite-runtime"
        )


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        """
        Initialize the KeyPoint Classifier.
        
        Args:
            model_path: Path to the TFLite model file. If None, uses default path.
            num_threads: Number of threads for TFLite interpreter.
        """
        if model_path is None:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'keypoint_classifier.tflite')
        
        print(f"    Creating TFLite Interpreter for: {model_path}")
        self.interpreter = Interpreter(model_path=model_path,
                                       num_threads=num_threads)
        print("    Interpreter created, allocating tensors...")

        self.interpreter.allocate_tensors()
        print("    Tensors allocated, getting input/output details...")
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("    Input/output details retrieved")

    def __call__(
        self,
        landmark_list,
    ):
        """
        Classify gesture from preprocessed landmark list.
        
        Args:
            landmark_list: Preprocessed landmark list (1D array of 42 values: 21 landmarks * 2 coords)
            
        Returns:
            result_index: Index of the predicted gesture class
        """
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

