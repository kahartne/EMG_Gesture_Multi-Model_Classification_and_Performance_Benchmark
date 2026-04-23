# EMG Gesture Classification & Performance Benchmark

This project implements an end-to-end machine learning and systems benchmarking pipeline
for EMG-based gesture classification.
It evaluates inference performance across different hardware and precision configurations
using both a Logistic Regression model and a Multi-Layer Perceptron (MLP).

---

## Overview

Raw electromyography (EMG) signals are processed into fixed-size windows and converted
into features:

- Mean Absolute Value (MAV)
- Root Mean Square (RMS)
- Zero Crossings (ZC)

These features are used to train classification models, which are then deployed in
multiple execution environments for performance analysis.

---

## Models
- Logistic Regression (baseline, low workload)
- Multi-Layer Perceptron (MLP) (2-layer neural network, higher workload)

---

## Experiments

This pipeline evaluates:
- CPU vs GPU execution
- FP32 vs INT8 precision
- SIMD vs scalar execution
- Model complexity (Logistic vs MLP)
- Workload scaling (small vs large batch sizes)

---

## Installation

### CPU Environment (default)
    '''bash
    pip install -r requirements.txt

### GPU Environment (optional)
    1. Install CUDA Toolkit (recommended: CUDA 11.8 or compatible)
    2. Install cuDNN 8.x and copy files into CUDA directory:
        - bin/ → contains cudnn64_8.dll
        - include/
        - lib/
    3. Install GPU dependencies:
        '''bash
        pip install -r requirements-gpu.txt
    *Note: GPU execution requires a properly configured CUDA + cuDNN environment

### Quick Start

Make sure you have data	est.csv
- To make data folder: mkdir data
- Then put test.csv in data folder (this is what the trained model runs on for benchmarking)
- Make sure test.csv only has columns: mav, rms, zc

Run the full pipeline:
    python emg_gesture_ml_pipeline_run_all.py

To run prediction generator alone:
python emg_gesture_prediction_generator.py ^
  --model models/[model name here].onnx ^
  --gesture data/[feature input file here].csv ^
  --out [prediction file here].csv ^
  --device [cpu|gpu]
  --scale [scale_factor (default=1)]

---

## Files (After Complete Run)
├── data/
│   └── test.csv                      # Inference dataset (features only)
├── models/                           # Generated artifacts
│   ├── logistic_regression.onnx      # FP32 logistic model
│   ├── logistic_regression_int8.onnx # INT8 logistic model
│   ├── mlp.onnx                      # FP32 MLP model
│   ├── mlp_int8.onnx                 # INT8 MLP model
│   ├── logistic_weights.txt          # Logistic weights (for C integration)
│   ├── logistic_bias.txt             # Logistic bias (for C integration)
│   ├── mlp_W0.txt, mlp_b0.txt, ...   # MLP layer weights (for C integration)
│   ├── features.txt                  # Feature order
│   ├── README.md                     # Project documentation
│   └── gesture_ml_models_onnx.zip    # ZIP of all trained models
├── results/                          # Performance results
│   ├── benchmark_results.csv
│   ├── classification_report_comparison.png
│   ├── logistic_confusion_matrix.png
│   ├── mlp_confusion_matrix.png
│   ├── time_comparison.png
│   ├── speedup.png
│   ├── throughput.png
│   ├── transfer_time.png
│   ├── load_vs_inference.png
│   ├── logistic_time.png
│   ├── predictions.png
│   ├── mlp_time.png
│   ├── fp32_model_comparison.png
│   └── int8_model_comparison.png
│
├── emg_gesture_ml_predictor.py                 # Train logistic regression + export weights
├── emg_gesture_build_onnx_from_weights.py      # Build ONNX model manually (MatMul + Add + ArgMax)
├── emg_gesture_build_onnx_mlp.py               # Build MLP ONNX model
├── emg_gesture_prediction_generator.py         # Run inference + output predictions
├── emg_gesture_prediction_quantization.py      # Apply INT8 quantization
├── emg_gesture_prediction_benchmark.py         # Benchmark CPU/GPU + FP32/INT8
├── emg_gesture_prediction_benchmark_results.py # Generate plots from results
├── emg_gesture_prediction_nvprof_results.py    # GPU kernel profiling analysis
├── emg_gesture_prediction_simd.py              # SIMD vs scalar experiment
├── emg_gesture_ml_pipeline_run_all.py          # Run full pipeline (one command)
│
├── features.csv                    # Training dataset (features + label)
└──
---

## Metrics Collected
- Total Time (ms)
- Load Time (ms)
- Inference Time (ms)
- Transfer Overhead (ms)
- Throughput (samples/sec)

---

## Inference Formula
Logistic Regression:
    z = W·x + b
    prediction = argmax(z)

---

MLP Inference:
x → Dense → ReLU → Dense → ReLU → Dense → argmax
Layer 1:
    z0 = W0·x + b0
    h0 = ReLU(z0)

Layer 2:
    z1 = W1·h0 + b1
    h1 = ReLU(zz)

Layer 3:
    z2 = W2·h1 + b2

    prediction = argmax(z2)

---

## Outputs
- benchmark_results.csv
- predictions.csv
- performance plots (figures)

---

## Notes
- Feature order MUST be
    mav
    rms
    zc
- ONNX models are manually constructed to ensure compatibility with INT8 quantization.
- INT8 quantization is applied using ONNX Runtime.
- Predictions are verified to remain within valid class ranges (0-12)
- Performance differences depend on workload size:
    - Logistic Regression shows minimal INT8 benefit
    - MLP shows measurable improvements due to higher compute intensity
    - GPU performance heavily dependent on workload size

---

## Key Observations
- CPU performs well for small workloads due to low overhead
- GPU acceleration becomes beneficial as workload size increases
- Logistic Regression shows minimal benefit from GPU acceleration
- MLP benefits more from parallel execution
- INT8 quantization improves CPU performance more than GPU performance

---

## Summary
This project demonstrates key computer architecture concepts:

- Parallelism and throughput
- SIMD vectorization
- CPU vs GPU tradeoffs
- Precision vs performance (FP32 vs INT8)
- Compute-bound vs overhead-bound workloads
