# QFALT
This is an implementation of the paper "QFALT: Quantization and Fault Aware Loss for Training Enables Performance Recovery with Unreliable Weights" presented at IJCNN 2024

## Requirements
Pytorch, numpy, scipy

## Description

1. *CNN_training_quant_aware_extra_test_runs.py* - quantization-aware training for 9-layer ALL_CNN_C model (activations are not quantized)
2. *CNN_training_var_aware_extra_test_runs_type2.py* - quantization and stuck-at fault aware training for 9-layer ALL_CNN_C model (activations are not quantized)
3. *CNN_training_quant_aware_extra_test_runs_w_act_quant_clean_v2.py* - Updated and generalized quantization-aware training, activations are also quantized, generic model supported, ON/OFF resistance state values implemented for test
