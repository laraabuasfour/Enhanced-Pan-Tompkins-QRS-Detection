# Enhanced-Pan-Tompkins-QRS-Detection

This project was developed as part of a DSP course project.

The project focuses on the **reproduction and enhancement of the Pan-Tompkins QRS detection algorithm** using real ECG data from the **MIT-BIH Arrhythmia Database**.

## Project Objectives

- Reproduce the classic Pan-Tompkins QRS detection algorithm
- Analyze each DSP stage of the algorithm
- Plot frequency response, phase response, pole-zero plots, and group delay
- Implement an LMS-based adaptive thresholding method
- Compare the original fixed-threshold method with the LMS-enhanced method
- Evaluate performance using real ECG records

## Implemented Processing Stages

The project includes the following stages:

1. Raw ECG loading and visualization
2. Bandpass filtering
3. Derivative filtering
4. Squaring
5. Moving window integration
6. Fixed threshold QRS detection
7. LMS adaptive threshold QRS detection
8. Performance evaluation

## Dataset

The ECG records are taken from the **MIT-BIH Arrhythmia Database** using the `wfdb` library.

Examples used in this project:
- Record 100
- Record 108

## Evaluation Metrics

The project evaluates QRS detection performance using:

- Sensitivity (Se)
- Positive Predictive Value (PPV)
- F1 Score
- Detection Error Rate (DER)
