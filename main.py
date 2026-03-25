import wfdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'

# Step 0: Load and visualize ECG signal from MIT-BIH record 100 using WFDB

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import freqz, tf2zpk, group_delay
import pandas as pd

# step0
# Load a segment from record 100 (first 10 seconds, 3600 samples at 360 Hz)
record = wfdb.rdrecord('100', sampto=3600, pn_dir='mitdb')
ecg_signal = record.p_signal[:, 0]  # Use channel 0
fs = record.fs  # Sampling frequency

# Plot original ECG signal
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(ecg_signal)) / fs, ecg_signal, label="Original ECG")
plt.title("Step 0: Raw ECG Signal (First 10 Seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()


# step1
# Define bandpass filter function (Butterworth)
def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# Apply bandpass filter to ECG signal
lowcut = 5  # Hz
highcut = 15  # Hz
filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

# Plot the bandpass filtered ECG
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(filtered_ecg)) / fs, filtered_ecg, label="Filtered ECG (5–15 Hz)", color='orange')
plt.title("Step 1: Bandpass Filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------magnitude and Phase response-----------------
# Frequency response (magnitude and phase) for bandpass filter
b, a = butter(2, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
w, h = freqz(b, a, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(abs(h)))
plt.title("Bandpass Filter: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Bandpass Filter: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# -----------------------Group Delay-------------------------
# Group Delay
w_gd, gd = group_delay((b, a))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Bandpass Filter: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# ---------------------Pole-Zero Plot-------------------------
# Pole-Zero Plot for BandPass filter
z, p, k = tf2zpk(b, a)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), label='Zeros', marker='o')
plt.scatter(np.real(p), np.imag(p), label='Poles', marker='x')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Bandpass Filter: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()


# step2
# Apply derivative filter
# Pan-Tompkins derivative: y(nT) = (1/8T)[-x(n-2T) - 2x(n-1T) + 2x(n+1T) + x(n+2T)]

def derivative_filter(signal, fs):
    derivative = np.zeros_like(signal)
    for i in range(2, len(signal) - 2):
        derivative[i] = (1 / (8 * (1 / fs))) * (
                -signal[i - 2] - 2 * signal[i - 1] + 2 * signal[i + 1] + signal[i + 2]
        )
    return derivative


# Apply derivative filter
deriv_ecg = derivative_filter(filtered_ecg, fs)

# Plot the derivative output
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(deriv_ecg)) / fs, deriv_ecg, label="Derivative Output", color='green')
plt.title("Step 2: Derivative Filter Output")
plt.xlabel("Time (s)")
plt.ylabel("Slope (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# -----------magnitude and Phase responses-----------------

# Derivative filter coefficients (FIR, T = 1/fs)
T = 1 / fs
b_deriv = (1 / (8 * T)) * np.array([-1, -2, 0, 2, 1])
a_deriv = np.array([1])

# Frequency response
w, h = freqz(b_deriv, a_deriv, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(h)))
plt.title("Derivative Filter: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Derivative Filter: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# ----------------Group Delay-----------------------
# Group Delay
w_gd, gd = group_delay((b_deriv, a_deriv))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Derivative Filter: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# --------------------Pole-Zero Plot----------------------
# Pole-Zero Plot
z, p, k = tf2zpk(b_deriv, a_deriv)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Derivative Filter: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

# step3
# Apply squaring function
# Squaring is not a linear filter — it doesn't have a frequency response, pole-zero plot, or group delay in the classical sense.
squared_ecg = deriv_ecg ** 2

# Plot the squared signal
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(squared_ecg)) / fs, squared_ecg, label="Squared Signal", color='purple')
plt.title("Step 3: Squared Signal Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# step4
# Apply moving window integration
def moving_window_integration(signal, window_size):
    integrated = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return integrated


# 150 ms window ≈ 54 samples at 360 Hz
window_size = int(0.150 * fs)
integrated_ecg = moving_window_integration(squared_ecg, window_size)

# Plot the result
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(integrated_ecg)) / fs, integrated_ecg, label="Integrated Signal", color='brown')
plt.title("Step 4: Moving Window Integration Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------magnitude and Phase response----------------------

# Coefficients of moving average filter
b_mwi = np.ones(window_size) / window_size
a_mwi = np.array([1])  # FIR filter has a = 1

# Frequency and phase response
w, h = freqz(b_mwi, a_mwi, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(h)))
plt.title("Moving Window Integrator: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Moving Window Integrator: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# ---------------------Group Delay---------------------
# Group Delay: Output magnitude and Output Phase
w_gd, gd = group_delay((b_mwi, a_mwi))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Moving Window Integrator: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------Pole-Zero Plot-------------------
# Pole-Zero Plot
z, p, k = tf2zpk(b_mwi, a_mwi)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Moving Window Integrator: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()


# Step 5: Fixed Threshold QRS Detection
def fixed_threshold_detection(integrated_ecg, fs):
    threshold = 0.6 * np.max(integrated_ecg)
    qrs_locs = []
    refractory_period = int(0.2 * fs)
    last_peak = -refractory_period
    for i in range(1, len(integrated_ecg) - 1):
        if (integrated_ecg[i] > threshold and
                integrated_ecg[i] > integrated_ecg[i - 1] and
                integrated_ecg[i] > integrated_ecg[i + 1] and
                (i - last_peak) > refractory_period):
            qrs_locs.append(i)
            last_peak = i
    return qrs_locs, threshold


# Plot Step 5 - Fixed Threshold Output
qrs_locations, fixed_threshold = fixed_threshold_detection(integrated_ecg, fs)

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(integrated_ecg)) / fs, integrated_ecg, label="Integrated Signal", color='brown')
plt.axhline(y=fixed_threshold, color='red', linestyle='--', label=f"Fixed Threshold ({fixed_threshold:.3f})")
plt.plot(np.array(qrs_locations) / fs, integrated_ecg[qrs_locations], 'ro', markersize=8, label="Detected QRS")
plt.title("Step 5: Fixed Threshold QRS Detection Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(f"Fixed Threshold QRS Detection:")
print(f"Threshold value: {fixed_threshold:.6f}")
print(f"Number of QRS detected: {len(qrs_locations)}")
print(f"QRS locations (seconds): {np.round(np.array(qrs_locations) / fs, 2)}")


# Step 6: LMS Adaptive Threshold - CORRECTED VERSION
class LMSAdaptiveThreshold:
    def __init__(self, learning_rate=0.001, filter_length=20):  # FIXED: __init__ instead of _init_
        self.mu = learning_rate
        self.M = filter_length
        self.weights = np.zeros(filter_length)

    def adapt_threshold(self, integrated_signal, known_qrs_locations=None):
        N = len(integrated_signal)
        adaptive_threshold = np.zeros(N)

        for n in range(self.M, N):
            # Get input vector (past M samples)
            x = integrated_signal[n - self.M:n]

            # Calculate predicted output
            y_pred = np.dot(self.weights, x)
            adaptive_threshold[n] = y_pred

            # If we have known QRS locations, use them for training
            if known_qrs_locations is not None:
                desired = self.create_desired_response(n, known_qrs_locations, integrated_signal)
                error = desired - y_pred
                self.weights += 2 * self.mu * error * x
            else:
                # Use a simple adaptive strategy without ground truth
                # Adapt based on local signal characteristics
                local_mean = np.mean(integrated_signal[max(0, n - 50):n + 1])
                desired = local_mean * 0.5  # Simple heuristic
                error = desired - y_pred
                self.weights += 2 * self.mu * error * x

        return adaptive_threshold

    def create_desired_response(self, n, qrs_locations, signal):
        window = 20
        for qrs_loc in qrs_locations:
            if abs(n - qrs_loc) <= window:
                return signal[n] * 0.8  # Higher threshold near QRS
        return signal[n] * 0.2  # Lower threshold elsewhere


def lms_detect_qrs(integrated_norm, thresholds_lms, fs):
    qrs_locs = []
    refractory_period = int(0.2 * fs)
    last_qrs = -refractory_period

    for i in range(1, len(integrated_norm) - 1):
        if (integrated_norm[i] > thresholds_lms[i] and
                integrated_norm[i] > integrated_norm[i - 1] and
                integrated_norm[i] > integrated_norm[i + 1] and
                (i - last_qrs) > refractory_period):
            qrs_locs.append(i)
            last_qrs = i
    return qrs_locs


# Normalize the integrated signal for LMS processing
integrated_norm = (integrated_ecg - np.mean(integrated_ecg)) / np.std(integrated_ecg)

# Initialize and run LMS adaptive threshold
lms = LMSAdaptiveThreshold(learning_rate=0.001, filter_length=20)

# For initial training, we can use the fixed threshold detections as a rough guide
# This is a practical approach when ground truth isn't immediately available
adaptive_threshold = lms.adapt_threshold(integrated_norm, known_qrs_locations=qrs_locations)

# Detect QRS using LMS threshold
qrs_locations_lms = lms_detect_qrs(integrated_norm, adaptive_threshold, fs)

# Plot individual LMS Threshold Output
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(integrated_norm)) / fs, integrated_norm, label='Integrated ECG (normalized)', color='brown')
plt.plot(np.arange(len(adaptive_threshold)) / fs, adaptive_threshold, label='LMS Adaptive Threshold', linestyle='--',
         color='blue', linewidth=2)
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, integrated_norm[qrs_locations_lms], 'ro', markersize=8,
             label='LMS Detected QRS')
plt.title("Step 6: LMS Adaptive Threshold QRS Detection Output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nLMS Adaptive Threshold QRS Detection:")
print(f"Number of QRS detected: {len(qrs_locations_lms)}")
if len(qrs_locations_lms) > 0:
    print(f"QRS locations (seconds): {np.round(np.array(qrs_locations_lms) / fs, 2)}")
else:
    print("No QRS complexes detected with LMS threshold")

# Normalize the fixed threshold to match the normalized integrated signal scale
fixed_threshold_normalized = (fixed_threshold - np.mean(integrated_ecg)) / np.std(integrated_ecg)

# Plotting Comparison
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(integrated_norm)) / fs, integrated_norm, label='Integrated ECG (normalized)', color='brown')
plt.plot(np.arange(len(adaptive_threshold)) / fs, adaptive_threshold, label='LMS Threshold', linestyle='--',
         color='blue')
plt.axhline(y=fixed_threshold_normalized, color='red', linestyle='--', label='Fixed Threshold (normalized)')
plt.plot(np.array(qrs_locations) / fs, integrated_norm[qrs_locations], 'go', markersize=6, label='Fixed Threshold QRS')
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, integrated_norm[qrs_locations_lms], 'ro', markersize=6, label='LMS QRS')
plt.title("LMS Adaptive Threshold vs Fixed Threshold QRS Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 7: Performance Evaluation

# Step 7.1: Load annotations for record 100
try:
    annotation = wfdb.rdann('100', 'atr', sampto=3600, pn_dir='mitdb')
    true_qrs_locations = annotation.sample
    true_qrs_locations = np.array(true_qrs_locations)
    print(f"\nLoaded {len(true_qrs_locations)} true QRS annotations")
    print(f"True QRS locations (seconds): {np.round(true_qrs_locations / fs, 2)}")
except Exception as e:
    print(f"Error loading annotations: {e}")
    print("Creating synthetic true QRS locations for demonstration...")
    # Create some example true locations based on typical heart rate
    true_qrs_locations = np.arange(360, 3600, 360)  # Every second approximately


# Define function to compare detected vs true peaks
def evaluate_performance(detected_locs, true_locs, fs, tolerance_ms=100):
    tolerance = int((tolerance_ms / 1000) * fs)
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    matched = set()

    # Count True Positives and False Positives
    for d in detected_locs:
        # Find closest true peak
        diffs = np.abs(true_locs - d)
        if len(diffs) == 0:
            continue
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance and min_idx not in matched:
            TP += 1
            matched.add(min_idx)
        else:
            FP += 1

    # Count False Negatives
    FN = len(true_locs) - len(matched)

    # Calculate metrics
    Se = TP / (TP + FN) if TP + FN > 0 else 0  # Sensitivity (Recall)
    PPV = TP / (TP + FP) if TP + FP > 0 else 0  # Positive Predictive Value (Precision)
    F1 = 2 * PPV * Se / (PPV + Se) if PPV + Se > 0 else 0  # F1 Score
    DER = (FP + FN) / len(true_locs) if len(true_locs) > 0 else 0  # Detection Error Rate

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Se": Se,
        "PPV": PPV,
        "F1": F1,
        "DER": DER
    }


# Evaluate fixed threshold
performance_fixed = evaluate_performance(qrs_locations, true_qrs_locations, fs)

# Evaluate LMS threshold
performance_lms = evaluate_performance(qrs_locations_lms, true_qrs_locations, fs)

# Display results
performance_df = pd.DataFrame([performance_fixed, performance_lms],
                              index=["Fixed Threshold", "LMS Adaptive Threshold"])
print("\n=== Step 7.1: Performance Evaluation ===")
print(performance_df.round(3))

# Additional visualization of detection results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(ecg_signal)) / fs, ecg_signal, label='Original ECG', alpha=0.7)
plt.plot(true_qrs_locations / fs, ecg_signal[true_qrs_locations], 'k^', markersize=8, label='True QRS')
plt.plot(np.array(qrs_locations) / fs, ecg_signal[qrs_locations], 'go', markersize=6, label='Fixed Threshold Detected')
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, ecg_signal[qrs_locations_lms], 'ro', markersize=6, label='LMS Detected')
plt.title("QRS Detection Results on Original ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#######################################################################

# Step 0: Load and visualize ECG signal from MIT-BIH record 100 using WFDB

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import freqz, tf2zpk, group_delay
import pandas as pd

# step0
# Load a segment from record 108 (first 10 seconds, 3600 samples at 360 Hz)
record = wfdb.rdrecord('108', sampto=3600, pn_dir='mitdb')
ecg_signal = record.p_signal[:, 0]  # Use channel 0
fs = record.fs  # Sampling frequency

print(f"Record 108 Information:")
print(f"Sampling frequency: {fs} Hz")
print(f"Signal length: {len(ecg_signal)} samples")
print(f"Duration: {len(ecg_signal) / fs:.1f} seconds")
print(f"Signal channels: {record.sig_name}")

# Plot original ECG signal
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(ecg_signal)) / fs, ecg_signal, label="Original ECG - Record 108")
plt.title("Step 0: Raw ECG Signal from MIT-BIH Record 108 (First 10 Seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.tight_layout()
plt.legend()
plt.show()


# step1
# Define bandpass filter function (Butterworth)
def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# Apply bandpass filter to ECG signal
lowcut = 5  # Hz
highcut = 15  # Hz
filtered_ecg = bandpass_filter(ecg_signal, lowcut, highcut, fs)

# Plot the bandpass filtered ECG
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(filtered_ecg)) / fs, filtered_ecg, label="Filtered ECG (5–15 Hz) - Record 108", color='orange')
plt.title("Step 1: Bandpass Filtered ECG - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------magnitude and Phase response-----------------
# Frequency response (magnitude and phase) for bandpass filter
b, a = butter(2, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
w, h = freqz(b, a, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(abs(h)))
plt.title("Bandpass Filter: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Bandpass Filter: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# -----------------------Group Delay-------------------------
# Group Delay
w_gd, gd = group_delay((b, a))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Bandpass Filter: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# ---------------------Pole-Zero Plot-------------------------
# Pole-Zero Plot for BandPass filter
z, p, k = tf2zpk(b, a)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), label='Zeros', marker='o')
plt.scatter(np.real(p), np.imag(p), label='Poles', marker='x')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Bandpass Filter: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()


# step2
# Apply derivative filter
# Pan-Tompkins derivative: y(nT) = (1/8T)[-x(n-2T) - 2x(n-1T) + 2x(n+1T) + x(n+2T)]

def derivative_filter(signal, fs):
    derivative = np.zeros_like(signal)
    for i in range(2, len(signal) - 2):
        derivative[i] = (1 / (8 * (1 / fs))) * (
                -signal[i - 2] - 2 * signal[i - 1] + 2 * signal[i + 1] + signal[i + 2]
        )
    return derivative


# Apply derivative filter
deriv_ecg = derivative_filter(filtered_ecg, fs)

# Plot the derivative output
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(deriv_ecg)) / fs, deriv_ecg, label="Derivative Output - Record 108", color='green')
plt.title("Step 2: Derivative Filter Output - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Slope (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# -----------magnitude and Phase responses-----------------

# Derivative filter coefficients (FIR, T = 1/fs)
T = 1 / fs
b_deriv = (1 / (8 * T)) * np.array([-1, -2, 0, 2, 1])
a_deriv = np.array([1])

# Frequency response
w, h = freqz(b_deriv, a_deriv, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(h)))
plt.title("Derivative Filter: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Derivative Filter: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# ----------------Group Delay-----------------------
# Group Delay
w_gd, gd = group_delay((b_deriv, a_deriv))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Derivative Filter: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# --------------------Pole-Zero Plot----------------------
# Pole-Zero Plot
z, p, k = tf2zpk(b_deriv, a_deriv)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Derivative Filter: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()

# step3
# Apply squaring function
# Squaring is not a linear filter — it doesn't have a frequency response, pole-zero plot, or group delay in the classical sense.
squared_ecg = deriv_ecg ** 2

# Plot the squared signal
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(squared_ecg)) / fs, squared_ecg, label="Squared Signal - Record 108", color='purple')
plt.title("Step 3: Squared Signal Output - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# step4
# Apply moving window integration
def moving_window_integration(signal, window_size):
    integrated = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return integrated


# 150 ms window ≈ 54 samples at 360 Hz
window_size = int(0.150 * fs)
integrated_ecg = moving_window_integration(squared_ecg, window_size)

# Plot the result
plt.figure(figsize=(12, 3))
plt.plot(np.arange(len(integrated_ecg)) / fs, integrated_ecg, label="Integrated Signal - Record 108", color='brown')
plt.title("Step 4: Moving Window Integration Output - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------magnitude and Phase response----------------------

# Coefficients of moving average filter
b_mwi = np.ones(window_size) / window_size
a_mwi = np.array([1])  # FIR filter has a = 1

# Frequency and phase response
w, h = freqz(b_mwi, a_mwi, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(12, 6))

# Magnitude Response
plt.subplot(3, 1, 1)
plt.plot(frequencies, 20 * np.log10(np.abs(h)))
plt.title("Moving Window Integrator: Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()

# Phase Response
plt.subplot(3, 1, 2)
plt.plot(frequencies, np.unwrap(np.angle(h)))
plt.title("Moving Window Integrator: Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()

# ---------------------Group Delay---------------------
# Group Delay: Output magnitude and Output Phase
w_gd, gd = group_delay((b_mwi, a_mwi))
plt.subplot(3, 1, 3)
plt.plot(w_gd * fs / (2 * np.pi), gd)
plt.title("Moving Window Integrator: Group Delay")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Samples")
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------Pole-Zero Plot-------------------
# Pole-Zero Plot
z, p, k = tf2zpk(b_mwi, a_mwi)
plt.figure(figsize=(5, 5))
plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')
unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
plt.gca().add_artist(unit_circle)
plt.title("Moving Window Integrator: Pole-Zero Plot")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.grid()
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()


# Step 5: Fixed Threshold QRS Detection
def fixed_threshold_detection(integrated_ecg, fs):
    threshold = 0.6 * np.max(integrated_ecg)
    qrs_locs = []
    refractory_period = int(0.2 * fs)
    last_peak = -refractory_period
    for i in range(1, len(integrated_ecg) - 1):
        if (integrated_ecg[i] > threshold and
                integrated_ecg[i] > integrated_ecg[i - 1] and
                integrated_ecg[i] > integrated_ecg[i + 1] and
                (i - last_peak) > refractory_period):
            qrs_locs.append(i)
            last_peak = i
    return qrs_locs, threshold


# Plot Step 5 - Fixed Threshold Output
qrs_locations, fixed_threshold = fixed_threshold_detection(integrated_ecg, fs)

plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(integrated_ecg)) / fs, integrated_ecg, label="Integrated Signal", color='brown')
plt.axhline(y=fixed_threshold, color='red', linestyle='--', label=f"Fixed Threshold ({fixed_threshold:.3f})")
plt.plot(np.array(qrs_locations) / fs, integrated_ecg[qrs_locations], 'ro', markersize=8, label="Detected QRS")
plt.title("Step 5: Fixed Threshold QRS Detection - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(f"Fixed Threshold QRS Detection - Record 108:")
print(f"Threshold value: {fixed_threshold:.6f}")
print(f"Number of QRS detected: {len(qrs_locations)}")
print(f"QRS locations (seconds): {np.round(np.array(qrs_locations) / fs, 2)}")


# Step 6: LMS Adaptive Threshold - CORRECTED VERSION
class LMSAdaptiveThreshold:
    def __init__(self, learning_rate=0.001, filter_length=20):  # FIXED: __init__ instead of _init_
        self.mu = learning_rate
        self.M = filter_length
        self.weights = np.zeros(filter_length)

    def adapt_threshold(self, integrated_signal, known_qrs_locations=None):
        N = len(integrated_signal)
        adaptive_threshold = np.zeros(N)

        for n in range(self.M, N):
            # Get input vector (past M samples)
            x = integrated_signal[n - self.M:n]

            # Calculate predicted output
            y_pred = np.dot(self.weights, x)
            adaptive_threshold[n] = y_pred

            # If we have known QRS locations, use them for training
            if known_qrs_locations is not None:
                desired = self.create_desired_response(n, known_qrs_locations, integrated_signal)
                error = desired - y_pred
                self.weights += 2 * self.mu * error * x
            else:
                # Use a simple adaptive strategy without ground truth
                # Adapt based on local signal characteristics
                local_mean = np.mean(integrated_signal[max(0, n - 50):n + 1])
                desired = local_mean * 0.5  # Simple heuristic
                error = desired - y_pred
                self.weights += 2 * self.mu * error * x

        return adaptive_threshold

    def create_desired_response(self, n, qrs_locations, signal):
        window = 20
        for qrs_loc in qrs_locations:
            if abs(n - qrs_loc) <= window:
                return signal[n] * 0.8  # Higher threshold near QRS
        return signal[n] * 0.2  # Lower threshold elsewhere


def lms_detect_qrs(integrated_norm, thresholds_lms, fs):
    qrs_locs = []
    refractory_period = int(0.2 * fs)
    last_qrs = -refractory_period

    for i in range(1, len(integrated_norm) - 1):
        if (integrated_norm[i] > thresholds_lms[i] and
                integrated_norm[i] > integrated_norm[i - 1] and
                integrated_norm[i] > integrated_norm[i + 1] and
                (i - last_qrs) > refractory_period):
            qrs_locs.append(i)
            last_qrs = i
    return qrs_locs


# Normalize the integrated signal for LMS processing
integrated_norm = (integrated_ecg - np.mean(integrated_ecg)) / np.std(integrated_ecg)

# Initialize and run LMS adaptive threshold
lms = LMSAdaptiveThreshold(learning_rate=0.001, filter_length=20)

# For initial training, we can use the fixed threshold detections as a rough guide
# This is a practical approach when ground truth isn't immediately available
adaptive_threshold = lms.adapt_threshold(integrated_norm, known_qrs_locations=qrs_locations)

# Detect QRS using LMS threshold
qrs_locations_lms = lms_detect_qrs(integrated_norm, adaptive_threshold, fs)

# Plot individual LMS Threshold Output
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(integrated_norm)) / fs, integrated_norm, label='Integrated ECG (normalized)', color='brown')
plt.plot(np.arange(len(adaptive_threshold)) / fs, adaptive_threshold, label='LMS Adaptive Threshold', linestyle='--',
         color='blue', linewidth=2)
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, integrated_norm[qrs_locations_lms], 'ro', markersize=8,
             label='LMS Detected QRS')
plt.title("Step 6: LMS Adaptive Threshold QRS Detection - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nLMS Adaptive Threshold QRS Detection - Record 108:")
print(f"Number of QRS detected: {len(qrs_locations_lms)}")
if len(qrs_locations_lms) > 0:
    print(f"QRS locations (seconds): {np.round(np.array(qrs_locations_lms) / fs, 2)}")
else:
    print("No QRS complexes detected with LMS threshold")

# Normalize the fixed threshold to match the normalized integrated signal scale
fixed_threshold_normalized = (fixed_threshold - np.mean(integrated_ecg)) / np.std(integrated_ecg)

# Plotting Comparison
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(integrated_norm)) / fs, integrated_norm, label='Integrated ECG (normalized)', color='brown')
plt.plot(np.arange(len(adaptive_threshold)) / fs, adaptive_threshold, label='LMS Threshold', linestyle='--',
         color='blue')
plt.axhline(y=fixed_threshold_normalized, color='red', linestyle='--', label='Fixed Threshold (normalized)')
plt.plot(np.array(qrs_locations) / fs, integrated_norm[qrs_locations], 'go', markersize=6, label='Fixed Threshold QRS')
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, integrated_norm[qrs_locations_lms], 'ro', markersize=6, label='LMS QRS')
plt.title("LMS Adaptive vs Fixed Threshold QRS Detection - Record 108")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Step 7: Performance Evaluation

# Step 7.1: Load annotations for record 108
try:
    annotation = wfdb.rdann('108', 'atr', sampto=3600, pn_dir='mitdb')
    true_qrs_locations = annotation.sample
    true_qrs_locations = np.array(true_qrs_locations)
    print(f"\nLoaded {len(true_qrs_locations)} true QRS annotations for Record 108")
    print(f"True QRS locations (seconds): {np.round(true_qrs_locations / fs, 2)}")

    # Display annotation types for Record 108
    print(f"Annotation symbols: {annotation.symbol}")
    print(f"Annotation types: {set(annotation.symbol)}")

except Exception as e:
    print(f"Error loading annotations for Record 108: {e}")
    print("Creating synthetic true QRS locations for demonstration...")
    # Record 108 typically has different rhythm patterns, so adjust accordingly
    # Create approximate QRS locations based on visual inspection
    true_qrs_locations = np.array([180, 540, 900, 1260, 1620, 1980, 2340, 2700, 3060, 3420])  # Example locations


# Define function to compare detected vs true peaks
def evaluate_performance(detected_locs, true_locs, fs, tolerance_ms=100):
    tolerance = int((tolerance_ms / 1000) * fs)
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    matched = set()

    # Count True Positives and False Positives
    for d in detected_locs:
        # Find closest true peak
        diffs = np.abs(true_locs - d)
        if len(diffs) == 0:
            continue
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= tolerance and min_idx not in matched:
            TP += 1
            matched.add(min_idx)
        else:
            FP += 1

    # Count False Negatives
    FN = len(true_locs) - len(matched)

    # Calculate metrics
    Se = TP / (TP + FN) if TP + FN > 0 else 0  # Sensitivity (Recall)
    PPV = TP / (TP + FP) if TP + FP > 0 else 0  # Positive Predictive Value (Precision)
    F1 = 2 * PPV * Se / (PPV + Se) if PPV + Se > 0 else 0  # F1 Score
    DER = (FP + FN) / len(true_locs) if len(true_locs) > 0 else 0  # Detection Error Rate

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Se": Se,
        "PPV": PPV,
        "F1": F1,
        "DER": DER
    }


# Evaluate fixed threshold
performance_fixed = evaluate_performance(qrs_locations, true_qrs_locations, fs)

# Evaluate LMS threshold
performance_lms = evaluate_performance(qrs_locations_lms, true_qrs_locations, fs)

# Display results
performance_df = pd.DataFrame([performance_fixed, performance_lms],
                              index=["Fixed Threshold", "LMS Adaptive Threshold"])
print("\n=== Step 7.1: Performance Evaluation - Record 108 ===")
print(performance_df.round(3))

# Display detailed analysis for Record 108
print(f"\n=== Detailed Analysis for MIT-BIH Record 108 ===")
print(f"Record 108 characteristics:")
print(f"- Contains various arrhythmias and abnormal beats")
print(f"- More challenging for QRS detection algorithms")
print(f"- Duration analyzed: 10 seconds")
print(f"- Sampling rate: {fs} Hz")

# Calculate heart rate estimates
if len(qrs_locations) > 1:
    rr_intervals_fixed = np.diff(np.array(qrs_locations)) / fs
    hr_fixed = 60 / np.mean(rr_intervals_fixed)
    print(f"\nHeart Rate Estimates:")
    print(f"Fixed Threshold: {hr_fixed:.1f} BPM")

if len(qrs_locations_lms) > 1:
    rr_intervals_lms = np.diff(np.array(qrs_locations_lms)) / fs
    hr_lms = 60 / np.mean(rr_intervals_lms)
    print(f"LMS Adaptive: {hr_lms:.1f} BPM")

if len(true_qrs_locations) > 1:
    rr_intervals_true = np.diff(np.array(true_qrs_locations)) / fs
    hr_true = 60 / np.mean(rr_intervals_true)
    print(f"True (Ground Truth): {hr_true:.1f} BPM")

# Additional visualization of detection results
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(ecg_signal)) / fs, ecg_signal, label='Original ECG', alpha=0.7)
plt.plot(true_qrs_locations / fs, ecg_signal[true_qrs_locations], 'k^', markersize=8, label='True QRS')
plt.plot(np.array(qrs_locations) / fs, ecg_signal[qrs_locations], 'go', markersize=6, label='Fixed Threshold Detected')
if len(qrs_locations_lms) > 0:
    plt.plot(np.array(qrs_locations_lms) / fs, ecg_signal[qrs_locations_lms], 'ro', markersize=6, label='LMS Detected')
plt.title("QRS Detection Results on Original ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()