import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Load ECG signal
record = wfdb.rdrecord('C:\\Users\\Acer\\Desktop\\Capstone\\Capstone\\data\\raw\\MIMIC-IV-ECG-1.0\\files\\p1000\\p10000032\\s40689238\\40689238')

# Access data
signal = record.p_signal  # Shape: (5000, 12) - 12 leads
sampling_rate = record.fs  # 500 Hz
lead_names = record.sig_name  # ['I', 'II', 'III', ...]

print(f"Signal shape: {signal.shape}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Lead names: {lead_names}")
print(f"Duration: {signal.shape[0] / sampling_rate} seconds")

# Create time axis
time = np.arange(signal.shape[0]) / sampling_rate

# Plot all 12 leads
fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
fig.suptitle('12-Lead ECG Visualization', fontsize=16, fontweight='bold')

for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
    ax.plot(time, signal[:, i], linewidth=0.5, color='black')
    ax.set_ylabel(f'{lead_name}\n(mV)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1])
    
    # Add grid similar to ECG paper
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.25, alpha=0.3)

axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('ecg_12_leads.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'ecg_12_leads.png'")

# Also create a focused view of the first 3 seconds with the standard limb leads
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig2.suptitle('Standard Limb Leads (I, II, III) - First 3 Seconds', fontsize=14, fontweight='bold')

time_limit = 3  # seconds
time_mask = time <= time_limit

standard_leads = ['I', 'II', 'III']
for i, (ax, lead_name) in enumerate(zip(axes2, standard_leads)):
    if lead_name in lead_names:
        lead_idx = lead_names.index(lead_name)
        ax.plot(time[time_mask], signal[time_mask, lead_idx], linewidth=1, color='red')
        ax.set_ylabel(f'{lead_name} (mV)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle=':', linewidth=0.25, alpha=0.3)

axes2[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
axes2[-1].set_xlim(0, time_limit)

plt.tight_layout()
plt.savefig('ecg_limb_leads_3sec.png', dpi=300, bbox_inches='tight')
print("Focused view saved as 'ecg_limb_leads_3sec.png'")

plt.show()
