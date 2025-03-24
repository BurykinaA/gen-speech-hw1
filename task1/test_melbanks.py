import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from melbanks import LogMelFilterBanks

def load_test_audio(target_sr=16000):
    duration = 2
    sr = target_sr
    t = torch.linspace(0, duration, int(sr * duration))
    
    signal = 0.5 * torch.sin(2 * np.pi * 440 * t)
    signal += 0.3 * torch.sin(2 * np.pi * 880 * t)
    signal += 0.2 * torch.sin(2 * np.pi * 1320 * t)
    
    signal += 0.05 * torch.randn_like(signal)
    signal = signal / torch.max(torch.abs(signal))
    signal = signal.unsqueeze(0)
    
    print(f"Generated synthetic audio - shape: {signal.shape}, Sample rate: {target_sr}Hz")
    return signal, target_sr

def test_exact_match():
    print("\n=== Testing exact match as per README requirements ===")
    signal, sr = load_test_audio()
    
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        hop_length=160,
        n_mels=80
    )(signal)
    
    logmelbanks = LogMelFilterBanks(
        samplerate=sr,
        hop_length=160,
        n_mels=80
    )(signal)
    
    shape_match = torch.log(melspec + 1e-6).shape == logmelbanks.shape
    print(f"Shape match: {shape_match}")
    print(f"  - torchaudio shape: {torch.log(melspec + 1e-6).shape}")
    print(f"  - custom shape: {logmelbanks.shape}")
    
    value_match = torch.allclose(torch.log(melspec + 1e-6), logmelbanks, rtol=1e-4, atol=1e-4)
    if not value_match:
        max_diff = torch.max(torch.abs(torch.log(melspec + 1e-6) - logmelbanks))
        print(f"Value match: False (max difference: {max_diff:.6f})")
    else:
        print("Value match: True")
    
    return shape_match and value_match

def test_with_explicit_parameters():
    print("\n=== Testing with explicitly matched parameters ===")
    signal, sr = load_test_audio()
    
    n_fft = 400
    hop_length = 160
    n_mels = 80
    power = 2.0
    norm = 'slaney'
    mel_scale = 'htk'
    
    torchaudio_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        norm=norm,
        mel_scale=mel_scale,
        center=True,
        pad_mode='reflect',
        f_min=0.0,
        f_max=sr/2
    )
    
    custom_melbanks = LogMelFilterBanks(
        n_fft=n_fft,
        samplerate=sr,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        norm_mel=norm,
        mel_scale=mel_scale,
        center=True,
        pad_mode='reflect',
        f_min_hz=0.0,
        f_max_hz=sr/2
    )
    
    with torch.no_grad():
        torchaudio_output = torchaudio_melspec(signal)
        torchaudio_log_output = torch.log(torchaudio_output + 1e-6)
        custom_output = custom_melbanks(signal)
    
    shape_match = torchaudio_log_output.shape == custom_output.shape
    print(f"Shape match: {shape_match}")
    print(f"  - torchaudio shape: {torchaudio_log_output.shape}")
    print(f"  - custom shape: {custom_output.shape}")
    
    value_match = torch.allclose(torchaudio_log_output, custom_output, rtol=1e-4, atol=1e-4)
    if not value_match:
        max_diff = torch.max(torch.abs(torchaudio_log_output - custom_output))
        print(f"Value match: False (max difference: {max_diff:.6f})")
    else:
        print("Value match: True")
    
    return shape_match, value_match, torchaudio_log_output, custom_output

def plot_comparison(torchaudio_output, custom_output, filename="mel_comparison.png"):
    print(f"\nCreating visual comparison plot: {filename}")
    
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(custom_output.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Custom LogMelFilterBanks Implementation')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    
    plt.subplot(1, 2, 2)
    plt.imshow(torchaudio_output.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Torchaudio MelSpectrogram Implementation (log)')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Comparison plot saved as '{filename}'")
    plt.close()

if __name__ == "__main__":
    print("Testing LogMelFilterBanks implementation...")
    
    readme_match = test_exact_match()
    shape_match, value_match, torchaudio_output, custom_output = test_with_explicit_parameters()
    plot_comparison(torchaudio_output, custom_output)
    
    if readme_match:
        print("\nSUCCESS")
    else:
        print("\nFAILURE")
