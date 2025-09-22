import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import types

INPUT_AUDIO_DIR = Path("src/assets/audio")
OUTPUT_IMAGE_DIR = Path("public/assets/spectrograms")

spec_config = types.SimpleNamespace(
    sampling_rate=44100,
    n_fft=2048,
    hop_size=512,
    n_mels=128,
    win_size=2048,
)

plot_config = types.SimpleNamespace(
    width_per_second=1.3,
    min_width=5.0,
    height=4.0,
)

def save_spectrogram_image(spec, output_path, config):
    plt.rcParams.update({'font.size': 16})
    
    duration_s = spec.shape[1] * config.hop_size / config.sampling_rate
    
    fig_width = max(plot_config.min_width, duration_s * plot_config.width_per_second)
    
    fig, ax = plt.subplots(figsize=(fig_width, plot_config.height))
    
    im = ax.imshow(spec,
                   aspect='auto', 
                   origin='lower', 
                   interpolation='none', 
                   cmap="viridis",
                   extent=[0, duration_s, 0, spec.shape[0]])
    
    fig.colorbar(im, ax=ax, format='%+2.0f dB', pad=0.02)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Bins")
    
    yticks = np.linspace(0, spec.shape[0], num=5, dtype=int)
    ax.set_yticks(yticks)
    
    ax.tick_params(axis='x', labelbottom=True)
    ax.tick_params(axis='y', labelleft=True)
    ax.grid(False)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path)
    plt.close(fig)

def main():
    print(f"Searching for audio files in: {INPUT_AUDIO_DIR}")
    
    audio_files = list(INPUT_AUDIO_DIR.rglob("*.wav")) + list(INPUT_AUDIO_DIR.rglob("*.mp3"))
    
    if not audio_files:
        print("No audio files found. Please check the INPUT_AUDIO_DIR path.")
        return

    print(f"{len(audio_files)} audio files found. Starting processing...")

    for audio_path in audio_files:
        try:
            y, sr = librosa.load(audio_path, sr=spec_config.sampling_rate)

            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=spec_config.n_fft,
                hop_length=spec_config.hop_size,
                n_mels=spec_config.n_mels,
                win_length=spec_config.win_size
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            relative_path = audio_path.relative_to(INPUT_AUDIO_DIR)
            output_path = OUTPUT_IMAGE_DIR / relative_path.with_suffix('.png')

            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_spectrogram_image(mel_spec_db, output_path, spec_config)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"✗ Error processing {audio_path}: {e}")

    print("\nProcessing complete!")


if __name__ == "__main__":
    OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    main()