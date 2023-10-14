import librosa
import matplotlib.pyplot as plt
import numpy as np
audio_file = "D:\Downloads\Bill Withers - Ain't No Sunshine (Official Audio).mp3"

audio, sample_rate = librosa.load(audio_file)

# Compute the mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

# Convert to dB scale (optional but commonly done for visualization)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

import matplotlib.pyplot as plt

# Generate and display the spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
plt.title('Mel Spectrogram')
plt.colorbar(format='%+2.0f dB')

# Save the graph as a PNG file
plt.savefig('spectrogram.png', bbox_inches='tight', pad_inches=0)

# Show the plot (optional)
plt.show()

