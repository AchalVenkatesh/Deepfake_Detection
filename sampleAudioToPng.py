import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display

audio_directory = 'D:\AIML\Datasets\DeepFake audio\KAGGLE\AUDIO\FAKE'

image_directory = 'D:\AIML\deepfake_detection\images\FAKE'

#Getting the audio files from the audio directory
audio_files = (f for f in os.listdir(audio_directory) if f.endswith('.wav') or f.endswith('mp3'))

i = 1

for audio_file in audio_files:
    
    address = os.path.join(audio_directory,audio_file)
    audio, sample_rate = librosa.load(address)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

    # Convert to dB scale (optional but commonly done for visualization)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Generate and display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Save the graph as a PNG file
    plt.savefig(os.path.join(image_directory,'sample'+str(i)), bbox_inches='tight', pad_inches=0)
    i=i+1
    plt.close()

audio_directory = 'D:\AIML\Datasets\DeepFake audio\KAGGLE\AUDIO\REAL'

image_directory = 'D:\AIML\deepfake_detection\images\REAL'

#Getting the audio files from the audio directory
audio_files = (f for f in os.listdir(audio_directory) if f.endswith('.wav') or f.endswith('mp3'))

i = 1

for audio_file in audio_files:
    address = os.path.join(audio_directory,audio_file)
    audio, sample_rate = librosa.load(address)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)

    # Convert to dB scale (optional but commonly done for visualization)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Generate and display the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sample_rate, cmap='viridis')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Save the graph as a PNG file
    plt.savefig(os.path.join(image_directory,'sample'+str(i)), bbox_inches='tight', pad_inches=0)
    i=i+1
    plt.close()