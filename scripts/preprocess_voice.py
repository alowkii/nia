from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

encoder = VoiceEncoder()

# Load your reference audio
wav = preprocess_wav("voice_samples/my_voice_sample.wav")
reference_embedding = encoder.embed_utterance(wav)
np.save("voice_samples/my_voice_embedding.npy", reference_embedding)