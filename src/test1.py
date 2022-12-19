import whisper

model = whisper.load_model("base")
model = whisper.load_model("../models/base.en.pt")
path =  "../input/2022_11_13_Carbon_14_Tears.mp3"

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(path)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
