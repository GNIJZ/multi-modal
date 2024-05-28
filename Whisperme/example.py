from faster_whisper import WhisperModel
import os

current_path = os.getcwd()
print(current_path)
path = "largev3/"

# Run on GPU with FP16
model = WhisperModel(model_size_or_path=path, device="cuda", compute_type="int8",local_files_only=True)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe("111.mp4", beam_size=5,
                                  language="zh", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
