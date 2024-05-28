import threading
from faster_whisper import WhisperModel
from io import BytesIO
import typing
import io
import collections
import wave
import os
import pyaudio
import webrtcvad
import keyboard
from transformers import pipeline

from Llm import chatspeak
from ThreadManager import ThreadManager
import logging

current_path = os.getcwd()
path = "largev3/"

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(name)s - %(levelname)s - %(message)s')


class Transcriber(object):
    def __init__(self,
                 model_size: str = path,
                 device: str = "cuda",
                 compute_type: str = "float16",
                 prompt: str = None,
                 ) -> None:
        """ FasterWhisper 语音转写

        Args:
            model_size (str): 模型大小，可选项为 "tiny", "base", "small", "medium", "large" 。
                更多信息参考：https://github.com/openai/whisper
            device (str, optional): 模型运行设备。
            compute_type (str, optional): 计算类型。默认为"default"。
            prompt (str, optional): 初始提示。如果需要转写简体中文，可以使用简体中文提示。
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.prompt = prompt

    def __enter__(self) -> 'Transcriber':

        self._model = WhisperModel(model_size_or_path=self.model_size,
                                   device=self.device,
                                   compute_type=self.compute_type,
                                   local_files_only=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def transcribe_audio(self, audio: bytes) -> typing.Generator[str, None, None]:
        segments, info = self._model.transcribe(audio=BytesIO(audio),
                                                initial_prompt=self.prompt,
                                                language="zh", vad_filter=True
                                                )
        if info.language != "zh":
            return {"error": "transcribe Chinese only"}
        for segment in segments:
            yield segment.text


class AudioRecorder(object):
    """ Audio recorder.
    Args:
        channels (int, optional): 通道数，默认为1（单声道）。
        sample_rate (int, optional): 采样率，默认为16000 Hz。
        chunk (int, optional): 缓冲区中的帧数，默认为256。
        frame_duration (int, optional): 每帧的持续时间（单位：毫秒），默认为30。
    """
    def __init__(self,
                 channels: int = 1,
                 sample_rate: int = 16000,
                 chunk: int = 256,
                 frame_duration: int = 30) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.frame_size = sample_rate * frame_duration // 1000
        self.audio_data = io.BytesIO()

    def __enter__(self) -> 'AudioRecorder':
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=self.channels,
                                      rate=self.sample_rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def record_audio(self) -> typing.Generator[bytes, None, None]:
        MAXLEN = 30
        watcher = collections.deque(maxlen=MAXLEN)
        triggered, ratio = False, 0.5

        while True:
            frame = self.stream.read(self.chunk)
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            watcher.append(is_speech)
            self.audio_data.write(frame)

            if not triggered:
                num_voiced = sum(watcher)
                if num_voiced > ratio * watcher.maxlen:
                    triggered = True
                    watcher.clear()
            else:
                num_unvoiced = len(watcher) - sum(watcher)
                if num_unvoiced > ratio * watcher.maxlen:
                    triggered = False
                    yield self.audio_data.getvalue()
                    self.audio_data.seek(0)
                    self.audio_data.truncate()

            if keyboard.is_pressed('s'):
                break

class SpeechThread(threading.Thread):
    def __init__(self, transcriber, audio_recorder, canchat):
        threading.Thread.__init__(self)
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.canchat = canchat

    def run(self):
        try:
            with self.audio_recorder as recorder:
                with self.transcriber as transcriber:
                    for audio in recorder.record_audio():
                        for seg in transcriber.transcribe_audio(audio):
                            print("User: ", seg)
                            if self.canchat:
                                response = chatspeak(seg, model="E:\Python_Pro\metaall\llama_ch",
                                                     file="E:/Python_Pro/metaall/Llm/temp_audio.mp3")
                                print("AI: ", response)
                            if keyboard.is_pressed('s'):
                                break
        except KeyboardInterrupt:
            print("键盘中断：终止程序...")
        except Exception as e:
            logging.error(e, exc_info=True, stack_info=True)


if __name__ == '__main__':
    try:
        audio_recorder = AudioRecorder(channels=1, sample_rate=16000)
        transcriber = Transcriber(model_size="model/largev3/")
        speech_thread = SpeechThread(transcriber, audio_recorder,True)
        manager = ThreadManager(function=speech_thread)
        manager.control_thread(True)
    except Exception as e:
        logging.error(e, exc_info=True)
