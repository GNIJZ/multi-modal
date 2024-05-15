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

        # self.trans= pipeline("translation_en_to_zh", model="model/opus_zh")
        # self.classifier= pipeline("text-classification", model='model/bert_emotion', top_k=1)

    def __enter__(self) -> 'Transcriber':

        self._model = WhisperModel(model_size_or_path=self.model_size,
                                   device=self.device,
                                   compute_type=self.compute_type,
                                   local_files_only=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def __call__(self, audio: bytes) -> typing.Generator[str, None, None]:
        segments, info = self._model.transcribe(audio=BytesIO(audio),
                                                initial_prompt=self.prompt,
                                                language="zh", vad_filter=True
                                                )

        if info.language != "zh":
            return {"error": "transcribe Chinese only"}

        for segment in segments:
            t = segment.text
            yield t
            # print(t)
            # if t.strip():
            #     yield t
                # print(t)
            # translated_data = self.trans(t)[0]['translation_text']
            # #prediction = self.classifier(translated_data, )
            # print(prediction[0])


class AudioRecorder(object):
    """ Audio recorder.
    Args:
        channels (int, 可选): 通道数，默认为1（单声道）。
        rate (int, 可选): 采样率，默认为16000 Hz。
        chunk (int, 可选): 缓冲区中的帧数，默认为256。
        frame_duration (int, 可选): 每帧的持续时间（单位：毫秒），默认为30。
    """

    def __init__(self,
                 channels: int = 1,
                 sample_rate: int = 16000,
                 chunk: int = 256,
                 frame_duration: int = 30) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.frame_size = (sample_rate * frame_duration // 1000)
        self.__frames: typing.List[bytes] = []

    def __enter__(self) -> 'AudioRecorder':

        self.vad = webrtcvad.Vad()
        # 设置 VAD 的敏感度。参数是一个 0 到 3 之间的整数。0 表示对非语音最不敏感，3 最敏感。
        self.vad.set_mode(1)

        self.audio = pyaudio.PyAudio()
        self.sample_width = self.audio.get_sample_size(pyaudio.paInt16)
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

    def __bytes__(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.__frames))
            self.__frames.clear()
        return buf.getvalue()

    def __iter__(self):
        """ Record audio until silence is detected.
        """
        MAXLEN = 30
        watcher = collections.deque(maxlen=MAXLEN)
        triggered, ratio = False, 0.5
        while True:
            frame = self.stream.read(self.frame_size)
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            watcher.append(is_speech)
            self.__frames.append(frame)
            if not triggered:
                num_voiced = len([x for x in watcher if x])
                if num_voiced > ratio * watcher.maxlen:
                    # logging.info("start recording...")
                    triggered = True
                    watcher.clear()
                    self.__frames = self.__frames[-MAXLEN:]
            else:
                num_unvoiced = len([x for x in watcher if not x])
                if num_unvoiced > ratio * watcher.maxlen:
                    # logging.info("stop recording...")
                    triggered = False
                    yield bytes(self)
            if keyboard.is_pressed('s'):
                break

class SpeechThread(threading.Thread):
    def __init__(self,transcriber,audio_recorder,transen,classifier):
        threading.Thread.__init__(self)
        self.transcriber = transcriber
        self.audio_recorder = audio_recorder
        self.transen = transen
        self.classifier =classifier
    def run(self):
        try:
            with self.audio_recorder as recorder:
                with self.transcriber as transcriber:
                    for audio in recorder:
                        for seg in transcriber(audio):
                            # logging.info(seg)
                            print(seg)
                            prediction=self.classifier(self.transen(seg)[0]['translation_text'], )
                            print(prediction)
                            if keyboard.is_pressed('s'):
                                break
        except KeyboardInterrupt:
            print("键盘中断：终止程序...")
        except Exception as e:
            logging.error(e, exc_info=True, stack_info=True)




# 这里添加 SpeechThread 和 ThreadManager 的定义代码

if __name__ == '__main__':
    try:
        # 实例化 ThreadManager
        audio_recorder = AudioRecorder(channels=1, sample_rate=16000)
        transcriber = Transcriber(model_size="model/largev3/")

        speech_thread = SpeechThread(transcriber, audio_recorder)

        manager = ThreadManager(function=speech_thread)

        # 启动线程
        manager.control_thread(True)



    except Exception as e:
        logging.error(e, exc_info=True)
