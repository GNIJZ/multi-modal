import io

import transformers
import torch
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# 设置模型和消息
def chatspeak(message):
    model_id = "llama"
    message=message
    messages = [
        {"role": "user", "content": message},
    ]

    # 设置文本生成管道
    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda:0"
    )

    # 生成对话
    prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # 提取生成的文本
    generated_text = outputs[0]["generated_text"]
    print(generated_text)
    conversation_lines = generated_text.strip().split("\n")

    # 提取最后一句话
    last_sentence = conversation_lines[-1]

    if last_sentence is not None:

        # 使用 gtts 生成语音
        tts = gTTS(text=last_sentence, lang='en')

        # 将语音保存到临时文件中
        temp_file_path = "temp_audio.mp3"
        tts.save(temp_file_path)

        # 播放生成的语音
        audio = AudioSegment.from_file(temp_file_path, format="mp3")
        play(audio)
