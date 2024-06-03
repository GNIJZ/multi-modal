1、目前语音识别采用了faster-whisper；
2、目前模态有面部识别和语音识别的特征；
3、删除了中英互转的模型；
4、使用了llama3中文微调版本；
5、开启单线程的语音线程，目前在A4000上推理一条的速度在上大概10秒；



-------------------------------------------------

tips：
语音本文采用largev3，自行到huggingface下载https://huggingface.co/openai/whisper-large-v3

llama的中文大模型地址https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v2/tree/main





--------------------------------------------------
后续加入面部的情绪识别、制作动态链接，使用c++作为程序后端

