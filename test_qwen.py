from funasr import AutoModel


audio_in = "https://github.com/QwenLM/Qwen-Audio/raw/main/assets/audio/1272-128104-0000.flac"
audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
model = AutoModel(model="Qwen-Audio",
                  disable_update=True,
#                  disable_log=False,
                  device="dlc",
                  )
res = model.generate(input=audio_in, prompt=prompt)
print(res)
