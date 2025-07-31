from funasr import AutoModel

url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"
#url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
model = AutoModel(model="iic/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch",
                  disable_update=True,
                  disable_log=False,
                  device="dlc",
                  )
res = model.generate(input=url, decoding_ctc_weight=0.0)
print(res)
