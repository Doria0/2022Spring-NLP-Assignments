import torch
import torchaudio.datasets
# import os
d = torchaudio.datasets.SPEECHCOMMANDS('./data', download=True)
d1 = torchaudio.datasets.SPEECHCOMMANDS('./data', url='speech_commands_v0.01' download=True)

# root = '.\data\SpeechCommands\speech_commands_v0.01'
# data = []
# flag = 0
# for root, dir, files in os.walk(root):
#     if flag >= 30:
#         break
#     for file in files:
#         f_path, cmd = os.path.join(root, file), root.split('\\')[-1]
#         print(root, cmd)
#         data.append((f_path, cmd))
#         flag = flag+1
