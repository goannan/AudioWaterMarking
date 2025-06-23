import os
import json

input_csv = "D:/dataset/LJSpeech-1.1/metadata.csv"
output_manifest = "D:/dataset/LJSpeech-1.1/train_manifest.json"

manifest = []
with open(input_csv, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        wav_path = f"D:/dataset/LJSpeech-1.1/LJSpeech_watermarked/{parts[0]}.wav"
        text = parts[2]
        manifest.append({
            "audio_filepath": wav_path,
            "text": text,
            "duration": 5.0  # 你可以使用 librosa 计算真实时长
        })

with open(output_manifest, 'w', encoding='utf-8') as f:
    for m in manifest:
        f.write(json.dumps(m) + '\n')
