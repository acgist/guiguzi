# RNNOISE

```
ffplay.exe -ar 48000 -ac 1 -f s16le .\audio.rnnoise.pcm

ffmpeg -i audio.mp3 -ar 48000 -ac 1 -f s16le -c:a pcm_s16le audio.pcm

ffmpeg -ar 48000 -ac 1 -f s16le -c:a pcm_s16le -i audio.pcm audio.mp3
```

## FFmpeg

```
# 版本 6.1.1
https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2024-07-31-12-50/ffmpeg-n6.1.1-329-g0bd31a8f91-linux64-gpl-shared-6.1.tar.xz
```
