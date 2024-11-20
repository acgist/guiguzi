# RNNOISE

```
ffplay.exe -ar 48000 -ac 1 -f s16le .\audio.rnnoise.pcm

ffmpeg -i audio.mp3 -ar 48000 -ac 1 -f s16le -c:a pcm_s16le audio.pcm

ffmpeg -ar 48000 -ac 1 -f s16le -c:a pcm_s16le -i audio.pcm audio.mp3
```
