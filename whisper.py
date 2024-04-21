import argparse
from transformers import pipeline
import torchaudio


def main(file):
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small",device="cuda")
    speech = torchaudio.load("speech.wav")[0][0].numpy()
    while(True):
        transcription = pipe(speech)['text']
    with open('transcription.txt', 'w') as file:
        file.write(transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe speech using whisper.")
    parser.add_argument("--file", type=str, default="speech.wav",
                        help="Audio file to transcribe.")
    args = parser.parse_args()
    main(args.file)
