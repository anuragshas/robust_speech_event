import gradio as gr
import librosa
from transformers import AutoFeatureExtractor, pipeline


def load_and_fix_data(input_file, model_sampling_rate):
    speech, sample_rate = librosa.load(input_file)
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]
    if sample_rate != model_sampling_rate:
        speech = librosa.resample(speech, sample_rate, model_sampling_rate)
    return speech


feature_extractor = AutoFeatureExtractor.from_pretrained(
    "anuragshas/wav2vec2-xls-r-1b-hi-with-lm"
)
sampling_rate = feature_extractor.sampling_rate

asr = pipeline(
    "automatic-speech-recognition", model="anuragshas/wav2vec2-xls-r-1b-hi-with-lm"
)


def predict_and_ctc_lm_decode(input_file):
    speech = load_and_fix_data(input_file, sampling_rate)
    transcribed_text = asr(speech, chunk_length_s=5, stride_length_s=1)
    return transcribed_text["text"]


gr.Interface(
    predict_and_ctc_lm_decode,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", label="Record your audio")
    ],
    outputs=[gr.outputs.Textbox()],
    examples=[["example1.wav"]],
    title="Hindi ASR using Wav2Vec2-1B with LM",
    description="Built during Robust Speech Event",
    layout="horizontal",
    theme="huggingface",
).launch(enable_queue=True)
