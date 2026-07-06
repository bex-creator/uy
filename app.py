import gradio as gr
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import numpy as np

print("Loading model... (This happens once when the server starts)")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_music(prompt):
    """This function runs when a user clicks the generate button on your website."""
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    
    # Generate 5-10 seconds of audio
    audio_values = model.generate(**inputs, max_new_tokens=512)
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # Extract the audio data
    audio_data = audio_values[0, 0].cpu().numpy()
    
    # Gradio expects audio returned as a tuple: (sample_rate, numpy_audio_array)
    return (sampling_rate, audio_data)

# Build the Website Interface
interface = gr.Interface(
    fn=generate_music,
    inputs=gr.Textbox(lines=2, placeholder="Type your vibe... e.g., 'Lo-fi chill beat with piano'"),
    outputs=gr.Audio(label="Generated Track"),
    title="🎵 My Custom AI Music Generator",
    description="Type a prompt below and let the AI build a song for you!",
    allow_flagging="never"
)

# Launch the website
if __name__ == "__main__":
    interface.launch()
