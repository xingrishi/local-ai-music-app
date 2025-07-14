import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import argparse
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_music(text_prompt, output_path=None):
    """
    Generates music from a text prompt using the MusicGen model
    and saves it as a .wav file.
    """
    if output_path is None:
        # Default to saving in the same directory as the script
        output_path = os.path.join(SCRIPT_DIR, "music_output.wav")

    """
    Generates music from a text prompt using the MusicGen model
    and saves it as a .wav file.
    """
    print("Loading model and processor...")
    # Load the processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    # Check for Apple Silicon (MPS) and use it if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) for acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    
    model.to(device)

    print(f"Generating music for prompt: '{text_prompt}'")
    # Process the text prompt
    inputs = processor(
        text=[text_prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate audio
    audio_values = model.generate(**inputs, max_new_tokens=256)
    
    # Get the sampling rate from the model config
    sampling_rate = model.config.audio_encoder.sampling_rate
    
    # Move audio to CPU and convert to numpy array
    audio_numpy = audio_values[0].cpu().numpy().squeeze()

    print(f"Saving audio to {output_path}...")
    # Save as a .wav file
    scipy.io.wavfile.write(output_path, rate=sampling_rate, data=audio_numpy)
    print("Done!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate music from a text prompt.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A calming piano melody with gentle rain in the background",
        help="The text prompt to generate music from."
    )
    args = parser.parse_args()

    # Generate music using the provided prompt
    generate_music(args.prompt)
