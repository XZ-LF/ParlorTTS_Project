import os
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

voice_profiles = {
    "Emma (Warm Female)": "A female speaker with a warm, soothing voice, speaking normally.",
    "Marcus (Deep Male)": "A male speaker with a deep, calm, and resonant voice, speaking normally.",
    "Alex (Fast & Clear)": "A speaker with a clear, fast-paced, professional voice, speaking normally."
}

# The pre-rendered Instant Soundboard items (Zero latency)
soundboard_items = {
    "sos": {"text": "SOS! I need help immediately. Please come here.", "prompt": "A clear voice, speaking very loudly, urgently, and frantically."},
    "laugh": {"text": "Haha, that is really funny!", "prompt": "A clear voice, speaking very happily and excitedly, laughing out loud."},
    "yes": {"text": "Yes, I completely agree.", "prompt": "A clear voice, speaking normally and calmly."},
    "no": {"text": "No, I don't think that is right.", "prompt": "A clear voice, speaking normally and sternly."},
    "thanks": {"text": "Thank you so much.", "prompt": "A clear voice, speaking warmly and affectionately."}
}

def generate_previews():
    os.makedirs("samples", exist_ok=True)
    
    missing_samples = []
    
    # Check missing voice profiles
    for name, description in voice_profiles.items():
        filename = f"samples/{name.split()[0].lower()}.wav"
        if not os.path.exists(filename):
            missing_samples.append((description, "Hi! I am one of the voices you can choose for this application.", filename))
            
    # Check missing soundboard items
    for key, data in soundboard_items.items():
        filename = f"samples/{key}.wav"
        if not os.path.exists(filename):
            missing_samples.append((data["prompt"], data["text"], filename))

    if not missing_samples:
        print("All audio samples and soundboard files already exist. Skipping generation.")
        return

    print("Missing files detected. Loading AI model in optimized FP16 mode... (This takes a moment)")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load in float16 for massive speed improvements on GPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler-tts-mini-v1", 
        torch_dtype=torch_dtype
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    # Generate missing files
    for description, text, filename in missing_samples:
        print(f"Generating -> {filename}...")
        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        sf.write(filename, generation.cpu().numpy().squeeze(), model.config.sampling_rate)
        
    print("All necessary AAC samples generated successfully!")

if __name__ == "__main__":
    generate_previews()