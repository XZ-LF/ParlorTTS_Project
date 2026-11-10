import os
import re
import json
import time
import warnings
import argparse

# Silence Gradio/Starlette HTTP_422 deprecation warning in terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")

import gradio as gr
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ==========================================
# 1. LOAD AI MODELS (FP16 + TF32 Optimized)
# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

print("Loading Parler-TTS Engine (FP16)...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    torch_dtype=torch_dtype
).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

print("Loading Emotion Classifier...")
emotion_classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

print("Loading Fast Stateless Grammar Model (flan-t5-small)...")
t5_model_id = "google/flan-t5-small"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_id)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(
    t5_model_id,
    torch_dtype=torch_dtype
).to(device)
t5_model.eval()

# ==========================================
# 2. EMOJI CATALOG & VOICE MAPPINGS
# ==========================================
os.makedirs("samples", exist_ok=True)
SOUNDBOARD_JSON = "samples/soundboard.json"

EMOJI_CATEGORIES = {
    "🌟 All": [
        "🚨", "🆘", "✋", "👍", "👎", "💖", "🙏", "👋",
        "😂", "😊", "😢", "😠", "😨", "😴", "🤕", "🤢",
        "💊", "🩺", "🚽", "🛏️", "🥶", "🥵", "💧", "🥤",
        "🍲", "🍎", "🍜", "☕", "🏠", "📺", "📱", "🎵",
        "🌳", "♿", "🔔", "🔊"
    ],
    "🚨 Health & Needs": [
        "🚨", "🆘", "🤕", "🤢", "💊", "🩺", "🚽", "🛏️",
        "😴", "🥶", "🥵", "♿"
    ],
    "🍽️ Food & Drink": [
        "💧", "🥤", "🍲", "🍎", "🍜", "☕", "🍞", "🍽️"
    ],
    "😊 Feelings & Social": [
        "👍", "👎", "💖", "🙏", "👋", "✋", "😂", "😊",
        "😢", "😠", "😨", "🫂"
    ],
    "🏠 Daily Life": [
        "🏠", "📺", "📱", "🎵", "📖", "🌳", "🚿", "🧥",
        "🔔", "🔊"
    ]
}

voice_profiles = {
    "Emma (Warm Female)": "A female speaker with a warm, soothing voice",
    "Marcus (Deep Male)": "A male speaker with a deep, calm, and resonant voice",
    "Alex (Fast & Clear)": "A speaker with a clear, fast-paced, professional voice"
}

emotion_to_prompt = {
    "neutral": "normally and calmly",
    "joy": "very happily, excitedly, and enthusiastically",
    "amusement": "with a cheerful chuckle and bright amusement",
    "sadness": "very sadly, softly, and depressingly",
    "anger": "very angrily, aggressively, and sternly",
    "annoyance": "with heavy annoyance and bitter frustration",
    "fear": "fearfully, nervously, and anxiously",
    "surprise": "with extreme surprise and shock",
    "love": "very warmly, gently, and affectionately",
    "caring": "gently, softly, and comfortingly"
}

DEFAULT_SOUNDBOARD = [
    {
        "id": "sos",
        "emoji": "🚨",
        "name": "SOS (Help!)",
        "text": "SOS! I need help immediately. Please come here.",
        "prompt": "A clear voice, speaking very loudly, urgently, and frantically.",
        "is_sos": True
    },
    {
        "id": "wait",
        "emoji": "✋",
        "name": "Hold On",
        "text": "Please wait a moment, I am typing a response.",
        "prompt": "A clear voice, speaking clearly and politely.",
        "is_sos": False
    },
    {
        "id": "yes",
        "emoji": "👍",
        "name": "Yes",
        "text": "Yes, I completely agree.",
        "prompt": "A clear voice, speaking normally and calmly.",
        "is_sos": False
    },
    {
        "id": "no",
        "emoji": "👎",
        "name": "No",
        "text": "No, I don't think that is right.",
        "prompt": "A clear voice, speaking normally and sternly.",
        "is_sos": False
    },
    {
        "id": "thanks",
        "emoji": "💖",
        "name": "Thanks",
        "text": "Thank you so much for helping me.",
        "prompt": "A clear voice, speaking warmly and affectionately.",
        "is_sos": False
    },
    {
        "id": "laugh",
        "emoji": "😂",
        "name": "Laugh",
        "text": "Haha, that is really funny!",
        "prompt": "A clear voice, speaking very happily and excitedly, laughing out loud.",
        "is_sos": False
    }
]

def load_soundboard():
    if os.path.exists(SOUNDBOARD_JSON):
        try:
            with open(SOUNDBOARD_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    save_soundboard(DEFAULT_SOUNDBOARD)
    return DEFAULT_SOUNDBOARD

def save_soundboard(items):
    with open(SOUNDBOARD_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def synthesize_to_file(prompt_desc, spoken_text, filepath):
    input_ids = tokenizer(prompt_desc, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(spoken_text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_np = generation.cpu().numpy().squeeze()
    sf.write(filepath, audio_np, model.config.sampling_rate)
    return model.config.sampling_rate, audio_np

def filter_emoji_choices(category):
    choices = EMOJI_CATEGORIES.get(category, EMOJI_CATEGORIES["🌟 All"])
    return gr.update(choices=choices, value=choices[0])

# ==========================================
# 3. SOUNDBOARD & EDITING FUNCTIONS
# ==========================================
def play_soundboard(sound_id, current_items):
    target = next((item for item in current_items if item["id"] == sound_id), None)
    if not target:
        return None, "⚠️ Soundboard button not found."

    filepath = f"samples/{sound_id}.wav"
    if os.path.exists(filepath):
        data, sr = sf.read(filepath, dtype="float32")
        return (sr, data), f"🔊 Speaking: {target['emoji']} {target['name']} — \"{target['text']}\""

    sr, data = synthesize_to_file(target["prompt"], target["text"], filepath)
    return (sr, data), f"🔊 Speaking: {target['emoji']} {target['name']} — \"{target['text']}\""

def save_current_audio_to_soundboard(emoji, name, last_gen, current_items):
    if not last_gen or "audio" not in last_gen:
        return current_items, "⚠️ Speak a sentence first before pinning it to your board.", gr.update()

    emoji = (emoji or "🔊").strip()
    spoken_text = last_gen.get("text", "").strip()
    default_label = " ".join(spoken_text.split()[:3]).capitalize() if spoken_text else "Custom Phrase"
    name = name.strip() or default_label

    new_id = f"custom_{int(time.time())}"
    filepath = f"samples/{new_id}.wav"
    sf.write(filepath, last_gen["audio"], last_gen["sr"])

    new_item = {
        "id": new_id,
        "emoji": emoji,
        "name": name,
        "text": spoken_text,
        "prompt": last_gen.get("prompt", ""),
        "is_sos": False
    }
    updated_items = current_items + [new_item]
    save_soundboard(updated_items)

    return (
        updated_items,
        f"✅ Pinned '{emoji} {name}' to Quick Speak Board!",
        gr.update(open=False)
    )

def open_edit_panel(sound_id, current_items):
    target = next((item for item in current_items if item["id"] == sound_id), None)
    if not target:
        return sound_id, gr.update(visible=False), "", "🔊", ""

    all_emojis = EMOJI_CATEGORIES["🌟 All"]
    current_emoji = target["emoji"] if target["emoji"] in all_emojis else "🔊"
    header_text = f"### ✏️ Edit Button: **{target['emoji']} {target['name']}**\n*Spoken Phrase: \"{target['text']}\"*"
    return (
        sound_id,
        gr.update(visible=True),
        header_text,
        gr.update(choices=all_emojis, value=current_emoji),
        target["name"]
    )

def save_edited_soundboard_item(edit_id, new_emoji, new_name, current_items):
    if not edit_id:
        return current_items, gr.update(visible=False), "⚠️ No button selected."

    new_emoji = (new_emoji or "🔊").strip()
    new_name = new_name.strip() or "Phrase"

    updated_items = []
    for item in current_items:
        if item["id"] == edit_id:
            updated_item = dict(item)
            updated_item["emoji"] = new_emoji
            updated_item["name"] = new_name
            updated_items.append(updated_item)
        else:
            updated_items.append(item)

    save_soundboard(updated_items)
    return updated_items, gr.update(visible=False), f"💾 Updated button to '{new_emoji} {new_name}'."

def delete_soundboard_item(edit_id, current_items):
    if not edit_id:
        return current_items, gr.update(visible=False), "⚠️ No button selected."

    target = next((item for item in current_items if item["id"] == edit_id), None)
    label_str = f"{target['emoji']} {target['name']}" if target else edit_id

    updated_items = [item for item in current_items if item["id"] != edit_id]
    save_soundboard(updated_items)

    filepath = f"samples/{edit_id}.wav"
    if edit_id.startswith("custom_") and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    return updated_items, gr.update(visible=False), f"🗑️ Removed '{label_str}' from board."

def cancel_edit():
    return None, gr.update(visible=False)

# ==========================================
# 4. FAST HYBRID AAC SENTENCE EXPANDER
# ==========================================
INSTANT_AAC_MAP = {
    "tired": "I am feeling really exhausted right now, and I need to rest for a little while.",
    "sleepy": "I am getting very sleepy right now, and I would like to go to sleep soon.",
    "exhausted": "I am completely exhausted right now, and I need to take a break and rest.",
    "hungry": "I am getting pretty hungry right now, could we please get something to eat?",
    "starving": "I am really hungry right now, could we please have some food soon?",
    "thirsty": "I am feeling quite thirsty right now, could I please have a glass of water?",
    "water": "Could I please have a glass of water to drink?",
    "food": "I am feeling hungry right now, could I please have something to eat?",
    "cold": "I am feeling quite cold right now, could you please bring me a blanket?",
    "hot": "I am feeling way too warm right now, could we please cool the room down?",
    "dizzy": "I am feeling a little dizzy right now, and I need to sit down and rest.",
    "sick": "I am not feeling very well right now, could you please stay with me?",
    "nauseous": "My stomach feels nauseous right now, and I need to rest quietly.",
    "pain": "I am in pain right now, could you please come help me?",
    "hurt": "Something is hurting me right now, could you please check on me?",
    "headache": "I have a bad headache right now, could I please have some medicine and rest?",
    "medicine": "It is time for my medicine, could you please help me get it?",
    "bathroom": "I need to use the bathroom right now, could you please help me get there?",
    "toilet": "I need to use the restroom right away, please help me get there.",
    "restroom": "I need to use the restroom right away, please help me get there.",
    "blanket": "I am feeling a little chilly, could you please bring me a blanket?",
    "help": "I need some help right now, could you please come over and assist me?",
    "help me": "Could you please come over and help me with this right now?",
    "i need help": "I need some help right now, could someone please come over and assist me?",
    "need help": "I need some help right now, could someone please come over and assist me?",
    "i love you": "I love you so much, and I really appreciate having you here with me.",
    "love you": "I love you so much, and I am really grateful to have you with me.",
    "thank you": "Thank you so much for your help, I really appreciate it.",
    "thanks": "Thank you so much for helping me out with that.",
    "sorry": "I am really sorry about that, thank you for being patient with me.",
    "wait": "Please wait just a moment, I am still preparing what I want to say.",
    "hold on": "Please hold on for a second while I finish typing my thought.",
    "bored": "I am feeling a little bored right now, could we do something together?",
    "lonely": "I am feeling a little lonely right now, would you mind keeping me company?",
    "scared": "I am feeling scared and anxious right now, please stay close to me.",
    "anxious": "I am feeling quite anxious right now, and I need a moment to calm down.",
    "overwhelmed": "I am feeling a bit overwhelmed right now, could we please slow down?",
    "happy": "I am feeling really happy and grateful right now!",
    "sad": "I am feeling quite sad right now, and I could really use some comfort.",
    "angry": "I am feeling very frustrated and upset about this right now.",
    "ready": "I am all ready to go whenever you are.",
    "done": "I am all finished with this now, thank you.",
    "home": "I would really like to go home now, if that is possible.",
    "outside": "I would really like to get some fresh air outside for a little while.",
    "bed": "I am ready to get into bed and rest now, could you please help me?"
}

BODY_PARTS = {"head", "stomach", "tummy", "belly", "back", "neck", "throat", "chest", "arm", "arms", "leg", "legs", "foot", "feet", "hand", "hands", "eye", "eyes", "ear", "ears", "tooth", "teeth", "shoulder"}
PAIN_WORDS = {"hurt", "hurts", "hurting", "pain", "ache", "aching", "sore", "cramp", "burning"}
ACTION_VERBS = {"eat", "drink", "go", "watch", "play", "listen", "read", "call", "see", "visit", "buy", "get", "take", "use", "open", "close", "turn", "sit", "stand", "lie", "lay", "sleep", "rest", "walk", "talk", "ask", "tell", "change", "wash", "clean", "charge", "fix", "find", "make"}
DESIRE_VERBS = {"want", "need", "like", "love", "hate", "prefer", "wish", "miss", "feel", "have"}
FEELING_ADJS = {"tired", "sleepy", "hungry", "thirsty", "cold", "hot", "warm", "chilly", "dizzy", "sick", "weak", "happy", "sad", "angry", "upset", "annoyed", "frustrated", "worried", "nervous", "scared", "afraid", "anxious", "bored", "lonely", "excited", "comfortable", "uncomfortable", "itchy", "full", "ready", "busy"}

def match_structured_slots(words):
    word_set = set(words)

    body_hit = next((w for w in words if w in BODY_PARTS), None)
    pain_hit = next((w for w in words if w in PAIN_WORDS), None)
    if body_hit and pain_hit:
        leftover = [w for w in words if w not in {body_hit, pain_hit, "my", "i", "have", "has", "is", "are", "very", "really", "so", "bad", "need", "want", "please"}]
        if leftover:
            return f"My {body_hit} is hurting quite a bit right now, could you please help me get {' '.join(leftover)}?"
        return f"My {body_hit} is hurting right now, could you please help me feel more comfortable?"

    if "hungry" in word_set and len(words) > 1:
        items = [w for w in words if w not in {"i", "am", "im", "feel", "feeling", "hungry", "want", "need", "to", "eat", "have", "get", "some", "please"}]
        if items:
            return f"I am feeling hungry right now, and I would really like to have some {' '.join(items)}."
        return INSTANT_AAC_MAP["hungry"]

    if "thirsty" in word_set and len(words) > 1:
        items = [w for w in words if w not in {"i", "am", "im", "feel", "feeling", "thirsty", "want", "need", "to", "drink", "have", "get", "some", "please"}]
        if items:
            return f"I am feeling quite thirsty right now, could I please have some {' '.join(items)} to drink?"
        return INSTANT_AAC_MAP["thirsty"]

    if "cold" in word_set and len(words) > 1:
        items = [w for w in words if w not in {"i", "am", "im", "feel", "feeling", "cold", "want", "need", "get", "bring", "me", "a", "the", "please"}]
        if items:
            return f"I am feeling quite cold right now, could you please help me with the {' '.join(items)}?"

    if "hot" in word_set and len(words) > 1:
        items = [w for w in words if w not in {"i", "am", "im", "feel", "feeling", "hot", "want", "need", "turn", "on", "the", "a", "please"}]
        if items:
            return f"I am feeling too warm right now, could you please help me with the {' '.join(items)}?"

    return None

def preprocess_shorthand_to_first_person(words):
    explicit_starters = {"i", "im", "i'm", "ive", "i've", "id", "i'd", "my", "me", "we", "you", "can", "could", "would", "please", "where", "what", "when", "why", "how", "who", "is", "are", "do", "does", "did", "lets", "let's"}

    processed_tokens = []
    for idx, w in enumerate(words):
        if idx > 0 and w in {"want", "need"} and words[idx - 1] not in {"i", "you", "we", "dont", "don't", "do", "not", "and"}:
            processed_tokens.extend(["and", "I", w])
        elif idx > 0 and w in ACTION_VERBS and words[idx - 1] in {"want", "need", "like", "love", "prefer"}:
            processed_tokens.extend(["to", w])
        else:
            processed_tokens.append(w)

    rebuilt = " ".join(processed_tokens)
    first_w = words[0]

    if first_w in explicit_starters:
        if first_w in {"i", "im", "i'm"} and len(words) > 1 and words[1] in FEELING_ADJS:
            return f"I am feeling {' '.join(processed_tokens[1:])}"
        return rebuilt
    if first_w in FEELING_ADJS:
        return f"I am feeling {rebuilt}"
    if first_w in DESIRE_VERBS:
        return f"I {rebuilt}"
    if first_w in ACTION_VERBS:
        return f"I would like to {rebuilt}"
    if first_w in BODY_PARTS:
        return f"My {rebuilt}"

    return f"I need help with {rebuilt}"

def expand_keywords(keywords):
    raw = " ".join(keywords.strip().split())
    if not raw:
        return ""

    clean = re.sub(r"[^a-zA-Z0-9\s']", "", raw.lower()).strip()
    if not clean:
        return raw

    if clean in INSTANT_AAC_MAP:
        return INSTANT_AAC_MAP[clean]

    words = clean.split()
    if len(words) == 2 and words[0] in {"i", "im", "i'm"} and words[1] in INSTANT_AAC_MAP:
        return INSTANT_AAC_MAP[words[1]]
    if len(words) == 3 and words[0] == "i" and words[1] in {"am", "feel"} and words[2] in INSTANT_AAC_MAP:
        return INSTANT_AAC_MAP[words[2]]

    slot_sentence = match_structured_slots(words)
    if slot_sentence:
        return slot_sentence

    seeded_sentence = preprocess_shorthand_to_first_person(words)
    t5_prompt = f"Make a complete, natural English sentence: {seeded_sentence}"

    inputs = t5_tokenizer(t5_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = t5_model.generate(
            inputs.input_ids,
            max_new_tokens=32,
            num_beams=1,
            do_sample=False
        )

    sentence = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    if not any(p in words for p in {"he", "him", "his", "she", "her", "they", "them"}):
        sentence = re.sub(r"^(He|She)\s+is\b", "I am", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(He|She)\s+wants\b", "I want", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(He|She)\s+needs\b", "I need", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^(He|She)\s+feels\b", "I feel", sentence, flags=re.IGNORECASE)

    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?")):
        sentence += "."

    out_words = sentence.split()
    if len(out_words) <= 6 and not sentence.endswith("?"):
        lower_s = sentence.lower()
        if any(k in lower_s for k in ("need", "help", "please", "could", "can", "water", "medicine", "bathroom", "doctor")):
            sentence = sentence.rstrip(".!") + ", could you please help me with that?"
        elif any(k in lower_s for k in ("want", "would like", "going to", "watch", "eat", "drink", "play", "go")):
            sentence = sentence.rstrip(".!") + " right now, if that is okay with you."
        elif any(k in lower_s for k in ("feel", "am ", "hurt", "pain", "tired", "sick")):
            sentence = sentence.rstrip(".!") + " right now, and I wanted to let you know."

    return sentence

# ==========================================
# 5. MAIN TTS GENERATOR LOGIC
# ==========================================
def update_audio_preview(voice_choice):
    if voice_choice == "Custom":
        return None
    filename = f"samples/{voice_choice.split()[0].lower()}.wav"
    if os.path.exists(filename):
        data, sr = sf.read(filename, dtype="float32")
        return (sr, data)
    return None

def generate_speech(user_text, voice_choice, custom_voice_text, emotion_override, custom_emotion_text, pace):
    if not user_text or not user_text.strip():
        return None, "⚠️ Type a word or sentence first.", {}, "", None, gr.update(), ""

    base_description = (
        custom_voice_text
        if voice_choice == "Custom" and custom_voice_text
        else voice_profiles.get(voice_choice, "A clear voice")
    )

    pace_prompt = ""
    if pace == "Slow":
        pace_prompt = ", speaking very slowly and deliberately"
    elif pace == "Fast":
        pace_prompt = ", speaking very quickly and briskly"

    confidence_dict = {}

    if emotion_override == "Auto":
        predictions = emotion_classifier(user_text)[0]
        for item in predictions:
            confidence_dict[item["label"].capitalize()] = item["score"]

        top_emotion = predictions[0]["label"]
        mapped_emotion = emotion_to_prompt.get(top_emotion, "normally")
        final_prompt = f"{base_description}, speaking {mapped_emotion}{pace_prompt}."
        debug_message = f"Auto-Detected Emotion: {top_emotion.upper()}\nPrompt: {final_prompt}"

    elif emotion_override == "Custom":
        custom_emotion = custom_emotion_text if custom_emotion_text else "normally"
        final_prompt = f"{base_description}, speaking {custom_emotion}{pace_prompt}."
        debug_message = f"Custom Prompt: {final_prompt}"

    else:
        modifier = f"very {emotion_override.lower()}" if emotion_override != "Neutral" else "normally"
        final_prompt = f"{base_description}, speaking {modifier}{pace_prompt}."
        debug_message = f"Manual Prompt: {final_prompt}"

    input_ids = tokenizer(final_prompt, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(user_text, return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

    audio_np = generation.cpu().numpy().squeeze()
    sr = model.config.sampling_rate
    audio_output = (sr, audio_np)

    clean_text = user_text.strip()
    suggested_label = " ".join(clean_text.split()[:3]).rstrip(".,!?").capitalize()

    last_gen_data = {
        "sr": sr,
        "audio": audio_np,
        "text": clean_text,
        "prompt": final_prompt
    }

    return (
        audio_output,
        f"🔊 Speaking: \"{clean_text}\"",
        confidence_dict,
        debug_message,
        last_gen_data,
        gr.update(open=True),
        suggested_label
    )

# ==========================================
# 6. ACCESSIBLE UI & EMOJI PICKER STYLING
# ==========================================
custom_css = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

/* Quick Speak Soundboard Grid */
.sb-pair { gap: 6px !important; flex-wrap: nowrap !important; align-items: stretch !important; margin-bottom: 4px !important; }
.sb-btn {
    min-height: 68px !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    text-align: left !important;
}
.sos-btn {
    background: #dc2626 !important;
    color: #ffffff !important;
    min-height: 68px !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
}
.sos-btn:hover { background: #b91c1c !important; }
.edit-icon-btn {
    min-width: 46px !important;
    max-width: 46px !important;
    min-height: 68px !important;
    padding: 0 !important;
    font-size: 1.1rem !important;
    border-radius: 14px !important;
}

/* Composer Action Buttons */
.composer-box textarea {
    font-size: 1.25rem !important;
    line-height: 1.5 !important;
}
.speak-main-btn {
    min-height: 64px !important;
    font-size: 1.22rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
}
.expand-main-btn, .clear-main-btn {
    min-height: 64px !important;
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
}

/* Interactive Emoji Tile Grid */
.emoji-picker-grid .wrap {
    display: grid !important;
    grid-template-columns: repeat(auto-fill, minmax(52px, 1fr)) !important;
    gap: 8px !important;
    padding-top: 6px !important;
}
.emoji-picker-grid label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 52px !important;
    font-size: 1.75rem !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 2px solid rgba(148, 163, 184, 0.3) !important;
    background: rgba(148, 163, 184, 0.08) !important;
    transition: transform 0.12s ease, border-color 0.12s ease, background 0.12s ease !important;
}
.emoji-picker-grid label:hover {
    transform: scale(1.06) !important;
    border-color: #7c3aed !important;
}
.emoji-picker-grid label input[type="radio"] {
    display: none !important;
}
.emoji-picker-grid label.selected,
.emoji-picker-grid label:has(input:checked) {
    border: 3px solid #7c3aed !important;
    background: rgba(124, 58, 237, 0.22) !important;
    transform: scale(1.08) !important;
}
"""

initial_sb = load_soundboard()

with gr.Blocks(title="Accessible AAC Communicator") as demo:
    soundboard_state = gr.State(initial_sb)
    editing_id_state = gr.State(None)
    last_generated_state = gr.State(None)

    # Top Header & Unified Audio Output Bar
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.Markdown("## 🗣️ Accessible AAC Communicator")
            status_banner = gr.Textbox(
                value="Ready — Tap a Quick Phrase on the left, or type on the right.",
                show_label=False,
                interactive=False,
                max_lines=1
            )
        with gr.Column(scale=4):
            shared_audio_player = gr.Audio(
                label="🔊 Active Voice Output (Plays Automatically)",
                interactive=False,
                autoplay=True
            )

    # Main 2-Column Accessible Workspace
    with gr.Row(equal_height=False):

        # ==========================================
        # LEFT PANEL: 1-TAP QUICK SPEAK BOARD
        # ==========================================
        with gr.Column(scale=5, variant="panel"):
            gr.Markdown("### ⚡ Quick Speak Board")
            gr.Markdown("*Tap any tile to speak immediately. Tap `✏️` to change its icon, rename, or delete.*")

            sb_grid_container = gr.Column()

            # Modal for Editing / Deleting an Existing Button
            with gr.Group(visible=False) as edit_modal_group:
                edit_header_md = gr.Markdown("### ✏️ Edit Button")
                edit_name_input = gr.Textbox(label="Button Label", max_lines=1)
                edit_cat_radio = gr.Radio(
                    choices=list(EMOJI_CATEGORIES.keys()),
                    value="🌟 All",
                    label="Filter Emoji Category"
                )
                edit_emoji_picker = gr.Radio(
                    choices=EMOJI_CATEGORIES["🌟 All"],
                    value="🔊",
                    label="Tap an Emoji Icon to Select",
                    elem_classes=["emoji-picker-grid"]
                )
                with gr.Row():
                    save_edit_btn = gr.Button("💾 Save Changes", variant="primary")
                    cancel_edit_btn = gr.Button("✖️ Cancel")
                delete_sb_btn = gr.Button("🗑️ Delete This Button", variant="stop")

            # Render Quick Speak Buttons with Inline Edit Triggers
            with sb_grid_container:
                @gr.render(inputs=soundboard_state)
                def render_soundboard_buttons(items):
                    for i in range(0, len(items), 2):
                        with gr.Row(key=f"sb_row_{i}"):
                            for item in items[i:i + 2]:
                                with gr.Row(key=f"sb_pair_{item['id']}", elem_classes=["sb-pair"]):
                                    btn_class = ["sos-btn"] if item.get("is_sos") else ["sb-btn"]
                                    play_btn = gr.Button(
                                        f"{item['emoji']}  {item['name']}",
                                        elem_classes=btn_class,
                                        scale=5,
                                        key=f"play_{item['id']}"
                                    )
                                    edit_btn = gr.Button(
                                        "✏️",
                                        elem_classes=["edit-icon-btn"],
                                        scale=0,
                                        min_width=46,
                                        key=f"edit_{item['id']}"
                                    )
                                    play_btn.click(
                                        fn=lambda s_id=item["id"]: play_soundboard(s_id, items),
                                        outputs=[shared_audio_player, status_banner]
                                    )
                                    edit_btn.click(
                                        fn=lambda s_id=item["id"]: open_edit_panel(s_id, items),
                                        outputs=[
                                            editing_id_state,
                                            edit_modal_group,
                                            edit_header_md,
                                            edit_emoji_picker,
                                            edit_name_input
                                        ]
                                    )

        # ==========================================
        # RIGHT PANEL: UNIFIED SMART COMPOSER
        # ==========================================
        with gr.Column(scale=7, variant="panel"):
            gr.Markdown("### 💬 Type & Speak")
            gr.Markdown("*Type a few shorthand words (like `tired` or `hungry noodle`) and tap **Auto-Expand**, or type a full sentence and tap **Speak Out Loud**.*")

            text_input = gr.Textbox(
                label="Your Message",
                placeholder="Type shorthand words (e.g., 'tired', 'stomach hurt medicine') or a full sentence here...",
                lines=3,
                elem_classes=["composer-box"]
            )

            # Big, high-contrast action buttons
            with gr.Row():
                expand_btn = gr.Button(
                    "✨ 1. Auto-Expand Words",
                    variant="secondary",
                     scale=2,
                    elem_classes=["expand-main-btn"]
                )
                generate_btn = gr.Button(
                    "🗣️ 2. Speak Out Loud",
                    variant="primary",
                    scale=3,
                    elem_classes=["speak-main-btn"]
                )
                clear_btn = gr.Button(
                    "🧹 Clear",
                    scale=1,
                    elem_classes=["clear-main-btn"]
                )

            # Simple, Accessible Voice & Tone Controls
            with gr.Group():
                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=list(voice_profiles.keys()) + ["Custom"],
                        value="Emma (Warm Female)",
                        label="🎙️ Voice"
                    )
                    emotion_dropdown = gr.Dropdown(
                        choices=["Auto", "Neutral", "Happily", "Sadly", "Angrily", "Custom"],
                        value="Auto",
                        label="🎭 Tone / Emotion"
                    )
                    pace_dropdown = gr.Radio(
                        choices=["Slow", "Normal", "Fast"],
                        value="Normal",
                        label="⏱️ Speed"
                    )
                with gr.Row():
                    custom_voice_input = gr.Textbox(label="Custom Voice Description", visible=False)
                    custom_emotion_input = gr.Textbox(label="Custom Emotion Description", visible=False)

            # Guided Drawer: Pin Last Spoken Sentence to Board with Visual Emoji Picker
            with gr.Accordion("📌 Pin Last Spoken Phrase to Quick Speak Board", open=False) as save_drawer_group:
                save_name_input = gr.Textbox(
                    label="1. Short Button Name",
                    placeholder="e.g., Need Water (auto-filled when you speak)",
                    max_lines=1
                )
                save_cat_radio = gr.Radio(
                    choices=list(EMOJI_CATEGORIES.keys()),
                    value="🌟 All",
                    label="2. Filter Emoji Category"
                )
                save_emoji_picker = gr.Radio(
                    choices=EMOJI_CATEGORIES["🌟 All"],
                    value="🔊",
                    label="3. Tap an Emoji Icon",
                    elem_classes=["emoji-picker-grid"]
                )
                save_to_sb_btn = gr.Button(
                    "➕ Add Phrase to Quick Speak Board",
                    variant="primary",
                    size="lg"
                )

            # Hidden-by-default Diagnostics & Voice Preview
            with gr.Accordion("⚙️ Voice Preview & AI Emotion Diagnostics", open=False):
                with gr.Row():
                    preview_player = gr.Audio(
                        label="Selected Voice Preview",
                        interactive=False,
                        value="samples/emma.wav" if os.path.exists("samples/emma.wav") else None
                    )
                    emotion_chart = gr.Label(
                        num_top_classes=4,
                        label="Detected Emotion Confidence"
                    )
                prompt_debug_box = gr.Textbox(
                    label="AI Conditioning Prompt Used",
                    interactive=False,
                    lines=2
                )

    # ==========================================
    # 7. EVENT WIRING
    # ==========================================
    # Composer actions
    expand_btn.click(fn=expand_keywords, inputs=text_input, outputs=text_input)
    clear_btn.click(fn=lambda: "", outputs=text_input)

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, voice_dropdown, custom_voice_input, emotion_dropdown, custom_emotion_input, pace_dropdown],
        outputs=[
            shared_audio_player,
            status_banner,
            emotion_chart,
            prompt_debug_box,
            last_generated_state,
            save_drawer_group,
            save_name_input
        ]
    )

    # Emoji category filters (for both Save Drawer and Edit Modal)
    save_cat_radio.change(fn=filter_emoji_choices, inputs=save_cat_radio, outputs=save_emoji_picker)
    edit_cat_radio.change(fn=filter_emoji_choices, inputs=edit_cat_radio, outputs=edit_emoji_picker)

    # Save current spoken audio to Quick Speak Board
    save_to_sb_btn.click(
        fn=save_current_audio_to_soundboard,
        inputs=[save_emoji_picker, save_name_input, last_generated_state, soundboard_state],
        outputs=[soundboard_state, status_banner, save_drawer_group]
    )

    # Edit / Delete Soundboard items
    save_edit_btn.click(
        fn=save_edited_soundboard_item,
        inputs=[editing_id_state, edit_emoji_picker, edit_name_input, soundboard_state],
        outputs=[soundboard_state, edit_modal_group, status_banner]
    )
    delete_sb_btn.click(
        fn=delete_soundboard_item,
        inputs=[editing_id_state, soundboard_state],
        outputs=[soundboard_state, edit_modal_group, status_banner]
    )
    cancel_edit_btn.click(
        fn=cancel_edit,
        outputs=[editing_id_state, edit_modal_group]
    )

    # Voice & Emotion Custom Toggles
    voice_dropdown.change(fn=update_audio_preview, inputs=voice_dropdown, outputs=preview_player)
    voice_dropdown.change(
        fn=lambda v: gr.update(visible=(v == "Custom")),
        inputs=voice_dropdown,
        outputs=custom_voice_input
    )
    emotion_dropdown.change(
        fn=lambda e: gr.update(visible=(e == "Custom")),
        inputs=emotion_dropdown,
        outputs=custom_emotion_input
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the Accessible AAC App")
    parser.add_argument("--no-redirect", action="store_true", help="Do not automatically open the browser.")
    args = parser.parse_args()

    print(f"Starting server... Auto-open browser: {not args.no_redirect}")
    demo.launch(
        inbrowser=not args.no_redirect,
        theme=gr.themes.Soft(),
        css=custom_css
    )