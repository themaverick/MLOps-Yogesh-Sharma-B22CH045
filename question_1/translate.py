from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
import os

MODEL_NAME  = "Helsinki-NLP/opus-mt-bn-en"
INPUT_FILE  = "input.txt"
REF_FILE    = "reference.txt"
OUTPUT_FILE = "output.txt"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)
print("Model loaded successfully.\n")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    bengali_sentences = [line.strip() for line in f if line.strip()]

print(f"Total sentences to translate: {len(bengali_sentences)}\n")

def translate(sentences, batch_size=8):
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        translated_tokens = model.generate(**inputs)
        decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(decoded)
    return translations

translations = translate(bengali_sentences)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for t in translations:
        f.write(t + "\n")

print(f"Translations saved to '{OUTPUT_FILE}'.\n")

print("=" * 60)
print("First sentence (Bengali)  :", bengali_sentences[0])
print("First sentence (English)  :", translations[0])
print("=" * 60 + "\n")

with open(REF_FILE, "r", encoding="utf-8") as f:
    references = [line.strip() for line in f if line.strip()]

bleu = sacrebleu.corpus_bleu(translations, [references])

print(f"SacreBLEU Score: {bleu.score:.2f}")
print(f"(Precision scores: {bleu.precisions})")
