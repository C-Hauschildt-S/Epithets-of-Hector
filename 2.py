from dicesapi import DicesAPI
from cltk import NLP
import csv
from collections import Counter

# Initialize CLTK
nlp = NLP(language="grc", suppress_banner=True)
print("CLTK NLP pipeline initialized.")

# Initialize DICES API
api = DicesAPI()
print("DICES API client initialized.")

# Fetch a small work (Theogony) that has actual text
all_speeches = api.getSpeeches().advancedFilter(
    lambda s: getattr(s.work, "title", "").lower() == "theogony"
)
print(f"Total speeches in Theogony: {len(all_speeches)}")

# Keep only speeches with text
all_speeches = [s for s in all_speeches if (hasattr(s, "l_text") and s.l_text) or (hasattr(s, "text") and s.text)]
print(f"Speeches with text: {len(all_speeches)}")

# Example: filter speeches by or to a character (e.g., Gaia)
target_character = "Gaia"
character_speeches = []
for speech in all_speeches:
    spkr_names = [spkr.name if hasattr(spkr, "name") else str(spkr) for spkr in getattr(speech, "spkr", [])]
    addr_names = [addr.name if hasattr(addr, "name") else str(addr) for addr in getattr(speech, "addr", [])]

    if target_character in spkr_names or target_character in addr_names:
        character_speeches.append(speech)

print(f"Found {len(character_speeches)} speeches about or to {target_character}.")

# CSV output
csv_file = r"C:\Users\carol\PycharmProjects\dices-client\Examples\SpeechTests\Theogony_speeches.csv"

# Function to extract multi-word formulas/epithets
def extract_epithets(doc, target="Γαῖα", window=2):
    tokens = [w.string for w in doc.words]
    lemmas = [w.lemma for w in doc.words]
    phrases = []

    for i, lemma in enumerate(lemmas):
        if lemma == target:
            start = max(i - window, 0)
            end = min(i + window + 1, len(tokens))
            phrase = " ".join(tokens[start:end])
            phrases.append(phrase)
    return phrases

epithet_counter = Counter()

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Speaker", "Addressee", "Locus", "Summary", "Tokens", "Lemmas", f"{target_character}_Epithets"])

    for speech in character_speeches:
        spkr_names = [spkr.name if hasattr(spkr, "name") else str(spkr) for spkr in getattr(speech, "spkr", [])]
        addr_names = [addr.name if hasattr(addr, "name") else str(addr) for addr in getattr(speech, "addr", [])]
        work_title = getattr(speech.work, "title", str(speech.work)) if hasattr(speech, "work") else "Unknown work"
        line_range = f"lines {getattr(speech, 'l_fi', '?')}-{getattr(speech, 'l_la', '?')}"
        locus = f"{work_title} ({line_range})"

        text = getattr(speech, "l_text", None) or getattr(speech, "text", None)
        if not text:
            continue

        summary = text[:100].replace("\n", " ") + "..."

        doc = nlp.analyze(text)
        tokens = [w.string for w in doc.words]
        lemmas = [w.lemma for w in doc.words]

        epithets = extract_epithets(doc)
        for phrase in epithets:
            epithet_counter[phrase] += 1

        for spkr in spkr_names:
            for addr in addr_names:
                writer.writerow([
                    spkr,
                    addr,
                    locus,
                    summary,
                    " ".join(tokens[:20]),
                    " ".join(lemmas[:20]),
                    "; ".join(epithets) if epithets else ""
                ])

print(f"Saved {target_character} speeches to {csv_file}")

# Print top epithets
print("\nTop epithets/formulas:")
for epithet, freq in epithet_counter.most_common(10):
    print(f"{epithet} ({freq}×)")

