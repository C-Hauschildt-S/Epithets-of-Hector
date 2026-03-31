from dicesapi import DicesAPI
from cltk import NLP
import csv
from collections import Counter

# Initialize CLTK for Ancient Greek
nlp = NLP(language="grc", suppress_banner=True)
print("CLTK NLP pipeline initialized.")

# Initialize DICES API
api = DicesAPI()
print("DICES API client initialized.")

# Fetch all speeches from DICES
all_speeches_raw = api.getSpeeches()
print(f"Total speeches retrieved from API: {len(all_speeches_raw)}")

# Check available work titles
titles = set(getattr(s.work, "title", None) for s in all_speeches_raw)
print(f"Available work titles in dataset: {titles}")

# Filter speeches for Iliad (case-insensitive)
all_speeches = all_speeches_raw.advancedFilter(
    lambda s: getattr(s.work, "title", "").lower() in ["iliad", "the iliad"]
)
print(f"Speeches filtered for Iliad: {len(all_speeches)}")

# Keep only speeches with text
all_speeches = [s for s in all_speeches if (hasattr(s, "l_text") and s.l_text) or (hasattr(s, "text") and s.text)]
print(f"Speeches with text: {len(all_speeches)}")

# Filter speeches by or to Hector
hector_speeches = []
for speech in all_speeches:
    spkr_names = [spkr.name if hasattr(spkr, "name") else str(spkr) for spkr in getattr(speech, "spkr", [])]
    addr_names = [addr.name if hasattr(addr, "name") else str(addr) for addr in getattr(speech, "addr", [])]

    if "Hector" in spkr_names or "Hector" in addr_names:
        hector_speeches.append(speech)

print(f"Found {len(hector_speeches)} speeches in the Iliad about or to Hector.")

# CSV output file
csv_file = r"C:\Users\carol\PycharmProjects\dices-client\Examples\SpeechTests\Hector_speeches_with_epithets.csv"

# Function to extract multi-word formulas / epithets around "Hector"
def extract_hector_epithets(doc, target="Ἕκτωρ", window=2):
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

# Collect all epithets for frequency analysis
epithet_counter = Counter()

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Speaker", "Addressee", "Locus", "Summary", "Tokens", "Lemmas", "Hector_Epithets"])

    for speech in hector_speeches:
        spkr_names = [spkr.name if hasattr(spkr, "name") else str(spkr) for spkr in getattr(speech, "spkr", [])]
        addr_names = [addr.name if hasattr(addr, "name") else str(addr) for addr in getattr(speech, "addr", [])]
        work_title = getattr(speech.work, "title", str(speech.work)) if hasattr(speech, "work") else "Unknown work"
        line_range = f"lines {getattr(speech, 'l_fi', '?')}-{getattr(speech, 'l_la', '?')}"
        locus = f"{work_title} ({line_range})"

        # Get speech text
        text = None
        if hasattr(speech, "l_text") and speech.l_text:
            text = speech.l_text
        elif hasattr(speech, "text") and speech.text:
            text = speech.text

        if not text:
            continue  # skip speeches without text

        summary = text[:100].replace("\n", " ") + "..."

        # CLTK NLP analysis
        doc = nlp.analyze(text)
        tokens = [w.string for w in doc.words]
        lemmas = [w.lemma for w in doc.words]

        # Extract Hector epithets / formulas
        epithets = extract_hector_epithets(doc)
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

print(f"Saved Hector speeches with epithets to {csv_file}")

# Print top 10 most frequent Hector formulas
print("\nTop 10 recurring Hector epithets/formulas:")
for epithet, freq in epithet_counter.most_common(10):
    print(f"{epithet} ({freq}×)")
