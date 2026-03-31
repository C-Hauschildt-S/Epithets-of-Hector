"""
Sanity check: find ALL sentences in the treebank where Hektor is mentioned
by name (Ἕκτωρ or Πριαμίδης), and compare against the epithet output CSV.
Reports any sentences where Hektor appears but no epithet row was generated.
"""
import csv
import unicodedata
import sys

# --- Reuse key functions from main script ---
sys.path.insert(0, r"C:\Users\carol\PycharmProjects\CLTK+DICES")

TREEBANK_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\tlg0012.tlg001.perseus-grc1.tb.xml"
OUT_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_morphosyntax_v2.csv"

def normalize_greek(s):
    nfkd = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in nfkd if unicodedata.category(c) not in ("Mn",))
    return stripped.upper()

HEKTOR_NORM = normalize_greek("Ἕκτωρ")
PRIAMIDES_NORM = normalize_greek("Πριαμίδης")
HEKTOR_LEMMAS = {HEKTOR_NORM}

# --- Parse treebank for ALL Hektor mentions ---
import xml.etree.ElementTree as ET

tree = ET.parse(TREEBANK_FILE)
root = tree.getroot()

treebank_mentions = []  # (book, line_start, line_end, hektor_forms, full_sentence)

for sentence in root.iter("sentence"):
    subdoc = sentence.get("subdoc", "")
    if not subdoc:
        continue

    parts = subdoc.split(".")
    book = int(parts[0])

    # Parse line range
    line_part = parts[1] if len(parts) > 1 else "0"
    if "-" in line_part:
        ls, le = line_part.split("-")
        line_start, line_end = int(ls), int(le)
    else:
        line_start = int(line_part)
        line_end = line_start

    words = sentence.findall("word")
    tokens = [w.get("form", "") for w in words]
    lemmas = [w.get("lemma", "") for w in words]

    hektor_forms = []
    for i, (tok, lem) in enumerate(zip(tokens, lemmas)):
        nt = normalize_greek(tok)
        nl = normalize_greek(lem)
        if nt in HEKTOR_LEMMAS or nl in HEKTOR_LEMMAS:
            hektor_forms.append(tok)

    if hektor_forms:
        treebank_mentions.append({
            "book": book,
            "line_start": line_start,
            "line_end": line_end,
            "hektor_forms": hektor_forms,
            "sentence": " ".join(tokens),
        })

# --- Load output CSV ---
output_lines = set()
with open(OUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        output_lines.add((int(row["Book"]), int(row["Line"])))

# --- Compare ---
total_word_occurrences = sum(len(m["hektor_forms"]) for m in treebank_mentions)
print(f"Treebank: {len(treebank_mentions)} sentences with Hektor mention ({total_word_occurrences} total word occurrences)")
print(f"Output CSV: {len(output_lines)} unique (Book, Line) pairs with epithets")
print()

covered = 0
not_covered = []

for m in treebank_mentions:
    # Check if ANY line in the sentence range has an epithet row
    found = False
    for line in range(m["line_start"], m["line_end"] + 1):
        if (m["book"], line) in output_lines:
            found = True
            break

    if found:
        covered += 1
    else:
        not_covered.append(m)

not_covered_count = len(not_covered)
print(f"Sentences WITH epithet rows: {covered}")
print(f"Sentences WITHOUT epithet rows: {not_covered_count}")
print()
print(f"=== {not_covered_count} Hektor mentions with NO epithet in output ===")
for m in not_covered:
    forms = ", ".join(m["hektor_forms"])
    print(f"  {m['book']}.{m['line_start']}-{m['line_end']}  [{forms}]  {m['sentence'][:120]}")