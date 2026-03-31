import os, re, csv, unicodedata, sys
from typing import Any, Dict, List
from glob import glob
from lxml import etree
from cltk import NLP
from betacode.conv import beta_to_uni

# ===== CONFIG =====
GLOB_PATTERN = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"
OUT_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_with_morphosyntax.csv"
WINDOW = 5  # distance window for epithets
MIN_LEN = 3
STOP_POS = {"PART", "CCONJ", "SCONJ", "ADP", "PRON", "DET", "AUX", "INTJ", "NUM", "PUNCT", "X"}
STOP_NORM = {"ΔΕ", "ΤΕ", "ΑΛΛΑ", "ΜΑΛΑ", "ΜΗ", "ΟΥ", "ΟΥΚ", "ΟΥΧ"}
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

# --------- CLTK NLP ----------
print("Initializing CLTK...")
nlp = NLP(language="grc", suppress_banner=True)
print("CLTK initialized.")


def analyze(text: str):
    """Analyze Greek text with CLTK."""
    try:
        doc = nlp.analyze(text)
    except Exception as e:
        print(f"CLTK analysis failed: {e}")
        tokens = text.split()
        return tokens, [""] * len(tokens), [""] * len(tokens), [None] * len(tokens)

    # Try to extract tokens from the doc
    words = getattr(doc, "words", None)
    if words:
        tokens = [str(w) for w in words]
        pos = [getattr(w, "pos", "") or getattr(w, "upos", "") or getattr(w, "xpos", "") or "" for w in words]
        lemmas = [getattr(w, "lemma", "") or "" for w in words]
        tokobjs = words
    else:
        # Fallback to tokens attribute
        toks = getattr(doc, "tokens", []) or []
        tokens, pos, lemmas, tokobjs = [], [], [], []

        for t in toks:
            if isinstance(t, str):
                tokens.append(t)
                pos.append("")
                lemmas.append("")
                tokobjs.append(None)
            else:
                tokens.append(getattr(t, "string", None) or getattr(t, "text", None) or str(t))
                pos.append(getattr(t, "pos", "") or getattr(t, "upos", "") or getattr(t, "xpos", "") or "")
                lemmas.append(getattr(t, "lemma", "") or "")
                tokobjs.append(t)

    if not tokens:
        tokens = text.split()
        pos = [""] * len(tokens)
        lemmas = [""] * len(tokens)
        tokobjs = [None] * len(tokens)

    return tokens, pos, lemmas, tokobjs


# --------- Normalization ----------
def normalize_greek(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς", "σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+", " ", s).strip()
    return s


def normalize_tokens(tokens: List[str]) -> List[str]:
    return [normalize_greek(re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]", "", t or "")) for t in tokens]


# --------- TEI parser ----------
def iter_lines(xml_path: str):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    for div in root.findall(".//div1"):
        book = int(div.get("n") or 0)
        ln = 0
        for l in div.findall(".//l"):
            ln += 1
            beta = "".join(l.itertext()).strip()
            if beta:
                yield book, ln, beta


# --------- Morph helpers ----------
def morph_from_obj(t: Any) -> Dict[str, str]:
    """Extract morphological features from CLTK token object."""
    d = {}
    if not t:
        return d

    # Try different CLTK attribute structures
    features = getattr(t, "features", None) or getattr(t, "feats", None)
    if features:
        if isinstance(features, dict):
            for k, v in features.items():
                if v:
                    d[k.lower()] = str(v).lower()
        elif isinstance(features, str):
            # Parse "Case=Nom|Number=Sing" format
            for p in features.split("|"):
                if "=" in p:
                    k, v = p.split("=", 1)
                    d[k.lower()] = v.lower()

    # Try morph string
    m = getattr(t, "morph", None)
    if m:
        for p in str(m).split("|"):
            if "=" in p:
                k, v = p.split("=", 1)
                d[k.lower()] = v.lower()

    # Try direct attributes
    for attr in ["Case", "Number", "Gender", "VerbForm", "Tense", "Mood", "Voice"]:
        v = getattr(t, attr, None)
        if v:
            d[attr.lower()] = str(v).lower()

    return d


def morph_field(m: Dict[str, str], k: str) -> str:
    """Get a morphology field value."""
    return m.get(k, "")


def agree_cng(m1: Dict[str, str], m2: Dict[str, str]) -> bool:
    """Check case/number/gender agreement."""
    if not m1 or not m2:
        return False
    ok = True
    if "case" in m1 and "case" in m2:
        ok &= (m1["case"] == m2["case"])
    if "number" in m1 and "number" in m2:
        ok &= (m1["number"] == m2["number"])
    if "gender" in m1 and "gender" in m2:
        ok &= (m1["gender"] == m2["gender"])
    return ok


def is_valid_epithet(pos: str, morph: Dict[str, str]) -> bool:
    """Check if token is an adjective or participle."""
    if not pos:  # If POS is empty, we can't validate
        return False

    # Adjectives
    if pos in ["ADJ", "A"]:
        return True

    # Participles (might be tagged as VERB with VerbForm=Part)
    if pos in ["VERB", "V"] and morph.get("verbform") == "part":
        return True

    return False


# --------- MAIN ----------
def main():
    rows = []

    # DEBUG
    debug_count = 0
    max_debug = 10
    lines_processed = 0
    lines_with_hector = 0

    # Test normalization
    HEKTOR = normalize_greek("Ἕκτωρ")
    print(f"Looking for HEKTOR normalized as: '{HEKTOR}'")
    print(f"Also checking variants...")

    # Track what normalized words we're seeing
    all_normalized_words = set()

    for path in sorted(glob(GLOB_PATTERN)):
        print(f"\nProcessing: {path}")

        for book, line, beta in iter_lines(path):
            lines_processed += 1

            # Show some sample lines regardless of Hector
            if debug_count < max_debug:
                print(f"\n--- Book {book}, Line {line} ---")
                print(f"Beta code: {beta}")

            try:
                greek = beta_to_uni(beta).strip()
            except Exception as e:
                print(f"Beta code conversion failed: {e}")
                continue

            if debug_count < max_debug:
                print(f"Converted: {greek}")

            tokens, pos, lemmas, tokobjs = analyze(greek)
            if not tokens:
                continue
            norm = normalize_tokens(tokens)

            if debug_count < max_debug:
                print(f"Tokens: {tokens[:10]}")
                print(f"Normalized: {norm[:10]}")
                debug_count += 1

            # Collect all normalized words to see what's in the text
            all_normalized_words.update(norm)

            # Find Hector mentions - try multiple variants
            hektor_variants = [
                normalize_greek("Ἕκτωρ"),  # Hektor
                normalize_greek("ἕκτωρ"),  # hektor (lowercase)
                normalize_greek("Ἕκτορ"),  # without final ρ
                normalize_greek("Ἑκτωρ"),  # different breathing
                "ΕΚΤΩΡ", "ΕΚΤΟΡ", "ΗΕΚΤΩΡ"  # various forms
            ]

            hektor_indices = []
            for variant in hektor_variants:
                hektor_indices.extend([i for i, n in enumerate(norm) if n == variant])

            hektor_indices = list(set(hektor_indices))  # Remove duplicates

            if not hektor_indices:
                # Also check if any token CONTAINS the name
                for i, n in enumerate(norm):
                    if "ΕΚΤΩΡ" in n or "ΕΚΤΟΡ" in n:
                        print(
                            f"\n!!! FOUND HECTOR-LIKE WORD: '{tokens[i]}' normalized to '{n}' at Book {book}, Line {line}")
                        hektor_indices.append(i)

            if not hektor_indices:
                continue

            lines_with_hector += 1
            print(f"\n★★★ FOUND HECTOR at Book {book}, Line {line}: {greek}")
            print(f"    Positions: {hektor_indices}, tokens: {[tokens[i] for i in hektor_indices]}")

            def near_hektor(i):
                return any(abs(i - h) <= WINDOW for h in hektor_indices)

            for i, tok in enumerate(tokens):
                if i in hektor_indices:
                    continue

                if not GREEK_RE.search(tok):
                    continue

                p = pos[i]
                n = norm[i]

                if len(n) < MIN_LEN or n in STOP_NORM:
                    continue

                tobj = tokobjs[i]
                cm = morph_from_obj(tobj)

                if not near_hektor(i):
                    continue

                rows.append({
                    "Book": book,
                    "Line": line,
                    "Hero": "HEKTOR",
                    "Epithet": tok,
                    "Lemma": lemmas[i],
                    "POS": p,
                    "Case": morph_field(cm, "case"),
                    "Number": morph_field(cm, "number"),
                    "Gender": morph_field(cm, "gender"),
                    "VerbForm": morph_field(cm, "verbform"),
                    "Tense": morph_field(cm, "tense"),
                    "Voice": morph_field(cm, "voice"),
                    "Mood": morph_field(cm, "mood"),
                    "Greek": greek
                })

            if lines_with_hector >= 5:
                break  # Stop after finding 5 Hector lines for debugging

        if lines_with_hector >= 5:
            break

    # ---------- Summary ----------
    print(f"\n\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"  Lines processed: {lines_processed}")
    print(f"  Lines with Hector: {lines_with_hector}")
    print(f"  Potential epithets found: {len(rows)}")

    # Show sample normalized words
    print(f"\nSample normalized words from text (first 50):")
    for w in sorted(list(all_normalized_words))[:50]:
        if w:  # Skip empty strings
            print(f"  '{w}'")

    if rows:
        os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
        with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {OUT_FILE} ({len(rows)} rows)")
    else:
        print("\nNo epithets found!")


if __name__ == "__main__":
    main()
