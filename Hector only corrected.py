import os, re, csv, unicodedata, sys
from typing import Any, Dict, List
from glob import glob
from lxml import etree
from cltk import NLP
from betacode.conv import beta_to_uni


# ===== CONFIG =====
GLOB_PATTERN   = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"
OUT_FILE       = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_with_morphosyntax.csv"
WINDOW         = 5      # distance window for epithets
MIN_LEN        = 3
STOP_POS       = {"PART","CCONJ","SCONJ","ADP","PRON","DET","AUX","INTJ","NUM","PUNCT","X","VERB"}
STOP_NORM      = {"ΔΕ","ΤΕ","ΑΛΛΑ","ΜΑΛΑ","ΜΗ","ΟΥ","ΟΥΚ","ΟΥΧ"}
GREEK_RE       = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

# --------- CLTK NLP ----------
nlp = NLP("grc", suppress_banner=True)


def analyze(text: str):
    doc = nlp.analyze(text)
    toks = getattr(doc, "tokens", []) or []

    tokens, pos, lemmas, tokobjs = [], [], [], []

    for t in toks:
        if isinstance(t, str):
            tokens.append(t)
            pos.append("")
            lemmas.append("")
            tokobjs.append(None)
        else:
            tokens.append(getattr(t,"string",None) or getattr(t,"text",None) or str(t))
            pos.append(getattr(t,"pos","") or getattr(t,"upos","") or "")
            lemmas.append(getattr(t,"lemma","") or "")
            tokobjs.append(t)

    if not tokens:
        tokens = text.split()
        pos = [""]*len(tokens)
        lemmas = [""]*len(tokens)
        tokobjs = [None]*len(tokens)

    return tokens, pos, lemmas, tokobjs

# --------- Normalization ----------
def normalize_greek(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς","σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+"," ", s).strip()
    return s

def normalize_tokens(tokens: List[str]) -> List[str]:
    return [normalize_greek(re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]","", t or "")) for t in tokens]

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
    # Check for 'features' or 'feats' dictionary
    features = getattr(t, "features", None) or getattr(t, "feats", None)
    if features and isinstance(features, dict):
        for k, v in features.items():
            if v:
                d[k.lower()] = str(v).lower()

    # Try morph string (format: "Case=Nom|Number=Sing|Gender=Masc")
    m = getattr(t, "morph", None)
    if m:
        for p in str(m).split("|"):
            if "=" in p:
                k, v = p.split("=", 1)
                d[k.lower()] = v.lower()

    # Try direct attributes (capitalized)
    for attr in ["Case", "Number", "Gender", "VerbForm", "Tense", "Mood", "Voice"]:
        v = getattr(t, attr, None)
        if v:
            d[attr.lower()] = str(v).lower()

    return d


def is_valid_epithet(pos: str, morph: Dict[str, str]) -> bool:
    """Check if token is an adjective or participle."""
    # Adjectives
    if pos == "ADJ":
        return True

    # Participles (might be tagged as VERB with VerbForm=Part)
    if pos == "VERB" and morph.get("verbform") == "part":
        return True

    return False


# --------- MAIN ----------
def main():
    rows = []

    for path in sorted(glob(GLOB_PATTERN)):
        for book, line, beta in iter_lines(path):
            try:
                greek = beta_to_uni(beta).strip()
            except Exception:
                greek = beta

            tokens, pos, lemmas, tokobjs = analyze(greek)
            if not tokens:
                continue
            norm = normalize_tokens(tokens)

            # Find Hector mentions
            HEKTOR = normalize_greek("Ἕκτωρ")
            hektor_indices = [i for i, n in enumerate(norm) if n == HEKTOR]
            if not hektor_indices:
                continue

            # Get Hector's morphology for agreement checking
            hektor_morphs = [morph_from_obj(tokobjs[i]) for i in hektor_indices]

            def near_hektor(i):
                return any(abs(i - h) <= WINDOW for h in hektor_indices)

            for i, tok in enumerate(tokens):
                # Skip if it's Hector itself
                if norm[i] == HEKTOR:
                    continue

                if not GREEK_RE.search(tok):
                    continue

                p = pos[i]
                n = norm[i]

                # Basic filters
                if len(n) < MIN_LEN or n in STOP_NORM:
                    continue

                # Get morphology
                tobj = tokobjs[i]
                cm = morph_from_obj(tobj)

                # ONLY accept adjectives and participles
                if not is_valid_epithet(p, cm):
                    continue

                # Must be near Hector
                if not near_hektor(i):
                    continue

                # Optional: Check case/number/gender agreement with nearest Hector
                # Uncomment to enforce agreement:
                # nearest_h = min(hektor_indices, key=lambda h: abs(i - h))
                # if not agree_cng(cm, hektor_morphs[hektor_indices.index(nearest_h)]):
                #     continue

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

    # ---------- CSV ----------
    if rows:
        os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
        with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {OUT_FILE} ({len(rows)} rows)")
    else:
        print("No output found.")
if __name__ == "__main__":
    main()

