# test_hector_epithets.py
import os, re, csv, unicodedata
from typing import Dict, List, Tuple
from lxml import etree

# ===== CONFIG =====
TREEBANK_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\tlg0012.tlg001.perseus-grc1.tb.xml"
OUT_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_with_morphosyntax.csv"
DEBUG_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_debug_filtered.csv"
WINDOW = 7  # INCREASED from 5 to 7
MIN_LEN = 3
STOP_NORM = {"ΔΕ", "ΤΕ", "ΑΛΛΑ", "ΜΑΛΑ", "ΜΗ", "ΟΥ", "ΟΥΚ", "ΟΥΧ"}
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

# ===== PERSEUS POSTAG DECODER =====
POSTAG_MAP = {
    0: {'pos': {
        'n': 'NOUN', 'v': 'VERB', 't': 'VERB', 'a': 'ADJ', 'd': 'ADV',
        'l': 'DET', 'g': 'DET', 'c': 'CONJ', 'r': 'ADP', 'p': 'PRON',
        'm': 'NUM', 'i': 'INTJ', 'e': 'INTJ', 'x': 'X'
    }},
    1: {'person': {'1': '1', '2': '2', '3': '3'}},
    2: {'number': {'s': 'Sing', 'p': 'Plur', 'd': 'Dual'}},
    3: {'tense': {'p': 'Pres', 'i': 'Impf', 'r': 'Perf', 'l': 'Plup', 't': 'Fut', 'f': 'Futperf', 'a': 'Aor'}},
    4: {'mood': {'i': 'Ind', 's': 'Sub', 'o': 'Opt', 'n': 'Inf', 'm': 'Imp', 'p': 'Part', 'g': 'Ger', 'd': 'Gerv',
                 'u': 'Sup'}},
    5: {'voice': {'a': 'Act', 'p': 'Pass', 'm': 'Mid', 'e': 'Mp'}},
    6: {'gender': {'m': 'Masc', 'f': 'Fem', 'n': 'Neut'}},
    7: {'case': {'n': 'Nom', 'g': 'Gen', 'd': 'Dat', 'a': 'Acc', 'v': 'Voc', 'l': 'Loc'}},
    8: {'degree': {'c': 'Comp', 's': 'Sup'}}
}


def decode_postag(postag: str) -> Dict[str, str]:
    """Decode Perseus 9-character morphology code."""
    morph = {}
    if not postag or len(postag) < 9:
        postag = postag + '-' * (9 - len(postag)) if postag else '---------'

    for i, char in enumerate(postag[:9]):
        if char != '-' and i in POSTAG_MAP:
            for key, mapping in POSTAG_MAP[i].items():
                if char in mapping:
                    morph[key] = mapping[char]

    return morph


# ===== NORMALIZATION =====
def normalize_greek(s: str) -> str:
    """Normalize Greek text: remove accents, uppercase."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς", "σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+", " ", s).strip()
    return s


def normalize_tokens(tokens: List[str]) -> List[str]:
    return [normalize_greek(re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]", "", t or "")) for t in tokens]


# ===== TREEBANK PARSER =====
def parse_perseus_treebank(filepath: str):
    """Parse Perseus treebank XML file, tracking line numbers for each token."""
    print(f"Loading treebank: {filepath}")
    tree = etree.parse(filepath)
    root = tree.getroot()

    sentences = root.findall('.//sentence')
    print(f"Found {len(sentences)} sentences in treebank")

    for sentence in sentences:
        doc_id = sentence.get('document_id', '')
        subdoc = sentence.get('subdoc', '')

        # Parse book and line (use first line of range)
        if subdoc:
            subdoc = subdoc.split('-')[0]
            parts = subdoc.split('.')
            book = int(parts[0]) if parts and parts[0].isdigit() else 0
            line_start = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        else:
            book, line_start = 0, 0

        tokens, lemmas, pos_tags, morphs, cites = [], [], [], [], []

        for word in sentence.findall('.//word'):
            form = word.get('form', '')
            lemma = word.get('lemma', '')
            postag = word.get('postag', '---------')
            cite = word.get('cite', '')  # e.g., "urn:cts:greekLit:tlg0012.tlg001:1.1"

            # Extract actual line number from cite
            actual_line = line_start
            if cite and ':' in cite:
                try:
                    line_ref = cite.split(':')[-1]  # "1.1"
                    if '.' in line_ref:
                        actual_line = int(line_ref.split('.')[1])
                except:
                    pass

            tokens.append(form)
            lemmas.append(lemma)
            cites.append(actual_line)

            morph = decode_postag(postag)
            pos_tags.append(morph.get('pos', ''))
            morphs.append(morph)

        if tokens:
            yield book, line_start, tokens, lemmas, pos_tags, morphs, cites


# ===== EPITHET VALIDATION =====
def is_valid_epithet(pos: str, morph: Dict[str, str]) -> bool:
    """Check if token is an adjective or participle."""
    if not pos:
        return False

    # Adjectives
    if pos == 'ADJ':
        return True

    # Participles
    if pos == 'VERB' and morph.get('mood') == 'Part':
        return True

    return False


def agree_cng(m1: Dict[str, str], m2: Dict[str, str]) -> bool:
    """Check case/number/gender agreement - RELAXED version."""
    if not m1 or not m2:
        return False

    # Count matching features
    matches = 0
    total = 0

    for feature in ['case', 'number', 'gender']:
        val1 = m1.get(feature)
        val2 = m2.get(feature)

        if val1 and val2:
            total += 1
            if val1 == val2:
                matches += 1
            else:
                # If they conflict, it's not agreement
                return False

    # Need at least 2 matching features, OR all features present match
    return matches >= 2 or (total > 0 and matches == total)


def morph_field(m: Dict[str, str], k: str) -> str:
    return m.get(k, '')


# ===== MAIN =====
def main():
    rows = []
    debug_rows = []  # Track what gets filtered

    lines_processed = 0
    lines_with_hector = 0

    for book, line_start, tokens, lemmas, pos_tags, morphs, cites in parse_perseus_treebank(TREEBANK_FILE):
        lines_processed += 1

        norm = normalize_tokens(tokens)

        # Find Hector
        HEKTOR = normalize_greek("Ἕκτωρ")
        hektor_indices = [i for i, n in enumerate(norm) if n == HEKTOR]

        if not hektor_indices:
            continue

        lines_with_hector += 1

        # Debug first few
        if lines_with_hector <= 5:
            print(f"\n{'=' * 70}")
            print(f"Book {book}, Lines {line_start}")
            print(f"Hector at token positions: {hektor_indices}")
            print(f"\nTokens (showing line numbers):")
            for i in range(min(30, len(tokens))):
                near = "★" if any(abs(i - h) <= WINDOW for h in hektor_indices) else " "
                is_hektor = "← HECTOR" if i in hektor_indices else ""
                print(
                    f"  {near} [{i:2d}] Line {cites[i]:3d} | {tokens[i]:20s} | POS={pos_tags[i]:8s} | {morphs[i]} {is_hektor}")

        def near_hektor(i):
            return any(abs(i - h) <= WINDOW for h in hektor_indices)

        # ===== COLLECT EPITHET CANDIDATES =====
        epithet_indices = []

        for i in range(len(tokens)):
            if i in hektor_indices:
                continue

            if not GREEK_RE.search(tokens[i]):
                continue

            n = norm[i]
            if len(n) < MIN_LEN or n in STOP_NORM:
                continue

            # Check if near Hector
            if not near_hektor(i):
                continue

            # Check POS
            if not is_valid_epithet(pos_tags[i], morphs[i]):
                # LOG what was filtered by POS
                debug_rows.append({
                    "Book": book,
                    "Line": cites[i],
                    "Token": tokens[i],
                    "Lemma": lemmas[i],
                    "POS": pos_tags[i],
                    "Morph": str(morphs[i]),
                    "FilterReason": "POS_not_ADJ_or_Part"
                })
                continue

            # Check agreement
            nearest_h = min(hektor_indices, key=lambda h: abs(i - h))
            hektor_morph = morphs[nearest_h]
            epithet_morph = morphs[i]

            if not agree_cng(epithet_morph, hektor_morph):
                # LOG what was filtered by agreement
                debug_rows.append({
                    "Book": book,
                    "Line": cites[i],
                    "Token": tokens[i],
                    "Lemma": lemmas[i],
                    "POS": pos_tags[i],
                    "Morph": str(morphs[i]),
                    "FilterReason": f"Agreement_fail (Hektor:{hektor_morph} vs Epithet:{epithet_morph})"
                })
                continue

            epithet_indices.append(i)

        # ===== GROUP CONSECUTIVE EPITHETS =====
        used = set()

        for i in epithet_indices:
            if i in used:
                continue

            # Build multi-word epithet
            epithet_tokens = [tokens[i]]
            epithet_lemmas = [lemmas[i]]
            epithet_pos = [pos_tags[i]]
            morph_list = [morphs[i]]
            indices = [i]
            epithet_line = cites[i]  # Line where epithet appears

            # Look for consecutive epithets
            j = i + 1
            while j in epithet_indices:
                epithet_tokens.append(tokens[j])
                epithet_lemmas.append(lemmas[j])
                epithet_pos.append(pos_tags[j])
                morph_list.append(morphs[j])
                indices.append(j)
                used.add(j)
                j += 1

            used.add(i)

            # Combine
            epithet_text = " ".join(epithet_tokens)
            epithet_lemma = " ".join(epithet_lemmas)
            epithet_pos_str = "+".join(epithet_pos)

            cm = morph_list[0]

            # FIXED: Show only tokens from the same line as the epithet
            same_line_tokens = [tokens[idx] for idx in range(len(tokens)) if cites[idx] == epithet_line]
            greek_context = " ".join(same_line_tokens)

            rows.append({
                "Book": book,
                "Line": epithet_line,  # Use actual line number, not sentence start
                "Hero": "HEKTOR",
                "Epithet": epithet_text,
                "Lemma": epithet_lemma,
                "POS": epithet_pos_str,
                "WordCount": len(epithet_tokens),
                "Case": morph_field(cm, "case"),
                "Number": morph_field(cm, "number"),
                "Gender": morph_field(cm, "gender"),
                "Tense": morph_field(cm, "tense"),
                "Voice": morph_field(cm, "voice"),
                "Mood": morph_field(cm, "mood"),
                "Greek": greek_context  # Only the line where epithet appears
            })

    # ===== OUTPUT =====
    print(f"\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"  Lines processed: {lines_processed}")
    print(f"  Lines with Hector: {lines_with_hector}")
    print(f"  Epithets found: {len(rows)}")
    print(f"  Tokens filtered out: {len(debug_rows)}")

    # Save main results
    if rows:
        os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
        with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {OUT_FILE}")

        print(f"\nSample epithets (first 15):")
        for row in rows[:15]:
            print(f"  Book {row['Book']}.{row['Line']}: {row['Epithet']} ({row['Lemma']})")

    # Save debug file
    if debug_rows:
        with open(DEBUG_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)
        print(f"\nWrote debug file: {DEBUG_FILE}")
        print(f"Check this file to see what epithets are being filtered (like ἀνδροφόνος, ἱππόδαμος)")


if __name__ == "__main__":
    main()