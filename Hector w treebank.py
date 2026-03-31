import os, re, csv, unicodedata, glob
from typing import Dict, List, Tuple
from lxml import etree
import betacode.conv as beta_conv

# ===== CONFIG =====
TREEBANK_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\tlg0012.tlg001.perseus-grc1.tb.xml"
WHITELIST_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\epithet_whitelist.csv"
OUT_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_morphosyntax_v2.csv"
CANDIDATES_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\epithet_candidates.csv"
ILIAD_XML_DIR = r"C:\Users\carol\PycharmProjects\CLTK+DICES"
USE_DICES = True
WINDOW = 5
MIN_LEN = 3
COMBO_SPAN_MAX = 8
STOP_NORM = {"ΔΕ", "ΤΕ", "ΑΛΛΑ", "ΜΑΛΑ", "ΜΗ", "ΟΥ", "ΟΥΚ", "ΟΥΧ"}
STOP_LEM = {"ΜΗΝΙΣ", "ΑΕΙΔΩ", "ΘΕΑ", "ΜΟΥΣΑ", "ΑΙΔΗΣ",
            "ΙΣΟΣ", "ΕΙΚΕΛΟΣ", "ΚΑΚΟΣ", "ΠΟΛΥΣ", "ΖΩΟΣ"}
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
PATR_SUFFIXES = ("ΙΔΗΣ", "ΙΑΔΗΣ", "ΪΔΗΣ", "ΪΑΔΗΣ")

PARTICIPLE_BLACKLIST_FILE = r"C:\Users\carol\PycharmProjects\CLTK+DICES\participle_blacklist.csv"

# False-positive Πριαμίδης references (other sons of Priam, not Hektor)
FALSE_POSITIVE_PRIAMIDES = {
    (13, 586),  # Helenos
    (20, 87),   # Lykaon
}

# Periphrastic Hektor references that treebank parses as modifying another noun
FORCE_INCLUDE_PRIAMIDES = {
    (7, 250),   # Πριαμίδαο κατ' ἀσπίδα — Hektor's shield
}

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
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς", "σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+", " ", s).strip()
    return s


def normalize_tokens(tokens: List[str]) -> List[str]:
    return [normalize_greek(re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]", "", t or "")) for t in tokens]


# ===== VERSE LINE INDEX (from TEI XML) =====
def build_line_index(xml_dir: str) -> Dict[tuple, List[str]]:
    """Parse iliad_book_XX.xml files to build (book, line) -> [normalized words] index."""
    index = {}
    for book_num in range(1, 25):
        filepath = os.path.join(xml_dir, f"iliad_book_{book_num:02d}.xml")
        if not os.path.exists(filepath):
            print(f"[WARN] Missing: {filepath}")
            continue
        tree = etree.parse(filepath)
        # Find all <l> elements — each is one verse line
        lines = tree.findall('.//{*}l')
        if not lines:
            lines = tree.findall('.//l')
        line_num = 0
        for l_elem in lines:
            line_num += 1
            # Use explicit n= attribute if present, otherwise count sequentially
            n_attr = l_elem.get('n')
            if n_attr and n_attr.isdigit():
                line_num = int(n_attr)
            # Get raw text content (strip nested elements like milestone)
            raw = ''.join(l_elem.itertext()).strip()
            # Convert Beta Code to Unicode
            try:
                uni = beta_conv.beta_to_uni(raw)
            except Exception:
                uni = raw
            # Normalize and tokenize
            words = [normalize_greek(w) for w in uni.split() if w.strip()]
            words = [w for w in words if w]
            index[(book_num, line_num)] = words
    print(f"Line index built: {len(index)} verse lines across 24 books")
    return index


def find_word_line(book: int, sent_start: int, sent_end: int,
                   word_form: str, word_idx: int, sent_len: int,
                   line_index: Dict[tuple, List[str]]) -> int:
    """Find the exact verse line for a word within a sentence spanning sent_start..sent_end."""
    norm_word = normalize_greek(word_form)
    if not norm_word:
        return sent_start

    # Search each line in the sentence's range for this word
    for line_num in range(sent_start, sent_end + 1):
        words_on_line = line_index.get((book, line_num), [])
        if norm_word in words_on_line:
            return line_num

    # Fallback: estimate by position ratio
    if sent_end > sent_start and sent_len > 1:
        ratio = word_idx / max(sent_len - 1, 1)
        estimated = sent_start + round(ratio * (sent_end - sent_start))
        return min(max(estimated, sent_start), sent_end)

    return sent_start


# ===== WHITELIST LOADING =====
def load_whitelist(filepath: str) -> Tuple[Dict[str, dict], List[Tuple[List[str], dict]]]:
    """Load epithet whitelist. Returns (single_word_dict, multi_word_list).
    single_word_dict: normalized_lemma -> {type, cluster, notes}
    multi_word_list: [(normalized_lemma_parts, {type, cluster, notes}), ...]
    """
    singles = {}
    multis = []

    if not os.path.exists(filepath):
        print(f"[WARN] Whitelist not found: {filepath}")
        return singles, multis

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = (row.get("lemma") or "").strip()
            if not lemma:
                continue
            info = {
                "type": (row.get("type") or "").strip(),
                "cluster": (row.get("cluster") or "").strip(),
                "notes": (row.get("notes") or "").strip(),
            }
            norm = normalize_greek(lemma)
            parts = norm.split()
            if len(parts) == 1:
                singles[norm] = info
            else:
                multis.append((parts, info))

    # Sort longest first so longer formulas match before shorter subsets
    multis.sort(key=lambda x: len(x[0]), reverse=True)
    print(f"Whitelist loaded: {len(singles)} single-word, {len(multis)} multi-word")
    return singles, multis


def load_participle_blacklist(filepath: str) -> set:
    """Load blacklisted participle lemmas (verbs that are never epithets)."""
    bl = set()
    if not os.path.exists(filepath):
        print(f"[WARN] Participle blacklist not found: {filepath}")
        return bl
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = (row.get("lemma") or "").strip()
            if lemma:
                bl.add(normalize_greek(lemma))
    print(f"Participle blacklist loaded: {len(bl)} lemmas")
    return bl


# ===== TREEBANK PARSER =====
def parse_perseus_treebank(filepath: str):
    print(f"Loading treebank: {filepath}")
    tree = etree.parse(filepath)
    root = tree.getroot()
    sentences = root.findall('.//sentence')
    print(f"Found {len(sentences)} sentences in treebank")

    for sentence in sentences:
        subdoc = sentence.get('subdoc', '')
        book, line_start, line_end = 0, 0, 0
        if subdoc:
            span_parts = subdoc.split('-')
            start_parts = span_parts[0].split('.')
            book = int(start_parts[0]) if start_parts and start_parts[0].isdigit() else 0
            line_start = int(start_parts[1]) if len(start_parts) > 1 and start_parts[1].isdigit() else 0
            if len(span_parts) > 1:
                end_parts = span_parts[1].split('.')
                if len(end_parts) == 2:
                    line_end = int(end_parts[1]) if end_parts[1].isdigit() else line_start
                elif len(end_parts) == 1:
                    line_end = int(end_parts[0]) if end_parts[0].isdigit() else line_start
            else:
                line_end = line_start

        tokens, lemmas, pos_tags, morphs, heads, rels = [], [], [], [], [], []
        for word in sentence.findall('.//word'):
            form = word.get('form', '')
            lemma = word.get('lemma', '')
            postag = word.get('postag', '---------')
            head = word.get('head', '0')
            relation = word.get('relation', '')
            tokens.append(form)
            lemmas.append(lemma)
            morph = decode_postag(postag)
            pos_tags.append(morph.get('pos', ''))
            morphs.append(morph)
            # head is 1-indexed word id; convert to 0-indexed token position
            heads.append(int(head) - 1 if head.isdigit() and int(head) > 0 else -1)
            rels.append(relation)

        if tokens:
            yield book, line_start, line_end, tokens, lemmas, pos_tags, morphs, heads, rels


# ===== DICES INTEGRATION =====
def build_speech_index():
    if not USE_DICES:
        return []
    try:
        from dicesapi import DicesAPI
        api = DicesAPI()
        speeches = api.getSpeeches().advancedFilter(
            lambda s: getattr(s.work, "title", "").lower() in {"iliad", "the iliad"}
        )
    except Exception as e:
        print(f"[WARN] DICES off: {e}")
        return []

    rx = [
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)-(?P<b2>\d+)\.(?P<l2>\d+)$"),
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)-(?P<l2>\d+)$"),
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)$"),
    ]
    idx = []
    for s in speeches:
        spk = [getattr(p, "name", str(p)) for p in getattr(s, "spkr", [])]
        adr = [getattr(p, "name", str(p)) for p in getattr(s, "addr", [])]
        urn = getattr(s, "urn", "") or ""
        b1 = l1 = b2 = l2 = None
        for r in rx:
            m = r.search(urn)
            if m:
                gd = m.groupdict()
                b1 = int(gd["b1"]); l1 = int(gd["l1"])
                b2 = int(gd.get("b2") or gd["b1"])
                l2 = int(gd.get("l2") or gd["l1"])
                break
        idx.append({"b1": b1, "l1": l1, "b2": b2, "l2": l2, "sp": spk, "ad": adr})
    print(f"DICES: loaded {len(idx)} speech ranges")
    return idx


def line_in_range(book, line, b1, l1, b2, l2):
    if None in (b1, l1, b2, l2):
        return False
    if b1 == b2 == book:
        return l1 <= line <= l2
    if book < b1 or book > b2:
        return False
    if book == b1 and line < l1:
        return False
    if book == b2 and line > l2:
        return False
    return True


def lookup_speech(idx, book, line):
    if not idx:
        return "", ""
    matches = [d for d in idx if line_in_range(book, line, d["b1"], d["l1"], d["b2"], d["l2"])]
    if not matches:
        return "", ""
    matches.sort(key=lambda d: (d["b2"] - d["b1"]) * 10000 + (d["l2"] - d["l1"]))
    best = matches[0]
    return "; ".join(best["sp"]) if best["sp"] else "", "; ".join(best["ad"]) if best["ad"] else ""


# ===== MORPHOLOGICAL HELPERS =====
def agree_cng(m1: Dict[str, str], m2: Dict[str, str]) -> bool:
    if not m1 or not m2:
        return False
    for feature in ['case', 'number', 'gender']:
        val1 = m1.get(feature)
        val2 = m2.get(feature)
        if val1 and val2 and val1 != val2:
            return False
    return any(m1.get(f) and m2.get(f) and m1[f] == m2[f] for f in ['case', 'number', 'gender'])


def is_valid_epithet(pos: str, morph: Dict[str, str], norm_lemma: str,
                     relation: str, part_blacklist: set) -> bool:
    # Blacklist check applies regardless of POS (some participles are tagged as ADJ)
    if norm_lemma in part_blacklist:
        return False
    if pos == 'ADJ':
        return True
    # Participles: must be attributive (ATR)
    if pos == 'VERB' and morph.get('mood') == 'Part':
        if relation == 'ATR':
            return True
    return False


def is_patronymic(n: str) -> bool:
    return any(n.endswith(s) for s in PATR_SUFFIXES)


def morph_field(m: Dict[str, str], k: str) -> str:
    return m.get(k, '')


# ===== EPITHET CLUSTER ASSIGNMENT =====
def assign_clusters(rows):
    """Assign cluster IDs to epithets that co-occur in the same sentence."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["_sent_id"]].append(i)

    cluster_id = 0
    for sent_id, indices in groups.items():
        if len(indices) >= 2:
            cluster_id += 1
            # Use first row's Book.Line as label
            first = rows[indices[0]]
            tag = f"{first['Book']}.{first['Line']}_C{cluster_id}"
            for idx in indices:
                rows[idx]["EpithetCluster"] = tag
        else:
            rows[indices[0]]["EpithetCluster"] = ""
    # Remove internal field
    for r in rows:
        del r["_sent_id"]


# ===== MULTI-WORD FORMULA MATCHING =====
def find_whitelist_formulas(norm_lemmas: List[str], multi_whitelist, tokens, lemmas):
    """Find multi-word whitelist entries by matching lemma OR surface sets within a window."""
    norm_surface = [normalize_greek(t) for t in tokens]
    results = []
    used_positions = set()
    for parts, info in multi_whitelist:
        part_set = set(parts)
        plen = len(parts)
        # Slide a window of up to COMBO_SPAN_MAX tokens
        for i in range(len(norm_lemmas)):
            end = min(i + COMBO_SPAN_MAX, len(norm_lemmas))
            # Collect matching positions (by lemma or surface) within the window
            matched = []
            remaining = list(parts)
            for j in range(i, end):
                if j in used_positions:
                    continue
                for k, p in enumerate(remaining):
                    if norm_lemmas[j] == p or norm_surface[j] == p:
                        matched.append(j)
                        remaining.pop(k)
                        break
                if not remaining:
                    break
            if not remaining and len(matched) == plen:
                matched.sort()
                phrase = " ".join(tokens[m] for m in matched)
                lemma_phrase = " ".join(lemmas[m] for m in matched)
                for m in matched:
                    used_positions.add(m)
                results.append({
                    "start": matched[0], "end": matched[-1],
                    "positions": matched,
                    "text": phrase, "lemma": lemma_phrase,
                    "info": info, "source": "whitelist",
                })
    return results


# ===== MAIN =====
def main():
    # Load whitelist and blacklist
    wl_singles, wl_multis = load_whitelist(WHITELIST_FILE)
    part_blacklist = load_participle_blacklist(PARTICIPLE_BLACKLIST_FILE)

    # Load DICES speech index
    print("Loading DICES speech data...")
    speech_index = build_speech_index()

    # Load verse line index from TEI XML files
    print("Building verse line index from XML files...")
    line_index = build_line_index(ILIAD_XML_DIR)

    rows = []
    candidates = []
    HEKTOR_NORM = normalize_greek("Ἕκτωρ")
    HEKTOR_LEMMAS = {HEKTOR_NORM, normalize_greek("Πριαμίδης")}

    for book, line_start, line_end, tokens, lemmas, pos_tags, morphs, heads, rels in parse_perseus_treebank(TREEBANK_FILE):
        norm = normalize_tokens(tokens)
        norm_lemmas = [normalize_greek(l) for l in lemmas]

        # Find Hector mentions (by surface or lemma)
        hektor_indices = []
        for i in range(len(tokens)):
            if norm[i] in HEKTOR_LEMMAS or norm_lemmas[i] in HEKTOR_LEMMAS:
                hektor_indices.append(i)

        if not hektor_indices:
            continue

        # DICES lookup
        speaker, addressee = lookup_speech(speech_index, book, line_start)

        hektor_set = set(hektor_indices)

        # Determine confidence: "high" if Ἕκτωρ explicitly present, "review" if only Πριαμίδης
        PRIAMIDES_NORM = normalize_greek("Πριαμίδης")
        has_explicit_hektor = any(
            norm[hi] == HEKTOR_NORM or norm_lemmas[hi] == HEKTOR_NORM
            for hi in hektor_indices
        )
        confidence = "high" if has_explicit_hektor else "review"

        # Πριαμίδης is both an identifier AND a patronymic epithet — record it
        for hi in hektor_indices:
            nl_hi = norm_lemmas[hi]
            n_hi = norm[hi]
            if nl_hi == PRIAMIDES_NORM or n_hi == PRIAMIDES_NORM:
                # Skip if this patronymic syntactically modifies a non-Hektor noun
                # e.g. "Πριαμίδης Ἕλενος" at 6.76 where ATR points to Helenos
                word_line = find_word_line(book, line_start, line_end, tokens[hi], hi, len(tokens), line_index)
                # Skip if this patronymic syntactically modifies a non-Hektor noun
                # e.g. "Πριαμίδης Ἕλενος" at 6.76 where ATR points to Helenos
                # (but allow known periphrastic references like Πριαμίδαο κατ' ἀσπίδα)
                if (book, word_line) not in FORCE_INCLUDE_PRIAMIDES:
                    parent_hi = heads[hi]
                    if 0 <= parent_hi < len(tokens) and parent_hi not in hektor_set:
                        if rels[hi] in ('ATR', 'ATV', 'ATV_CO', 'APOS') and pos_tags[parent_hi] in ('NOUN', 'PRON', 'ADJ'):
                            continue
                # Skip known false positives (other Priamids)
                if (book, word_line) in FALSE_POSITIVE_PRIAMIDES:
                    continue
                rows.append({
                    "_sent_id": (book, line_start),
                    "Book": book, "Line": word_line,
                    "Hero": "HEKTOR",
                    "Epithet": tokens[hi], "Lemma": lemmas[hi],
                    "EpithetCluster": "",
                    "Speaker": speaker, "Addressee": addressee,
                    "Case": morph_field(morphs[hi], "case"),
                    "Number": morph_field(morphs[hi], "number"),
                    "Gender": morph_field(morphs[hi], "gender"),
                    "Tense": "", "Voice": "", "Mood": "",
                    "POS": pos_tags[hi],
                    "Source": "whitelist",
                    "Confidence": confidence,
                    "Greek": " ".join(tokens),
                })

        def near_hektor(idx):
            return any(abs(idx - h) <= WINDOW for h in hektor_indices)

        def syntactically_linked_to_hektor(idx):
            """Check if token at idx modifies Hektor via dependency tree."""
            # Direct: this word's head is Hektor
            if heads[idx] in hektor_set:
                return True
            # Reverse: Hektor's head is this word (e.g. predicate adjective)
            if any(heads[h] == idx for h in hektor_indices):
                return True
            # One step up: this word's head's head is Hektor (e.g. through conjunction)
            parent = heads[idx]
            if 0 <= parent < len(heads) and heads[parent] in hektor_set:
                return True
            return False

        def modifies_other_noun(idx):
            """Check if token at idx syntactically modifies a non-Hektor noun."""
            parent = heads[idx]
            if parent < 0 or parent >= len(tokens):
                return False
            # If this word is ATR to a noun/proper noun that is NOT Hektor
            if rels[idx] in ('ATR', 'ATV', 'ATV_CO') and parent not in hektor_set:
                parent_pos = pos_tags[parent]
                if parent_pos in ('NOUN', 'PRON', 'ADJ'):
                    return True
            return False

        # ===== 1) MULTI-WORD WHITELIST MATCHES =====
        used_positions = set()
        multi_hits = find_whitelist_formulas(norm_lemmas, wl_multis, tokens, lemmas)

        def formula_modifies_other_person(positions):
            """Check if any word in the formula syntactically depends on a non-Hektor
            named entity, or shares an apposition node with one."""
            pos_set = set(positions)
            for p in positions:
                parent = heads[p]
                if parent < 0 or parent >= len(tokens):
                    continue
                if parent in hektor_set or parent in pos_set:
                    continue
                # Direct: parent is a non-Hektor noun
                if pos_tags[parent] in ('NOUN', 'PRON') and norm_lemmas[parent] not in STOP_NORM:
                    return True
                if rels[p] in ('ATR', 'APOS', 'ADV_AP', 'OBJ_AP'):
                    if pos_tags[parent] in ('NOUN', 'PRON', 'ADJ'):
                        return True
                # Shared parent: other nouns with the same head (apposition siblings)
                # e.g. Κεβριόνην and υἱόν both point to the same APOS node
                for j in range(len(tokens)):
                    if j in pos_set or j in hektor_set:
                        continue
                    if heads[j] == heads[p] and j != p:
                        if pos_tags[j] == 'NOUN' and rels[j] in ('OBJ_AP', 'SBJ_AP', 'ADV_AP_CO',
                                                                    'APOS', 'ExD', 'OBJ_AP_CO'):
                            # Don't reject vocative formulas just because a clause
                            # subject (SBJ_AP) shares the same elliptic head —
                            # e.g. "Ἕκτορ υἱὲ Πριάμοιο ... Ζεύς με ... προέηκε"
                            if rels[j] == 'SBJ_AP' and rels[p] in ('ExD_AP', 'ExD'):
                                continue
                            return True
            # Also check: ExD siblings (vocative/direct address context)
            for p in positions:
                if rels[p] == 'ExD':
                    for j in range(len(tokens)):
                        if j in pos_set or j in hektor_set:
                            continue
                        if heads[j] == heads[p] and rels[j] == 'ExD' and pos_tags[j] == 'NOUN':
                            return True
            # Adjacent non-Hektor noun in ExD context (different heads but same address)
            min_pos = min(positions)
            max_pos = max(positions)
            for j in range(max(0, min_pos - 2), min(len(tokens), max_pos + 3)):
                if j in pos_set or j in hektor_set:
                    continue
                if pos_tags[j] == 'NOUN' and rels[j] == 'ExD':
                    return True
            return False

        for mh in multi_hits:
            positions = mh.get("positions", list(range(mh["start"], mh["end"] + 1)))
            if not any(near_hektor(p) for p in positions):
                continue
            # Reject if formula syntactically belongs to another person
            if formula_modifies_other_person(positions):
                continue
            # For formulas: check gender agreement to avoid cross-gender misattribution
            # (e.g. πόδας ὠκέα Ἶρις should not match for Hektor)
            nearest_h = min(hektor_indices, key=lambda h: min(abs(p - h) for p in positions))
            h_gender = morphs[nearest_h].get("gender")
            if h_gender:
                formula_genders = [morphs[p].get("gender") for p in positions if morphs[p].get("gender")]
                if formula_genders and all(g != h_gender for g in formula_genders):
                    continue
            used_positions.update(positions)
            # Pick morphology from the word that agrees with Hektor,
            # not blindly the first word (avoids Gen from Πριάμοιο, Acc from βοὴν)
            nearest_h = min(hektor_indices, key=lambda h: min(abs(p - h) for p in positions))
            h_morph = morphs[nearest_h]
            cm = morphs[positions[0]]  # fallback
            for p in positions:
                if agree_cng(morphs[p], h_morph):
                    cm = morphs[p]
                    break
            info = mh["info"]
            formula_line = find_word_line(book, line_start, line_end, tokens[positions[0]], positions[0], len(tokens), line_index)
            rows.append({
                "_sent_id": (book, line_start),
                "Book": book, "Line": formula_line,
                "Hero": "HEKTOR",
                "Epithet": mh["text"], "Lemma": mh["lemma"],
                "EpithetCluster": "",
                "Speaker": speaker, "Addressee": addressee,
                "Case": morph_field(cm, "case"),
                "Number": morph_field(cm, "number"),
                "Gender": morph_field(cm, "gender"),
                "Tense": morph_field(cm, "tense"),
                "Voice": morph_field(cm, "voice"),
                "Mood": morph_field(cm, "mood"),
                "POS": "+".join(pos_tags[p] for p in positions),
                "Source": "whitelist",
                "Confidence": confidence,
                "Greek": " ".join(tokens),
            })

        # ===== 2) SINGLE-WORD EPITHET DETECTION =====
        for i in range(len(tokens)):
            if i in hektor_indices or i in used_positions:
                continue
            if not GREEK_RE.search(tokens[i]):
                continue

            n = norm[i]
            nl = norm_lemmas[i]
            if len(n) < MIN_LEN or n in STOP_NORM or nl in STOP_LEM:
                continue
            if not near_hektor(i):
                continue

            # Check whitelist first (by normalized lemma)
            in_whitelist = nl in wl_singles or n in wl_singles

            # Reject if this word syntactically modifies another noun (not Hektor)
            if modifies_other_noun(i):
                continue

            # Reject predicative/adverbial adjectives: these describe states, not epithets
            if rels[i] in ('PNOM', 'OCOMP', 'ATV', 'ATV_CO', 'AtvV'):
                continue

            # Determine if this is a valid epithet candidate
            is_adj_or_part = is_valid_epithet(pos_tags[i], morphs[i], nl, rels[i], part_blacklist)
            is_patr = is_patronymic(n) or is_patronymic(nl)

            if not in_whitelist and not is_adj_or_part and not is_patr:
                continue

            # Syntactic check: use dependency tree to verify this modifies Hektor
            # For whitelisted items, still prefer syntactic link but allow proximity fallback
            has_syn_link = syntactically_linked_to_hektor(i)

            epithet_morph = morphs[i]

            if not has_syn_link and not is_patr:
                # No syntactic link — require morphological agreement as fallback
                nearest_h = min(hektor_indices, key=lambda h: abs(i - h))
                hektor_morph = morphs[nearest_h]
                if not agree_cng(epithet_morph, hektor_morph):
                    continue

            # Determine cluster
            wl_key = nl if nl in wl_singles else (n if n in wl_singles else None)
            if wl_key:
                source = "whitelist"
            elif is_patr:
                source = "auto-patronymic"
            else:
                source = "auto-detected"

            word_line = find_word_line(book, line_start, line_end, tokens[i], i, len(tokens), line_index)
            row_data = {
                "_sent_id": (book, line_start),
                "Book": book, "Line": word_line,
                "Hero": "HEKTOR",
                "Epithet": tokens[i], "Lemma": lemmas[i],
                "EpithetCluster": "",
                "Speaker": speaker, "Addressee": addressee,
                "Case": morph_field(epithet_morph, "case"),
                "Number": morph_field(epithet_morph, "number"),
                "Gender": morph_field(epithet_morph, "gender"),
                "Tense": morph_field(epithet_morph, "tense"),
                "Voice": morph_field(epithet_morph, "voice"),
                "Mood": morph_field(epithet_morph, "mood"),
                "POS": pos_tags[i],
                "Source": source,
                "Confidence": confidence,
                "Greek": " ".join(tokens),
            }

            if in_whitelist or is_patr:
                rows.append(row_data)
            else:
                # Auto-detected candidate → goes to candidates file for review
                rows.append(row_data)
                candidates.append({
                    "Book": book, "Line": word_line, "Lemma": lemmas[i],
                    "Form": tokens[i], "POS": pos_tags[i],
                    "Case": morph_field(epithet_morph, "case"),
                    "Number": morph_field(epithet_morph, "number"),
                    "Gender": morph_field(epithet_morph, "gender"),
                    "Context": " ".join(tokens),
                })

    # ===== ASSIGN CLUSTERS =====
    assign_clusters(rows)

    # ===== OUTPUT =====
    print(f"\n{'=' * 70}")
    print(f"RESULTS:")
    print(f"  Total epithet rows: {len(rows)}")
    print(f"  From whitelist: {sum(1 for r in rows if r['Source'] == 'whitelist')}")
    print(f"  Auto-patronymic: {sum(1 for r in rows if r['Source'] == 'auto-patronymic')}")
    print(f"  Auto-detected: {sum(1 for r in rows if r['Source'] == 'auto-detected')}")
    print(f"  New candidates for review: {len(candidates)}")

    if rows:
        fieldnames = ["Book", "Line", "Hero", "Epithet", "Lemma",
                       "EpithetCluster", "Speaker", "Addressee",
                       "Case", "Number", "Gender", "Tense", "Voice", "Mood",
                       "POS", "Source", "Confidence", "Greek"]
        rows.sort(key=lambda r: (r["Book"], r["Line"]))
        with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {OUT_FILE} ({len(rows)} rows)")

        print(f"\nSample (first 15):")
        for row in rows[:15]:
            src = f" [{row['Source']}]" if row['Source'] != 'whitelist' else ""
            spk = f" (Speaker: {row['Speaker']})" if row['Speaker'] else ""
            print(f"  {row['Book']}.{row['Line']}: {row['Epithet']} ({row['Lemma']}) — {row['Case']} {row['EpithetCluster']}{src}{spk}")

    if candidates:
        cand_fields = ["Book", "Line", "Lemma", "Form", "POS", "Case", "Number", "Gender", "Context"]
        # Deduplicate by lemma
        seen = set()
        unique_cands = []
        for c in candidates:
            key = c["Lemma"]
            if key not in seen:
                seen.add(key)
                unique_cands.append(c)
        with open(CANDIDATES_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cand_fields)
            writer.writeheader()
            writer.writerows(unique_cands)
        print(f"\nWrote {CANDIDATES_FILE} ({len(unique_cands)} unique candidates to review)")
        print("  → Review these and add good ones to epithet_whitelist.csv")


if __name__ == "__main__":
    main()