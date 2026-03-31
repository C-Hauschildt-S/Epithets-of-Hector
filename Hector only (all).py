# 01_extractor_with_sentiment_hector_only.py
# Hector-only strict epithet extractor + (lexicon + contextual) sentiment
# Output: iliad_with_sentiment.csv

import os, re, csv, unicodedata, sys
from typing import Any, Dict, List, Tuple
from glob import glob
from lxml import etree
from cltk import NLP

# === contextual sentiment (PhilTa + classifier) ===
try:
    from philta_ctx import predict_sentiment  # returns (score, label) or None
    HAS_CTX = True
except Exception:
    HAS_CTX = False

# ===== CONFIG =====
GLOB_PATTERN    = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"
OUT_FILE        = r"C:\Users\carol\PycharmProjects\CLTK+DICES\hector_only_all_parameters.csv"
USE_DICES       = True
WINDOW          = 5          # distance window to Hector for epithet detection
COMBO_SPAN_MAX  = 8          # span for explicit formulas
SENT_WINDOW     = 6          # lexicon sentiment window (also used for contextual local window)
# ==================

GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

# --------- normalization helpers ---------
def normalize_greek(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς","σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+"," ", s).strip()
    return s

def normalize_tokens(tokens: List[str]) -> List[str]:
    # keep this very cheap: strip non-greek word chars, normalize once
    out=[]
    for s in tokens:
        s2 = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]","", s or "")
        out.append(normalize_greek(s2))
    return out

def roman_to_int(s: str) -> int:
    s=s.upper(); vals={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    tot=0
    for i,ch in enumerate(s):
        tot += -vals[ch] if (i+1<len(s) and vals[ch]<vals[s[i+1]]) else vals[ch]
    return tot

def parse_book_n(nval: str) -> int:
    try: return int(nval)
    except Exception: return roman_to_int(nval)

# --------- Beta→Greek ----------
def load_beta_to_unicode():
    try:
        from betacode.conv import beta_to_uni
        return beta_to_uni
    except Exception:
        pass
    try:
        from betacode import betacode_to_greek
        return betacode_to_greek
    except Exception:
        pass
    try:
        from betacode import beta2unicode
        return beta2unicode
    except Exception:
        pass
    return None

b2u = load_beta_to_unicode()
if not b2u:
    print("FATAL: pip install -U betacode"); sys.exit(1)

# --------- TEI lines ----------
def iter_lines(xml_path: str):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    for div in root.findall(".//div1"):
        nval = div.get("n")
        if not nval: continue
        book = parse_book_n(nval)
        counter = 0
        for l in div.findall(".//l"):
            counter += 1
            n = l.get("n")
            ln = int(n) if (n and n.isdigit()) else counter
            beta = "".join(l.itertext()).strip()
            if beta:
                yield book, ln, beta

# --------- Analyzer (CLTK + Stanza fallback) ----------
nlp = NLP("grc", suppress_banner=True)

def cltk_analyze(text: str):
    doc = nlp.analyze(text)
    toks = getattr(doc, "tokens", []) or []
    tokens = [getattr(t,"string",None) or getattr(t,"text",None) or str(t) for t in toks] or text.split()
    pos    = [getattr(t,"pos","") or getattr(t,"upos","") or "" for t in toks]
    lemmas = [getattr(t,"lemma","") or "" for t in toks]
    return tokens, pos, lemmas, toks

_ST = None
def stanza_analyze(text: str):
    global _ST
    if _ST is None:
        import stanza
        _ST = stanza.Pipeline("grc", processors="tokenize,pos,lemma", tokenize_no_ssplit=True, verbose=False)
    doc = _ST(text)
    tokens=[]; pos=[]; lemmas=[]; toks=[]
    for s in doc.sentences:
        for w in s.words:
            tokens.append(w.text); pos.append(w.upos or ""); lemmas.append(w.lemma or ""); toks.append(w)
    return tokens, pos, lemmas, toks

def robust_analyze(text: str):
    # Exception-safe fallback, then quality fallback
    try:
        tokens, pos, lemmas, toks = cltk_analyze(text)
    except Exception:
        return stanza_analyze(text)

    total = len(tokens)
    with_lemma = sum(1 for x in lemmas if x)
    with_pos   = sum(1 for x in pos if x)
    if total == 0 or (with_lemma/max(1,total) < 0.2 and with_pos/max(1,total) < 0.2):
        return stanza_analyze(text)
    return tokens, pos, lemmas, toks

# --------- Morph helpers ----------
def morph_from_obj(t: Any) -> Dict[str,str]:
    if t is None:
        return {}
    d={}
    m = getattr(t, "morph", None)
    if m:
        for part in str(m).split("|"):
            if "=" in part:
                k,v = part.split("=",1); d[k.lower()] = v.lower()
    f = getattr(t, "feats", None)
    if f and not d:
        for part in str(f).split("|"):
            if "=" in part:
                k,v = part.split("=",1); d[k.lower()] = v.lower()
    for k in ("Case","Number","Gender"):
        v = getattr(t, k, None)
        if isinstance(v,str) and v:
            d[k.lower()] = v.lower()
    return d

def agree_cng(m1: Dict[str,str], m2: Dict[str,str]) -> bool:
    if not m1 or not m2: return False
    ok=True
    if "case" in m1 and "case" in m2: ok &= (m1["case"] == m2["case"])
    if "number" in m1 and "number" in m2: ok &= (m1["number"] == m2["number"])
    if "gender" in m1 and "gender" in m2: ok &= (m1["gender"] == m2["gender"])
    return ok

# --------- Hector-only hero detection ----------
HEKTOR = "ΕΚΤΩΡ"
HEKTOR_LEMMAS_RAW = {"ΕΚΤΩΡ","ΠΡΙΑΜΙΔΗΣ"}  # keep what you had for Hector
HEKTOR_LEMMAS = {normalize_greek(x) for x in HEKTOR_LEMMAS_RAW}

# --------- Epithet rules ----------
STOP_SURF = {"ΜΗΝΙΝ","ΜΗΝΙΣ","ΑΕΙΔΕ","ΘΕΑ","ΘΕΑΝ","ΘΕΑΣ","ΘΕΑΙ","ΜΟΥΣΑ","ΜΟΥΣΑΝ","ΜΟΥΣΗΣ","ΑΝΔΡΑ","ΑΝΔΡΟΣ","ΑΝΗΡ"}
STOP_LEM  = {"ΜΗΝΙΣ","ΑΕΙΔΩ","ΘΕΑ","ΜΟΥΣΑ"}

WHITELIST_NOMINAL = {
    "ΓΕΡΗΝΙΟΣ","ΙΠΠΟΤΑ","ΠΥΛΙΟΣ","ΒΑΣΙΛΕΥΣ","ΓΕΡΩΝ",
    "ΜΕΓΑΣ","ΤΑΧΥΣ","ΩΚΥΣ","ΞΑΝΘΟΣ","ΚΡΑΤΕΡΟΣ","ΦΑΙΔΙΜΟΣ",
    "ΔΑΙΦΡΩΝ","ΚΟΡΥΘΑΙΟΛΟΣ","ΑΝΤΙΘΕΟΣ","ΔΙΟΓΕΝΗΣ","ΑΡΗΙΦΙΛΟΣ","ΜΕΓΑΛΗΤΩΡ","ΘΕΟΕΙΔΗΣ",
    "ΠΗΛΕΙΔΗΣ","ΠΗΛΕΙΑΔΗΣ","ΠΡΙΑΜΙΔΗΣ","ΑΤΡΕΙΔΗΣ","ΤΥΔΕΙΔΗΣ","ΤΕΛΑΜΩΝΙΟΣ","ΛΑΕΡΤΙΑΔΗΣ","ΑΓΧΙΣΙΑΔΗΣ"
}
UNIV_ADJ = {"ΔΙΟΣ","ΜΕΓΑΣ","ΩΚΥΣ","ΤΑΧΥΣ","ΚΡΑΤΕΡΟΣ","ΦΑΙΔΙΜΟΣ","ΞΑΝΘΟΣ","ΔΑΙΦΡΩΝ","ΑΝΤΙΘΕΟΣ","ΘΕΟΕΙΔΗΣ"}

PATR_SUFFIXES = ("ΙΔΗΣ","ΙΑΔΗΣ","ΪΔΗΣ","ΪΑΔΗΣ")
def is_patronymic(n: str) -> bool:
    return any(n.endswith(s) for s in PATR_SUFFIXES)

def is_dios_epithet(lemma: str, pos_tag: str) -> bool:
    ln = normalize_greek(lemma) if lemma else ""
    if ln == "ΖΕΥΣ":     # Διός (Zeus, gen.) ≠ δῖος (adj.)
        return False
    if pos_tag == "ADJ":
        return True
    if ln == "ΔΙΟΣ":
        return True
    return False

# --------- Formulae ----------
FORMULAS = [
    ("ΑΝΑΞ","ΑΝΔΡΩΝ"),
    ("ΠΟΙΜΗΝ","ΛΑΩΝ"),
    ("ΠΟΔΑΣ","ΩΚΥΣ"),
    ("ΒΟΗΝ","ΑΓΑΘΟΣ"),
]

def find_formulas(tokens: List[str], norm: List[str]) -> List[Dict]:
    idx = {}
    for i,n in enumerate(norm):
        idx.setdefault(n, []).append(i)
    out=[]
    seen=set()
    for a,b in FORMULAS:
        if a not in idx or b not in idx:
            continue
        for i in idx[a]:
            # first b within span
            j = None
            for cand in idx[b]:
                if cand > i and (cand - i) <= COMBO_SPAN_MAX:
                    j = cand
                    break
            if j is None:
                continue
            phrase = " ".join(tokens[k] for k in range(i, j+1))
            norm_phrase = normalize_greek(phrase)
            key = (i, j, norm_phrase)
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind":"formula","start":i,"end":j,"center":(i+j)/2.0,
                        "text":phrase,"norm":norm_phrase})
    return out

# --------- DICES (optional) ----------
def build_speech_index():
    if not USE_DICES: return []
    try:
        from dicesapi import DicesAPI
        api = DicesAPI()
        speeches = api.getSpeeches().advancedFilter(lambda s: getattr(s.work,"title","").lower() in {"iliad","the iliad"})
    except Exception as e:
        print("[WARN] DICES off:", e); return []
    rx = [
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)-(?P<b2>\d+)\.(?P<l2>\d+)$"),
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)-(?P<l2>\d+)$"),
        re.compile(r":(?P<b1>\d+)\.(?P<l1>\d+)$"),
    ]
    idx=[]
    for s in speeches:
        spk=[getattr(p,"name",str(p)) for p in getattr(s,"spkr",[])]
        adr=[getattr(p,"name",str(p)) for p in getattr(s,"addr",[])]
        urn=getattr(s,"urn","") or ""
        b1=l1=b2=l2=None
        for r in rx:
            m=r.search(urn)
            if m:
                gd=m.groupdict(); b1=int(gd["b1"]); l1=int(gd["l1"])
                b2=int(gd.get("b2") or gd["b1"]); l2=int(gd.get("l2") or gd["l1"])
                break
        idx.append({"b1":b1,"l1":l1,"b2":b2,"l2":l2,"sp":spk,"ad":adr})
    return idx

def line_in_range(book,line,b1,l1,b2,l2):
    if None in (b1,l1,b2,l2): return False
    if b1 == b2 == book: return l1 <= line <= l2
    if book < b1 or book > b2: return False
    if book == b1 and line < l1: return False
    if book == b2 and line > l2: return False
    return True

def lookup_speech(idx, book, line):
    if not idx: return [], []
    c=[d for d in idx if line_in_range(book,line,d["b1"],d["l1"],d["b2"],d["l2"])]
    if not c: return [], []
    c.sort(key=lambda d: (d["b2"]-d["b1"])*10000 + (d["l2"]-d["l1"]))
    return c[0]["sp"], c[0]["ad"]

# --------- Sentiment mini-lexicon ----------
POS_LEMMA = {
    "ἀγαθός": 1.0, "κλέος": 0.6, "νίκη": 0.8, "χαρά": 0.9, "ἥρως": 0.4,
    "φίλος": 0.5, "εὐκλέεια": 0.7, "ἀρετή": 0.7,
    "ἱππόδαμος": 0.4,
}
NEG_LEMMA = {
    "κακός": -1.0, "ἄλγος": -0.7, "χόλος": -0.8, "λύπη": -0.7, "φόβος": -0.6,
    "πένθος": -0.8, "μῆνις": -0.9, "ὀδύνη": -0.8, "φονεύω": -0.7,
    "ἀνδροφόνος": -0.5,
}
LEXICON = {**POS_LEMMA, **NEG_LEMMA}

def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", s)

def lex_key(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower().strip()
    s = s.replace("ς","σ")  # normalize sigma for matching
    s = re.sub(r"[’'·.,;:!?…]+$", "", s)
    return s

LEXICON_NORM     = { lex_key(k): v for k, v in LEXICON.items() }
LEXICON_STRIPPED = { lex_key(strip_diacritics(k)): v for k, v in LEXICON.items() }

EPIC_ENDINGS = (
    ("οιο", "ος"),
    ("οισι", "οις"),
    ("οισιν", "οις"),
    ("οιοιο", "ου"),
)
def epic_lemma_fallback(key: str) -> str:
    for old, new in EPIC_ENDINGS:
        if key.endswith(old):
            return key[: -len(old)] + new
    return key

def lemmatize_tokens_grc(tokens: List[str], lemmas: List[str]) -> List[str]:
    out=[]
    L = max(len(tokens), len(lemmas))
    for i in range(L):
        t = tokens[i] if i < len(tokens) else ""
        l = lemmas[i] if i < len(lemmas) else ""
        out.append(unicodedata.normalize("NFC", l) if l else unicodedata.normalize("NFC", t))
    return out

def _lex_lookup(lm: str):
    k = lex_key(lm)
    v = LEXICON_NORM.get(k)
    if v is None:
        v = LEXICON_STRIPPED.get(lex_key(strip_diacritics(lm)))
    if v is None:
        k2 = epic_lemma_fallback(k)
        if k2 != k:
            v = LEXICON_NORM.get(k2)
    if v is None:
        ks = lex_key(strip_diacritics(lm))
        ks2 = epic_lemma_fallback(ks)
        if ks2 != ks:
            v = LEXICON_STRIPPED.get(ks2)
    return v

def global_sentiment(lemmas_line: List[str]) -> Tuple[float, str, int]:
    score=0.0; hits=0
    for lm in lemmas_line:
        v = _lex_lookup(lm)
        if v is not None:
            score += v; hits += 1
    avg = (score / hits) if hits else 0.0
    label = "pos" if avg > 0.1 else "neg" if avg < -0.1 else "neu"
    return avg, label, hits

def targeted_sentiment_unique_indices(lemmas_line: List[str], hero_indices: List[int], window: int) -> Tuple[float, str, int]:
    if not hero_indices:
        return 0.0, "neu", 0
    L = len(lemmas_line)
    covered=set()
    for idx in hero_indices:
        a = max(0, idx - window); b = min(L, idx + window + 1)
        covered.update(range(a,b))
    score=0.0; hits=0
    for i in covered:
        v = _lex_lookup(lemmas_line[i])
        if v is not None:
            score += v; hits += 1
    avg = (score / hits) if hits else 0.0
    label = "pos" if avg > 0.1 else "neg" if avg < -0.1 else "neu"
    return avg, label, hits

# --------- MAIN ----------
def main():
    probe = b2u("mh=nin a)/eide qea\\ *phlhi+a/dew *)axilh=os")
    if not GREEK_RE.search(probe):
        print("FATAL: Beta→Greek not working"); sys.exit(1)

    files = sorted(glob(GLOB_PATTERN))
    if not files:
        print("No TEI files; fix GLOB_PATTERN"); sys.exit(1)

    sp_index = build_speech_index() if USE_DICES else []
    rows: List[Dict[str, Any]] = []

    for path in files:
        for book, line_no, beta in iter_lines(path):
            greek = b2u(beta).strip()
            if not greek or not GREEK_RE.search(greek):
                continue

            # Analyze once
            tokens, pos, lemmas, tokobjs = robust_analyze(greek)
            if not tokens:
                continue

            norm = normalize_tokens(tokens)

            # Hector detection: prefer lemmas if usable, else surface
            with_lemma = sum(1 for x in lemmas if x)
            use_lemma = len(tokens) > 0 and (with_lemma/len(tokens)) >= 0.2

            hektor_indices=[]
            hektor_morphs=[]

            if use_lemma:
                for i, lem in enumerate(lemmas):
                    if normalize_greek(lem) in HEKTOR_LEMMAS:
                        hektor_indices.append(i)
                        tobj = tokobjs[i] if (tokobjs and i < len(tokobjs)) else None
                        hektor_morphs.append(morph_from_obj(tobj))
            else:
                for i, n in enumerate(norm):
                    if n in HEKTOR_LEMMAS:
                        hektor_indices.append(i)
                        tobj = tokobjs[i] if (tokobjs and i < len(tokobjs)) else None
                        hektor_morphs.append(morph_from_obj(tobj))

            if not hektor_indices:
                continue

            # Precompute "within WINDOW of Hector"
            # Cheap because Hector indices are few
            def within_window(i: int) -> bool:
                for h in hektor_indices:
                    if abs(i - h) <= WINDOW:
                        return True
                return False

            # Sentiment lemmas
            line_lemmas = lemmatize_tokens_grc(tokens, lemmas)
            g_score, g_label, _g_hits = global_sentiment(line_lemmas)

            ctx_score, ctx_label = (None, None)
            if HAS_CTX:
                res = predict_sentiment(greek)
                if res is not None:
                    ctx_score, ctx_label = res

            # Single-token epithet candidates
            singles=[]
            for i, s in enumerate(tokens):
                if not s or not GREEK_RE.search(s):
                    continue

                p = pos[i] if i < len(pos) else ""
                l = lemmas[i] if i < len(lemmas) else ""
                n = norm[i]
                ln = normalize_greek(l) if l else ""

                if n in STOP_SURF or ln in STOP_LEM:
                    continue

                if not within_window(i):
                    continue

                ok=False
                if p == "ADJ":
                    if n == "ΔΙΟΣ" and not is_dios_epithet(l, p):
                        ok = False
                    else:
                        ok = True
                if not ok and (ln in UNIV_ADJ or n in UNIV_ADJ):
                    ok = True
                if not ok and (n in WHITELIST_NOMINAL or is_patronymic(n) or is_patronymic(ln)):
                    ok = True
                if not ok:
                    continue

                # Agreement check: only enforce if we actually have morph on both sides
                if p == "ADJ" and (n not in WHITELIST_NOMINAL) and (ln not in UNIV_ADJ):
                    tobj = tokobjs[i] if (tokobjs and i < len(tokobjs)) else None
                    cm = morph_from_obj(tobj)
                    # enforce agreement only if adjective has features and at least one Hector token has features
                    if cm and any(hm for hm in hektor_morphs):
                        agrees = any(agree_cng(cm, hm) for hm in hektor_morphs if hm)
                        if not agrees:
                            continue

                singles.append({"kind":"single","start":i,"end":i,"center":float(i),
                                "text":s,"norm":n,"lemma":l,"pos":p})

            # Multi-word formulas (optional)
            formulas = find_formulas(tokens, norm)

            # Merge + dedup (use normalized key)
            cands=[]
            seen=set()
            for c in singles + formulas:
                key=(c["start"], c["end"], c.get("norm",""))
                if key in seen:
                    continue
                seen.add(key)
                # Require formulas also near Hector (center distance cutoff)
                if c["kind"] == "formula":
                    if not any(abs(h - c["center"]) <= WINDOW for h in hektor_indices):
                        continue
                cands.append(c)

            if not cands:
                continue

            spk, adr = lookup_speech(sp_index, book, line_no) if sp_index else ([], [])

            # Targeted sentiments (Hector)
            t_score, t_label, t_hits = targeted_sentiment_unique_indices(line_lemmas, hektor_indices, SENT_WINDOW)

            # Contextual target sentiment (PhilTa) on local window around first Hector mention
            ctx_t_score, ctx_t_label = (None, None)
            if HAS_CTX:
                idx0 = hektor_indices[0]
                L = len(tokens)
                a = max(0, idx0 - SENT_WINDOW); b = min(L, idx0 + SENT_WINDOW + 1)
                local_text = " ".join(tokens[a:b])
                res = predict_sentiment(local_text)
                if res is not None:
                    ctx_t_score, ctx_t_label = res

            # Emit one row per candidate, Hero fixed to Hector
            for c in cands:
                rows.append({
                    "Book": book,
                    "Line": line_no,
                    "Locus": f"{book}.{line_no}",
                    "Hero": HEKTOR,
                    "Epithet": c["text"],
                    "Speaker": "; ".join(spk) if spk else "",
                    "Addressee": "; ".join(adr) if adr else "",
                    "Greek": greek,
                    "SentimentGlobalScore": f"{g_score:.3f}",
                    "SentimentGlobal": g_label,
                    "TargetSentimentScore": f"{t_score:.3f}",
                    "TargetSentimentLabel": t_label,
                    "TargetSentimentHits": t_hits,
                    "CtxSentimentScore": "" if ctx_score is None else f"{ctx_score:.3f}",
                    "CtxSentimentLabel": "" if ctx_label is None else ctx_label,
                    "CtxTargetSentimentScore": "" if ctx_t_score is None else f"{ctx_t_score:.3f}",
                    "CtxTargetSentimentLabel": "" if ctx_t_label is None else ctx_t_label,
                })

    # Write CSV
    if rows:
        os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
        with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Book","Line","Locus","Hero","Epithet","Speaker","Addressee","Greek",
                    "SentimentGlobalScore","SentimentGlobal",
                    "TargetSentimentScore","TargetSentimentLabel","TargetSentimentHits",
                    "CtxSentimentScore","CtxSentimentLabel",
                    "CtxTargetSentimentScore","CtxTargetSentimentLabel",
                ]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {OUT_FILE} ({len(rows)} rows)")
    else:
        print("No rows produced — check Hector detection / epithet candidate logic.")

if __name__ == "__main__":
    main()
