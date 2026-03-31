## 01_mini_extractor.py — strict, clean epithet extractor
# CLTK primary + Stanza fallback; optional DICES for Speaker/Addressee
# Outputs: iliad_MINI_occurrences.csv with Book, Line, Locus, Hero, Epithet, Speaker, Addressee, Greek

import os, re, csv, unicodedata, sys
from typing import Any, Dict, List, Set, Tuple
from glob import glob
from lxml import etree
from cltk import NLP

# ===== CONFIG =====
GLOB_PATTERN    = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"
OUT_OCCURRENCES = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_MINI_occurrences.csv"
USE_DICES       = True
WINDOW          = 5
COMBO_SPAN_MAX  = 8
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
    return [normalize_greek(re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]","",s)) for s in tokens]

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

def cltk_analyze(text):
    doc = nlp.analyze(text)
    toks = getattr(doc, "tokens", []) or []
    tokens = [getattr(t,"string",None) or getattr(t,"text",None) or str(t) for t in toks] or text.split()
    pos    = [getattr(t,"pos","") or getattr(t,"upos","") or "" for t in toks]
    lemmas = [getattr(t,"lemma","") or "" for t in toks]
    return tokens, pos, lemmas, toks

_ST = None
def stanza_analyze(text):
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

def robust_analyze(text):
    tokens, pos, lemmas, toks = cltk_analyze(text)
    total = len(tokens)
    with_lemma = sum(1 for x in lemmas if x)
    with_pos   = sum(1 for x in pos if x)
    if total == 0 or (with_lemma/max(1,total) < 0.2 and with_pos/max(1,total) < 0.2):
        tokens, pos, lemmas, toks = stanza_analyze(text)
    return tokens, pos, lemmas, toks

# --------- Morph helpers ----------
def morph_from_obj(t: Any) -> Dict[str,str]:
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

# --------- Heroes & aliases ----------
ACHILLES   = "ΑΧΙΛΛΕΥΣ";  AGAMEMNON  = "ΑΓΑΜΕΜΝΩΝ"; MENELAOS   = "ΜΕΝΕΛΑΟΣ"
ODYSSEUS   = "ΟΔΥΣΣΕΥΣ";  AJAX_TEL   = "ΑΙΑΣ";       AJAX_OIL   = "ΑΪΑΣ"
DIOMEDES   = "ΔΙΟΜΗΔΗΣ";  NESTOR     = "ΝΕΣΤΩΡ";     HEKTOR     = "ΕΚΤΩΡ"
PARIS      = "ΠΑΡΙΣ";     ALEXANDROS = "ΑΛΕΞΑΝΔΡΟΣ"; AENEAS     = "ΑΙΝΕΙΑΣ"
PATROKLOS  = "ΠΑΤΡΟΚΛΟΣ"; SARPEDON   = "ΣΑΡΠΗΔΩΝ";   GLAUKOS    = "ΓΛΑΥΚΟΣ"
PRIAMOS    = "ΠΡΙΑΜΟΣ";   HELENOS    = "ΕΛΕΝΟΣ";     IDOMENEUS  = "ΙΔΟΜΕΝΕΥΣ"

HERO_LEMMAS = {
    ACHILLES:   {"ΑΧΙΛΛΕΥΣ","ΠΗΛΕΙΔΗΣ","ΠΗΛΕΪΔΗΣ","ΠΗΛΕΙΑΔΗΣ"},
    AGAMEMNON:  {"ΑΓΑΜΕΜΝΩΝ","ΑΤΡΕΙΔΗΣ","ΑΤΡΕΪΔΗΣ"},
    MENELAOS:   {"ΜΕΝΕΛΑΟΣ","ΑΤΡΕΙΔΗΣ","ΑΤΡΕΪΔΗΣ"},
    ODYSSEUS:   {"ΟΔΥΣΣΕΥΣ","ΛΑΕΡΤΙΑΔΗΣ"},
    AJAX_TEL:   {"ΑΙΑΣ","ΤΕΛΑΜΩΝΙΟΣ","ΤΕΛΑΜΩΝΙΑΔΗΣ"},
    AJAX_OIL:   {"ΑΪΑΣ","ΟΙΛΕΙΔΗΣ","ΟΙΛΕΪΔΗΣ"},
    DIOMEDES:   {"ΔΙΟΜΗΔΗΣ","ΤΥΔΕΙΔΗΣ","ΤΥΔΕΪΔΗΣ"},
    NESTOR:     {"ΝΕΣΤΩΡ","ΓΕΡΗΝΙΟΣ","ΠΥΛΙΟΣ"},
    HEKTOR:     {"ΕΚΤΩΡ","ΠΡΙΑΜΙΔΗΣ"},
    PARIS:      {"ΠΑΡΙΣ","ΑΛΕΞΑΝΔΡΟΣ"},
    ALEXANDROS: {"ΑΛΕΞΑΝΔΡΟΣ"},
    AENEAS:     {"ΑΙΝΕΙΑΣ","ΑΓΧΙΣΙΑΔΗΣ"},
    PATROKLOS:  {"ΠΑΤΡΟΚΛΟΣ"},
    SARPEDON:   {"ΣΑΡΠΗΔΩΝ"},
    GLAUKOS:    {"ΓΛΑΥΚΟΣ"},
    PRIAMOS:    {"ΠΡΙΑΜΟΣ"},
    HELENOS:    {"ΕΛΕΝΟΣ"},
    IDOMENEUS:  {"ΙΔΟΜΕΝΕΥΣ"},
}
HERO_CANON = { ALEXANDROS: PARIS }
def canonical(h): return HERO_CANON.get(h, h)

def is_hero_lemma(lem_norm: str) -> bool:
    for lems in HERO_LEMMAS.values():
        if lem_norm in lems: return True
    return False

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
def is_patronymic(n): return any(n.endswith(s) for s in PATR_SUFFIXES)

def is_dios_epithet(lemma, pos_tag, surface):
    ln = normalize_greek(lemma) if lemma else ""
    if ln == "ΖΕΥΣ": return False
    if pos_tag == "ADJ": return True
    if ln == "ΔΙΟΣ": return True
    if surface and surface[:1].islower(): return True
    return False

def nearest(i, hero_positions): return min(abs(i-h) for h in hero_positions) if hero_positions else 999

# --------- Formulae (explicit, no hero name inside) ----------
FORMULAS = [
    ("ΑΝΑΞ","ΑΝΔΡΩΝ"),     # ἄναξ ἀνδρῶν
    ("ΠΟΙΜΗΝ","ΛΑΩΝ"),     # ποιμὴν λαῶν
    ("ΠΟΔΑΣ","ΩΚΥΣ"),      # πόδας ὠκὺς
    ("ΒΟΗΝ","ΑΓΑΘΟΣ"),     # βοὴν ἀγαθός
]

def find_formulas(tokens: List[str], norm: List[str]) -> List[Dict]:
    idx = {}
    for i,n in enumerate(norm): idx.setdefault(n, []).append(i)
    cands=[]
    for a,b in FORMULAS:
        if a not in idx or b not in idx: continue
        for i in idx[a]:
            # prefer adjacency; otherwise allow within COMBO_SPAN_MAX with order
            js = [j for j in idx[b] if j>i and (j-i)<=COMBO_SPAN_MAX]
            if not js: continue
            j = js[0]
            phrase = " ".join(tokens[k] for k in range(i, j+1))
            cands.append({"kind":"formula","start":i,"end":j,"center":(i+j)/2.0,
                          "text":phrase,"norm":normalize_greek(phrase)})
    # de-dup
    seen=set(); out=[]
    for c in cands:
        key=(c["start"], c["end"], c["norm"])
        if key not in seen: seen.add(key); out.append(c)
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

# --------- main ----------
def main():
    files = sorted(glob(GLOB_PATTERN))
    if not files:
        print("No TEI files; fix GLOB_PATTERN"); sys.exit(1)
    # Beta→Greek quick probe
    probe = b2u("mh=nin a)/eide qea\\ *phlhi+a/dew *)axilh=os")
    if not GREEK_RE.search(probe):
        print("FATAL: Beta→Greek not working"); sys.exit(1)

    sp_index = build_speech_index()
    rows=[]

    for path in files:
        for book, line_no, beta in iter_lines(path):
            greek = b2u(beta).strip()
            if not GREEK_RE.search(greek):
                continue

            tokens, pos, lemmas, tokobjs = robust_analyze(greek)
            if not tokens:
                continue
            norm = normalize_tokens(tokens)

            # locate heroes (prefer lemma)
            hero_hits=[]
            hero_morphs=[]
            with_lemma = sum(1 for x in lemmas if x)
            use_lemma = len(tokens)>0 and (with_lemma/len(tokens))>=0.2
            if use_lemma:
                for i, lem in enumerate(lemmas):
                    ln = normalize_greek(lem)
                    for hero, lems in HERO_LEMMAS.items():
                        if ln in lems:
                            hero_hits.append((canonical(hero), i))
                            hero_morphs.append((i, morph_from_obj(tokobjs[i])))
                            break
            else:
                for i, n in enumerate(norm):
                    for hero, forms in HERO_LEMMAS.items():  # even in fallback, try lemma names if available
                        if n in forms:
                            hero_hits.append((canonical(hero), i))
                            hero_morphs.append((i, morph_from_obj(tokobjs[i])))
                            break
            if not hero_hits:
                continue
            hero_positions = [i for _,i in hero_hits]

            # ---- single-token epithets (strict) ----
            singles=[]
            for i, s in enumerate(tokens):
                if not s or not GREEK_RE.search(s): continue
                p = pos[i] if i<len(pos) else ""
                l = lemmas[i] if i<len(lemmas) else ""
                n = norm[i]; ln = normalize_greek(l) if l else ""

                if n in STOP_SURF or ln in STOP_LEM:  # ban list
                    continue

                ok=False
                # 1) ADJ only (base rule)
                if p == "ADJ":
                    # δῖος guard vs Διός ‘of Zeus’
                    if n == "ΔΙΟΣ" and not is_dios_epithet(l,p,s):
                        ok = False
                    else:
                        ok = True

                # 2) Allow universal adj even if POS is noisy
                if not ok and ln in UNIV_ADJ:
                    ok = True

                # 3) Nominal whitelist & patronymics
                if not ok and (n in WHITELIST_NOMINAL or is_patronymic(n) or is_patronymic(ln)):
                    ok = True

                if not ok:
                    continue

                # Must be near a hero
                if nearest(i, hero_positions) > WINDOW:
                    continue

                # If this is an ADJ and not in whitelist/universal, require agreement with SOME hero on the line
                if p == "ADJ" and (n not in WHITELIST_NOMINAL) and (ln not in UNIV_ADJ):
                    cm = morph_from_obj(tokobjs[i])
                    agrees = any(agree_cng(cm, hm) for _,hm in hero_morphs)
                    if not agrees:
                        continue

                singles.append({"kind":"single","start":i,"end":i,"center":float(i),
                                "text":s,"norm":n,"lemma":l,"pos":p})

            # ---- explicit multi-word formulas (no hero names inside) ----
            formulas = find_formulas(tokens, norm)

            # Collect candidates (no generic ADJ+PROPN pairs to avoid including hero names)
            cands=[]; seen=set()
            for c in singles + formulas:
                key=(c["start"],c["end"],c["text"])
                if key in seen: continue
                seen.add(key); cands.append(c)
            if not cands:
                continue

            # DICES speakers
            spk, adr = lookup_speech(sp_index, book, line_no) if sp_index else ([], [])

            # Attach: one candidate -> one hero (best by distance, alias bias, and formula bias)
            used_spans=set()

            def ag_bias(cnorm, hero):
                return 1.5 if ("ΑΝΑΞ" in cnorm and hero==AGAMEMNON) else 0.0

            for c in cands:
                span_key=(book,line_no,c["start"],c["end"])
                if span_key in used_spans:
                    continue
                best=None; best_score=1e9
                for hero, hpos in hero_hits:
                    dist = abs(hpos - c["center"])
                    score = dist
                    # adjacency preference
                    if dist == 1:
                        if c["end"] == hpos - 1: score -= 1.25
                        if c["start"] == hpos + 1: score -= 0.75
                    # alias preference (strong)
                    if c.get("norm") in {"ΑΤΡΕΙΔΗΣ","ΠΗΛΕΙΔΗΣ","ΠΡΙΑΜΙΔΗΣ","ΤΥΔΕΙΔΗΣ"} and hero in {AGAMEMNON, ACHILLES, PRIAMOS, DIOMEDES}:
                        score -= 5.0
                    # formula nudge
                    score -= ag_bias(c.get("norm",""), hero)

                    if score < best_score:
                        best_score=score; best=hero
                if not best:
                    continue
                used_spans.add(span_key)
                rows.append({
                    "Book": book, "Line": line_no, "Locus": f"{book}.{line_no}",
                    "Hero": best, "Epithet": c["text"],
                    "Speaker": "; ".join(spk) if spk else "",
                    "Addressee": "; ".join(adr) if adr else "",
                    "Greek": greek
                })

    os.makedirs(os.path.dirname(OUT_OCCURRENCES) or ".", exist_ok=True)
    rows.sort(key=lambda r: (r["Book"], r["Line"], r["Hero"]))
    with open(OUT_OCCURRENCES,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["Book","Line","Locus","Hero","Epithet","Speaker","Addressee","Greek"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"Wrote {OUT_OCCURRENCES} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
