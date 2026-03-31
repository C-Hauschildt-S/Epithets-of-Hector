# Epithets.py — strict epithet detector with CLTK→Stanza fallback and DICES join
# Requirements: lxml, cltk>=1.5, betacode, stanza  (dicesapi optional)

import os, re, csv, unicodedata, sys
from typing import Any, Dict, List, Set, Tuple
from glob import glob
from lxml import etree
from cltk import NLP

# ================== SETTINGS ==================
GLOB_PATTERN    = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"
OUT_OCCURRENCES = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_epithet_occurrences.csv"
OUT_COVERAGE    = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_epithet_coverage.csv"

WINDOW          = 6
COMBO_SPAN_MAX  = 8
USE_DICES       = True
DEBUG_SAMPLES   = 2    # set 0 to silence
# ==============================================

# ---------- Beta code -> Unicode ----------
def _load_beta_to_unicode():
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
    try:
        from cltk.alphabet.grc.beta_to_unicode import beta_to_unicode as _b2u
        return _b2u
    except Exception:
        pass
    return None

beta_to_unicode = _load_beta_to_unicode()
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]+")

def probe_conversion_or_die():
    if not beta_to_unicode:
        print("[FATAL] No Beta→Greek converter. Run: pip install -U betacode")
        sys.exit(1)
    sample_beta = "mh=nin a)/eide qea\\ *phlhi+a/dew *)axilh=os"
    sample_gr = beta_to_unicode(sample_beta)
    if not GREEK_RE.search(sample_gr):
        print("[FATAL] Beta→Greek conversion failed. Run: pip install -U betacode")
        sys.exit(1)

# ---------- Normalization ----------
def normalize_greek(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s).replace("ς","σ").upper()
    s = re.sub(r"[^\w\u0370-\u03FF\u1F00-\u1FFF]+"," ", s).strip()
    return s

# ---------- TEI reader ----------
def roman_to_int(s: str) -> int:
    s = s.upper()
    vals={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    tot=0
    for i,ch in enumerate(s):
        if i+1<len(s) and vals[ch]<vals[s[i+1]]: tot-=vals[ch]
        else: tot+=vals[ch]
    return tot

def parse_book_n(nval: str) -> int:
    try: return int(nval)
    except Exception: return roman_to_int(nval)

def iter_books_and_lines(xml_path: str):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    for div in root.findall(".//div1"):
        nval = div.get("n")
        if not nval: continue
        book = parse_book_n(nval)
        counter = 0
        for l in div.findall(".//l"):
            counter += 1
            n = l.get("
