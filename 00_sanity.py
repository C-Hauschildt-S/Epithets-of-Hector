# 00_sanity.py
import re, sys
from glob import glob
from lxml import etree

# —— CONFIG ——
GLOB_PATTERN = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"

# —— Beta→Greek ——
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
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

if not b2u:
    print("FATAL: install betacode: pip install -U betacode"); sys.exit(1)

# —— pick first 3 TEI lines ——
files = sorted(glob(GLOB_PATTERN))
if not files:
    print("FATAL: No TEI XMLs found; fix GLOB_PATTERN"); sys.exit(1)

def iter_lines(xml):
    tree = etree.parse(xml)
    for div in tree.findall(".//div1"):
        cnt = 0
        for l in div.findall(".//l"):
            cnt += 1
            yield div.get("n"), l.get("n") or str(cnt), "".join(l.itertext()).strip()

samples = []
for p in files:
    for b, ln, beta in iter_lines(p):
        samples.append((int(b) if b.isdigit() else b, int(ln), beta))
        if len(samples) >= 3: break
    if len(samples) >= 3: break

print("— Beta→Greek probe —")
for b, ln, beta in samples:
    greek = b2u(beta)
    print(f"{b}.{ln}  β: {beta}")
    print(f"{b}.{ln}  γ: {greek}\n")
    if not GREEK_RE.search(greek):
        print("FATAL: Beta→Greek failed (no Greek chars)"); sys.exit(1)

# —— CLTK first, Stanza fallback ——
from cltk import NLP
nlp = NLP("grc", suppress_banner=True)

def cltk_analyze(text):
    doc = nlp.analyze(text)
    toks = getattr(doc, "tokens", []) or []
    return [ (getattr(t, "string", None) or getattr(t,"text",None) or str(t),
              getattr(t,"pos","") or getattr(t,"upos",""),
              getattr(t,"lemma","") or "")
             for t in toks ]

def stanza_analyze(text):
    import stanza
    nlp_s = stanza.Pipeline("grc", processors="tokenize,pos,lemma", tokenize_no_ssplit=True, verbose=False)
    doc = nlp_s(text)
    out=[]
    for s in doc.sentences:
        for w in s.words:
            out.append((w.text, w.upos or "", w.lemma or ""))
    return out

print("— Analyzer probe (CLTK → optional Stanza) —")
for b, ln, beta in samples:
    greek = b2u(beta)
    rows = cltk_analyze(greek)
    filled = sum(1 for _,p,l in rows if p or l)
    if filled < max(1,len(rows)) * 0.2:
        print("CLTK sparse — falling back to Stanza")
        rows = stanza_analyze(greek)
    print(f"{b}.{ln}:")
    for tok, pos, lem in rows[:10]:
        print(f"   {tok:<18} POS={pos:<6} LEMMA={lem}")
    print()
print("OK: conversion + analyzer working.")
