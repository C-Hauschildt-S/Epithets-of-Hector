# ---------- Beta Code → Greek via CLTK ----------
try:
    # CLTK 1.x path
    from cltk.alphabet.grc.beta_to_unicode import Replacer
    _BETA = Replacer()
    def beta2grc(s: str) -> str:
        # CLTK expects plain Beta Code like: mh=nin a)/eide qea\
        return _BETA.beta_code(s)
    print("[DEBUG] Beta→Greek converter: CLTK Replacer")
except Exception as e:
    print("[WARNING] CLTK Beta→Greek converter NOT available:", e)
    def beta2grc(s: str) -> str:
        return s

# ---- PROBE: does Beta→Greek actually work? ----
_probe_beta = "mh=nin a)/eide qea\\ *phlhi+a/dew *)axilh=os"
print("[DEBUG] Beta→Greek probe:")
print("  beta : ", _probe_beta)
print("  greek:", beta2grc(_probe_beta))
