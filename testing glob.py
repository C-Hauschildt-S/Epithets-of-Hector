import glob, os

# 👇 adjust this pattern to match your actual file names
XML_GLOB = r"C:\Users\carol\PycharmProjects\CLTK+DICES\iliad_book_*.xml"

matches = sorted(glob.glob(XML_GLOB))
print(f"[DEBUG] Matched {len(matches)} files:")
for p in matches[:10]:
    print("  -", p)
if not matches:
    raise SystemExit("No files matched XML_GLOB. Fix the path/pattern.")
