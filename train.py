import pandas as pd

rows = [
    {"text":"Ἀτρεΐδης ἄναξ ἀνδρῶν …", "label":"pos"},
    {"text":"μῆνιν ἄειδε θεά Πηληϊάδεω Ἀχιλῆος", "label":"neg"},
    {"text":"κῦδος ἔχουσιν Ἀχαιοί …", "label":"pos"},
    {"text":"πένθος ἔλαβεν …", "label":"neg"},
    {"text":"καί …", "label":"neu"},
]

df = pd.DataFrame(rows)
df.to_csv("sentiment_train.csv", index=False, encoding="utf-8")
print("Wrote sentiment_train.csv")
