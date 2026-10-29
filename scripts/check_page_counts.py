import fitz

for name in ['manuscript.pdf', 'supplementary.pdf', 'full_submission_merged.pdf']:
    doc = fitz.open(name)
    print(f"{name}: {len(doc)} pages")
