#Turns the open-access JATS XML papers in ml/rag/papers/ into corpus_chunks.json.
#JATS is nice because the section structure is already there (<body> -> <sec><title> -> <p>),
#so im chunking on those boundaries instead of a blind character splitter. Each chunk carries
#its section trail as a prefix + metadata (doc/pmcid/doi/license/section) so the next steps
#(sentence-transformer embed -> pgvector) have everything they need.
import os, re, json, glob
import xml.etree.ElementTree as ET
from html.entities import name2codepoint

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
OUT_PATH = os.path.join(os.path.dirname(__file__), "corpus_chunks.json")
TOKEN_BUDGET = 350   #target chunk size. rough, tuned later against retrieval eval

#Sections that are pure noise for a metrics-interpretation RAG - skip them entirely
SKIP_SECTIONS = ("reference", "acknowledg", "funding", "competing interest",
                 "conflict of interest", "author contribution", "data availability",
                 "supplementary", "abbreviation", "ethics", "consent", "availability of data")

def rough_tokens(text):
    #cheap proxy: ~0.75 tokens per word. swap for the embed model's tokenizer if i ever need it exact
    return int(len(text.split()) / 0.75)

def lname(el):
    #localname - strips any {namespace} prefix so i can match 'sec'/'p'/'title' regardless of ns
    return el.tag.rsplit("}", 1)[-1]

#Inline/block noise i never want in chunk text - citation markers, figures, tables, math.
#Skipping these subtrees during traversal (instead of regex-stripping the raw string) keeps the
#XML balanced, and crucially still keeps each noise element's TAIL text (the words after it).
NOISE_TAGS = {"xref", "fig", "table-wrap", "disp-formula", "inline-formula", "tex-math",
              "graphic", "media", "supplementary-material", "math"}

def _collect_text(e, out):
    if lname(e) in NOISE_TAGS:
        return
    if e.text:
        out.append(e.text)
    for c in e:
        _collect_text(c, out)
        if c.tail:            #tail lives outside the child, so we keep it even when the child is noise
            out.append(c.tail)

def text_of(el):
    out = []
    _collect_text(el, out)
    return re.sub(r"\s+", " ", "".join(out)).strip()

#JATS uses tons of named entities (&deg; &alpha; &plusmn;...) defined in a DTD the stdlib parser
#wont load, so it would crash. I resolve the known ones to unicode myself before parsing, leaving
#the 5 real XML built-ins alone and dropping the rare unknown so the parse cant break.
_XML_BUILTIN = {"amp", "lt", "gt", "quot", "apos"}
def resolve_entities(s):
    def repl(m):
        name = m.group(1)
        if name in _XML_BUILTIN:
            return m.group(0)
        cp = name2codepoint.get(name)
        return chr(cp) if cp else ""
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9]+);", repl, s)

def preprocess(raw):
    #string-level cleanup BEFORE parsing is ONLY safe for things that dont unbalance tags:
    #drop the DTD ref and resolve named entities. Noise elements are skipped later during traversal
    #(regex-stripping whole tag blocks was unbalancing the XML and crashing the parser).
    raw = re.sub(r"<!DOCTYPE.*?>", "", raw, flags=re.S)
    return resolve_entities(raw)

def load_manifest():
    #doc name -> {pmcid, license} from manifest.tsv. The manifest is authoritative for pmcid/license
    #(the XML's pmc article-id attribute is inconsistent across publishers).
    meta = {}
    man = os.path.join(PAPERS_DIR, "manifest.tsv")
    if os.path.exists(man):
        for line in open(man, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith("PMC"):
                meta[parts[1]] = {"pmcid": parts[0], "license": parts[2]}
    return meta

def get_meta(article, doc_name, manifest):
    #doi/title/year come from the XML front matter; pmcid + license come from the manifest
    doi = title = year = None
    for e in article.iter():
        if lname(e) == "article-id" and e.get("pub-id-type") == "doi" and not doi:
            doi = (e.text or "").strip()
        elif lname(e) == "article-title" and title is None:
            title = text_of(e)
        elif lname(e) == "year" and year is None:
            year = (e.text or "").strip()
    m = manifest.get(doc_name, {})
    return {"doc_id": doc_name, "pmcid": m.get("pmcid"), "doi": doi,
            "title": title, "year": year, "license": m.get("license", "unknown")}

def collect_sections(elem, trail, groups):
    #recurse the section tree. each <sec> extends the title trail; direct <p>s under a level get
    #grouped under that trail, and a nested <sec> flushes the current buffer then recurses deeper.
    title_el = next((c for c in elem if lname(c) == "title"), None)
    title_txt = text_of(title_el) if title_el is not None else ""
    new_trail = trail + ([title_txt] if title_txt else [])
    paras = []
    for child in elem:
        tag = lname(child)
        if tag == "sec":
            if paras:
                groups.append((new_trail, paras)); paras = []
            collect_sections(child, new_trail, groups)
        elif tag == "p":
            txt = text_of(child)
            if txt:
                paras.append(txt)
    if paras:
        groups.append((new_trail, paras))

def make_chunk(trail_str, paras, meta, idx):
    body = "\n\n".join(paras)
    #prefix the section trail so the chunk is self-contained out of context (contextual chunking)
    text = f"[{meta['title']} | {trail_str}]\n{body}"
    return {"chunk_id": f"{meta['doc_id']}#{idx:03d}",
            "doc_id": meta["doc_id"], "pmcid": meta["pmcid"], "doi": meta["doi"],
            "license": meta["license"], "section": trail_str,
            "text": text, "n_tokens": rough_tokens(body)}

def pack(groups, meta):
    #within each section, pack paragraphs up to the token budget, starting a fresh chunk when full
    chunks, idx = [], 0
    for trail, paras in groups:
        if any(any(s in seg.lower() for s in SKIP_SECTIONS) for seg in trail):
            continue
        trail_str = " > ".join(trail) if trail else "Body"
        buf = []
        for p in paras:
            buf.append(p)
            if rough_tokens(" ".join(buf)) >= TOKEN_BUDGET:
                chunks.append(make_chunk(trail_str, buf, meta, idx)); idx += 1; buf = []
        if buf:
            chunks.append(make_chunk(trail_str, buf, meta, idx)); idx += 1
    return chunks

def chunk_file(path, manifest):
    doc_name = os.path.splitext(os.path.basename(path))[0]
    raw = open(path, encoding="utf-8").read()
    root = ET.fromstring(preprocess(raw))
    article = next((e for e in root.iter() if lname(e) == "article"), root)
    meta = get_meta(article, doc_name, manifest)

    #grab the abstract (dense with findings) + the whole body as one pool of section groups
    groups = []
    for e in article.iter():
        if lname(e) == "abstract":
            collect_sections(e, ["Abstract"], groups)
        elif lname(e) == "body":
            collect_sections(e, [], groups)
    return pack(groups, meta)

def main():
    manifest = load_manifest()
    all_chunks = []
    for path in sorted(glob.glob(os.path.join(PAPERS_DIR, "*.xml"))):
        try:
            chunks = chunk_file(path, manifest)
            all_chunks.extend(chunks)
            print(f"{os.path.basename(path):40s} -> {len(chunks):3d} chunks")
        except Exception as e:
            print(f"{os.path.basename(path):40s} -> FAILED: {type(e).__name__}: {e}")
    #drop tiny fragments (stray one-liners, leftover labels) - they just pollute retrieval
    all_chunks = [c for c in all_chunks if c["n_tokens"] >= 10]
    json.dump(all_chunks, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    toks = [c["n_tokens"] for c in all_chunks] or [0]
    print(f"\nwrote {len(all_chunks)} chunks -> {OUT_PATH}")
    print(f"tokens/chunk: min {min(toks)}  avg {sum(toks)//len(toks)}  max {max(toks)}")

if __name__ == "__main__":
    main()
