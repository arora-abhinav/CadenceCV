#Parses the front matter of each JATS XML paper in ml/rag/papers/ and writes documents.json -
#one record per paper, shaped exactly like the documents table so i can just load it and insert.
#Im using beautifulsoup w/ the lxml 'xml' parser here (not ElementTree like the chunker) because the
#front matter is shallow and bs4's .find/.get_text is way less fiddly for pulling single fields.
import os, glob, json
from bs4 import BeautifulSoup

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
OUT_PATH = os.path.join(os.path.dirname(__file__), "documents.json")

#The CC url in the XML <license> maps cleanly onto the license strings the schema expects
LICENSE_FROM_URL = {
    "by": "CC BY",
    "by-nc-nd": "CC BY-NC-ND",
}

def load_manifest():
    #pmcid + license from manifest.tsv - its the authoritative source for these two (the XML's pmc
    #article-id + license wording drift between publishers, the manifest is the one i curated by hand)
    meta = {}
    man = os.path.join(PAPERS_DIR, "manifest.tsv")
    for line in open(man, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[0].startswith("PMC"):
            meta[parts[1]] = {"pmcid": parts[0], "license": parts[2]}
    return meta

def get_doi(am):
    #doi lives as an <article-id pub-id-type="doi"> in the article-meta
    tag = am.find("article-id", attrs={"pub-id-type": "doi"})
    return tag.get_text(strip=True) if tag else None

def get_title(am):
    #the real article title is the one inside <title-group>, NOT any stray <article-title> in the refs
    tg = am.find("title-group")
    if tg and tg.find("article-title"):
        return tg.find("article-title").get_text(" ", strip=True)
    return None

def get_first_author(am):
    #first <contrib contrib-type="author"> - build "Given Surname" so it reads like a normal citation name
    author = am.find("contrib", attrs={"contrib-type": "author"})
    if not author:
        return None
    surname = author.find("surname")
    given = author.find("given-names")
    surname = surname.get_text(strip=True) if surname else ""
    given = given.get_text(strip=True) if given else ""
    return (given + " " + surname).strip() or None

def get_year(am):
    #a paper carries several <pub-date>s (online vs print vs volume). i want the citation year, so i
    #prefer the print/collection year and only fall back to the epub (online-first) date if thats all there is
    dates = {}
    for pd in am.find_all("pub-date"):
        kind = pd.get("pub-type") or pd.get("date-type")
        year = pd.find("year")
        if year:
            dates[kind] = year.get_text(strip=True)
    for pref in ("ppub", "collection", "epub"):
        if pref in dates:
            return int(dates[pref])
    #nothing matched the usual labels - just take whatever year we found
    return int(next(iter(dates.values()))) if dates else None

def get_journal(soup):
    #journal-title sits up in <journal-meta>, separate from the article-meta the rest comes from
    jt = soup.find("journal-title")
    return jt.get_text(" ", strip=True) if jt else None

def parse_paper(path, manifest):
    doc_id = os.path.splitext(os.path.basename(path))[0]
    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "xml")
    am = soup.find("article-meta")

    man = manifest.get(doc_id, {})
    pmcid = man.get("pmcid")
    license_ = man.get("license")

    #every field the documents table has, in schema order. the front matter cant tell me the study-design
    #stuff (design/n_participants/population/... need the actual paper read), so those go out as null for
    #me to backfill by hand later - keeps each record the full shape so the insert is a straight column map
    return {
        "doc_id": doc_id,                       #not a db column, just so i can eyeball which paper is which
        "doi": get_doi(am),
        "pmcid": pmcid,
        "title": get_title(am),
        "first_author": get_first_author(am),
        "year": get_year(am),
        "journal": get_journal(soup),
        "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/",
        "license": license_,
        #BY-NC-ND text cant be shown to the user, only cited as a pointer (see the schema comment)
        "text_surfaceable": license_ != "CC BY-NC-ND",
        #--- everything below needs the paper actually read, so null for now ---
        "design": None,
        "n_participants": None,
        "population": None,
        "conditions": None,
        "speed_min_ms": None,
        "speed_max_ms": None,
        "cohort_id": None,
    }

def main():
    manifest = load_manifest()
    docs = []
    for path in sorted(glob.glob(os.path.join(PAPERS_DIR, "*.xml"))):
        try:
            docs.append(parse_paper(path, manifest))
            print(f"{os.path.basename(path):40s} -> ok")
        except Exception as e:
            print(f"{os.path.basename(path):40s} -> FAILED: {type(e).__name__}: {e}")
    json.dump(docs, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nwrote {len(docs)} documents -> {OUT_PATH}")

if __name__ == "__main__":
    main()
