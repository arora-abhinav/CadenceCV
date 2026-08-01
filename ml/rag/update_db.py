from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer
import numpy

#Same presaved encoder as the insert script - needed here because updating a chunk's text means its
#embedding is now stale, so i re-encode while im at it (avoiding a redownload of the weights)
transformer_path = os.path.join(os.getcwd(), 'ml/rag/transformer_models', 'sentence_transformer.pkl')
if not os.path.exists(transformer_path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save(transformer_path)
else:
    model = SentenceTransformer(transformer_path)

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True)

#Same raw-sql-in-plain-text style as chunk_into_db.py. This script is the UPDATE counterpart: it re-syncs
#rows that ALREADY exist to whatever the json now says (e.g. i backfilled design/n_participants on the
#documents, or re-chunked and the text/embedding changed). INSERT..ON CONFLICT DO NOTHING skips existing
#rows, so it cant push corrections - thats why the updates live here.
with open("/Users/abhinavarora/Desktop/CadenceCV/ml/rag/corpus_chunks.json", "r") as file:
    chunks = json.load(file)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/rag/documents.json", "r") as file:
    documents = json.load(file)


def update_document_in_db(documents):
    with engine.connect() as conn:
        for doc in documents:
            #match the row on pmcid (the identity key i insert on) and overwrite every non-key column with
            #the current json value. doi/pmcid arent in the SET - pmcid is the WHERE, and i dont want to be
            #reassigning dois. rows whose pmcid isnt in the table yet just match nothing and are skipped.
            conn.execute(
                text(
                    """
                    UPDATE documents SET
                        title            = :title,
                        first_author     = :first_author,
                        year             = :year,
                        journal          = :journal,
                        url              = :url,
                        license          = :license,
                        text_surfaceable = :text_surfaceable,
                        design           = :design,
                        n_participants   = :n_participants,
                        population       = :population,
                        conditions       = :conditions,
                        speed_min_ms     = :speed_min_ms,
                        speed_max_ms     = :speed_max_ms,
                        cohort_id        = :cohort_id
                    WHERE pmcid = :pmcid
                    """
                ),
                {
                    "pmcid": doc["pmcid"],
                    "title": doc["title"],
                    "first_author": doc["first_author"],
                    "year": doc["year"],
                    "journal": doc["journal"],
                    "url": doc["url"],
                    "license": doc["license"],
                    "text_surfaceable": doc["text_surfaceable"],
                    "design": doc["design"],
                    "n_participants": doc["n_participants"],
                    "population": doc["population"],
                    "conditions": doc["conditions"],
                    "speed_min_ms": doc["speed_min_ms"],
                    "speed_max_ms": doc["speed_max_ms"],
                    "cohort_id": doc["cohort_id"],
                },
            )
        #with engine.connect() doesnt autocommit in sqlalchemy 2.0, so the writes vanish without this
        conn.commit()


def update_chunks_in_db(data):
    with engine.connect() as conn:
        for chunk in data:
            #if the text changed the old embedding is wrong, so re-encode -> stringify -> CAST to VECTOR,
            #exactly like the insert path
            embedding = str(model.encode(chunk["text"]).tolist())

            #same derivations as the insert: order index off the chunk_id, abstract-vs-body off the section
            chunk_index = int(chunk["chunk_id"].split("#")[-1])
            part = "abstract" if chunk["section"].lower().startswith("abstract") else "body"

            #match on chunk_id (the primary key) and overwrite everything else, pmcid included in case a
            #re-chunk moved the chunk under a different paper
            conn.execute(
                text(
                    """
                    UPDATE chunks SET
                        pmcid       = :pmcid,
                        chunk_index = :chunk_index,
                        part        = :part,
                        section     = :section,
                        text        = :text,
                        n_tokens    = :n_tokens,
                        embedding   = CAST(:embedding AS VECTOR)
                    WHERE chunk_id = :chunk_id
                    """
                ),
                {
                    "chunk_id": chunk["chunk_id"],
                    "pmcid": chunk["pmcid"],
                    "chunk_index": chunk_index,
                    "part": part,
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "n_tokens": chunk["n_tokens"],
                    "embedding": embedding,
                },
            )
        conn.commit()


#Documents FIRST - if a chunk's pmcid moved onto a paper that only exists after a doc update, the FK needs
#that parent to be current before the chunk update lands
update_document_in_db(documents)
update_chunks_in_db(chunks)
