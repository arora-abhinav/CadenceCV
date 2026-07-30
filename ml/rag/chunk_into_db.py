from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import json
from sentence_transformers import SentenceTransformer
import numpy

#Avoiding redownloading weights again
transformer_path = os.path.join(os.getcwd(), 'ml/rag/transformer_models', 'sentence_transformer.pkl')
if not os.path.exists(transformer_path):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.save(transformer_path)
else:
    model = SentenceTransformer(transformer_path)

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")
print(DATABASE_URL)

engine = create_engine(DATABASE_URL, echo=True)

#My preferred way of operating with sql alchemy is to write stuff in raw plain text
with open("/Users/abhinavarora/Desktop/CadenceCV/ml/rag/corpus_chunks.json", "r") as file:
    chunks = json.load(file)

with open("/Users/abhinavarora/Desktop/CadenceCV/ml/rag/documents.json", "r") as file:
    documents = json.load(file)

#Papers go in FIRST. chunks.doc_id is a FK onto documents.doc_id, so the parent rows have to exist
#before i can hang chunks off them. Each dict here is already the full documents-table shape (from the parser).
def insert_document_into_db(documents):
    with engine.connect() as conn:
        for doc in documents:
            #ON CONFLICT keeps this re-runnable: pmcid is UNIQUE so a repeat run just skips instead of erroring
            conn.execute(
                text(
                    """
                    INSERT INTO documents (
                        doi, pmcid, title, first_author, year, journal, url, license,
                        text_surfaceable, design, n_participants, population, conditions,
                        speed_min_ms, speed_max_ms, cohort_id
                    )
                    VALUES (
                        :doi, :pmcid, :title, :first_author, :year, :journal, :url, :license,
                        :text_surfaceable, :design, :n_participants, :population, :conditions,
                        :speed_min_ms, :speed_max_ms, :cohort_id
                    )
                    ON CONFLICT (pmcid) DO NOTHING
                    """
                ),
                {
                    "doi": doc["doi"],
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
        #Executing the changes
        conn.commit()


def insert_chunks_into_db(data):
    with engine.connect() as conn:
        for chunk in data:
            #.tolist() required since ogvector doesnt support numpy arrays
            embedding = str(model.encode(chunk["text"]).tolist())

            #chunk_id looks like 'anderson2022_step_rate#007' - the bit after the # is its order index
            chunk_index = int(chunk["chunk_id"].split("#")[-1])
            #Separating abstract vs body.
            part = "abstract" if chunk["section"].lower().startswith("abstract") else "body"

            #pmcid is the FK straight onto documents(pmcid) now, and the chunk json already has it, so i
            #just bind it in - no more looking up a SERIAL doc_id first
            conn.execute(
                text(
                    """
                    INSERT INTO chunks (
                        chunk_id, pmcid, chunk_index, part, section, text, n_tokens, embedding
                    )
                    VALUES (
                        :chunk_id, :pmcid, :chunk_index, :part, :section, :text, :n_tokens,
                        CAST(:embedding AS VECTOR)
                    )
                    ON CONFLICT (chunk_id) DO NOTHING
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
        #same deal - explicit commit or the whole batch rolls back on close
        conn.commit()

