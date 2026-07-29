CREATE TABLE chunks (
    chunk_id       TEXT PRIMARY KEY,          -- 'rivadulla2021_footnet#029'
    doc_id         TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index    INT  NOT NULL,             -- 29 — ordering + neighbor expansion

    -- where in the document this came from
    part           TEXT NOT NULL,             -- 'abstract'|'body'|'peer-review'|'author-response'
                                              --   |'editor-report'|'caption'|'table'|'supplementary'
    section_path   TEXT[],                    -- {'Materials & methods','Data collection'}
    section_index  INT,                       -- order of section within doc

    -- text: raw and embedding input kept separate
    text           TEXT NOT NULL,             -- what goes in the prompt
    context_prefix TEXT,                      -- '[Title | Materials & methods > Data collection]'
    n_tokens       INT  NOT NULL,
    embedding      VECTOR(384),               -- match your encoder: bge-small = 384

    -- ingestion hygiene
    content_hash   TEXT NOT NULL,             -- sha256(text) — dedup, change detection
    retrievable    BOOLEAN NOT NULL DEFAULT TRUE,

    UNIQUE (doc_id, chunk_index)
);

CREATE INDEX idx_chunks_doc ON chunks (doc_id);