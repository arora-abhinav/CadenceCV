CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,
    -- pmcid is the FK instead of the SERIAL doc_id: the chunk json already carries pmcid, so i can insert
    -- straight in without looking up an int id first. ON UPDATE CASCADE is necessary in case the pmcid changes
    -- since they're somewhat inconsistent/ ON DELETE CASCADE simply ensures that if a pmcid is gone from the documents
    -- table, so will the fields inside chunks that reference that pmcid.
    pmcid       TEXT NOT NULL REFERENCES documents(pmcid) ON UPDATE CASCADE ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    part        TEXT NOT NULL,
    section     TEXT,
    text        TEXT NOT NULL,
    n_tokens    INT,
    
    --384 is the dimension of the embedded vector by sentence transformers
    embedding   VECTOR(384)
);