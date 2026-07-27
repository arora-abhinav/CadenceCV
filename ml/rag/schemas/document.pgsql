CREATE TABLE documents (
    doc_id         SERIAL PRIMARY KEY,

    -- identity
    doi TEXT UNIQUE,
    pmcid TEXT UNIQUE,
    title TEXT NOT NULL,
    first_author TEXT,
    year SMALLINT,
    journal TEXT,
    url TEXT NOT NULL,

    -- licensing
    license TEXT NOT NULL, -- 'CC BY' | 'CC BY-NC-ND'
    text_surfaceable  BOOLEAN NOT NULL DEFAULT TRUE,  -- FALSE for BY-NC-ND: chunk internally, cite as pointer

    -- provenance printed into the generation prompt (never a WHERE clause)
    design TEXT, -- 'meta-analysis' | 'rct' | 'cohort' | 'cross-sectional' | 'validation'
    n_participants INT, -- NULL for reviews
    population TEXT,  -- free text: 'recreational, healthy, mixed sex'
    conditions TEXT[], 

    -- the only fields used for retrieval scoring (soft penalty, NULL-tolerant)
    speed_min_ms REAL,
    speed_max_ms REAL,

    -- cohort dedup: Malisoux 2023 + 2024 share the Luxembourg RCT cohort
    cohort_id TEXT,

    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);