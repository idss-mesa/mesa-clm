-- 0001_mesa_clm: the mesa-clm decision provenance sidecar (DESIGN D11), Postgres dialect.
--
-- GENERATED from mesa_clm.provenance.store.DUCKDB_DDL by
-- mesa_clm.provenance.migrate.render_postgres_migration(); do not edit by hand. The DuckDB
-- dialect is the source: this file is its word-for-word translation (JSON -> JSONB,
-- DOUBLE -> DOUBLE PRECISION; ids are TEXT in both) plus the Postgres-only indexes, and
-- tests/unit/test_provenance_ddl.py asserts the packaged file equals the rendering and that
-- the two dialects expose the same tables, columns, nullability and CHECK constraints.
--
-- The vocabulary CHECKs (level, calibration, method, shape, outcome, write_status,
-- accepted_by, via, label_source, run status) are rendered from mesa_clm.vocab; the
-- ``labels`` table is mesa_clm.provenance.labels.LABELS_DDL verbatim (D1, D30). Every UNIQUE
-- key is over NOT NULL columns ('' sentinels) because DuckDB 1.5.x has no
-- UNIQUE NULLS NOT DISTINCT; the same shape is kept here so the dialects stay identical.
-- No foreign keys: rows are written in one transaction per run and pruned per run.
-- This schema never touches schema `mesa` (mesa-ducklake); the join into the AVU history is
-- (project_id, snapshot_id, irods_path, attribute, value, unit) on avu_links.

BEGIN;

CREATE SCHEMA IF NOT EXISTS mesa_clm;

CREATE TABLE IF NOT EXISTS mesa_clm.schema_versions (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now());

CREATE TABLE IF NOT EXISTS mesa_clm.runs (
        run_id TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'decided', 'applied', 'partial', 'failed', 'abandoned')),
        started_at TIMESTAMPTZ NOT NULL,
        finished_at TIMESTAMPTZ,
        terminal_at TIMESTAMPTZ,
        exported_at TIMESTAMPTZ,
        card_name TEXT NOT NULL,
        card_sha256 TEXT NOT NULL,
        irods_path TEXT,
        project_id TEXT,
        planner TEXT NOT NULL,
        planner_model TEXT,
        planner_prompt_sha256 TEXT,
        plan_json JSONB NOT NULL,
        planner_fallback BOOLEAN NOT NULL DEFAULT FALSE,
        provider TEXT NOT NULL,
        tier TEXT NOT NULL,
        clm_model TEXT NOT NULL,
        encoder_model TEXT NOT NULL,
        clm_commit TEXT NOT NULL DEFAULT '',
        schema_sha256 TEXT NOT NULL,
        encoder_fp TEXT NOT NULL,
        clm_model_fp TEXT NOT NULL,
        serving_lock_sha TEXT,
        framings_lock_sha TEXT NOT NULL,
        artifacts_version TEXT,
        labels_sha256 TEXT,
        mesa_clm_version TEXT NOT NULL,
        policy_profile TEXT NOT NULL,
        config_sha256 TEXT NOT NULL,
        history_backend TEXT CHECK (history_backend IS NULL OR history_backend IN ('direct', 'spool', 'none')),
        history_waiver_actor TEXT,
        vm_id TEXT NOT NULL,
        degraded BOOLEAN NOT NULL DEFAULT FALSE,
        n_decisions INTEGER,
        n_clm_calls INTEGER,
        n_encoder_tokens BIGINT,
        seconds DOUBLE PRECISION);

CREATE TABLE IF NOT EXISTS mesa_clm.decisions (
        decision_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        group_id TEXT,
        parent_decision_id TEXT,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        question_key TEXT NOT NULL,
        framing_id TEXT NOT NULL,
        shape TEXT NOT NULL CHECK (shape IN ('rank_fit', 'choice')),
        k INTEGER NOT NULL,
        scope TEXT NOT NULL CHECK (scope IN ('dataset', 'column', 'site', 'avu')),
        column_name TEXT,
        site_code TEXT,
        state_sha256 TEXT NOT NULL,
        state_json JSONB NOT NULL,
        target_sha256 TEXT NOT NULL,
        context_sha256 TEXT NOT NULL,
        context_tokens INTEGER NOT NULL,
        truncated BOOLEAN NOT NULL DEFAULT FALSE,
        method TEXT NOT NULL CHECK (method IN ('clm', 'fake', 'rule', 'planner', 'ols_rank', 'claude:structured_output', 'unavailable')),
        model TEXT NOT NULL DEFAULT '',
        level TEXT NOT NULL CHECK (level IN ('none', 'zero_shot', 'calibrated', 'probe', 'head')),
        calibration TEXT NOT NULL CHECK (calibration IN ('none', 'uncalibrated', 'platt', 'temperature')),
        probs JSONB,
        raw_probs JSONB,
        confidence DOUBLE PRECISION,
        clm_confidence DOUBLE PRECISION,
        s_c DOUBLE PRECISION,
        p_fit DOUBLE PRECISION,
        margin DOUBLE PRECISION,
        anchor_index INTEGER,
        answer_index INTEGER NOT NULL,
        answer TEXT NOT NULL,
        options JSONB NOT NULL,
        rank INTEGER,
        artifact_version TEXT,
        feature_spec TEXT,
        encoder_fp TEXT NOT NULL,
        clm_model_fp TEXT NOT NULL,
        schema_sha256 TEXT NOT NULL,
        latency_ms DOUBLE PRECISION,
        threshold_auto DOUBLE PRECISION,
        threshold_propose DOUBLE PRECISION,
        outcome TEXT NOT NULL CHECK (outcome IN ('auto', 'proposed', 'human', 'escalated', 'abstain', 'rejected', 'rule', 'decider_unavailable')),
        reason TEXT,
        ts TIMESTAMPTZ NOT NULL,
        CHECK ((probs IS NULL) = (calibration = 'none')),
        CHECK (method NOT IN ('claude:structured_output', 'ols_rank', 'planner', 'rule', 'unavailable') OR (probs IS NULL AND level = 'none')),
        CHECK (level <> 'zero_shot' OR calibration = 'uncalibrated'),
        CHECK (level NOT IN ('calibrated', 'probe', 'head') OR calibration IN ('platt', 'temperature')),
        CHECK (method <> 'ols_rank' OR (outcome IN ('proposed', 'abstain') AND rank IS NOT NULL)),
        CHECK (shape <> 'choice' OR anchor_index IS NULL));

CREATE TABLE IF NOT EXISTS mesa_clm.decision_options (
        decision_id TEXT NOT NULL,
        option_index INTEGER NOT NULL,
        option_key TEXT NOT NULL,
        option_text TEXT NOT NULL,
        rank INTEGER,
        s_c DOUBLE PRECISION,
        p_fit DOUBLE PRECISION,
        prob DOUBLE PRECISION,
        raw_prob DOUBLE PRECISION,
        masked BOOLEAN NOT NULL DEFAULT FALSE,
        action_sha256 TEXT,
        PRIMARY KEY (decision_id, option_index),
        UNIQUE (decision_id, option_key));

CREATE TABLE IF NOT EXISTS mesa_clm.decision_groups (
        group_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        question_key TEXT,
        scope TEXT NOT NULL CHECK (scope IN ('dataset', 'column', 'site', 'avu')),
        column_name TEXT,
        site_code TEXT,
        aspect TEXT,
        ontology_id TEXT,
        search_json JSONB NOT NULL,
        n_candidates INTEGER NOT NULL,
        winner_decision_id TEXT,
        top_p_fit DOUBLE PRECISION,
        group_margin DOUBLE PRECISION,
        level TEXT CHECK (level IS NULL OR level IN ('none', 'zero_shot', 'calibrated', 'probe', 'head')),
        method TEXT CHECK (method IS NULL OR method IN ('clm', 'fake', 'rule', 'planner', 'ols_rank', 'claude:structured_output', 'unavailable')),
        outcome TEXT NOT NULL CHECK (outcome IN ('auto', 'proposed', 'human', 'escalated', 'abstain', 'rejected', 'rule', 'decider_unavailable')),
        anchor_won BOOLEAN NOT NULL DEFAULT FALSE,
        escalated_from TEXT,
        ts TIMESTAMPTZ NOT NULL);

CREATE TABLE IF NOT EXISTS mesa_clm.avu_links (
        link_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        group_id TEXT,
        decision_id TEXT,
        project_id TEXT,
        snapshot_id BIGINT,
        spool_batch_id TEXT,
        irods_path TEXT,
        target_type TEXT NOT NULL DEFAULT 'data_object',
        attribute TEXT NOT NULL,
        value TEXT NOT NULL,
        unit TEXT NOT NULL DEFAULT '',
        op TEXT NOT NULL DEFAULT 'add' CHECK (op IN ('add', 'delete')),
        term_curie TEXT,
        term_iri TEXT,
        term_label TEXT,
        ontology_id TEXT,
        aspect TEXT,
        column_name TEXT NOT NULL DEFAULT '',
        site_code TEXT NOT NULL DEFAULT '',
        value_kind TEXT,
        source TEXT,
        write_status TEXT NOT NULL CHECK (write_status IN ('proposed', 'accepted', 'dry_run', 'writing', 'written', 'spooled', 'reverted', 'mirror_failed', 'local_only')),
        accepted_by TEXT CHECK (accepted_by IS NULL OR accepted_by IN ('policy', 'human', 'agent')),
        duplicate_of TEXT,
        written_at TIMESTAMPTZ,
        UNIQUE (run_id, attribute, value, unit, column_name, site_code),
        CHECK (write_status <> 'accepted' OR accepted_by IS NOT NULL));

CREATE TABLE IF NOT EXISTS mesa_clm.human_overrides (
        override_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        group_id TEXT,
        decision_id TEXT,
        link_id TEXT,
        actor TEXT NOT NULL,
        via TEXT NOT NULL CHECK (via IN ('elicitation', 'cli', 'tool')),
        action TEXT NOT NULL CHECK (action IN ('pick', 'none', 'accept', 'reject', 'decline', 'restore')),
        chosen_decision_id TEXT,
        chosen_option_key TEXT,
        elicitation_key TEXT,
        label_source TEXT CHECK (label_source IS NULL OR label_source IN ('curator', 'curator_implicit', 'agent_pick', 'consensus_all', 'consensus_majority', 'consensus_negative', 'teacher', 'teacher_implicit', 'gold')),
        labels_written INTEGER NOT NULL DEFAULT 0,
        offered JSONB NOT NULL,
        ts TIMESTAMPTZ NOT NULL,
        CHECK (via <> 'tool' OR label_source IS NULL OR label_source = 'agent_pick'));

CREATE TABLE IF NOT EXISTS mesa_clm.labels (
        label_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        task_key TEXT NOT NULL,
        target_sha256 TEXT NOT NULL,
        option_key TEXT NOT NULL DEFAULT '',
        label_source TEXT NOT NULL CHECK (label_source IN ('curator', 'curator_implicit', 'agent_pick', 'consensus_all', 'consensus_majority', 'consensus_negative', 'teacher', 'teacher_implicit', 'gold')),
        label TEXT NOT NULL,
        label_index INTEGER NOT NULL CHECK (label_index >= 0),
        weight DOUBLE PRECISION NOT NULL CHECK (weight >= 0 AND weight <= 1),
        state_sha256 TEXT NOT NULL,
        state_json JSONB NOT NULL,
        card TEXT NOT NULL DEFAULT '',
        product_code TEXT NOT NULL DEFAULT '',
        leak_group TEXT NOT NULL DEFAULT '',
        fold_eligible BOOLEAN NOT NULL DEFAULT TRUE,
        bench_card BOOLEAN NOT NULL DEFAULT FALSE,
        origin TEXT NOT NULL DEFAULT '',
        actor TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (task_key, target_sha256, option_key, label_source));

CREATE TABLE IF NOT EXISTS mesa_clm.audits (
        audit_id TEXT PRIMARY KEY,
        task_key TEXT NOT NULL,
        artifact_version TEXT NOT NULL,
        n INTEGER NOT NULL CHECK (n >= 0),
        n_cards INTEGER NOT NULL CHECK (n_cards >= 0),
        cards JSONB NOT NULL,
        reviewer TEXT NOT NULL,
        n_errors INTEGER NOT NULL CHECK (n_errors >= 0),
        cp95_upper DOUBLE PRECISION NOT NULL CHECK (cp95_upper >= 0 AND cp95_upper <= 1),
        risk DOUBLE PRECISION NOT NULL CHECK (risk >= 0 AND risk <= 1),
        passed BOOLEAN NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        CHECK (n_errors <= n));

CREATE TABLE IF NOT EXISTS mesa_clm.clm_calls (
        call_id TEXT PRIMARY KEY,
        run_id TEXT,
        endpoint TEXT NOT NULL,
        model TEXT NOT NULL,
        n_questions INTEGER NOT NULL,
        n_candidates INTEGER NOT NULL,
        input_tokens INTEGER,
        latency_ms DOUBLE PRECISION NOT NULL,
        cache_hit BOOLEAN,
        status TEXT NOT NULL CHECK (status IN ('ok', 'error', 'timeout', 'unavailable')),
        ts TIMESTAMPTZ NOT NULL);

CREATE INDEX IF NOT EXISTS runs_owner_started ON mesa_clm.runs (owner, started_at);
CREATE INDEX IF NOT EXISTS decisions_run_seq ON mesa_clm.decisions (run_id, seq);
CREATE INDEX IF NOT EXISTS decisions_question_key_ts ON mesa_clm.decisions (question_key, ts);
CREATE INDEX IF NOT EXISTS decisions_group ON mesa_clm.decisions (group_id);
CREATE INDEX IF NOT EXISTS decision_groups_run ON mesa_clm.decision_groups (run_id);
CREATE INDEX IF NOT EXISTS avu_links_run ON mesa_clm.avu_links (run_id);
CREATE INDEX IF NOT EXISTS avu_links_path ON mesa_clm.avu_links (irods_path, attribute, value, unit);
CREATE INDEX IF NOT EXISTS avu_links_snapshot ON mesa_clm.avu_links (project_id, snapshot_id);
CREATE INDEX IF NOT EXISTS human_overrides_run ON mesa_clm.human_overrides (run_id);
CREATE INDEX IF NOT EXISTS labels_task_source ON mesa_clm.labels (task_key, label_source);
CREATE INDEX IF NOT EXISTS clm_calls_run ON mesa_clm.clm_calls (run_id);

INSERT INTO mesa_clm.schema_versions (version) VALUES (1) ON CONFLICT DO NOTHING;

COMMIT;
