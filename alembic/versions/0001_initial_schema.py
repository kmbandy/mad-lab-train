"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-02
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";

        CREATE TYPE job_status AS ENUM (
            'pending', 'queued', 'running', 'paused',
            'completed', 'failed', 'cancelled'
        );

        CREATE TYPE stage_status AS ENUM (
            'pending', 'running', 'paused', 'completed', 'failed', 'skipped'
        );

        CREATE TYPE stage_type AS ENUM (
            'dataset_prep', 'data_gen', 'finetune', 'pretrain',
            'quant', 'merge', 'prune', 'eval', 'convert', 'upload'
        );

        CREATE TYPE execution_target AS ENUM ('local', 'ec2');

        CREATE TABLE runs (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name              TEXT NOT NULL,
            template_name     TEXT NOT NULL,
            status            job_status NOT NULL DEFAULT 'pending',
            execution_target  execution_target NOT NULL DEFAULT 'local',
            ec2_config        JSONB,
            priority          INTEGER NOT NULL DEFAULT 100,
            scheduled_for     TIMESTAMPTZ,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            queued_at         TIMESTAMPTZ,
            started_at        TIMESTAMPTZ,
            ended_at          TIMESTAMPTZ,
            retain_logs_until TIMESTAMPTZ NOT NULL,
            error             TEXT
        );

        CREATE INDEX idx_runs_status ON runs(status);
        CREATE INDEX idx_runs_created_at ON runs(created_at DESC);

        CREATE TABLE stages (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            sequence    INTEGER NOT NULL,
            stage_type  stage_type NOT NULL,
            status      stage_status NOT NULL DEFAULT 'pending',
            input_path  TEXT,
            output_path TEXT,
            started_at  TIMESTAMPTZ,
            ended_at    TIMESTAMPTZ,
            error       TEXT,
            UNIQUE (run_id, sequence)
        );

        CREATE INDEX idx_stages_run_id ON stages(run_id);

        CREATE TABLE stage_configs (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            stage_id    UUID NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
            stage_type  stage_type NOT NULL,
            config      JSONB NOT NULL,
            UNIQUE (stage_id)
        );

        CREATE INDEX idx_stage_configs_run_id ON stage_configs(run_id);

        CREATE TABLE checkpoints (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id        UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            stage_id      UUID NOT NULL REFERENCES stages(id) ON DELETE CASCADE,
            sequence      INTEGER NOT NULL,
            is_clean      BOOLEAN NOT NULL DEFAULT TRUE,
            artifact_path TEXT NOT NULL,
            metadata      JSONB NOT NULL DEFAULT '{}',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (stage_id, sequence)
        );

        CREATE INDEX idx_checkpoints_run_id ON checkpoints(run_id);
        CREATE INDEX idx_checkpoints_stage_id ON checkpoints(stage_id);

        CREATE TABLE events (
            id          BIGSERIAL PRIMARY KEY,
            run_id      UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            stage_id    UUID REFERENCES stages(id) ON DELETE CASCADE,
            stage_type  stage_type,
            event_type  TEXT NOT NULL,
            data        JSONB NOT NULL DEFAULT '{}',
            ts          TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX idx_events_run_id ON events(run_id);
        CREATE INDEX idx_events_run_id_ts ON events(run_id, ts DESC);
        CREATE INDEX idx_events_stage_id ON events(stage_id);

        CREATE TABLE templates (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT UNIQUE NOT NULL,
            label       TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            chain       JSONB NOT NULL,
            is_builtin  BOOLEAN NOT NULL DEFAULT FALSE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """)


def downgrade() -> None:
    op.execute("""
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS checkpoints;
        DROP TABLE IF EXISTS stage_configs;
        DROP TABLE IF EXISTS stages;
        DROP TABLE IF EXISTS runs;
        DROP TABLE IF EXISTS templates;
        DROP TYPE IF EXISTS execution_target;
        DROP TYPE IF EXISTS stage_type;
        DROP TYPE IF EXISTS stage_status;
        DROP TYPE IF EXISTS job_status;
    """)
