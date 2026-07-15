"""add moeify stage type

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-15
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE stage_type ADD VALUE IF NOT EXISTS 'moeify'")


def downgrade() -> None:
    # PostgreSQL cannot remove a single enum value without rebuilding every
    # dependent column. Leaving the value in place is safe on code rollback.
    pass
