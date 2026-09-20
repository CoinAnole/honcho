"""add last_reinforced_at and established_evidence

Track B PR3 write-time established pass durability:
- documents.last_reinforced_at (nullable timestamptz) for successful
  established reinforce with a new evidence digest
- established_evidence uniqueness ledger (established_id, evidence_digest)
  so reinforce retries are ON CONFLICT DO NOTHING no-ops

Supersession edges stay on documents.internal_metadata (superseded_by /
supersedes) rather than a separate edge table.

Revision ID: c8d4f2a1b3e5
Revises: a7c3e9f1b2d4
Create Date: 2026-09-20

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema, table_exists

revision: str = "c8d4f2a1b3e5"
down_revision: str | None = "a7c3e9f1b2d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if not column_exists("documents", "last_reinforced_at", inspector):
        op.add_column(
            "documents",
            sa.Column(
                "last_reinforced_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            schema=schema,
        )

    if not table_exists("established_evidence", inspector):
        op.create_table(
            "established_evidence",
            sa.Column("established_id", sa.TEXT, nullable=False),
            sa.Column("evidence_digest", sa.TEXT, nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("established_id", "evidence_digest"),
            sa.ForeignKeyConstraint(
                ["established_id"],
                [f"{schema}.documents.id"],
                ondelete="CASCADE",
            ),
            schema=schema,
        )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_exists("established_evidence", inspector):
        op.drop_table("established_evidence", schema=schema)

    if column_exists("documents", "last_reinforced_at", inspector):
        op.drop_column("documents", "last_reinforced_at", schema=schema)
