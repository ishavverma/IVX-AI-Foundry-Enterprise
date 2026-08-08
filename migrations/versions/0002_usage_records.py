"""Add organization usage accounting."""

from alembic import op
import sqlalchemy as sa

revision = "0002_usage_records"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_records",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("organization_id", sa.String(length=36), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unit", sa.String(length=32), nullable=False, server_default="units"),
        sa.Column("resource_id", sa.String(length=36), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_records_organization_id", "usage_records", ["organization_id"])
    op.create_index("ix_usage_records_metric", "usage_records", ["metric"])


def downgrade() -> None:
    op.drop_index("ix_usage_records_metric", table_name="usage_records")
    op.drop_index("ix_usage_records_organization_id", table_name="usage_records")
    op.drop_table("usage_records")
