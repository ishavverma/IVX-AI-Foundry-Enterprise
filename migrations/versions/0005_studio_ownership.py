"""Track ownership for studio artifacts."""

from alembic import op
import sqlalchemy as sa

revision = "0005_studio_ownership"
down_revision = "0004_artifact_ownership"
branch_labels = None
depends_on = None

TABLES = ("conversation_sessions", "prompt_templates", "agent_definitions", "mcp_servers")


def upgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("created_by", sa.String(length=36), nullable=True))
            batch.create_index(f"ix_{table}_created_by", ["created_by"])
            batch.create_foreign_key(f"fk_{table}_created_by_users", "users", ["created_by"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    for table in reversed(TABLES):
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(f"fk_{table}_created_by_users", type_="foreignkey")
            batch.drop_index(f"ix_{table}_created_by")
            batch.drop_column("created_by")
