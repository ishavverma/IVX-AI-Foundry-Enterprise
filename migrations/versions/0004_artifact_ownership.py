"""Track ownership for evaluation, endpoint, and registry artifacts."""

from alembic import op
import sqlalchemy as sa

revision = "0004_artifact_ownership"
down_revision = "0003_checkpoint_ownership"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("evaluation_runs", "registry_assets", "inference_endpoints"):
        with op.batch_alter_table(table) as batch:
            batch.add_column(sa.Column("created_by", sa.String(length=36), nullable=True))
            batch.create_index(f"ix_{table}_created_by", ["created_by"])
            batch.create_foreign_key(f"fk_{table}_created_by_users", "users", ["created_by"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    for table in ("inference_endpoints", "registry_assets", "evaluation_runs"):
        with op.batch_alter_table(table) as batch:
            batch.drop_constraint(f"fk_{table}_created_by_users", type_="foreignkey")
            batch.drop_index(f"ix_{table}_created_by")
            batch.drop_column("created_by")
