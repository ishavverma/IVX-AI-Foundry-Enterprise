"""Track checkpoint ownership for tenant isolation."""

from alembic import op
import sqlalchemy as sa

revision = "0003_checkpoint_ownership"
down_revision = "0002_usage_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("checkpoints") as batch:
        batch.add_column(sa.Column("created_by", sa.String(length=36), nullable=True))
        batch.create_index("ix_checkpoints_created_by", ["created_by"])
        batch.create_foreign_key("fk_checkpoints_created_by_users", "users", ["created_by"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("checkpoints") as batch:
        batch.drop_constraint("fk_checkpoints_created_by_users", type_="foreignkey")
        batch.drop_index("ix_checkpoints_created_by")
        batch.drop_column("created_by")
