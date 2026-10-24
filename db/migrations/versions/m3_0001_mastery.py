"""M3: the mastery table, its idempotency guard, and KT model versions.
Revision ID: m3_0001
Revises: None
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "m3_0001"
down_revision = None # REQUIRED: replace with the output of `alembic heads`
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "mastery",
        sa.Column("learner_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("learners.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("skill_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("skills.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("estimate", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        # n_responses drives confidence; storing it means a dashboard can grey
        # out a skill measured from two answers instead of pretending it knows
        sa.Column("n_responses", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.CheckConstraint("estimate >= 0 AND estimate <= 1",
                           name="ck_mastery_is_a_probability"),
    )
    op.create_index("ix_mastery_skill", "mastery", ["skill_id", "estimate"])

    # At-least-once delivery means the consumer WILL see an interaction twice
    # after a restart. Without this table a mastery estimate drifts upward every
    # time a pod recycles, and nothing in the system notices.
    op.create_table(
        "mastery_applied",
        sa.Column("interaction_id", sa.BigInteger(), primary_key=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "kt_model_versions",
        sa.Column("version", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False), # sakt | bkt
        sa.Column("auc", sa.Float(), nullable=False),
        sa.Column("baseline_auc", sa.Float(), nullable=False),
        sa.Column("lift_points", sa.Float(), nullable=False),
        sa.Column("promoted", sa.Boolean(), nullable=False,
                  server_default=sa.false()),
        sa.Column("metrics", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now()),
    )

    # Ensure classes and enrollments exist for class weak skills query
    op.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(100) NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            class_id UUID REFERENCES classes(id) ON DELETE CASCADE,
            learner_id UUID REFERENCES learners(id) ON DELETE CASCADE,
            PRIMARY KEY (class_id, learner_id)
        )
    """)
    # Ensure skill_edges table exists for DAG queries
    op.execute("""
        CREATE TABLE IF NOT EXISTS skill_edges (
            prereq_id VARCHAR(100) NOT NULL,
            skill_id VARCHAR(100) NOT NULL,
            PRIMARY KEY (prereq_id, skill_id)
        )
    """)
    op.execute("""
        INSERT INTO skill_edges (prereq_id, skill_id)
        SELECT CAST(prerequisite_id AS VARCHAR(100)), CAST(skill_id AS VARCHAR(100))
        FROM skill_prerequisites
        ON CONFLICT DO NOTHING
    """)

def downgrade():
    op.drop_table("kt_model_versions")
    op.drop_table("mastery_applied")
    op.drop_table("mastery")

