"""
AFE — Database Models Package

Importing this module registers every ORM model with the SQLAlchemy
`Base.metadata` so that `init_db()` can create the full schema.
"""

from core.database.models.user import User, Role, UserRole        # noqa: F401
from core.database.models.organization import Organization         # noqa: F401
from core.database.models.usage import UsageRecord                 # noqa: F401
from core.database.models.project import Project                   # noqa: F401
from core.database.models.audit import AuditLog                    # noqa: F401
from core.database.models.notification import Notification         # noqa: F401
from core.database.models.task_job import TaskJob                  # noqa: F401
from core.database.models.dataset import Dataset                  # noqa: F401
from core.database.models.tokenizer import TokenizerModel          # noqa: F401
from core.database.models.model_def import ModelDefinition          # noqa: F401
from core.database.models.experiment import Experiment, Checkpoint  # noqa: F401
from core.database.models.evaluation import EvaluationRun, RegistryAsset, InferenceEndpoint  # noqa: F401
from core.database.models.studio import (  # noqa: F401
    ConversationSession,
    ChatMessage,
    PromptTemplate,
    AgentDefinition,
    MCPServerConfig,
)

__all__ = [
    "User",
    "Role",
    "UserRole",
    "Organization",
    "UsageRecord",
    "Project",
    "AuditLog",
    "Notification",
    "TaskJob",
    "Dataset",
    "TokenizerModel",
    "ModelDefinition",
    "Experiment",
    "Checkpoint",
    "EvaluationRun",
    "RegistryAsset",
    "InferenceEndpoint",
    "ConversationSession",
    "ChatMessage",
    "PromptTemplate",
    "AgentDefinition",
    "MCPServerConfig",
]
