"""String constants used across KDEAI modules."""


class DbKind:
    """Database kind identifiers for meta validation."""

    WORKSPACE_TM = "workspace_tm"
    REFERENCE_TM = "reference_tm"
    EXAMPLES = "examples"
    GLOSSARY = "glossary"


class TmScope:
    """Translation Memory lookup scope identifiers."""

    SESSION = "session"
    WORKSPACE = "workspace"
    REFERENCE = "reference"


class ReviewStatus:
    """Entry review status values."""

    REVIEWED = "reviewed"
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    UNREVIEWED = "unreviewed"


class PlanAction:
    """Plan entry action types."""

    COPY_TM = "copy_tm"
    LLM = "llm"
    NEEDS_LLM = "needs_llm"


# Schema versions
WORKSPACE_TM_SCHEMA_VERSION = "1"
REFERENCE_TM_SCHEMA_VERSION = "1"
EXAMPLES_SCHEMA_VERSION = "1"
GLOSSARY_SCHEMA_VERSION = "1"
