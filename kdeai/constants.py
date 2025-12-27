"""String constants used across KDEAI modules."""

from typing import Literal


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
    SKIP = "skip"


class CacheMode:
    """Cache mode values."""

    ON = "on"
    OFF = "off"


class AssetMode:
    """Asset mode values."""

    OFF = "off"
    AUTO = "auto"
    REQUIRED = "required"


class ApplyMode:
    """Apply mode values."""

    STRICT = "strict"
    REBASE = "rebase"


class OverwritePolicy:
    """Overwrite policy values."""

    CONSERVATIVE = "conservative"
    ALLOW_NONEMPTY = "allow-nonempty"
    ALLOW_REVIEWED = "allow-reviewed"
    ALL = "all"


class PostIndex:
    """Post-index toggle values."""

    ON = "on"
    OFF = "off"


CacheModeLiteral = Literal[CacheMode.ON, CacheMode.OFF]
AssetModeLiteral = Literal[AssetMode.OFF, AssetMode.AUTO, AssetMode.REQUIRED]
ApplyModeLiteral = Literal[ApplyMode.STRICT, ApplyMode.REBASE]
OverwritePolicyLiteral = Literal[
    OverwritePolicy.CONSERVATIVE,
    OverwritePolicy.ALLOW_NONEMPTY,
    OverwritePolicy.ALLOW_REVIEWED,
    OverwritePolicy.ALL,
]
PostIndexLiteral = Literal[PostIndex.ON, PostIndex.OFF]


# Schema versions
WORKSPACE_TM_SCHEMA_VERSION = "1"
REFERENCE_TM_SCHEMA_VERSION = "1"
EXAMPLES_SCHEMA_VERSION = "1"
GLOSSARY_SCHEMA_VERSION = "1"
