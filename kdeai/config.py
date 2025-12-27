from __future__ import annotations

from pathlib import Path
from typing import Literal
import hashlib
import json

from kdeai import hash as kdehash
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator

OverwritePolicy = Literal["conservative", "allow-nonempty", "allow-reviewed", "all"]
TmScope = Literal["session", "workspace", "reference"]
ExamplesScope = Literal["workspace", "reference"]
GlossaryScope = Literal["reference"]
AssetMode = Literal["off", "auto", "required"]
ApplyMode = Literal["strict", "rebase"]
ReviewStatus = Literal["reviewed", "draft", "needs_review", "unreviewed"]

def compute_canonical_hash(data: dict) -> str:
    canonical = kdehash.canonical_json_bytes(data)
    return hashlib.sha256(canonical).hexdigest()


class _BaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingPolicy(_BaseModel):
    model_id: StrictStr
    dim: StrictInt = Field(gt=0)
    distance: StrictStr
    encoding: StrictStr
    normalization: StrictStr
    input_canonicalization: StrictStr
    require_finite: StrictBool = True

    @field_validator("distance")
    @classmethod
    def _distance_lowercase(cls, value: str) -> str:
        lowered = value.lower()
        if value != lowered:
            raise ValueError("embedding_policy.distance must be lower-case")
        if lowered not in {"cosine", "l2"}:
            raise ValueError("embedding_policy.distance must be one of: cosine, l2")
        return lowered

    @field_validator("normalization")
    @classmethod
    def _normalization_value(cls, value: str) -> str:
        lowered = value.lower()
        if value != lowered:
            raise ValueError("embedding_policy.normalization must be lower-case")
        if lowered not in {"none", "l2_normalize"}:
            raise ValueError(
                "embedding_policy.normalization must be one of: none, l2_normalize"
            )
        return lowered

    @field_validator("encoding")
    @classmethod
    def _encoding_value(cls, value: str) -> str:
        if value != "float32_le":
            raise ValueError("embedding_policy.encoding must be 'float32_le'")
        return value

    @field_validator("input_canonicalization")
    @classmethod
    def _input_canonicalization_value(cls, value: str) -> str:
        if value != "source_text_v1":
            raise ValueError("embedding_policy.input_canonicalization must be 'source_text_v1'")
        return value


class ExamplesEligibility(_BaseModel):
    min_review_status: ReviewStatus
    allow_ai_generated: StrictBool


class ExamplesConfig(_BaseModel):
    mode_default: AssetMode
    lookup_scopes: list[ExamplesScope]
    top_n: StrictInt = Field(gt=0)
    embedding_policy: EmbeddingPolicy
    eligibility: ExamplesEligibility


class GlossaryConfig(_BaseModel):
    mode_default: AssetMode
    lookup_scopes: list[GlossaryScope]
    spacy_model: StrictStr
    normalization_id: StrictStr
    max_terms: StrictInt = Field(gt=0)


class PromptConfig(_BaseModel):
    generation_model_id: StrictStr | None = None
    examples: ExamplesConfig
    glossary: GlossaryConfig


class ApplyTaggingItem(_BaseModel):
    add_flags: list[StrictStr]
    add_ai_flag: StrictBool
    comment_prefix_key: StrictStr


class ApplyTagging(_BaseModel):
    tm_copy: ApplyTaggingItem
    llm: ApplyTaggingItem


class ApplyConfig(_BaseModel):
    mode_default: ApplyMode
    overwrite_default: OverwritePolicy
    tagging: ApplyTagging
    validation_patterns: list[StrictStr] = Field(default_factory=list)


class CommentPrefixes(_BaseModel):
    tool: StrictStr
    ai: StrictStr
    tm: StrictStr
    review: StrictStr


class MarkersConfig(_BaseModel):
    ai_flag: StrictStr
    comment_prefixes: CommentPrefixes


class LanguagesConfig(_BaseModel):
    source: StrictStr
    targets: list[StrictStr]


class TMSelection(_BaseModel):
    review_status_order: list[ReviewStatus]
    prefer_human: StrictBool


class TMConfig(_BaseModel):
    lookup_scopes: list[TmScope]
    selection: TMSelection


class BusyTimeouts(_BaseModel):
    read: StrictInt = Field(ge=0)
    write: StrictInt = Field(ge=0)


class WorkspaceTMConfig(_BaseModel):
    synchronous: StrictStr
    busy_timeout_ms: BusyTimeouts


class SqliteConfig(_BaseModel):
    workspace_tm: WorkspaceTMConfig


class Config(_BaseModel):
    format: Literal[2]
    languages: LanguagesConfig
    markers: MarkersConfig
    tm: TMConfig
    prompt: PromptConfig
    apply: ApplyConfig
    sqlite: SqliteConfig
    config_hash: StrictStr = ""
    embed_policy_hash: StrictStr = ""

    def model_post_init(self, __context: object) -> None:
        data = self.model_dump(mode="json", exclude={"config_hash", "embed_policy_hash"})
        self.config_hash = compute_canonical_hash(data)
        policy = {
            "model_id": self.prompt.examples.embedding_policy.model_id,
            "dim": self.prompt.examples.embedding_policy.dim,
            "distance": self.prompt.examples.embedding_policy.distance,
            "encoding": self.prompt.examples.embedding_policy.encoding,
            "input_canonicalization": self.prompt.examples.embedding_policy.input_canonicalization,
            "normalization": self.prompt.examples.embedding_policy.normalization,
            "require_finite": self.prompt.examples.embedding_policy.require_finite,
        }
        self.embed_policy_hash = compute_canonical_hash(policy)

    def examples_runtime_settings(
        self,
    ) -> tuple[list[ExamplesScope], int, AssetMode, ExamplesEligibility]:
        examples = self.prompt.examples
        return (
            list(examples.lookup_scopes),
            int(examples.top_n),
            examples.mode_default,
            examples.eligibility,
        )

    @property
    def data(self) -> dict:
        return self.model_dump(mode="json", exclude={"config_hash", "embed_policy_hash"})


def load_config(config_path: Path) -> Config:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return Config.model_validate(raw)


def load_config_from_root(root: Path) -> Config:
    return load_config(root / ".kdeai" / "config.json")
