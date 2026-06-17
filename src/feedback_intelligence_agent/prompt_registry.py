"""Versioned prompt registry.

Prompts are production assets: they are named, versioned, validated against
their declared variables, and covered by snapshot tests. This module provides
the generic registry machinery; the actual prompt definitions live in
``feedback_intelligence_agent.prompts``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Formatter

LATEST_VERSION = "latest"
"""Alias that always resolves to the most recently registered version."""


class PromptNotFoundError(LookupError):
    """Raised when a prompt name or version is not registered."""


class PromptVariableError(ValueError):
    """Raised when render variables do not match a template's declared variables."""


def template_placeholders(template: str) -> set[str]:
    """Return the named ``str.format`` placeholders used by a template."""
    placeholders: set[str] = set()
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name is None:
            continue
        if not field_name.isidentifier():
            raise PromptVariableError(
                f"template placeholders must be simple names, got {field_name!r}"
            )
        placeholders.add(field_name)
    return placeholders


@dataclass(frozen=True)
class PromptTemplate:
    """A named, versioned prompt template with declared variables.

    ``required_variables`` must be provided on every render. ``optional_variables``
    map variable names to default values used when the caller omits them. The
    union of both must exactly match the placeholders found in ``template``.
    """

    name: str
    version: str
    template: str
    required_variables: tuple[str, ...] = ()
    optional_variables: Mapping[str, str] = field(default_factory=dict)
    changelog: str = ""

    def __post_init__(self) -> None:
        """Validate that declared variables exactly match template placeholders."""
        overlap = set(self.required_variables) & set(self.optional_variables)
        if overlap:
            raise PromptVariableError(
                f"prompt '{self.name}' version '{self.version}' declares variables as "
                f"both required and optional: {', '.join(sorted(overlap))}"
            )
        declared = set(self.required_variables) | set(self.optional_variables)
        placeholders = template_placeholders(self.template)
        if declared != placeholders:
            undeclared = sorted(placeholders - declared)
            unused = sorted(declared - placeholders)
            details = []
            if undeclared:
                details.append(f"placeholders missing from declaration: {', '.join(undeclared)}")
            if unused:
                details.append(f"declared variables missing from template: {', '.join(unused)}")
            raise PromptVariableError(
                f"prompt '{self.name}' version '{self.version}' is inconsistent: "
                + "; ".join(details)
            )

    def render(self, **variables: str) -> str:
        """Render the template, raising a clear error on missing or unknown variables."""
        missing = sorted(set(self.required_variables) - set(variables))
        if missing:
            raise PromptVariableError(
                f"missing required variable(s) for prompt '{self.name}' "
                f"version '{self.version}': {', '.join(missing)}"
            )
        known = set(self.required_variables) | set(self.optional_variables)
        unknown = sorted(set(variables) - known)
        if unknown:
            raise PromptVariableError(
                f"unknown variable(s) for prompt '{self.name}' "
                f"version '{self.version}': {', '.join(unknown)}"
            )
        values = {**dict(self.optional_variables), **variables}
        return self.template.format(**values)


class PromptRegistry:
    """In-memory registry mapping prompt names to their versioned templates."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._templates: dict[str, dict[str, PromptTemplate]] = {}

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """Register a template and return it, rejecting duplicate name/version pairs."""
        if template.version == LATEST_VERSION:
            raise ValueError(f"'{LATEST_VERSION}' is a reserved version alias")
        versions = self._templates.setdefault(template.name, {})
        if template.version in versions:
            raise ValueError(
                f"prompt '{template.name}' version '{template.version}' is already registered"
            )
        versions[template.version] = template
        return template

    def get(self, name: str, version: str = LATEST_VERSION) -> PromptTemplate:
        """Return a template by name and version (or the latest registered version)."""
        versions = self._templates.get(name)
        if not versions:
            available = ", ".join(sorted(self._templates)) or "none"
            raise PromptNotFoundError(f"unknown prompt '{name}'; available prompts: {available}")
        if version == LATEST_VERSION:
            return next(reversed(list(versions.values())))
        template = versions.get(version)
        if template is None:
            raise PromptNotFoundError(
                f"unknown version '{version}' for prompt '{name}'; "
                f"available versions: {', '.join(versions)}"
            )
        return template

    def names(self) -> tuple[str, ...]:
        """Return all registered prompt names, sorted alphabetically."""
        return tuple(sorted(self._templates))

    def versions(self, name: str) -> tuple[str, ...]:
        """Return the registered versions for a prompt in registration order."""
        return tuple(version.version for version in self.list_templates(name))

    def list_templates(self, name: str | None = None) -> tuple[PromptTemplate, ...]:
        """Return registered templates, optionally filtered by prompt name."""
        if name is not None:
            if name not in self._templates:
                available = ", ".join(sorted(self._templates)) or "none"
                raise PromptNotFoundError(
                    f"unknown prompt '{name}'; available prompts: {available}"
                )
            return tuple(self._templates[name].values())
        return tuple(
            template
            for prompt_name in self.names()
            for template in self._templates[prompt_name].values()
        )

    def render(self, name: str, version: str = LATEST_VERSION, /, **variables: str) -> str:
        """Resolve a template and render it with the given variables."""
        return self.get(name, version).render(**variables)
