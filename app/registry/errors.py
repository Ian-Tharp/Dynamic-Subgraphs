from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegistryValidationIssue:
    code: str
    message: str
    node_id: str | None = None
    field: str | None = None


class RegistryValidationError(Exception):
    def __init__(self, issues: list[RegistryValidationIssue]) -> None:
        self.issues = issues
        summary = "; ".join(i.message for i in issues[:5])
        if len(issues) > 5:
            summary += f" (+{len(issues) - 5} more)"
        super().__init__(summary)
