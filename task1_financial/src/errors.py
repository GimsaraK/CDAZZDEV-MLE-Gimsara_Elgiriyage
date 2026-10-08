"""Exceptions raised inside the pipeline (always caught before reaching the notebook user)."""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'yes implement the plan @TASK1A_PLAN.md', Date: 2026-10-08 (see CITATIONS.md Entry 4)


class PipelineError(Exception):
    """Base class for expected, handled pipeline failures."""


class DataUnavailableError(PipelineError):
    """Market data could not be obtained from the live source or the snapshot."""
