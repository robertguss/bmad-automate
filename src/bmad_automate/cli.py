"""
BMAD Workflow Automation CLI.

Automates the BMAD (Business Method for Agile Development) workflow cycle
for stories defined in sprint-status.yaml. For each actionable story, the
script orchestrates a full development cycle:

    create-story -> dev-story -> code-review -> git-commit-push

Features:
    - Beautiful terminal UX with progress bars and colored output (via Rich)
    - Proper YAML parsing for sprint-status.yaml (no fragile grep/sed)
    - Robust subprocess handling with configurable timeouts and retries
    - Dry-run mode for previewing what would be executed
    - Flexible story selection (by status, limit, specific keys, or resume point)
    - Step-level control (skip any combination of steps)
    - Graceful Ctrl+C handling with partial summary
    - Comprehensive logging to file for debugging

Requirements:
    - Python 3.11+
    - Claude CLI installed and configured
    - BMAD workflows available (bmad:bmm:workflows:*)

Usage:
    bmad-automate [options] [story_keys...]

See --help for full options or the README for comprehensive documentation.
"""

import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Constants
DEFAULT_SPRINT_STATUS = "_bmad-output/implementation-artifacts/sprint-status.yaml"
DEFAULT_STORY_DIR = "_bmad-output/implementation-artifacts"
DEFAULT_LOG_FILE = "bmad-automation.log"
DEFAULT_RETRIES = 1
DEFAULT_TIMEOUT = 600  # 10 minutes

# Rich console for output
console = Console()


class StepStatus(Enum):
    """
    Status of a single step execution within a story.

    Values:
        SUCCESS: Step completed without errors (exit code 0).
        FAILED: Step failed (non-zero exit, timeout, or exception).
        SKIPPED: Step was not executed (user skip flag or auto-skip).
    """

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class StoryStatus(Enum):
    """
    Overall status of a story after all steps have been processed.

    Values:
        COMPLETED: All steps succeeded (or were skipped intentionally).
        FAILED: At least one step failed, stopping further execution.
        SKIPPED: Story was skipped entirely (e.g., dry-run mode).
    """

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """
    Result of executing a single step within a story.

    Attributes:
        name: Step identifier (e.g., 'create-story', 'dev-story').
        status: Execution status (SUCCESS, FAILED, or SKIPPED).
        duration: Time taken in seconds (0.0 if skipped).
        error: Error message if failed, empty string otherwise.
    """

    name: str
    status: StepStatus
    duration: float = 0.0
    error: str = ""


@dataclass
class StoryResult:
    """
    Result of processing all steps for a single story.

    Attributes:
        key: Story identifier (e.g., '3-3-account-translation').
        status: Overall story status (COMPLETED, FAILED, or SKIPPED).
        steps: List of individual step results in execution order.
        duration: Total time taken for all steps in seconds.
        failed_step: Name of the step that failed, if any.
    """

    key: str
    status: StoryStatus
    steps: list[StepResult] = field(default_factory=list)
    duration: float = 0.0
    failed_step: str = ""


@dataclass
class Config:
    """
    Configuration container for the automation script.

    Controls all aspects of script behavior including paths, execution
    options, story selection, and step control.
    """

    # Paths
    sprint_status: Path = Path(DEFAULT_SPRINT_STATUS)
    story_dir: Path = Path(DEFAULT_STORY_DIR)
    log_file: Path = Path(DEFAULT_LOG_FILE)

    # Execution control
    dry_run: bool = False
    yes: bool = False
    verbose: bool = False
    quiet: bool = False

    # Story selection
    limit: int = 0  # 0 = unlimited
    start_from: str = ""
    specific_stories: list[str] = field(default_factory=list)
    epic: int = 0  # 0 = all epics, otherwise filter to specific epic number

    # Step control
    skip_create: bool = False
    skip_dev: bool = False
    skip_review: bool = False
    skip_commit: bool = False

    # Retry/Timeout
    retries: int = DEFAULT_RETRIES
    timeout: int = DEFAULT_TIMEOUT


# Typer app instance
app = typer.Typer(
    name="bmad-automate",
    help="Automated BMAD Workflow Orchestrator",
    add_completion=False,
    rich_markup_mode="rich",
)


# Global state for signal handling
_interrupted = False
_current_story = ""
_results: list[StoryResult] = []
_start_time: float = 0.0
_config: Config | None = None


def signal_handler(signum: int, frame) -> None:  # noqa: ANN001
    """
    Handle interrupt signals (Ctrl+C, SIGTERM) gracefully.

    Sets the global _interrupted flag to True, allowing the main loop
    to complete the current operation before exiting with a summary.

    Args:
        signum: Signal number received.
        frame: Current stack frame (unused).
    """
    global _interrupted
    _interrupted = True
    console.print(
        "\n[yellow]Interrupt received. Finishing current operation...[/yellow]"
    )


def setup_signal_handlers() -> None:
    """
    Register signal handlers for graceful shutdown.

    Registers handlers for SIGINT (Ctrl+C) and SIGTERM to allow
    the script to complete current work and display a summary
    before exiting.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "45s" or "3m 21s".

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(201)
        '3m 21s'
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    remaining = int(seconds % 60)
    return f"{minutes}m {remaining:02d}s"


def log_to_file(message: str, config: Config) -> None:
    """
    Append a timestamped message to the log file.

    Args:
        message: Message to log.
        config: Configuration containing the log file path.

    The log format is: [YYYY-MM-DD HH:MM:SS] message
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.log_file, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def get_actionable_stories(config: Config) -> dict[str, list[str]]:
    """
    Parse sprint-status.yaml and return stories grouped by actionable status.

    Reads the sprint-status.yaml file and extracts story keys that have
    one of the actionable statuses: 'in-progress', 'ready-for-dev', or 'backlog'.
    Story keys must match the pattern: digit-digit-kebab-case (e.g., '3-3-account').

    Args:
        config: Configuration containing the sprint_status file path.

    Returns:
        Dictionary with status names as keys and lists of story keys as values.
        Keys: 'in-progress', 'ready-for-dev', 'backlog'

    Raises:
        SystemExit: If sprint-status.yaml doesn't exist or has invalid format.
    """
    if not config.sprint_status.exists():
        console.print(
            f"[red]Error: Sprint status file not found: {config.sprint_status}[/red]"
        )
        sys.exit(2)

    with open(config.sprint_status) as f:
        data = yaml.safe_load(f)

    if not data or "development_status" not in data:
        console.print("[red]Error: Invalid sprint-status.yaml format[/red]")
        sys.exit(2)

    dev_status = data["development_status"]

    # Pattern for story keys: digit-digit-kebab-case (e.g., 3-3-account-translation)
    story_pattern = re.compile(r"^\d+-\d+-.+$")

    # Actionable statuses in priority order
    actionable_statuses = ["in-progress", "ready-for-dev", "backlog"]
    stories_by_status: dict[str, list[str]] = {s: [] for s in actionable_statuses}

    for key, status in dev_status.items():
        if story_pattern.match(key) and status in actionable_statuses:
            stories_by_status[status].append(key)

    return stories_by_status


def get_all_story_keys(config: Config) -> set[str]:
    """
    Get all story keys from sprint-status.yaml regardless of status.

    Used for validating user-specified story keys exist in the project.
    Unlike get_actionable_stories(), this returns ALL stories including
    'done', 'blocked', etc.

    Args:
        config: Configuration containing the sprint_status file path.

    Returns:
        Set of all story keys matching the digit-digit-kebab pattern.
        Returns empty set if file doesn't exist or is invalid.
    """
    if not config.sprint_status.exists():
        return set()

    with open(config.sprint_status) as f:
        data = yaml.safe_load(f)

    if not data or "development_status" not in data:
        return set()

    dev_status = data["development_status"]
    story_pattern = re.compile(r"^\d+-\d+-.+$")

    return {key for key in dev_status.keys() if story_pattern.match(key)}


def filter_stories(
    stories_by_status: dict[str, list[str]], config: Config
) -> list[str]:
    """
    Apply filters to produce the final ordered list of stories to process.

    Handles story selection in this order:
    1. Specific stories: If user provides story keys, validate and use those.
    2. Auto-detect: Otherwise, combine stories in priority order.
    3. Apply --epic filter to limit to a specific epic.
    4. Apply --start-from and --limit filters to the result.

    Priority order for auto-detect: in-progress > ready-for-dev > backlog

    Args:
        stories_by_status: Dictionary of stories grouped by status.
        config: Configuration with filter settings.

    Returns:
        Ordered list of story keys to process.
    """
    # If specific stories provided, validate they exist (any status)
    if config.specific_stories:
        all_keys = get_all_story_keys(config)
        valid_stories = [s for s in config.specific_stories if s in all_keys]
        if len(valid_stories) != len(config.specific_stories):
            missing = set(config.specific_stories) - set(valid_stories)
            console.print(
                f"[yellow]Warning: Stories not found in sprint-status.yaml: "
                f"{missing}[/yellow]"
            )
        # Apply epic filter to specific stories too
        if config.epic > 0:
            epic_prefix = f"{config.epic}-"
            valid_stories = [s for s in valid_stories if s.startswith(epic_prefix)]
        return valid_stories

    # Build ordered list: in-progress first, then ready-for-dev, then backlog
    stories = (
        stories_by_status.get("in-progress", [])
        + stories_by_status.get("ready-for-dev", [])
        + stories_by_status.get("backlog", [])
    )

    # Apply epic filter (e.g., --epic 3 filters to stories starting with "3-")
    if config.epic > 0:
        epic_prefix = f"{config.epic}-"
        stories = [s for s in stories if s.startswith(epic_prefix)]
        if not stories:
            console.print(
                f"[yellow]Warning: No stories found for epic {config.epic}[/yellow]"
            )

    # Apply start-from filter
    if config.start_from:
        try:
            start_idx = stories.index(config.start_from)
            stories = stories[start_idx:]
        except ValueError:
            console.print(
                f"[yellow]Warning: Start story '{config.start_from}' "
                "not found, processing all[/yellow]"
            )

    # Apply limit
    if config.limit > 0:
        stories = stories[: config.limit]

    return stories


def get_story_path(story_key: str, config: Config) -> Path:
    """
    Construct the file path for a story's markdown file.

    Args:
        story_key: Story identifier (e.g., '3-3-account-translation').
        config: Configuration containing the story_dir path.

    Returns:
        Path to the story file: {story_dir}/{story_key}.md
    """
    return config.story_dir / f"{story_key}.md"


def run_step(
    step_name: str,
    command: str,
    story_key: str,
    config: Config,
) -> StepResult:
    """
    Execute a single workflow step with retry and timeout handling.

    Runs a shell command (typically a Claude CLI invocation) and handles:
    - Dry-run mode (just prints what would run)
    - Retries on failure (configurable via config.retries)
    - Timeout enforcement (configurable via config.timeout)
    - Interrupt handling (checks _interrupted flag)
    - Logging to file (stdout, stderr, success/failure)

    Args:
        step_name: Human-readable step name (e.g., 'dev-story').
        command: Shell command to execute.
        story_key: Story identifier for logging.
        config: Configuration with timeout, retries, and output settings.

    Returns:
        StepResult with status, duration, and error details if failed.
    """
    start_time = time.time()

    if config.dry_run:
        console.print(
            f"  [dim][DRY-RUN][/dim] Would run: [magenta]{step_name}[/magenta]"
        )
        console.print(f"  [dim]Command: {command}[/dim]")
        return StepResult(name=step_name, status=StepStatus.SKIPPED, duration=0.0)

    log_to_file(f"Running {step_name} for {story_key}", config)
    log_to_file(f"Command: {command}", config)

    for attempt in range(config.retries + 1):
        if _interrupted:
            return StepResult(
                name=step_name,
                status=StepStatus.FAILED,
                error="Interrupted",
                duration=time.time() - start_time,
            )

        try:
            if not config.quiet:
                attempt_str = (
                    f" (attempt {attempt + 1}/{config.retries + 1})"
                    if attempt > 0
                    else ""
                )
                console.print(
                    f"  [dim]Running[/dim] [magenta]{step_name}[/magenta]"
                    f"{attempt_str}..."
                )

            result = subprocess.run(
                command,
                shell=True,
                capture_output=not config.verbose,
                text=True,
                timeout=config.timeout,
            )

            # Log output
            if result.stdout:
                log_to_file(f"STDOUT:\n{result.stdout}", config)
            if result.stderr:
                log_to_file(f"STDERR:\n{result.stderr}", config)

            if result.returncode == 0:
                duration = time.time() - start_time
                log_to_file(
                    f"SUCCESS: {step_name} ({format_duration(duration)})", config
                )
                return StepResult(
                    name=step_name, status=StepStatus.SUCCESS, duration=duration
                )
            else:
                error = result.stderr or f"Exit code: {result.returncode}"
                log_to_file(f"FAILED: {step_name} - {error}", config)

                if attempt < config.retries:
                    console.print(f"  [yellow]Retrying {step_name}...[/yellow]")
                    continue

                return StepResult(
                    name=step_name,
                    status=StepStatus.FAILED,
                    error=error,
                    duration=time.time() - start_time,
                )

        except subprocess.TimeoutExpired:
            error = f"Timeout after {config.timeout}s"
            log_to_file(f"TIMEOUT: {step_name} - {error}", config)
            return StepResult(
                name=step_name,
                status=StepStatus.FAILED,
                error=error,
                duration=time.time() - start_time,
            )

        except Exception as e:
            error = str(e)
            log_to_file(f"ERROR: {step_name} - {error}", config)
            return StepResult(
                name=step_name,
                status=StepStatus.FAILED,
                error=error,
                duration=time.time() - start_time,
            )

    # Should not reach here, but just in case
    return StepResult(
        name=step_name,
        status=StepStatus.FAILED,
        error="Unknown error",
        duration=time.time() - start_time,
    )


def process_story(story_key: str, config: Config) -> StoryResult:
    """
    Process all workflow steps for a single story.

    Executes the full BMAD workflow cycle for one story:
    1. create-story: Generate story markdown file (auto-skipped if exists)
    2. dev-story: Implement the story following the markdown spec
    3. code-review: Review implementation and auto-fix issues
    4. git-commit: Commit and push changes

    Each step can be skipped via config flags. Execution stops on first failure.

    Args:
        story_key: Story identifier (e.g., '3-3-account-translation').
        config: Configuration with step skip flags and other settings.

    Returns:
        StoryResult with overall status and individual step results.
    """
    global _current_story
    _current_story = story_key

    start_time = time.time()
    story_path = get_story_path(story_key, config)
    steps: list[StepResult] = []

    log_to_file(f"=== Starting story: {story_key} ===", config)

    # Define step prompts (broken up for readability)
    dev_prompt = (
        f"/bmad:bmm:workflows:dev-story - Work on story file: {story_path}. "
        "Complete all tasks. Run tests after each implementation. "
        "Do not ask clarifying questions - use best judgment based on "
        "existing patterns."
    )
    review_prompt = (
        f"/bmad:bmm:workflows:code-review - Review story: {story_path}. "
        "IMPORTANT: When presenting options, always choose option 1 to "
        "auto-fix all issues immediately. Do not wait for user input."
    )
    commit_prompt = (
        f"Commit all changes for story {story_key} with a descriptive "
        "message. Then push to the current branch."
    )

    # Auto-skip create-story if story file already exists
    skip_create = config.skip_create
    if not skip_create and story_path.exists():
        if not config.quiet:
            console.print("  [dim]Story file exists, skipping create-story[/dim]")
        skip_create = True

    # Define steps
    # Note: --dangerously-skip-permissions is required for non-interactive automation
    # since Claude would otherwise wait for permission approvals that never come
    # Build create-story command
    create_cmd = (
        "claude --dangerously-skip-permissions -p "
        f'"/bmad:bmm:workflows:create-story - Create story: {story_key}"'
    )

    step_definitions = [
        (
            "create-story",
            skip_create,
            create_cmd,
        ),
        (
            "dev-story",
            config.skip_dev,
            f'claude --dangerously-skip-permissions -p "{dev_prompt}"',
        ),
        (
            "code-review",
            config.skip_review,
            f'claude --dangerously-skip-permissions -p "{review_prompt}"',
        ),
        (
            "git-commit",
            config.skip_commit,
            f'claude --dangerously-skip-permissions -p "{commit_prompt}"',
        ),
    ]

    failed_step = ""
    for step_name, skip, command in step_definitions:
        if _interrupted:
            break

        if skip:
            if not config.quiet:
                console.print(
                    f"  [yellow]Skipping[/yellow] [magenta]{step_name}[/magenta]"
                )
            steps.append(StepResult(name=step_name, status=StepStatus.SKIPPED))
            continue

        result = run_step(step_name, command, story_key, config)
        steps.append(result)

        if result.status == StepStatus.FAILED:
            failed_step = step_name
            break

    duration = time.time() - start_time

    # Determine overall status
    if any(s.status == StepStatus.FAILED for s in steps):
        status = StoryStatus.FAILED
    elif config.dry_run or all(s.status == StepStatus.SKIPPED for s in steps):
        status = StoryStatus.SKIPPED
    else:
        status = StoryStatus.COMPLETED

    log_to_file(
        f"=== Story {story_key}: {status.value} ({format_duration(duration)}) ===",
        config,
    )

    return StoryResult(
        key=story_key,
        status=status,
        steps=steps,
        duration=duration,
        failed_step=failed_step,
    )


def print_story_summary(result: StoryResult, config: Config) -> None:
    """
    Print a formatted summary of a completed story to the console.

    Displays the story key, total duration, status, and individual step results
    with color-coded status indicators. Respects quiet mode.

    Args:
        result: StoryResult containing status and step details.
        config: Configuration (checked for quiet mode).
    """
    if config.quiet:
        return

    # Status symbol and color
    if result.status == StoryStatus.COMPLETED:
        status_text = "[green]COMPLETED[/green]"
        symbol = "[green]OK[/green]"
    elif result.status == StoryStatus.FAILED:
        status_text = f"[red]FAILED[/red] ({result.failed_step})"
        symbol = "[red]XX[/red]"
    else:
        status_text = "[yellow]SKIPPED[/yellow]"
        symbol = "[yellow]--[/yellow]"

    duration_str = format_duration(result.duration)
    console.print(
        f"\n  {symbol} [cyan]{result.key}[/cyan] | {duration_str} | {status_text}"
    )

    # Show step details
    for step in result.steps:
        if step.status == StepStatus.SUCCESS:
            step_symbol = "[green]OK[/green]"
            duration_str = f"[dim]{format_duration(step.duration)}[/dim]"
        elif step.status == StepStatus.FAILED:
            step_symbol = "[red]XX[/red]"
            duration_str = f"[dim]{format_duration(step.duration)}[/dim]"
        else:
            step_symbol = "[yellow]--[/yellow]"
            duration_str = "[dim]skipped[/dim]"

        console.print(f"     {step_symbol} {step.name:<15} {duration_str}")


def print_dry_run_preview(stories: list[str], config: Config) -> None:
    """
    Print a detailed preview of what would be executed in dry-run mode.

    Displays a configuration table and numbered list of stories that would
    be processed. Used when --dry-run flag is specified.

    Args:
        stories: List of story keys that would be processed.
        config: Configuration to display (paths, timeouts, enabled steps).
    """
    console.print(
        Panel(
            "[bold cyan]DRY RUN MODE[/bold cyan] - No changes will be made",
            style="cyan",
        )
    )
    console.print()

    # Show configuration
    table = Table(title="Configuration", show_header=False, box=None)
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("Sprint Status", str(config.sprint_status))
    table.add_row("Story Directory", str(config.story_dir))
    table.add_row("Stories to Process", str(len(stories)))
    table.add_row("Retries", str(config.retries))
    table.add_row("Timeout", f"{config.timeout}s")

    steps_enabled = []
    if not config.skip_create:
        steps_enabled.append("create-story")
    if not config.skip_dev:
        steps_enabled.append("dev-story")
    if not config.skip_review:
        steps_enabled.append("code-review")
    if not config.skip_commit:
        steps_enabled.append("git-commit")
    table.add_row("Steps", " -> ".join(steps_enabled))

    console.print(table)
    console.print()

    # Show stories
    console.print("[bold]Stories to process:[/bold]")
    for i, story in enumerate(stories, 1):
        console.print(f"  {i}. [cyan]{story}[/cyan]")

    console.print()


def confirm_start(stories: list[str], config: Config) -> bool:
    """
    Display a preview and prompt user for confirmation before starting.

    Shows the number of stories, enabled steps, and story list. Waits for
    user input unless interrupted with Ctrl+C or EOF.

    Args:
        stories: List of story keys to be processed.
        config: Configuration (for displaying enabled steps and log path).

    Returns:
        True if user confirms (Enter or 'y'), False if declined ('n') or interrupted.
    """
    console.print()
    console.print(Panel("[bold]BMAD Automation Preview[/bold]", style="blue"))
    console.print()

    console.print(f"  Stories to process: [bold]{len(stories)}[/bold]")

    steps_enabled = []
    if not config.skip_create:
        steps_enabled.append("create-story")
    if not config.skip_dev:
        steps_enabled.append("dev-story")
    if not config.skip_review:
        steps_enabled.append("code-review")
    if not config.skip_commit:
        steps_enabled.append("git-commit")
    console.print(f"  Steps: {' -> '.join(steps_enabled)}")
    console.print(f"  Log file: {config.log_file}")
    console.print()

    console.print("  [bold]Stories:[/bold]")
    for story in stories:
        console.print(f"    [dim]-[/dim] [cyan]{story}[/cyan]")

    console.print()

    try:
        response = console.input("[yellow]Proceed? [Y/n]:[/yellow] ")
        if response.lower() in ("n", "no"):
            console.print("[dim]Aborted.[/dim]")
            return False
        return True
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Aborted.[/dim]")
        return False


def print_final_summary(
    results: list[StoryResult], config: Config, total_duration: float
) -> None:
    """
    Print the final summary report after all stories have been processed.

    Displays:
    - Header panel (green for success, red for failures, yellow for skipped)
    - Duration and story counts (completed, failed, skipped)
    - Success rate percentage
    - Results table with per-story status

    Args:
        results: List of StoryResult objects for all processed stories.
        config: Configuration (for log file path display).
        total_duration: Total elapsed time in seconds.
    """
    console.print()

    # Count results
    completed = sum(1 for r in results if r.status == StoryStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == StoryStatus.FAILED)
    skipped = sum(1 for r in results if r.status == StoryStatus.SKIPPED)
    total = len(results)

    # Header panel
    if failed == 0 and completed > 0:
        header_style = "green"
        header_text = "BMAD AUTOMATION COMPLETE"
    elif failed > 0:
        header_style = "red"
        header_text = "BMAD AUTOMATION FINISHED WITH FAILURES"
    else:
        header_style = "yellow"
        header_text = "BMAD AUTOMATION SUMMARY"

    console.print(Panel(f"[bold]{header_text}[/bold]", style=header_style))
    console.print()

    # Summary stats
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Label", style="dim")
    stats_table.add_column("Value")

    stats_table.add_row("Duration", f"[bold]{format_duration(total_duration)}[/bold]")
    stories_str = (
        f"[green]{completed} completed[/green], "
        f"[red]{failed} failed[/red], "
        f"[yellow]{skipped} skipped[/yellow]"
    )
    stats_table.add_row("Stories", stories_str)

    if total > 0 and (completed + failed) > 0:
        success_rate = completed / (completed + failed) * 100
        rate_color = (
            "green" if success_rate >= 80 else "yellow" if success_rate >= 50 else "red"
        )
        stats_table.add_row(
            "Success Rate", f"[{rate_color}]{success_rate:.0f}%[/{rate_color}]"
        )

    console.print(stats_table)
    console.print()

    # Results table
    if results:
        results_table = Table(title="Story Results")
        results_table.add_column("Story", style="cyan")
        results_table.add_column("Time", justify="right")
        results_table.add_column("Status")

        for result in results:
            if result.status == StoryStatus.COMPLETED:
                status_text = Text("Done", style="green")
            elif result.status == StoryStatus.FAILED:
                status_text = Text(f"Failed ({result.failed_step})", style="red")
            else:
                status_text = Text("Skipped", style="yellow")

            results_table.add_row(
                result.key, format_duration(result.duration), status_text
            )

        console.print(results_table)

    console.print()
    console.print(f"[dim]Log file: {config.log_file}[/dim]")


@app.command()
def main(
    # Positional arguments
    stories: Annotated[
        Optional[list[str]],
        typer.Argument(help="Specific story keys to process"),
    ] = None,
    # General options
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n", help="Preview what would run without executing"
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip interactive confirmation prompt"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Enable verbose output (show full command output)"
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Minimal output (only errors and summary)"),
    ] = False,
    # Story selection
    epic: Annotated[
        int,
        typer.Option(help="Only process stories for this epic number (e.g., --epic 3)"),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(help="Process at most N stories (0 = unlimited)"),
    ] = 0,
    start_from: Annotated[
        str,
        typer.Option(help="Resume from specific story key (skip earlier stories)"),
    ] = "",
    # Step control
    skip_create: Annotated[
        bool,
        typer.Option("--skip-create", help="Skip create-story step"),
    ] = False,
    skip_dev: Annotated[
        bool,
        typer.Option("--skip-dev", help="Skip dev-story step"),
    ] = False,
    skip_review: Annotated[
        bool,
        typer.Option("--skip-review", help="Skip code-review step"),
    ] = False,
    skip_commit: Annotated[
        bool,
        typer.Option("--skip-commit", help="Skip git commit/push step"),
    ] = False,
    # Retry/Timeout
    retries: Annotated[
        int,
        typer.Option(help=f"Retries per step (default: {DEFAULT_RETRIES})"),
    ] = DEFAULT_RETRIES,
    timeout: Annotated[
        int,
        typer.Option(help=f"Timeout per step in seconds (default: {DEFAULT_TIMEOUT})"),
    ] = DEFAULT_TIMEOUT,
    # Paths
    sprint_status: Annotated[
        Path,
        typer.Option(help="Path to sprint-status.yaml"),
    ] = Path(DEFAULT_SPRINT_STATUS),
    story_dir: Annotated[
        Path,
        typer.Option(help="Path to story files directory"),
    ] = Path(DEFAULT_STORY_DIR),
    log_file: Annotated[
        Path,
        typer.Option(help="Path to log file"),
    ] = Path(DEFAULT_LOG_FILE),
) -> None:
    """
    Automated BMAD Workflow Orchestrator.

    Process stories through the BMAD workflow cycle:
    create-story -> dev-story -> code-review -> git-commit

    Examples:

        # Dry run to see what would be processed
        bmad-automate --dry-run

        # Process next 3 stories with confirmation
        bmad-automate --limit 3

        # Process all stories in epic 3
        bmad-automate --epic 3

        # Process single story
        bmad-automate 3-3-account-translation

        # Non-interactive with verbose output
        bmad-automate --yes --verbose --limit 5
    """
    global _results, _start_time, _config

    # Build config from CLI arguments
    config = Config(
        sprint_status=sprint_status,
        story_dir=story_dir,
        log_file=log_file,
        dry_run=dry_run,
        yes=yes,
        verbose=verbose,
        quiet=quiet,
        limit=limit,
        start_from=start_from,
        specific_stories=stories or [],
        epic=epic,
        skip_create=skip_create,
        skip_dev=skip_dev,
        skip_review=skip_review,
        skip_commit=skip_commit,
        retries=retries,
        timeout=timeout,
    )
    _config = config

    # Set up signal handlers
    setup_signal_handlers()

    # Get and filter stories
    stories_by_status = get_actionable_stories(config)

    total_actionable = sum(len(v) for v in stories_by_status.values())
    if not total_actionable and not config.specific_stories:
        console.print(
            "[yellow]No actionable stories found in sprint-status.yaml "
            "(looking for: in-progress, ready-for-dev, backlog)[/yellow]"
        )
        raise typer.Exit(0)

    filtered_stories = filter_stories(stories_by_status, config)

    if not filtered_stories:
        console.print("[yellow]No stories to process after applying filters[/yellow]")
        raise typer.Exit(0)

    # Dry run mode
    if config.dry_run:
        print_dry_run_preview(filtered_stories, config)
        # Still process stories to show what commands would run
        for story in filtered_stories:
            if _interrupted:
                break
            result = process_story(story, config)
            _results.append(result)
        raise typer.Exit(0)

    # Confirmation
    if not config.yes:
        if not confirm_start(filtered_stories, config):
            raise typer.Exit(0)

    # Initialize log file
    log_to_file("=" * 50, config)
    log_to_file("BMAD Automation Started", config)
    log_to_file(f"Stories to process: {len(filtered_stories)}", config)
    log_to_file("=" * 50, config)

    _start_time = time.time()

    # Process stories with progress
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing stories...", total=len(filtered_stories)
        )

        for i, story in enumerate(filtered_stories):
            if _interrupted:
                console.print("\n[yellow]Interrupted by user[/yellow]")
                break

            progress.update(
                task,
                description=f"[cyan]Story {i + 1}/{len(filtered_stories)}: {story}",
            )

            result = process_story(story, config)
            _results.append(result)

            print_story_summary(result, config)

            progress.advance(task)

            # Stop on failure unless we want to continue
            if result.status == StoryStatus.FAILED:
                console.print(f"\n[red]Story {story} failed, stopping automation[/red]")
                break

    # Final summary
    total_duration = time.time() - _start_time
    print_final_summary(_results, config, total_duration)

    log_to_file("=" * 50, config)
    log_to_file("BMAD Automation Finished", config)
    log_to_file(f"Duration: {format_duration(total_duration)}", config)
    log_to_file("=" * 50, config)

    # Exit code
    if any(r.status == StoryStatus.FAILED for r in _results):
        raise typer.Exit(1)
    elif _interrupted:
        raise typer.Exit(130)


if __name__ == "__main__":
    app()
