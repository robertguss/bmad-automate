# BMAD Automate

Automated BMAD (Business Method for Agile Development) Workflow Orchestrator for Claude Code.

Processes stories through a complete development cycle:

```
create-story -> dev-story -> code-review -> git-commit
```

## Installation

### From GitHub (recommended)

```bash
uv tool install git+https://github.com/robertguss/bmad-automate
```

### From PyPI (when published)

```bash
uv tool install bmad-automate
```

### For development

```bash
git clone https://github.com/robertguss/bmad-automate.git
cd bmad-automate
uv sync
```

## Usage

Once installed, run from any BMAD project directory:

```bash
# Show help
bmad-automate --help

# Dry run to preview what would be processed
bmad-automate --dry-run

# Process next story
bmad-automate --limit 1

# Process all stories in epic 3
bmad-automate --epic 3

# Process specific story
bmad-automate 3-3-account-translation

# Non-interactive with verbose output
bmad-automate --yes --verbose --limit 5
```

## Options

### General Options

| Flag            | Description                              |
| --------------- | ---------------------------------------- |
| `-n, --dry-run` | Preview what would run without executing |
| `-y, --yes`     | Skip interactive confirmation prompt     |
| `-v, --verbose` | Show full Claude output during execution |
| `-q, --quiet`   | Minimal output (only errors and summary) |

### Story Selection

| Flag               | Description                               |
| ------------------ | ----------------------------------------- |
| `--epic N`         | Only process stories for epic N           |
| `--limit N`        | Process at most N stories (0 = unlimited) |
| `--start-from KEY` | Resume from specific story key            |
| `[stories...]`     | Specific story keys to process            |

### Step Control

| Flag            | Description               |
| --------------- | ------------------------- |
| `--skip-create` | Skip create-story step    |
| `--skip-dev`    | Skip dev-story step       |
| `--skip-review` | Skip code-review step     |
| `--skip-commit` | Skip git commit/push step |

### Retry & Timeout

| Flag          | Default | Description                 |
| ------------- | ------- | --------------------------- |
| `--retries N` | 1       | Retries per step on failure |
| `--timeout N` | 600     | Timeout per step in seconds |

### Paths

| Flag              | Default                                                    | Description           |
| ----------------- | ---------------------------------------------------------- | --------------------- |
| `--sprint-status` | `_bmad-output/implementation-artifacts/sprint-status.yaml` | Sprint status file    |
| `--story-dir`     | `_bmad-output/implementation-artifacts`                    | Story files directory |
| `--log-file`      | `bmad-automation.log`                                      | Log file path         |

## Requirements

- Python 3.11+
- Claude CLI installed and configured
- BMAD workflows available (`bmad:bmm:workflows:*`)

## Project Structure

The tool expects a BMAD project structure with:

```
your-project/
├── _bmad-output/
│   └── implementation-artifacts/
│       ├── sprint-status.yaml    # Story statuses
│       ├── 3-1-feature-name.md   # Story files
│       └── ...
└── ...
```

## Story Selection Priority

Stories are processed in this order:

1. **in-progress** - Resume interrupted work first
2. **ready-for-dev** - Stories ready to implement
3. **backlog** - New stories to start

## Smart Behaviors

- **Auto-skip create**: If story file already exists, create-story is skipped
- **Permission handling**: Uses `--dangerously-skip-permissions` for autonomous execution
- **Graceful interruption**: Ctrl+C shows partial summary before exiting

## Upgrading

```bash
uv tool upgrade bmad-automate
```

## License

MIT
