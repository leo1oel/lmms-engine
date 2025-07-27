# CLAUDE.md - LMMs Engine Repository Guide

## Repository Overview

**LMMs Engine** is a training framework for Large Multimodal Models (LMMs) developed by LMMs-Lab. This is a Python package focused on highly efficient training of multimodal models with support for various architectures and training paradigms.

## Key Information

- **Package**: `lmms_engine`
- **Primary Language**: Python ( >= 3.12)
- **License**: MIT
- **Main Entry Point**: `src/lmms_engine/launch/cli.py`
- **Installation**: `python3 -m pip install -e .`
- **Launch Command**: `lmms_launch` or `python -m lmms_engine.launch.cli`

## Configuration System

Training is configured via JSON files with this structure:

```json
[
	{
		"type": "trainer",
		"config": {
			"trainer_type": "hf_trainer",
			"dataset_config": {
				/* Dataset configuration */
			},
			"model_config": {
				/* Model configuration */
			}
			/* TrainingArguments parameters */
		}
	}
]
```

## Development Commands

### Training Launch

```bash
# Recommended: torchrun
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr="127.0.0.1" --master_port="8000" \
  -m lmms_engine.launch.cli --config path/to/config.json

# Alternative: accelerate
accelerate launch --use_fsdp [options] \
  -m lmms_engine.launch.cli --config path/to/config.json
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[all]"  # Includes preference learning and storage

# Performance optimizations
pip install flash-attn --no-build-isolation
pip install liger-kernel
```

## Architecture & Design Principles

The framework follows three main design patterns:

1. **Factory Pattern**: Used for creating components (models, trainers, datasets, processors)
2. **Builder Pattern**: Components are built on-demand during training initialization
3. **MVC-like Structure**: Controller manages pipeline execution (missing View component)

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale
