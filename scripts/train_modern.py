#!/usr/bin/env python3
"""Modern training script using dependency injection and clean architecture."""

import argparse
import sys
from pathlib import Path
import traceback
from loguru import logger

from src.adaptive_rl.config import load_and_validate_config, ConfigurationError
from src.adaptive_rl.core.container import setup_default_container
from src.adaptive_rl.pipelines.modern_builder import ModernPipelineBuilder


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


def main():
    """Main training function with modern architecture."""
    parser = argparse.ArgumentParser(description="Train adaptive RL agents with teacher guidance")
    parser.add_argument("config", type=Path, help="Path to experiment configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--validate-only", action="store_true", help="Only validate config without training")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without actual training")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_and_validate_config(args.config)
        logger.info(f"✅ Configuration validated successfully")
        logger.info(f"Experiment: {config.experiment.name}")
        logger.info(f"Environment: {config.environment.env_id}")
        logger.info(f"Algorithm: {config.algorithm.name}")
        logger.info(f"Scheduler: {config.scheduler.name}")
        logger.info(f"Device: {config.experiment.device.value}")

        if args.validate_only:
            logger.info("Configuration validation complete. Exiting.")
            return

        # Setup dependency injection container
        logger.info("Setting up dependency injection container")
        container = setup_default_container()

        # Build pipeline
        logger.info("Building experiment pipeline")
        builder = ModernPipelineBuilder(container)
        pipeline = builder.build_pipeline(config)

        if args.dry_run:
            logger.info("Dry run complete. Pipeline built successfully.")
            return

        # Execute experiment
        logger.info("🚀 Starting experiment execution")
        results = pipeline.run()

        # Log results
        if results["status"] == "success":
            logger.success("✅ Experiment completed successfully!")
            exp_results = results["results"]
            logger.info(f"Total episodes: {exp_results['total_episodes']}")
            logger.info(f"Total timesteps: {exp_results['total_timesteps']}")
            logger.info(f"Final metrics: {exp_results.get('final_metrics', {})}")
        else:
            logger.error(f"❌ Experiment failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)

    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("🛑 Experiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        logger.debug(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()