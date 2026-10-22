"""Helpers for the Wolverine Glacier OGGM notebooks.

config       glacier IDs, data sources, working directory, offline test mode
phase        rain/snow phase curves and their monthly expectation
massbalance  PhaseAwareTIModel (OGGM's monthly TI model with a configurable phase scheme),
             recalibration, seasonal balances
dynamics     velocity partition, inversion with sliding, HydroSliding (water-dependent sliding)
experiments  stationary-climate runs and their summaries
schnaps      SCHNAPS output -> phase curves, elevation-band diagnostics, OGGM climate files
plotting     figure style and helpers
"""
__version__ = "0.1.0"
