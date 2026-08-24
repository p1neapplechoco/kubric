"""Package marker for intervention command-line and demo entry modules.

Purpose: make dataset and three-branch demo scripts available as Python modules.
Public API: module entry points are invoked with ``python -m scripts.<name>``;
this package root intentionally exports no callable API.
Dependencies: none at package import time; each entry module owns its dependencies.
Trust boundary: package import performs no validation, simulation, rendering, media
composition, publication, or provenance attestation.
"""
