"""Example plugin for the RPLiDAR production platform.

This plugin demonstrates the lightweight plugin interface and registers a callback
that logs when the platform initializes.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lidar.platform import ProductionPlatform


def register_plugin(platform: "ProductionPlatform") -> bool:
    platform.log_event("ExamplePlugin loaded")
    return True
