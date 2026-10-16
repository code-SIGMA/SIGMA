#!/usr/bin/env python3
"""
env_loader.py

Generic library for loading configuration/environment data from a JSON file,
with a clear distinction between required and optional keys per "group".

Example:

    from env_loader import EnvLoader

    loader = EnvLoader("/root/env.json")

    general = loader.group("general", required=["run_id"])
    database = loader.group(
        "database",
        required=["db_host", "db_port", "db_name", "db_user", "db_password"],
    )

    run_id = general.get("run_id")

    db = database.build(Database, kwarg_map={
        "db_host": "db_host",
        "db_port": "db_port",
        "db_name": "db_name",
        "db_user": "db_user",
        "db_password": "db_password",
    })
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List

logger = logging.getLogger(__name__)


class EnvLoader:
    """Load a JSON environment file once and provide access to groups."""

    def __init__(self, path: str):
        self.path = path
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            logger.warning("Env file not found at %s", self.path)
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load env file %s: %s", self.path, e)
            self._data = {}

    @property
    def raw(self) -> Dict[str, Any]:
        """Access all loaded raw data if needed."""
        return self._data

    def group(
        self,
        name: str,
        required: Optional[List[str]] = None,
        optional: Optional[Dict[str, Any] | List[str]] = None,
    ) -> "EnvGroup":
        """
        Define a group of keys.

        required: List of keys that MUST exist for the group
                   to be considered "loaded".
        optional: Either a list of key names (default = None) or a
                   dict {key: default_value} for individual defaults.
        """
        required = required or []

        if optional is None:
            optional_defaults: Dict[str, Any] = {}
        elif isinstance(optional, dict):
            optional_defaults = optional
        else:
            optional_defaults = {k: None for k in optional}

        missing = [k for k in required if k not in self._data]
        loaded = len(missing) == 0

        data: Dict[str, Any] = {}
        for k in required:
            data[k] = self._data.get(k)
        for k, default in optional_defaults.items():
            data[k] = self._data.get(k, default)

        if not loaded:
            logger.info(
                "Env group '%s' not fully loaded, missing keys: %s",
                name,
                missing,
            )

        return EnvGroup(name=name, loaded=loaded, missing=missing, data=data)


@dataclass
class EnvGroup:
    """Result of a loaded (or incomplete) environment group."""

    name: str
    loaded: bool
    missing: List[str]
    data: Dict[str, Any] = field(default_factory=dict)
    instance: Any = None  # populated by build()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __bool__(self) -> bool:
        return self.loaded

    def build(
        self,
        factory: Callable[..., Any],
        kwarg_map: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        """
        Call factory(**kwargs) if the group is loaded, and
        store the result in self.instance.

        kwarg_map: Maps {constructor_kwarg_name: env_key_name}.
                   If None, pass all group keys unchanged.

        Return None and log the exception if construction fails
        or the group is not loaded.
        """
        if not self.loaded:
            logger.warning(
                "Skipping build for group '%s': missing keys %s",
                self.name,
                self.missing,
            )
            return None

        if kwarg_map is None:
            kwargs = dict(self.data)
        else:
            kwargs = {
                ctor_kwarg: self.data[env_key]
                for ctor_kwarg, env_key in kwarg_map.items()
            }

        try:
            self.instance = factory(**kwargs)
            return self.instance
        except Exception:
            logger.exception("Failed to build instance for group '%s'", self.name)
            return None
