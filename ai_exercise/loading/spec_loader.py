"""Fetch all 7 StackOne OpenAPI specs."""

from typing import Any

import requests

from ai_exercise.constants import SETTINGS


def fetch_specs() -> list[tuple[str, dict[str, Any]]]:
    """Fetch all StackOne OpenAPI specs.

    Returns:
        List of (spec_name, spec_dict) tuples.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for url in SETTINGS.spec_urls:
        spec_name = url.split("/")[-1].replace(".json", "")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            results.append((spec_name, response.json()))
            print(f"Fetched {spec_name}.json")
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
    return results
