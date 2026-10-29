import logging
import requests
from typing import Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FPLClient:
    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AutoFPL-Manager/1.0", "Accept": "application/json"}
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def get_bootstrap_statistic(self) -> Dict[str, Any]:
        endpoint = f"{self.BASE_URL}/bootstrap-static/"
        logger.info(f"Fetching primary FPL data from {endpoint}...")

        response = self.session.get(endpoint, timeout=10)
        response.raise_for_status()

        logger.info("Succesfully fetched bootstrap-static data.")
        return response.json()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def get_player_summary(self, player_id: int) -> Dict[str, Any]:
        """
        Get player history results.
        """
        endpoint = f"{self.BASE_URL}/element-summary/{player_id}/"
        response = self.session.get(endpoint, timeout=10)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    client = FPLClient()
    data = client.get_bootstrap_statistic()
    logger.info(f"Test OK: Found {len(data.get('elements', []))} players.")
