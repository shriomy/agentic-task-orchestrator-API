import time
from typing import Any
import requests

from ..config import settings


class AmadeusClient:
    def __init__(self) -> None:
        self.client_id = settings.amadeus_client_id
        self.client_secret = settings.amadeus_client_secret
        self.environment = settings.amadeus_environment
        self.access_token: str | None = None
        self.expires_at: float = 0.0
        self.base_url = self._get_base_url()

    def _get_base_url(self) -> str:
        return "https://test.api.amadeus.com" if self.environment == "test" else "https://api.amadeus.com"

    def _auth_url(self) -> str:
        return f"{self.base_url}/v1/security/oauth2/token"

    def _ensure_token(self) -> None:
        if self.access_token is None or time.time() > self.expires_at - 60:
            self._refresh_token()

    def _refresh_token(self) -> None:
        response = requests.post(
            self._auth_url(),
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        self.access_token = payload["access_token"]
        self.expires_at = time.time() + payload.get("expires_in", 1800)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_token()
        response = requests.get(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.access_token}"},
            params=params,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


amadeus_client = AmadeusClient()
