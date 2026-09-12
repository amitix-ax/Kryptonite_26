"""
Credential management for live data sources.

Nothing in this module hardcodes a secret. Every credential is read from
environment variables (optionally loaded from a local `.env` file via
python-dotenv, which is itself gitignored) or from the standard
per-service credential store (`~/.netrc` for Earthdata,
`~/.copernicusmarine/.copernicusmarine-credentials` for Copernicus
Marine). If a required credential is missing, we raise a clear,
actionable `MissingCredentialsError` -- we never silently skip a source
or substitute fabricated data (same "never silently invalid" principle
used throughout the physics/data modules).

Required credentials, all free to obtain:

    CDS (ERA5 wind):
        CDSAPI_URL, CDSAPI_KEY
        Register at https://cds.climate.copernicus.eu/

    NASA Earthdata (AMSR2 sea-ice concentration via NSIDC, BedMachine
    Antarctica bathymetry via NSIDC):
        EARTHDATA_USERNAME, EARTHDATA_PASSWORD  (or EARTHDATA_TOKEN)
        Register at https://urs.earthdata.nasa.gov/

    Copernicus Marine Service (ocean currents):
        COPERNICUSMARINE_SERVICE_USERNAME, COPERNICUSMARINE_SERVICE_PASSWORD
        Register at https://data.marine.copernicus.eu/register
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class MissingCredentialsError(RuntimeError):
    """Raised when a fetcher cannot find the credentials it needs. Never
    caught silently -- the caller must supply credentials or explicitly
    skip that data source."""


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Best-effort load of a local .env file, if python-dotenv is
    installed and the file exists. Never required -- environment
    variables set any other way work identically."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = Path(path)
    if p.exists():
        load_dotenv(p)


@dataclass
class CDSCredentials:
    url: str
    key: str

    @classmethod
    def from_env(cls) -> "CDSCredentials":
        url = os.environ.get("CDSAPI_URL", "https://cds.climate.copernicus.eu/api")
        key = os.environ.get("CDSAPI_KEY")
        if not key:
            raise MissingCredentialsError(
                "CDSAPI_KEY is not set. Register at "
                "https://cds.climate.copernicus.eu/ and set CDSAPI_KEY "
                "(and optionally CDSAPI_URL) as environment variables, or "
                "in a ~/.cdsapirc file per the cdsapi package's own convention."
            )
        return cls(url=url, key=key)


@dataclass
class EarthdataCredentials:
    username: str | None
    password: str | None
    token: str | None

    def has_any(self) -> bool:
        return bool(self.token or (self.username and self.password))

    @classmethod
    def from_env(cls) -> "EarthdataCredentials":
        creds = cls(
            username=os.environ.get("EARTHDATA_USERNAME"),
            password=os.environ.get("EARTHDATA_PASSWORD"),
            token=os.environ.get("EARTHDATA_TOKEN"),
        )
        if not creds.has_any():
            raise MissingCredentialsError(
                "No NASA Earthdata credentials found. Register (free) at "
                "https://urs.earthdata.nasa.gov/ and set either "
                "EARTHDATA_TOKEN, or EARTHDATA_USERNAME + EARTHDATA_PASSWORD, "
                "as environment variables. Alternatively, run "
                "`earthaccess.login(persist=True)` once interactively to "
                "populate ~/.netrc, which earthaccess.login('all') will "
                "then find automatically."
            )
        return creds


@dataclass
class CopernicusMarineCredentials:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "CopernicusMarineCredentials":
        username = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        password = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
        if not (username and password):
            raise MissingCredentialsError(
                "COPERNICUSMARINE_SERVICE_USERNAME / "
                "COPERNICUSMARINE_SERVICE_PASSWORD are not set. Register "
                "(free) at https://data.marine.copernicus.eu/register and "
                "set both as environment variables, or run "
                "`copernicusmarine login` once interactively to cache them."
            )
        return cls(username=username, password=password)
