from __future__ import annotations

from typing import Any

import gspread
from authlib.integrations.requests_client import AssertionSession
from cachelib import BaseCache
from google.auth.credentials import AnonymousCredentials

from .rating_table import RatingTable


class GoogleDocApi:
    def __init__(
        self,
        base_url: str,
        gdoc_credentials: dict[str, Any],
        public_worksheet_id: str,
        public_scoreboard_sheet: int,
        cache: BaseCache,
    ):
        """
        :param base_url:
        :param gdoc_credentials:
        :param public_worksheet_id:
        :param public_scoreboard_sheet:
        :param cache:
        """
        self._url = base_url
        self._gdoc_credentials = gdoc_credentials
        self._public_worksheet_id = public_worksheet_id
        self._public_scoreboard_sheet = public_scoreboard_sheet

        self._assertion_session = self._create_assertion_session()

        self._public_scores_sheet = self._get_sheet(public_worksheet_id, public_scoreboard_sheet)
        self._cache = cache

    def _create_assertion_session(self) -> AssertionSession:
        """Create AssertionSession to auto refresh access to google api"""
        scopes = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = self._gdoc_credentials

        header = {"alg": "RS256"}
        if key_id := credentials.get("private_key_id", None):
            header["kid"] = key_id

        # Google puts scope in payload
        claims = {"scope": " ".join(scopes)}
        return AssertionSession(
            token_endpoint=credentials["token_uri"],
            issuer=credentials["client_email"],
            subject=None,
            audience=credentials["token_uri"],
            grant_type=AssertionSession.JWT_BEARER_GRANT_TYPE,
            scope=" ".join(scopes),
            claims=claims,
            key=credentials["private_key"],
            header=header,
        )

    def _get_sheet(
        self,
        worksheet_id: str,
        sheet_id: int,
    ) -> gspread.Worksheet:
        gs: gspread.Client = gspread.Client(AnonymousCredentials(), session=self._assertion_session)
        worksheet: gspread.Spreadsheet = gs.open_by_key(worksheet_id)
        return worksheet.get_worksheet(sheet_id)

    def fetch_rating_table(self) -> "RatingTable":
        return RatingTable(self._public_scores_sheet, self._cache)

    def get_spreadsheet_url(self) -> str:
        return f"{self._url}/spreadsheets/d/{self._public_worksheet_id}#gid={self._public_scoreboard_sheet}"
