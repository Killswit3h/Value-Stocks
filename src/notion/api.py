from __future__ import annotations

import json
from typing import Any, Dict, List

import requests

from ..logging_utils import get_logger


logger = get_logger(__name__)


class NotionAPI:
    def __init__(self, api_key: str, db_id: str, cfg: dict):
        self.key = api_key
        self.db_id = db_id
        self.cfg = cfg
        self.base = "https://api.notion.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        })

    # ---- Databases ----
    def get_database(self) -> dict:
        r = self.session.get(f"{self.base}/databases/{self.db_id}")
        r.raise_for_status()
        return r.json()

    def update_database(self, properties_patch: dict) -> dict:
        r = self.session.patch(f"{self.base}/databases/{self.db_id}", data=json.dumps({"properties": properties_patch}))
        r.raise_for_status()
        return r.json()

    def query_database(self, filter_: dict) -> dict:
        body = {"filter": filter_}
        r = self.session.post(f"{self.base}/databases/{self.db_id}/query", data=json.dumps(body))
        r.raise_for_status()
        return r.json()

    # ---- Pages ----
    def create_page(self, parent: dict, title: str, children: list[dict]) -> str:
        body = {
            "parent": parent,
            "properties": {
                "Ticker": {"title": [{"text": {"content": title}}]},
            },
            "children": children,
        }
        r = self.session.post(f"{self.base}/pages", data=json.dumps(body))
        r.raise_for_status()
        return r.json().get("id")

    def get_page_children(self, page_id: str) -> list[dict]:
        r = self.session.get(f"{self.base}/blocks/{page_id}/children")
        r.raise_for_status()
        return r.json().get("results", [])

    def delete_block(self, block_id: str) -> None:
        r = self.session.delete(f"{self.base}/blocks/{block_id}")
        r.raise_for_status()

    def append_children(self, page_id: str, children: list[dict]) -> None:
        r = self.session.patch(f"{self.base}/blocks/{page_id}/children", data=json.dumps({"children": children}))
        r.raise_for_status()

    def update_page_properties(self, page_id: str, properties: dict) -> None:
        r = self.session.patch(f"{self.base}/pages/{page_id}", data=json.dumps({"properties": properties}))
        r.raise_for_status()

    # ---- Upserts ----
    def upsert_daily_page(self, title: str, children: list[dict]) -> str:
        # We create the daily page as a row in the target database (title property = Ticker).
        # Then overwrite its children blocks to keep idempotency.
        existing = self.query_database({
            "property": "Ticker",
            "title": {"equals": title}
        })
        if existing.get("results"):
            page_id = existing["results"][0]["id"]
            # delete all children then append
            for blk in self.get_page_children(page_id):
                try:
                    self.delete_block(blk["id"])
                except Exception:
                    pass
            if children:
                self.append_children(page_id, children)
            return page_id
        # create new page in database
        body = {
            "parent": {"database_id": self.db_id},
            "properties": {
                "Ticker": {"title": [{"text": {"content": title}}]},
            },
            "children": children,
        }
        r = self.session.post(f"{self.base}/pages", data=json.dumps(body))
        r.raise_for_status()
        return r.json().get("id")

    def upsert_db_row(self, c: dict, date_str: str) -> str:
        # upsert by Ticker + Report Date
        tick = c.get("ticker")
        existing = self.query_database({
            "and": [
                {"property": "Ticker", "title": {"equals": tick}},
                {"property": "Report Date", "date": {"equals": date_str}},
            ]
        })
        props = build_db_row_properties(self, c, date_str)
        if existing.get("results"):
            page_id = existing["results"][0]["id"]
            self.update_page_properties(page_id, props)
            return page_id
        body = {
            "parent": {"database_id": self.db_id},
            "properties": props,
        }
        r = self.session.post(f"{self.base}/pages", data=json.dumps(body))
        r.raise_for_status()
        return r.json().get("id")

