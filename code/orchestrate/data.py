"""Loads the challenge dataset once and exposes indexed lookups for the tools in tools.py.

Everything here is read-only and derived straight from dataset/*.csv -- no organizer-only
files, no hardcoded labels. Loaded lazily (module-level singleton) so importing this module
has no side effects until a lookup is actually needed.
"""

import os
from functools import lru_cache

import pandas as pd

from orchestrate.config import DATASET_DIR
from orchestrate.errors import DatasetError


class Dataset:
    def __init__(self, root: str = DATASET_DIR):
        self.root = root
        self.users = self._read("users.csv").set_index("user_id", drop=False)
        self.groups = self._read("groups.csv").set_index("group_id", drop=False)
        self.group_members = self._read("group_members.csv")
        self.business_accounts = self._read("business_accounts.csv").set_index("business_id", drop=False)
        self.user_business_history = self._read("user_business_history.csv")
        self.message_history = self._read("message_history.csv")
        self.message_events = self._read("message_events.csv").set_index(["user_id", "message_id"])
        self.images = self._read("images.csv").set_index("image_id", drop=False)
        self.voice_notes = self._read("voice_notes.csv").set_index("voice_note_id", drop=False)
        self.daily_notification_summary = self._read("daily_notification_summary.csv")

    def _read(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.root, filename)
        if not os.path.exists(path):
            raise DatasetError(f"Required dataset file is missing: {path}. Check DATASET_DIR in config.py.")
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise DatasetError(f"Dataset file is empty or unreadable: {path}.", cause=exc) from exc
        if df.empty:
            raise DatasetError(f"Dataset file has no rows: {path}.")
        return df

    def media_path(self, media_type: str, media_id: str) -> str | None:
        if media_type == "image" and media_id in self.images.index:
            return os.path.join(self.root, self.images.loc[media_id, "file_path"])
        if media_type == "voice" and media_id in self.voice_notes.index:
            return os.path.join(self.root, self.voice_notes.loc[media_id, "file_path"])
        return None

    def messages_by_id(self, user_id: str, message_ids: list[str]) -> list[dict]:
        """Fetch specific past messages by ID (for this user), each with its event outcome.
        Used by evaluate.py to show a judge the actual content behind a cited
        evidence_message_ids value, rather than trusting the citation blindly.
        """
        rows = self.message_history[
            (self.message_history.user_id == user_id) & (self.message_history.message_id.isin(message_ids))
        ]
        records = []
        for _, row in rows.iterrows():
            record = row.to_dict()
            key = (user_id, row["message_id"])
            record["events"] = self.message_events.loc[key].to_dict() if key in self.message_events.index else None
            records.append(record)
        return records


@lru_cache(maxsize=1)
def get_dataset() -> Dataset:
    return Dataset()
