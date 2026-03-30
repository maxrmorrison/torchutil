
from typing import TYPE_CHECKING

import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import inspect


try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None

import huggingface_hub as hf
import orjson
import pandas as pd

def patch_sql(trackio):

    if TYPE_CHECKING:
        from trackio import SQLiteStorage
        from trackio.commit_scheduler import CommitScheduler
        from trackio.dummy_commit_scheduler import DummyCommitScheduler
        from trackio.utils import (
            MEDIA_DIR,
            TRACKIO_DIR,
            deserialize_values,
            serialize_values,
        )
        import trackio.media.utils as utils
        import trackio.media.media as media
        import trackio.run as run
        import trackio.server as server
        import trackio.media as media_base
        import trackio
    else:
        SQLiteStorage = trackio.SQLiteStorage
        CommitScheduler = trackio.commit_scheduler.CommitScheduler
        DummyCommitScheduler = trackio.dummy_commit_scheduler.DummyCommitScheduler
        TRACKIO_DIR = trackio.TRACKIO_DIR
        MEDIA_DIR = trackio.utils.MEDIA_DIR
        deserialize_values = trackio.utils.deserialize_values
        serialize_values = trackio.utils.serialize_values
        utils = trackio.media.utils
        media = trackio.media.media
        run = trackio.run
        server = trackio.server
        media_base = trackio.media

    def disallow_method(name):
        def raiser(*args, **kwargs):
            raise NotImplementedError(f'SQLiteStorage.{name} is not allowed when using torchutil')
        setattr(SQLiteStorage, name, raiser)




    DB_EXT = ".db"

    # We never want to try to import a dataset, so tell trackio that we already tried.
    SQLiteStorage._dataset_import_attempted = True


    # This function checks for a "run" variable in the caller frame or the caller's caller frame and uses that
    #  to get individual run database file paths
    def get_project_db_path(project: str) -> Path:
        """Get the database path for a specific project."""
        # This is so incredibly cursed... but also genius?
        filename = SQLiteStorage.get_project_db_filename(project)
        caller_frame = inspect.currentframe().f_back
        run = caller_frame.f_locals.get('run', caller_frame.f_locals.get('run_name'))
        if not run:
            # try one level deeper
            caller_frame = caller_frame.f_back
            if caller_frame:
                run = caller_frame.f_locals.get('run', caller_frame.f_locals.get('run_name'))
                if not run:
                    raise NotImplementedError('torchutil.trackio_utils.trackio.SQLiteStorage.get_project_db_path can only be called from a frame with a local variable `run` or a frame whose parent has a local variable `run`')
        return TRACKIO_DIR / run / filename
    SQLiteStorage.get_project_db_path = get_project_db_path

    # def export_to_parquet():
    #     """
    #     Exports all projects' DB files as Parquet under the same path but with extension ".parquet".
    #     Also exports system_metrics to separate parquet files with "_system.parquet" suffix.
    #     Also exports configs to separate parquet files with "_configs.parquet" suffix.
    #     """
    #     if not SQLiteStorage._dataset_import_attempted:
    #         return
    #     if not TRACKIO_DIR.exists():
    #         return

    #     all_paths = os.listdir(TRACKIO_DIR)
    #     db_names = [f for f in all_paths if f.endswith(DB_EXT)]
    #     for db_name in db_names:
    #         db_path = TRACKIO_DIR / db_name
    #         parquet_path = db_path.with_suffix(".parquet")
    #         system_parquet_path = db_path.with_suffix("") / ""
    #         system_parquet_path = TRACKIO_DIR / (db_path.stem + "_system.parquet")
    #         configs_parquet_path = TRACKIO_DIR / (db_path.stem + "_configs.parquet")
    #         if (not parquet_path.exists()) or (
    #             db_path.stat().st_mtime > parquet_path.stat().st_mtime
    #         ):
    #             with sqlite3.connect(str(db_path)) as conn:
    #                 df = pd.read_sql("SELECT * FROM metrics", conn)
    #             if not df.empty:
    #                 metrics = df["metrics"].copy()
    #                 metrics = pd.DataFrame(
    #                     metrics.apply(
    #                         lambda x: deserialize_values(orjson.loads(x))
    #                     ).values.tolist(),
    #                     index=df.index,
    #                 )
    #                 del df["metrics"]
    #                 for col in metrics.columns:
    #                     df[col] = metrics[col]
    #                 df.to_parquet(
    #                     parquet_path,
    #                     write_page_index=True,
    #                     use_content_defined_chunking=True,
    #                 )

    #         if (not system_parquet_path.exists()) or (
    #             db_path.stat().st_mtime > system_parquet_path.stat().st_mtime
    #         ):
    #             with sqlite3.connect(str(db_path)) as conn:
    #                 try:
    #                     sys_df = pd.read_sql("SELECT * FROM system_metrics", conn)
    #                 except Exception:
    #                     sys_df = pd.DataFrame()
    #             if not sys_df.empty:
    #                 sys_metrics = sys_df["metrics"].copy()
    #                 sys_metrics = pd.DataFrame(
    #                     sys_metrics.apply(
    #                         lambda x: deserialize_values(orjson.loads(x))
    #                     ).values.tolist(),
    #                     index=sys_df.index,
    #                 )
    #                 del sys_df["metrics"]
    #                 for col in sys_metrics.columns:
    #                     sys_df[col] = sys_metrics[col]
    #                 sys_df.to_parquet(
    #                     system_parquet_path,
    #                     write_page_index=True,
    #                     use_content_defined_chunking=True,
    #                 )

    #         if (not configs_parquet_path.exists()) or (
    #             db_path.stat().st_mtime > configs_parquet_path.stat().st_mtime
    #         ):
    #             with sqlite3.connect(str(db_path)) as conn:
    #                 try:
    #                     configs_df = pd.read_sql("SELECT * FROM configs", conn)
    #                 except Exception:
    #                     configs_df = pd.DataFrame()
    #             if not configs_df.empty:
    #                 config_data = configs_df["config"].copy()
    #                 config_data = pd.DataFrame(
    #                     config_data.apply(
    #                         lambda x: deserialize_values(orjson.loads(x))
    #                     ).values.tolist(),
    #                     index=configs_df.index,
    #                 )
    #                 del configs_df["config"]
    #                 for col in config_data.columns:
    #                     configs_df[col] = config_data[col]
    #                 configs_df.to_parquet(
    #                     configs_parquet_path,
    #                     write_page_index=True,
    #                     use_content_defined_chunking=True,
    #                 )

    # def _cleanup_wal_sidecars(db_path: Path) -> None:
    #     """Remove leftover -wal/-shm files for a DB basename (prevents disk I/O errors)."""
    #     for suffix in ("-wal", "-shm"):
    #         sidecar = Path(str(db_path) + suffix)
    #         try:
    #             if sidecar.exists():
    #                 sidecar.unlink()
    #         except Exception:
    #             pass

    # def import_from_parquet():
    #     """
    #     Imports to all DB files that have matching files under the same path but with extension ".parquet".
    #     Also imports system_metrics from "_system.parquet" files.
    #     Also imports configs from "_configs.parquet" files.
    #     """
    #     if not TRACKIO_DIR.exists():
    #         return

    #     all_paths = os.listdir(TRACKIO_DIR)
    #     parquet_names = [
    #         f
    #         for f in all_paths
    #         if f.endswith(".parquet")
    #         and not f.endswith("_system.parquet")
    #         and not f.endswith("_configs.parquet")
    #     ]
    #     imported_projects = {Path(name).stem for name in parquet_names}
    #     for pq_name in parquet_names:
    #         parquet_path = TRACKIO_DIR / pq_name
    #         db_path = parquet_path.with_suffix(DB_EXT)

    #         SQLiteStorage._cleanup_wal_sidecars(db_path)

    #         df = pd.read_parquet(parquet_path)
    #         if "metrics" not in df.columns:
    #             metrics = df.copy()
    #             structural_cols = [
    #                 "id",
    #                 "timestamp",
    #                 "run_name",
    #                 "step",
    #                 "log_id",
    #                 "space_id",
    #             ]
    #             df = df[[c for c in structural_cols if c in df.columns]]
    #             for col in structural_cols:
    #                 if col in metrics.columns:
    #                     del metrics[col]
    #             metrics = orjson.loads(metrics.to_json(orient="records"))
    #             df["metrics"] = [orjson.dumps(serialize_values(row)) for row in metrics]

    #         with sqlite3.connect(str(db_path), timeout=30.0) as conn:
    #             df.to_sql("metrics", conn, if_exists="replace", index=False)
    #             conn.commit()

    #     system_parquet_names = [f for f in all_paths if f.endswith("_system.parquet")]
    #     for pq_name in system_parquet_names:
    #         parquet_path = TRACKIO_DIR / pq_name
    #         db_name = pq_name.replace("_system.parquet", DB_EXT)
    #         db_path = TRACKIO_DIR / db_name
    #         project_name = db_path.stem
    #         if project_name not in imported_projects and not db_path.exists():
    #             continue

    #         df = pd.read_parquet(parquet_path)
    #         if "metrics" not in df.columns:
    #             metrics = df.copy()
    #             other_cols = ["id", "timestamp", "run_name"]
    #             df = df[[c for c in other_cols if c in df.columns]]
    #             for col in other_cols:
    #                 if col in metrics.columns:
    #                     del metrics[col]
    #             metrics = orjson.loads(metrics.to_json(orient="records"))
    #             df["metrics"] = [orjson.dumps(serialize_values(row)) for row in metrics]

    #         with sqlite3.connect(str(db_path), timeout=30.0) as conn:
    #             df.to_sql("system_metrics", conn, if_exists="replace", index=False)
    #             conn.commit()

    #     configs_parquet_names = [f for f in all_paths if f.endswith("_configs.parquet")]
    #     for pq_name in configs_parquet_names:
    #         parquet_path = TRACKIO_DIR / pq_name
    #         db_name = pq_name.replace("_configs.parquet", DB_EXT)
    #         db_path = TRACKIO_DIR / db_name
    #         project_name = db_path.stem
    #         if project_name not in imported_projects and not db_path.exists():
    #             continue

    #         df = pd.read_parquet(parquet_path)
    #         if "config" not in df.columns:
    #             config_data = df.copy()
    #             other_cols = ["id", "run_name", "created_at"]
    #             df = df[[c for c in other_cols if c in df.columns]]
    #             for col in other_cols:
    #                 if col in config_data.columns:
    #                     del config_data[col]
    #             config_data = orjson.loads(config_data.to_json(orient="records"))
    #             df["config"] = [
    #                 orjson.dumps(serialize_values(row)) for row in config_data
    #             ]

    #         with sqlite3.connect(str(db_path), timeout=30.0) as conn:
    #             df.to_sql("configs", conn, if_exists="replace", index=False)
    #             conn.commit()

    def get_scheduler():
        """
        Get the scheduler for the database based on the environment variables.
        This applies to both local and Spaces.
        """
        with SQLiteStorage._scheduler_lock:
            if SQLiteStorage._current_scheduler is not None:
                return SQLiteStorage._current_scheduler
            hf_token = os.environ.get("HF_TOKEN")
            dataset_id = os.environ.get("TRACKIO_DATASET_ID")
            space_repo_name = os.environ.get("SPACE_REPO_NAME")
            if dataset_id is None or space_repo_name is None:
                scheduler = DummyCommitScheduler()
            else:
                scheduler = CommitScheduler(
                    repo_id=dataset_id,
                    repo_type="dataset",
                    folder_path=TRACKIO_DIR,
                    private=True,
                    allow_patterns=[
                        "*.parquet",
                        "*_system.parquet",
                        "*_configs.parquet",
                        "media/**/*",
                    ],
                    squash_history=True,
                    token=hf_token,
                    on_before_commit=SQLiteStorage.export_to_parquet,
                )
            SQLiteStorage._current_scheduler = scheduler
            return scheduler

    def get_alerts(
        project: str,
        run_name: str | None = None,
        level: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        if not run_name:
            results = []
            for run in SQLiteStorage.get_runs(project):
                results += get_alerts(project, run, level, since)
            return results

        db_path = SQLiteStorage.get_project_db_path(project)
        if not db_path.exists():
            return []

        with SQLiteStorage._get_connection(db_path) as conn:
            cursor = conn.cursor()
            try:
                query = (
                    "SELECT timestamp, run_name, title, text, level, step FROM alerts"
                )
                conditions = []
                params = []
                if run_name is not None:
                    conditions.append("run_name = ?")
                    params.append(run_name)
                if level is not None:
                    conditions.append("level = ?")
                    params.append(level)
                if since is not None:
                    conditions.append("timestamp > ?")
                    params.append(since)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " ORDER BY timestamp DESC"
                cursor.execute(query, params)

                rows = cursor.fetchall()
                return [
                    {
                        "timestamp": row["timestamp"],
                        "run": row["run_name"],
                        "title": row["title"],
                        "text": row["text"],
                        "level": row["level"],
                        "step": row["step"],
                    }
                    for row in rows
                ]
            except sqlite3.OperationalError as e:
                if "no such table: alerts" in str(e):
                    return []
                raise
    SQLiteStorage.get_alerts = get_alerts

    def get_alert_count(project: str) -> int:
        count = 0
        for run in SQLiteStorage.get_runs(project):
            db_path = SQLiteStorage.get_project_db_path(project)
            if not db_path.exists():
                continue
            with SQLiteStorage._get_connection(db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM alerts")
                    count += cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    continue
        return count
    SQLiteStorage.get_alert_count = get_alert_count

    def has_system_metrics(project: str) -> bool:
        """Check if a project has any system metrics logged."""
        for run in SQLiteStorage.get_runs():
            db_path = SQLiteStorage.get_project_db_path(project)
            if not db_path.exists():
                continue

            with SQLiteStorage._get_connection(db_path) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT COUNT(*) FROM system_metrics LIMIT 1")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        return True
                except sqlite3.OperationalError:
                    continue
        return False
    SQLiteStorage.has_system_metrics = has_system_metrics


    disallow_method('load_from_dataset')

    def get_projects() -> list[str]:
        """
        Get list of all projects by scanning the database files in the trackio directory.
        """
        if not SQLiteStorage._dataset_import_attempted:
            SQLiteStorage.load_from_dataset()

        projects: set[str] = set()
        if not TRACKIO_DIR.exists():
            return []

        for db_file in TRACKIO_DIR.rglob(f"*{DB_EXT}"):
            project_name = db_file.stem
            projects.add(project_name)
        return sorted(projects)
    SQLiteStorage.get_projects = get_projects

    def get_runs(project: str) -> list[str]:
        """Get list of all runs for a project, ordered by creation time."""
        return [p.parent.name for p in TRACKIO_DIR.rglob(SQLiteStorage.get_project_db_filename(project))]
    SQLiteStorage.get_runs = get_runs

    def get_max_steps_for_runs(project: str) -> dict[str, int]:
        """Get the maximum step for each run in a project."""
        runs = SQLiteStorage.get_runs(project)
        result = {}
        for run in runs:
            result[run] = SQLiteStorage.get_max_step_for_run(project, run)
        return result
    SQLiteStorage.get_max_steps_for_runs = get_max_steps_for_runs

    def delete_run(project: str, run: str) -> bool:
        """Delete a run from the database (metrics, config, and system_metrics)."""
        db_path = SQLiteStorage.get_project_db_path(project)
        if not db_path.exists():
            return False
        
        with SQLiteStorage._get_process_lock(project):
            shutil.rmtree(db_path.parent)
        return True
    SQLiteStorage.delete_run = delete_run

    # def _update_media_paths(obj, old_prefix, new_prefix): # TODO
    #     """Update media file paths in nested data structures."""
    #     if isinstance(obj, dict):
    #         if obj.get("_type") in [
    #             "trackio.image",
    #             "trackio.video",
    #             "trackio.audio",
    #         ]:
    #             old_path = obj.get("file_path", "")
    #             if isinstance(old_path, str):
    #                 normalized_path = old_path.replace("\\", "/")
    #                 if normalized_path.startswith(old_prefix):
    #                     new_path = normalized_path.replace(old_prefix, new_prefix, 1)
    #                     return {**obj, "file_path": new_path}
    #         return {
    #             key: SQLiteStorage._update_media_paths(value, old_prefix, new_prefix)
    #             for key, value in obj.items()
    #         }
    #     elif isinstance(obj, list):
    #         return [
    #             SQLiteStorage._update_media_paths(item, old_prefix, new_prefix)
    #             for item in obj
    #         ]
    #     return obj
    disallow_method('_update_media_paths')

    # def _rewrite_metrics_rows(metrics_rows, new_run_name, old_prefix, new_prefix): # TODO
    #     """Deserialize metrics rows, update media paths, and reserialize."""
    #     result = []
    #     for row in metrics_rows:
    #         metrics_data = orjson.loads(row["metrics"])
    #         metrics_deserialized = deserialize_values(metrics_data)
    #         updated = SQLiteStorage._update_media_paths(
    #             metrics_deserialized, old_prefix, new_prefix
    #         )
    #         result.append(
    #             (
    #                 row["timestamp"],
    #                 new_run_name,
    #                 row["step"],
    #                 orjson.dumps(serialize_values(updated)),
    #             )
    #         )
    #     return result
    disallow_method('_update_metrics_rows')

    # def _move_media_dir(source: Path, target: Path): # TODO
    #     """Move a media directory from source to target."""
    #     if source.exists():
    #         target.parent.mkdir(parents=True, exist_ok=True)
    #         if target.exists():
    #             shutil.rmtree(target)
    #         shutil.move(str(source), str(target))
    disallow_method('_move_media_dir')

    # def rename_run(project: str, old_name: str, new_name: str) -> None: # TODO
    #     """Rename a run within the same project.

    #     Raises:
    #         ValueError: If the new name is empty, the old run doesn't exist,
    #                     or a run with the new name already exists.
    #         RuntimeError: If the database operation fails.
    #     """
    #     if not new_name or not new_name.strip():
    #         raise ValueError("New run name cannot be empty")

    #     new_name = new_name.strip()

    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         raise ValueError(f"Project '{project}' does not exist")

    #     with SQLiteStorage._get_process_lock(project):
    #         with SQLiteStorage._get_connection(db_path) as conn:
    #             cursor = conn.cursor()

    #             cursor.execute(
    #                 "SELECT COUNT(*) FROM metrics WHERE run_name = ?", (old_name,)
    #             )
    #             if cursor.fetchone()[0] == 0:
    #                 raise ValueError(
    #                     f"Run '{old_name}' does not exist in project '{project}'"
    #                 )

    #             cursor.execute(
    #                 "SELECT COUNT(*) FROM metrics WHERE run_name = ?", (new_name,)
    #             )
    #             if cursor.fetchone()[0] > 0:
    #                 raise ValueError(
    #                     f"A run named '{new_name}' already exists in project '{project}'"
    #                 )

    #             try:
    #                 cursor.execute(
    #                     "SELECT timestamp, step, metrics FROM metrics WHERE run_name = ?",
    #                     (old_name,),
    #                 )
    #                 metrics_rows = cursor.fetchall()

    #                 old_prefix = f"{project}/{old_name}/"
    #                 new_prefix = f"{project}/{new_name}/"

    #                 updated_rows = SQLiteStorage._rewrite_metrics_rows(
    #                     metrics_rows, new_name, old_prefix, new_prefix
    #                 )

    #                 cursor.execute(
    #                     "DELETE FROM metrics WHERE run_name = ?", (old_name,)
    #                 )
    #                 cursor.executemany(
    #                     "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
    #                     updated_rows,
    #                 )

    #                 cursor.execute(
    #                     "UPDATE configs SET run_name = ? WHERE run_name = ?",
    #                     (new_name, old_name),
    #                 )

    #                 try:
    #                     cursor.execute(
    #                         "UPDATE system_metrics SET run_name = ? WHERE run_name = ?",
    #                         (new_name, old_name),
    #                     )
    #                 except sqlite3.OperationalError:
    #                     pass

    #                 try:
    #                     cursor.execute(
    #                         "UPDATE alerts SET run_name = ? WHERE run_name = ?",
    #                         (new_name, old_name),
    #                     )
    #                 except sqlite3.OperationalError:
    #                     pass

    #                 conn.commit()

    #                 SQLiteStorage._move_media_dir(
    #                     MEDIA_DIR / project / old_name,
    #                     MEDIA_DIR / project / new_name,
    #                 )
    #             except sqlite3.Error as e:
    #                 raise RuntimeError(
    #                     f"Database error while renaming run '{old_name}' to '{new_name}': {e}"
    #                 ) from e
    disallow_method('rename_run')

    # def move_run(project: str, run: str, new_project: str) -> bool: # TODO
    #     """Move a run from one project to another."""
    #     source_db_path = SQLiteStorage.get_project_db_path(project)
    #     if not source_db_path.exists():
    #         return False

    #     target_db_path = SQLiteStorage.init_db(new_project)

    #     with SQLiteStorage._get_process_lock(project):
    #         with SQLiteStorage._get_process_lock(new_project):
    #             with SQLiteStorage._get_connection(source_db_path) as source_conn:
    #                 source_cursor = source_conn.cursor()

    #                 source_cursor.execute(
    #                     "SELECT timestamp, step, metrics FROM metrics WHERE run_name = ?",
    #                     (run,),
    #                 )
    #                 metrics_rows = source_cursor.fetchall()

    #                 source_cursor.execute(
    #                     "SELECT config, created_at FROM configs WHERE run_name = ?",
    #                     (run,),
    #                 )
    #                 config_row = source_cursor.fetchone()

    #                 try:
    #                     source_cursor.execute(
    #                         "SELECT timestamp, metrics FROM system_metrics WHERE run_name = ?",
    #                         (run,),
    #                     )
    #                     system_metrics_rows = source_cursor.fetchall()
    #                 except sqlite3.OperationalError:
    #                     system_metrics_rows = []

    #                 try:
    #                     source_cursor.execute(
    #                         "SELECT timestamp, title, text, level, step, alert_id FROM alerts WHERE run_name = ?",
    #                         (run,),
    #                     )
    #                     alert_rows = source_cursor.fetchall()
    #                 except sqlite3.OperationalError:
    #                     alert_rows = []

    #                 if not metrics_rows and not config_row and not system_metrics_rows:
    #                     return False

    #                 with SQLiteStorage._get_connection(target_db_path) as target_conn:
    #                     target_cursor = target_conn.cursor()

    #                     old_prefix = f"{project}/{run}/"
    #                     new_prefix = f"{new_project}/{run}/"
    #                     updated_rows = SQLiteStorage._rewrite_metrics_rows(
    #                         metrics_rows, run, old_prefix, new_prefix
    #                     )

    #                     target_cursor.executemany(
    #                         "INSERT INTO metrics (timestamp, run_name, step, metrics) VALUES (?, ?, ?, ?)",
    #                         updated_rows,
    #                     )

    #                     if config_row:
    #                         target_cursor.execute(
    #                             """
    #                             INSERT OR REPLACE INTO configs (run_name, config, created_at)
    #                             VALUES (?, ?, ?)
    #                             """,
    #                             (run, config_row["config"], config_row["created_at"]),
    #                         )

    #                     for row in system_metrics_rows:
    #                         try:
    #                             target_cursor.execute(
    #                                 """
    #                                 INSERT INTO system_metrics (timestamp, run_name, metrics)
    #                                 VALUES (?, ?, ?)
    #                                 """,
    #                                 (row["timestamp"], run, row["metrics"]),
    #                             )
    #                         except sqlite3.OperationalError:
    #                             pass

    #                     for row in alert_rows:
    #                         try:
    #                             target_cursor.execute(
    #                                 """
    #                                 INSERT OR IGNORE INTO alerts (timestamp, run_name, title, text, level, step, alert_id)
    #                                 VALUES (?, ?, ?, ?, ?, ?, ?)
    #                                 """,
    #                                 (
    #                                     row["timestamp"],
    #                                     run,
    #                                     row["title"],
    #                                     row["text"],
    #                                     row["level"],
    #                                     row["step"],
    #                                     row["alert_id"],
    #                                 ),
    #                             )
    #                         except sqlite3.OperationalError:
    #                             pass

    #                     target_conn.commit()

    #                     SQLiteStorage._move_media_dir(
    #                         MEDIA_DIR / project / run,
    #                         MEDIA_DIR / new_project / run,
    #                     )

    #                     source_cursor.execute(
    #                         "DELETE FROM metrics WHERE run_name = ?", (run,)
    #                     )
    #                     source_cursor.execute(
    #                         "DELETE FROM configs WHERE run_name = ?", (run,)
    #                     )
    #                     try:
    #                         source_cursor.execute(
    #                             "DELETE FROM system_metrics WHERE run_name = ?", (run,)
    #                         )
    #                     except sqlite3.OperationalError:
    #                         pass
    #                     try:
    #                         source_cursor.execute(
    #                             "DELETE FROM alerts WHERE run_name = ?", (run,)
    #                         )
    #                     except sqlite3.OperationalError:
    #                         pass
    #                     source_conn.commit()

    #                     return True
    disallow_method('move_run')


    def get_all_run_configs(project: str) -> dict[str, dict]:
        """Get configurations for all runs in a project."""
        db_path = SQLiteStorage.get_project_db_path(project)
        if not db_path.exists():
            return {}

        with SQLiteStorage._get_connection(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT run_name, config FROM configs
                    """
                )

                results = {}
                for row in cursor.fetchall():
                    config = orjson.loads(row["config"])
                    results[row["run_name"]] = deserialize_values(config)
                return results
            except sqlite3.OperationalError as e:
                if "no such table: configs" in str(e):
                    return {}
                raise

    def set_project_metadata(project: str, key: str, value: str) -> None:
        db_path = SQLiteStorage.init_db(project)
        with SQLiteStorage._get_process_lock(project):
            with SQLiteStorage._get_connection(db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO project_metadata (key, value) VALUES (?, ?)",
                    (key, value),
                )
                conn.commit()

    def get_project_metadata(project: str, key: str) -> str | None:
        db_path = SQLiteStorage.get_project_db_path(project)
        if not db_path.exists():
            return None
        with SQLiteStorage._get_connection(db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT value FROM project_metadata WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                return row[0] if row else None
            except sqlite3.OperationalError:
                return None

    # def get_space_id(project: str) -> str | None: # TODO
    #     return SQLiteStorage.get_project_metadata(project, "space_id")
    disallow_method('get_space_id')


    # def has_pending_data(project: str) -> bool:
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return False
    #     with SQLiteStorage._get_connection(db_path) as conn:
    #         cursor = conn.cursor()
    #         try:
    #             cursor.execute(
    #                 "SELECT EXISTS(SELECT 1 FROM metrics WHERE space_id IS NOT NULL LIMIT 1)"
    #             )
    #             if cursor.fetchone()[0]:
    #                 return True
    #         except sqlite3.OperationalError:
    #             pass
    #         try:
    #             cursor.execute(
    #                 "SELECT EXISTS(SELECT 1 FROM system_metrics WHERE space_id IS NOT NULL LIMIT 1)"
    #             )
    #             if cursor.fetchone()[0]:
    #                 return True
    #         except sqlite3.OperationalError:
    #             pass
    #         try:
    #             cursor.execute("SELECT EXISTS(SELECT 1 FROM pending_uploads LIMIT 1)")
    #             if cursor.fetchone()[0]:
    #                 return True
    #         except sqlite3.OperationalError:
    #             pass
    #         return False
    disallow_method('has_pending_data') # TODO


    # def get_pending_logs(project: str) -> dict | None:
    #     return SQLiteStorage._get_pending(
    #         project, "metrics", extra_fields=["step"], include_config=True
    #     )
    disallow_method('get_pending_logs') # TODO

    # def clear_pending_logs(project: str, metric_ids: list[int]) -> None:
    #     SQLiteStorage._clear_pending(project, "metrics", metric_ids)
    disallow_method('clear_pending_logs') # TODO

    # def get_pending_system_logs(project: str) -> dict | None:
    #     return SQLiteStorage._get_pending(project, "system_metrics")
    disallow_method('get_pending_system_logs') # TODO

    # def _get_pending(
    #     project: str,
    #     table: str,
    #     extra_fields: list[str] | None = None,
    #     include_config: bool = False,
    # ) -> dict | None:
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return None
    #     extra_cols = ", ".join(extra_fields) + ", " if extra_fields else ""
    #     with SQLiteStorage._get_connection(db_path) as conn:
    #         cursor = conn.cursor()
    #         try:
    #             cursor.execute(
    #                 f"""SELECT id, timestamp, run_name, {extra_cols}metrics, log_id, space_id
    #                 FROM {table} WHERE space_id IS NOT NULL"""
    #             )
    #         except sqlite3.OperationalError:
    #             return None
    #         rows = cursor.fetchall()
    #         if not rows:
    #             return None
    #         logs = []
    #         ids = []
    #         for row in rows:
    #             metrics = deserialize_values(orjson.loads(row["metrics"]))
    #             entry = {
    #                 "project": project,
    #                 "run": row["run_name"],
    #                 "metrics": metrics,
    #                 "timestamp": row["timestamp"],
    #                 "log_id": row["log_id"],
    #             }
    #             for field in extra_fields or []:
    #                 entry[field] = row[field]
    #             if include_config:
    #                 entry["config"] = None
    #             logs.append(entry)
    #             ids.append(row["id"])
    #         return {"logs": logs, "ids": ids, "space_id": rows[0]["space_id"]}
    disallow_method('_get_pending') # TODO

    # def clear_pending_system_logs(project: str, metric_ids: list[int]) -> None:
    #     SQLiteStorage._clear_pending(project, "system_metrics", metric_ids)
    disallow_method('clear_pending_system_logs') # TODO
    

    # def _clear_pending(project: str, table: str, ids: list[int]) -> None:
    #     if not ids:
    #         return
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return
    #     with SQLiteStorage._get_process_lock(project):
    #         with SQLiteStorage._get_connection(db_path) as conn:
    #             placeholders = ",".join("?" * len(ids))
    #             conn.execute(
    #                 f"DELETE FROM {table} WHERE id IN ({placeholders})",
    #                 ids,
    #             )
    #             conn.commit()
    disallow_method('_clear_pending') # TODO

    # def get_pending_uploads(project: str) -> dict | None:
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return None
    #     with SQLiteStorage._get_connection(db_path) as conn:
    #         cursor = conn.cursor()
    #         try:
    #             cursor.execute(
    #                 """SELECT id, space_id, run_name, step, file_path, relative_path
    #                 FROM pending_uploads"""
    #             )
    #         except sqlite3.OperationalError:
    #             return None
    #         rows = cursor.fetchall()
    #         if not rows:
    #             return None
    #         uploads = []
    #         ids = []
    #         for row in rows:
    #             uploads.append(
    #                 {
    #                     "project": project,
    #                     "run": row["run_name"],
    #                     "step": row["step"],
    #                     "file_path": row["file_path"],
    #                     "relative_path": row["relative_path"],
    #                 }
    #             )
    #             ids.append(row["id"])
    #         return {"uploads": uploads, "ids": ids, "space_id": rows[0]["space_id"]}
    disallow_method('get_pending_uploads') # TODO

    # def clear_pending_uploads(project: str, upload_ids: list[int]) -> None:
    #     if not upload_ids:
    #         return
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return
    #     with SQLiteStorage._get_process_lock(project):
    #         with SQLiteStorage._get_connection(db_path) as conn:
    #             placeholders = ",".join("?" * len(upload_ids))
    #             conn.execute(
    #                 f"DELETE FROM pending_uploads WHERE id IN ({placeholders})",
    #                 upload_ids,
    #             )
    #             conn.commit()
    disallow_method('clear_pending_uploads') # TODO

    # def add_pending_upload(
    #     project: str,
    #     space_id: str,
    #     run_name: str | None,
    #     step: int | None,
    #     file_path: str,
    #     relative_path: str | None,
    # ) -> None:
    #     db_path = SQLiteStorage.init_db(project)
    #     with SQLiteStorage._get_process_lock(project):
    #         with SQLiteStorage._get_connection(db_path) as conn:
    #             conn.execute(
    #                 """INSERT INTO pending_uploads
    #                 (space_id, run_name, step, file_path, relative_path, created_at)
    #                 VALUES (?, ?, ?, ?, ?, ?)""",
    #                 (
    #                     space_id,
    #                     run_name,
    #                     step,
    #                     file_path,
    #                     relative_path,
    #                     datetime.now(timezone.utc).isoformat(),
    #                 ),
    #             )
    #             conn.commit()
    disallow_method('add_pending_upload') # TODO

    # def get_all_logs_for_sync(project: str) -> list[dict]:
    #     return SQLiteStorage._get_all_for_sync(
    #         project,
    #         "metrics",
    #         order_by="run_name, step",
    #         extra_fields=["step"],
    #         include_config=True,
    #     )
    disallow_method('get_all_logs_for_sync') # TODO

    # def get_all_system_logs_for_sync(project: str) -> list[dict]:
    #     return SQLiteStorage._get_all_for_sync(
    #         project, "system_metrics", order_by="run_name, timestamp"
    #     )
    disallow_method('get_all_system_logs_for_sync') # TODO

    # def _get_all_for_sync(
    #     project: str,
    #     table: str,
    #     order_by: str,
    #     extra_fields: list[str] | None = None,
    #     include_config: bool = False,
    # ) -> list[dict]:
    #     db_path = SQLiteStorage.get_project_db_path(project)
    #     if not db_path.exists():
    #         return []
    #     extra_cols = ", ".join(extra_fields) + ", " if extra_fields else ""
    #     with SQLiteStorage._get_connection(db_path) as conn:
    #         cursor = conn.cursor()
    #         try:
    #             cursor.execute(
    #                 f"""SELECT timestamp, run_name, {extra_cols}metrics, log_id
    #                 FROM {table} ORDER BY {order_by}"""
    #             )
    #         except sqlite3.OperationalError:
    #             return []
    #         rows = cursor.fetchall()
    #         results = []
    #         for row in rows:
    #             metrics = deserialize_values(orjson.loads(row["metrics"]))
    #             entry = {
    #                 "project": project,
    #                 "run": row["run_name"],
    #                 "metrics": metrics,
    #                 "timestamp": row["timestamp"],
    #                 "log_id": row["log_id"],
    #             }
    #             for field in extra_fields or []:
    #                 entry[field] = row[field]
    #             if include_config:
    #                 entry["config"] = None
    #             results.append(entry)
    #         return results
    disallow_method('_get_all_for_sync') # TODO




    def get_project_media_path(
        project: str,
        run: str | None = None,
        step: int | None = None,
        relative_path: str | Path | None = None,
    ) -> Path:
        """
        Get the full path where uploaded files are stored for a Trackio project (and create the directory if it doesn't exist).
        If a run is not provided, the files are stored in a project-level directory with the given relative path.

        Args:
            project: The project name
            run: The run name
            step: The step number
            relative_path: The relative path within the directory (only used if run is not provided)

        Returns:
            The full path to the media file
        """
        if step is not None and run is None:
            raise ValueError("Uploading files at a specific step requires a run")

        path = MEDIA_DIR / project
        if run:
            path = MEDIA_DIR / run / project
            if step is not None:
                path /= str(step)
        else:
            path /= "files"
            if relative_path:
                path /= relative_path
        path.mkdir(parents=True, exist_ok=True)
        return path.absolute()
    utils.get_project_media_path = get_project_media_path
    media.get_project_media_path = get_project_media_path
    run.get_project_media_path = get_project_media_path
    server.get_project_media_path = get_project_media_path
    media_base.get_project_media_path = get_project_media_path
    trackio.get_project_media_path = get_project_media_path