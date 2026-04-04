#!/usr/bin/env python3
"""
DuckDB REST proxy — exposes POST /query and GET /health.
Receives SQL via JSON body, executes against local DuckDB, returns rows as JSON.
"""

import os
import json
import logging
import duckdb
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("duckdb-proxy")

app = Flask(__name__)
PARQUET_DIR = os.environ.get("PARQUET_DIR", "/data/parquet")
PORT = int(os.environ.get("PORT", 8082))

# Persistent DuckDB connection (in-memory).
con = duckdb.connect(":memory:")

# Bootstrap: auto-discover parquet/CSV files.
def bootstrap():
    import glob
    patterns = [
        f"{PARQUET_DIR}/**/*.parquet",
        f"{PARQUET_DIR}/**/*.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    if files:
        csv_files = [f for f in files if f.endswith(".csv")]
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if csv_files:
            glob_csv = f"{PARQUET_DIR}/**/*.csv"
            try:
                con.execute(f"""
                    CREATE OR REPLACE VIEW bars AS
                    SELECT * FROM read_csv_auto('{glob_csv}', header=true, union_by_name=true)
                """)
                log.info(f"Loaded {len(csv_files)} CSV files into bars view")
            except Exception as e:
                log.warning(f"CSV load failed: {e}")
        if parquet_files:
            glob_pq = f"{PARQUET_DIR}/**/*.parquet"
            try:
                con.execute(f"""
                    CREATE OR REPLACE VIEW bars_parquet AS
                    SELECT * FROM read_parquet('{glob_pq}')
                """)
                log.info(f"Loaded {len(parquet_files)} parquet files")
            except Exception as e:
                log.warning(f"Parquet load failed: {e}")
    else:
        log.info("No data files found — creating empty bars table")
        con.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                timestamp TIMESTAMPTZ,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                source VARCHAR,
                timeframe VARCHAR
            )
        """)

bootstrap()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "parquet_dir": PARQUET_DIR})


@app.route("/query", methods=["POST"])
def query():
    body = request.get_json(force=True, silent=True)
    if not body or "query" not in body:
        return jsonify({"error": "missing 'query' field"}), 400

    sql = body["query"]
    log.debug(f"Query: {sql[:200]}")

    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        data = [dict(zip(columns, row)) for row in rows]
        return jsonify({"rows": data, "count": len(data), "columns": columns})
    except duckdb.Error as e:
        log.warning(f"Query error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/reload", methods=["POST"])
def reload():
    """Re-scan parquet directory and rebuild views."""
    try:
        bootstrap()
        return jsonify({"status": "reloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    log.info(f"DuckDB proxy starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
