"""PharmaSales FastMCP Server with comprehensive analytical tools, resources, and visualization-guided prompts."""

import base64
import json
import math
import random
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from pydantic import Field

from fastmcp import FastMCP, Context
from mcpagentdemo.db import get_connection

mcp = FastMCP(
    name="PharmaSales Analytics MCP",
    version="1.0.0",
    instructions="""An analytics MCP for the synthetic Indian pharma sales DuckDB dataset (2021-11-01 through 2026-10-27).

Start with `pharma://context/guide` and `pharma://schema/tables` to learn the data grain, schema, joins, metrics, and caveats. Use `list_database_objects` and `profile_table` when you need to discover tables or inspect unfamiliar columns. Use `execute_sql` for read-only, parameter-free analytical SQL; results are capped at the requested row count. For monthly forecast alternatives use `compare_forecast_models`; for coverage and drift use `forecast_accuracy_monitor`; for coherent product/segment totals use `reconciled_sales_forecast`; for assumed price, discount, volume, or cost changes use `sales_driver_scenario_forecast`. Use `forecast_numeric_series` or `predict_numeric_regression` for other catalog-backed numeric outcomes. Do not describe simulations as observed forecasts or causal effects.

For charts, read `pharma://design/color-themes`, then call `generate_chart_ready_dataset` and use the matching visualization prompt. Present monetary amounts only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions; 1 L = ₹100,000 and 1 Cr = ₹10,000,000. Label currency, state date ranges, distinguish counts from amounts, and explain the denominator behind percentages. Tool data may retain exact INR values for calculations; convert for human-facing summaries and charts. Treat the dataset as synthetic business data: it contains no real company, customer, or patient records, and it does not support medical or market claims. State uncertainty and avoid causal claims from descriptive comparisons."""
)

def _sanitize_val(val: Any) -> Any:
    if isinstance(val, (date, datetime)):
        return val.isoformat()
    if isinstance(val, Decimal):
        converted = float(val)
        return converted if math.isfinite(converted) else None
    if isinstance(val, float) and not math.isfinite(val):
        return None
    if isinstance(val, bytes):
        return "base64:" + base64.b64encode(val).decode("ascii")
    if isinstance(val, dict):
        return {str(key): _sanitize_val(value) for key, value in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_val(value) for value in val]
    return val

def _run_query(sql: str, params: list | tuple | None = None, max_rows: int | None = None) -> list[dict[str, Any]]:
    con = get_connection(read_only=True)
    try:
        cur = con.execute(sql, params or [])
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(max_rows) if max_rows is not None else cur.fetchall()
        return [
            {col: _sanitize_val(val) for col, val in zip(cols, row)}
            for row in rows
        ]
    finally:
        con.close()


def _quote_identifier(identifier: str) -> str:
    """Quote one SQL identifier after it has been matched to catalog metadata."""
    return '"' + identifier.replace('"', '""') + '"'


def _resolve_relation(con: Any, name: str) -> tuple[str, str, str]:
    """Resolve a caller-provided table or view name against the local catalog."""
    parts = [part.strip() for part in name.split(".")]
    if len(parts) == 1:
        schema_name, relation_name = "main", parts[0]
    elif len(parts) == 2:
        schema_name, relation_name = parts
    else:
        raise ValueError("Use a table name or schema.table name.")
    matches = con.execute(
        """SELECT table_schema, table_name, table_type
           FROM information_schema.tables
           WHERE lower(table_schema) = lower(?) AND lower(table_name) = lower(?)""",
        [schema_name, relation_name],
    ).fetchall()
    if not matches:
        raise ValueError(f"Table or view '{name}' was not found in the current database.")
    actual_schema, actual_name, relation_type = matches[0]
    return actual_schema, actual_name, relation_type


def _relation_sql(schema_name: str, relation_name: str) -> str:
    return f"{_quote_identifier(schema_name)}.{_quote_identifier(relation_name)}"


def _validate_readonly_query(query: str) -> str:
    """Accept one SELECT-like statement; DuckDB also disables external access."""
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Query must not be empty.")
    # Mask quoted strings and SQL comments before checking statement boundaries.
    masked = re.sub(
        r"(?s)/\*.*?\*/|--[^\n]*|'(?:''|\\.|[^'])*'|\"(?:\"\"|\\.|[^\"])*\"",
        lambda match: " " * len(match.group(0)),
        cleaned,
    )
    semicolons = [match.start() for match in re.finditer(";", masked)]
    if semicolons and (len(semicolons) != 1 or masked[semicolons[0] + 1:].strip()):
        raise ValueError("Only one SQL statement is permitted.")
    if semicolons:
        cleaned = cleaned[:semicolons[0]].rstrip()
    normalized = re.sub(
        r"(?s)/\*.*?\*/|--[^\n]*|'(?:''|\\.|[^'])*'|\"(?:\"\"|\\.|[^\"])*\"",
        " ",
        cleaned,
    ).lstrip()
    if not re.match(r"(?is)^(SELECT|WITH|DESCRIBE|DESC|SHOW|EXPLAIN)\b", normalized):
        raise ValueError("Only SELECT, WITH, DESCRIBE, SHOW, or EXPLAIN queries are permitted.")
    return cleaned


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small dense system with pivoted Gaussian elimination."""
    size = len(values)
    augmented = [list(matrix[row]) + [values[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] += 1e-8
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-15:
            raise ValueError("Regression features are singular; reduce the model complexity.")
        augmented[column] = [item / divisor for item in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    augmented[row][item] - factor * augmented[column][item]
                    for item in range(size + 1)
                ]
    return [augmented[row][-1] for row in range(size)]


def _fit_ridge_regression(features: list[list[float]], values: list[float], ridge: float = 1e-6) -> list[float]:
    if not features or len(features) != len(values):
        raise ValueError("Regression requires matching non-empty feature and value rows.")
    width = len(features[0])
    gram = [[0.0] * width for _ in range(width)]
    target = [0.0] * width
    for row, value in zip(features, values):
        if len(row) != width:
            raise ValueError("Regression feature rows have inconsistent widths.")
        for left in range(width):
            target[left] += row[left] * value
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(1, width):
        gram[index][index] += ridge
    return _solve_linear_system(gram, target)


def _fit_logistic_regression(features: list[list[float]], labels: list[int], ridge: float = 0.1) -> list[float]:
    """Fit a small binary logistic model with damped Newton/IRLS updates."""
    if not features or len(features) != len(labels) or len(set(labels)) < 2:
        raise ValueError("Logistic regression needs matching rows and both outcome classes.")
    width = len(features[0])
    coefficients = [0.0] * width
    for _ in range(60):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]
        for row, label in zip(features, labels):
            score = max(-35.0, min(35.0, sum(a * b for a, b in zip(row, coefficients))))
            probability = 1.0 / (1.0 + math.exp(-score))
            weight = max(probability * (1.0 - probability), 1e-8)
            residual = label - probability
            for left in range(width):
                gradient[left] += row[left] * residual
                for right in range(width):
                    hessian[left][right] += row[left] * row[right] * weight
        for index in range(1, width):
            gradient[index] -= ridge * coefficients[index]
            hessian[index][index] += ridge
        update = _solve_linear_system(hessian, gradient)
        coefficients = [coefficient + delta for coefficient, delta in zip(coefficients, update)]
        if max(abs(delta) for delta in update) < 1e-6:
            break
    return coefficients


def _predict_logistic_probability(features: list[float], coefficients: list[float]) -> float:
    score = max(-35.0, min(35.0, sum(a * b for a, b in zip(features, coefficients))))
    return 1.0 / (1.0 + math.exp(-score))


def _binary_model_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not labels:
        return {"rows": 0, "positive_rate_pct": None, "brier_score": None, "log_loss": None, "auc": None}
    clipped = [min(1.0 - 1e-12, max(1e-12, value)) for value in probabilities]
    brier = sum((probability - label) ** 2 for label, probability in zip(labels, clipped)) / len(labels)
    log_loss = -sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, clipped)
    ) / len(labels)
    positives = sum(labels)
    negatives = len(labels) - positives
    auc = None
    if positives and negatives:
        ranked = sorted(zip(clipped, labels), key=lambda item: item[0])
        rank_sum = 0.0
        start = 0
        while start < len(ranked):
            end = start + 1
            while end < len(ranked) and ranked[end][0] == ranked[start][0]:
                end += 1
            average_rank = ((start + 1) + end) / 2
            rank_sum += average_rank * sum(label for _, label in ranked[start:end])
            start = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    top_count = max(1, math.ceil(len(labels) * 0.1))
    top_labels = [label for _, label in sorted(zip(clipped, labels), key=lambda item: item[0], reverse=True)[:top_count]]
    return {
        "rows": len(labels),
        "positive_rate_pct": round(100 * positives / len(labels), 3),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "auc": round(auc, 5) if auc is not None else None,
        "top_decile_precision_pct": round(100 * sum(top_labels) / len(top_labels), 3),
        "top_decile_lift": round((sum(top_labels) / len(top_labels)) / (positives / len(labels)), 3) if positives else None,
    }


MONTHLY_FORECAST_MODELS = {
    "seasonal_naive": "Same month last year",
    "last_value": "Last observed value",
    "moving_average_3": "Trailing 3-month average",
    "moving_average_6": "Trailing 6-month average",
    "trend_seasonal": "Ridge trend with annual seasonality",
}


def _month_index(period: date) -> int:
    return period.year * 12 + period.month - 1


def _advance_month(period: date, steps: int = 1) -> date:
    index = _month_index(period) + steps
    year, month = divmod(index, 12)
    return date(year, month + 1, 1)


def _monthly_model_prediction(
    model_id: str,
    history: list[tuple[date, float]],
    target_period: date,
) -> float | None:
    if not history:
        return None
    history = sorted(history, key=lambda item: item[0])
    values = [value for _, value in history]
    if model_id == "seasonal_naive":
        source = date(target_period.year - 1, target_period.month, 1)
        return next((value for period, value in history if period == source), None)
    if model_id == "last_value":
        return values[-1]
    if model_id in {"moving_average_3", "moving_average_6"}:
        window = 3 if model_id.endswith("_3") else 6
        tail = values[-window:]
        return sum(tail) / len(tail)
    if model_id != "trend_seasonal":
        return None
    origin = min(_month_index(period) for period, _ in history)
    span = max(1, max(_month_index(period) for period, _ in history) - origin)
    use_season = len(history) >= 24

    def row_features(period: date) -> list[float]:
        features = [1.0, (_month_index(period) - origin) / span]
        if use_season:
            angle = 2.0 * math.pi * (period.month - 1) / 12
            features.extend([math.sin(angle), math.cos(angle)])
        return features

    coefficients = _fit_ridge_regression(
        [row_features(period) for period, _ in history],
        values,
        ridge=1e-5,
    )
    return sum(left * right for left, right in zip(row_features(target_period), coefficients))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _monthly_sales_segment_rows(group_by: str) -> tuple[date, list[dict[str, Any]]]:
    dimensions = {
        "overall": None,
        "therapeutic_area": "therapeutic_area",
        "brand_name": "brand_name",
        "channel": "channel",
        "region": "region",
    }
    if group_by not in dimensions:
        raise ValueError(f"group_by must be one of: {', '.join(dimensions)}.")
    latest = _run_query("SELECT date_trunc('month', max(invoice_date))::DATE AS partial_month FROM fact_sales")
    partial_month = latest[0]["partial_month"] if latest else None
    if not partial_month:
        raise ValueError("No invoice dates are available.")
    partial_month = partial_month if isinstance(partial_month, date) else date.fromisoformat(str(partial_month)[:10])
    dimension = dimensions[group_by]
    segment_sql = "'All sales'" if dimension is None else f"coalesce(cast({dimension} AS VARCHAR), '(missing)')"
    group_sql = "period" if dimension is None else "period, segment"
    sql = f"""
    SELECT date_trunc('month', invoice_date)::DATE AS period,
      {segment_sql} AS segment,
      sum(gross_sales_inr) AS gross_sales_inr,
      sum(discount_inr) AS discount_inr,
      sum(net_sales_inr) AS net_sales_inr,
      sum(cogs_inr) AS cogs_inr,
      sum(gross_profit_inr) AS gross_profit_inr,
      sum(quantity_packs) AS packs,
      count(*) AS invoice_lines
    FROM sales_full
    WHERE invoice_date < ? AND coalesce(returned, false) = false
    GROUP BY {group_sql}
    ORDER BY segment, period
    """
    return partial_month, _run_query(sql, [partial_month])


# ==============================================================================
# RESOURCES
# ==============================================================================

@mcp.resource("pharma://schema/tables")
def get_tables_schema() -> str:
    """Returns the full database schema for all DuckDB tables and views, including column names, data types, and nullability."""
    con = get_connection(read_only=True)
    try:
        tables = con.execute(
            """SELECT table_schema, table_name, table_type
               FROM information_schema.tables
               WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
               ORDER BY table_schema, table_name"""
        ).fetchall()
        schema_dict = {}
        for schema_name, table_name, relation_type in tables:
            cols = con.execute(f"DESCRIBE {_relation_sql(schema_name, table_name)}").fetchall()
            schema_dict[f"{schema_name}.{table_name}"] = {
                "kind": relation_type,
                "columns": [
                    {"name": c[0], "type": c[1], "nullable": c[2] == "YES"} for c in cols
                ],
            }
        return json.dumps(schema_dict, indent=2)
    finally:
        con.close()


@mcp.resource("pharma://context/guide")
def get_agent_context_guide() -> str:
    """Gives agents the dataset grain, join map, metric semantics, provenance, and safe analysis guidance."""
    guide = {
        "dataset": {
            "description": "Synthetic Indian pharma sales data in INR; not real company, pharmacy, or patient data.",
            "coverage": "2021-11-01 through 2026-10-27",
            "fact_grain": "One row per invoice line in fact_sales.",
            "approximate_scale": {"invoice_lines": 10_000_000, "customers": 60_000, "skus": 84, "sales_reps": 160},
        },
        "joins": [
            "fact_sales.sku_id = dim_sku.sku_id",
            "fact_sales.customer_id = dim_customer.customer_id",
            "fact_sales.invoice_date = dim_date.date",
            "dim_customer.sales_rep_id = dim_rep.sales_rep_id",
        ],
        "metric_semantics": {
            "net_sales_inr": "Net sales after invoice discounts, in INR.",
            "gross_profit_inr": "Gross profit in INR; use SUM(gross_profit_inr) / SUM(net_sales_inr) for weighted gross margin.",
            "gross_margin_pct": "Weighted gross profit divided by net sales, multiplied by 100. Do not average row-level percentages for an aggregate.",
            "avg_discount_pct": "Weighted discount amount divided by gross sales, multiplied by 100.",
            "total_packs": "SUM(quantity_packs); this is pack volume, not unit count.",
            "monthly_summary.invoices": "This column is COUNT(*) of fact_sales rows, so it counts invoice lines, not distinct invoice_no values.",
            "returns": "Use returned = true and credit_note_inr for return incidence and credits.",
            "currency_display": "Show monetary values only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions/billions. 1 L = ₹100,000; 1 Cr = ₹10,000,000. Raw *_inr tool fields remain exact INR for computation.",
        },
        "forecasting_and_prediction": {
            "forecast_sales": "Seasonal-naive monthly forecast of sales, gross profit, or invoice lines; uses the prior-year month and a recent chronological holdout.",
            "compare_forecast_models": "Compares seasonal naive, last value, trailing averages, and trend-seasonal regression on shared rolling-origin months; selects by WAPE and returns same-period alternatives.",
            "forecast_accuracy_monitor": "Calibrates an empirical absolute-error band on earlier one-step forecasts, then reports recent band coverage and MAE drift against the later monitor window.",
            "forecast_numeric_series": "Trend regression for any catalog-backed DATE/TIMESTAMP plus numeric field, aggregated by daily, weekly, monthly, quarterly, or yearly periods; supports optional top-group forecasts and a chronological holdout.",
            "predict_numeric_regression": "Regularized linear regression for a numeric target and one to eight numeric predictors, with a later-date holdout and per-month deterministic sample.",
            "sales_driver_scenario_forecast": "Creates low/base/high seasonal sales and profit scenarios from caller-supplied gross-price, discount, volume, and unit-cost changes; promotion response is not inferred.",
            "reconciled_sales_forecast": "Uses bottom-up seasonal forecasts so child groups plus Other equal their parent total at currency precision.",
            "customer_lapse_risk_model": "Logistic ranking for no purchase in the next 180 days using prior 180-day recency, frequency, sales, and discount features; includes a later temporal holdout.",
            "return_risk_model": "Logistic invoice-line return ranking using sale-time fields and a deterministic monthly sample; most recent 180 days are held out.",
            "monte_carlo_discount_simulation": "Conditional discount scenario distribution driven by an elasticity mean and standard deviation supplied by the caller; elasticity is not estimated by the simulation.",
            "limitations": "All source records are synthetic. Backtests estimate historical error, not guaranteed future accuracy. No inventory-on-hand or stockout data is available. Predictive scores are not causal explanations or clinical outcomes.",
        },
        "calendar": "dim_date.fiscal_year follows the Indian April–March fiscal year; use month/date ordering rather than alphabetical month labels.",
        "caveats": [
            "GST values and the September 2025 cutoff are dataset assumptions; the Trastuzumab nil-rate rule and nutraceutical rate are not independently verified.",
            "Expiry date uses an approximate 30.4375 days per shelf-life month.",
            "Descriptive correlations and comparisons do not establish causation.",
            "This dataset is not evidence for clinical, patient-outcome, or real-world market claims.",
        ],
        "workflow": [
            "Read pharma://schema/tables or call list_database_objects before writing SQL.",
            "Inspect unfamiliar tables with profile_table; use quoted, catalog-verified identifiers.",
            "Use parameterized purpose-built tools where available; execute_sql accepts one read-only query and caps returned rows.",
            "For charts, read pharma://design/color-themes and include the date range, metric definition, units, and a concise evidence-based takeaway.",
            "For forecasting, select a tool matching the target and grain, retain its time-based holdout metrics and uncertainty caveats, and separate historical forecasts from assumption-driven simulations.",
        ],
    }
    return json.dumps(guide, indent=2)


@mcp.resource("pharma://metadata/readme")
def get_dataset_metadata() -> str:
    """Returns the dataset's original business rules, provenance, assumptions, and caveats."""
    rows = _run_query("SELECT item, description FROM README ORDER BY item")
    return json.dumps({row["item"]: row["description"] for row in rows}, indent=2)


@mcp.resource("pharma://table/{table_name}")
def get_table_resource(table_name: str) -> str:
    """Returns the schema, row count, and a small sample for one catalog-verified table or view."""
    con = get_connection(read_only=True)
    try:
        schema_name, actual_name, relation_type = _resolve_relation(con, table_name)
        qualified = _relation_sql(schema_name, actual_name)
        columns = con.execute(f"DESCRIBE {qualified}").fetchall()
        row_count = con.execute(f"SELECT count(*) FROM {qualified}").fetchone()[0]
        sample = con.execute(f"SELECT * FROM {qualified} LIMIT 5").fetchall()
        sample_columns = [column[0] for column in columns]
        return json.dumps({
            "table": f"{schema_name}.{actual_name}",
            "kind": relation_type,
            "row_count": row_count,
            "columns": [
                {"name": c[0], "type": c[1], "nullable": c[2] == "YES"} for c in columns
            ],
            "sample_rows": [
                {name: _sanitize_val(value) for name, value in zip(sample_columns, row)}
                for row in sample
            ],
        }, indent=2)
    finally:
        con.close()


@mcp.resource("pharma://catalog/skus")
def get_sku_catalog() -> str:
    """Returns the complete catalog of all 84 pharmaceutical SKUs across 7 therapeutic areas with dosage forms, MRPs, and cold-chain flags."""
    return json.dumps(_run_query("SELECT * FROM dim_sku ORDER BY therapeutic_area, brand_name"), indent=2)

@mcp.resource("pharma://summary/monthly")
def get_monthly_kpis() -> str:
    """Returns high-level monthly aggregate sales, gross profit, margin percentage, and average discount percentages across 2021-2026."""
    return json.dumps(_run_query("SELECT * FROM monthly_summary ORDER BY month"), indent=2)

@mcp.resource("pharma://design/color-themes")
def get_visualization_design_system() -> str:
    """Returns ready-to-use accessible light, dark, and color-vision-safe chart themes and pharma category tokens."""
    design_system = {
        "guidance": "Use color as a redundant encoding with labels or markers. Keep the same business category on the same color across charts. Check contrast in the consuming client.",
        "themes": {
            "clinical_light": {
                "background": "#FFFFFF", "surface": "#F8FAFC", "text": "#1E293B",
                "muted_text": "#475569", "grid": "#E2E8F0", "font": "Inter, system-ui, sans-serif",
                "series": ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#332288", "#999933"],
            },
            "executive_dark": {
                "background": "#0F172A", "surface": "#1E293B", "text": "#F8FAFC",
                "muted_text": "#CBD5E1", "grid": "#334155", "font": "Inter, system-ui, sans-serif",
                "series": ["#56B4E9", "#00C08B", "#F28E2B", "#E78AC3", "#8DD3C7", "#FFD166", "#A78BFA", "#F87171"],
            },
            "color_vision_safe": {
                "background": "#FFFFFF", "text": "#1E293B", "grid": "#CBD5E1",
                "series": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"],
            },
        },
        "semantic_colors": {
            "revenue": "#0072B2", "profit": "#009E73", "discount_or_negative_variance": "#D55E00",
            "warning": "#E69F00", "neutral": "#475569", "cold_chain": "#56B4E9", "ambient": "#64748B",
        },
        "therapeutic_area_colors": {
            "Oncology": "#7C3AED", "Anti-infectives": "#0D9488", "Cardiology": "#E11D48",
            "Diabetes": "#0284C7", "Dermatology": "#D97706", "Pain & Respiratory": "#059669",
            "Nutraceuticals": "#647A00",
        },
        "channel_colors": {
            "Hospital": "#4338CA", "Wholesale": "#0284C7", "Retail": "#0D9488",
            "Online": "#B45309", "Institutional": "#64748B",
        },
        "chart_rules": {
            "time_series": "Use a line chart with chronological axis ordering; annotate policy or dataset events only when supported by source metadata.",
            "category_comparison": "Use sorted horizontal bars for long labels; include a zero baseline for bar charts.",
            "part_to_whole": "Use a sorted bar or treemap when there are many categories; avoid pie charts beyond five slices.",
            "dual_axis": "Avoid unless necessary; label both axes and units clearly. Prefer indexed values or small multiples.",
            "uncertainty": "Show intervals or the historical range where forecasts or sparse segments are presented.",
        },
        "number_formats": {"currency": "Use ₹L or ₹Cr only; never abbreviate as M/million or B/billion. 1 L = ₹100,000; 1 Cr = ₹10,000,000.", "percentage": "Dataset *_pct metrics are already on a 0-100 scale; append a literal % (12.3 -> 12.3%). Divide by 100 only for a formatter that expects a 0-1 fraction.", "count": "#,##0"},
    }
    return json.dumps(design_system, indent=2)


# ==============================================================================
# PROMPTS (Guided Agent Workflows & Data Viz Generation)
# ==============================================================================

@mcp.prompt()
def data_visualization_dashboard(
    dashboard_topic: str = "Executive Sales & Margin Overview",
    target_format: str = "echarts",
    theme: str = "clinical_light",
    currency_unit: str = "crore",
) -> str:
    """Creates a reusable brief for an accessible, evidence-based dashboard in the requested chart format and theme."""
    return f"""Create an analytical dashboard about: {dashboard_topic}
Output format: {target_format}. Theme: {theme} from `pharma://design/color-themes`. Monetary scale: {currency_unit} (choose `lakh` or `crore`).

First read `pharma://context/guide` and `pharma://design/color-themes`. Use `generate_chart_ready_dataset` or a focused analytics tool for the underlying values. Do not invent or silently transform data.

Return:
1. A short dashboard title, audience, date coverage, and one-sentence purpose.
2. A small set of charts chosen to answer the question, with metric definitions, units, axes, category ordering, tooltips, legend labels, and accessible color tokens.
3. A valid {target_format} specification (state if the format is pseudocode) and a concise factual takeaway supported by the returned values. For forecasts, separate observed history from projections and show returned empirical bands; for simulations, label scenario inputs and outcome ranges.
4. If the client can render interactive controls, expose a chart-type selector backed by `chart_alternatives[].id`. Reuse the exact returned rows when switching views; do not change the aggregation or issue a second query.

Use `currency_unit={currency_unit}` in `generate_chart_ready_dataset`. Display monetary values only in ₹ Lakhs (L) or ₹ Crores (Cr)—never millions/billions. Convert raw INR values with 1 L = 100,000 INR and 1 Cr = 10,000,000 INR; state the chosen scale. Use weighted gross margin (SUM(gross_profit_inr)/SUM(net_sales_inr)) and weighted discount rates. Order months chronologically, use a zero baseline for bars, label both axes if a dual axis is unavoidable, and pair color with labels or markers. Avoid pie charts with more than five categories. This dataset is synthetic business data; make no real-world market, clinical, patient-outcome, or causal claims."""


@mcp.prompt()
def advanced_analytics_workflow(
    business_question: str,
    output_style: str = "decision brief",
) -> str:
    """Guides an agent through careful dataset discovery, validation, analysis, and communication for an open-ended question."""
    return f"""Answer this business question using the connected DuckDB context: {business_question}
Preferred output: {output_style}.

Workflow:
1. Read `pharma://context/guide` and inspect `pharma://schema/tables` or call `list_database_objects`.
2. Confirm the question's grain, date window, denominator, and relevant join keys. Use `profile_table` before relying on unfamiliar fields.
3. Use a focused tool where possible: `compare_sales_periods`, `sales_driver_decomposition`, `customer_rfm_segments`, `customer_cohort_retention`, `basket_affinity_analysis`, `discount_scenario_simulator`, `monte_carlo_discount_simulation`, `sales_driver_scenario_forecast`, `forecast_sales`, `compare_forecast_models`, `forecast_accuracy_monitor`, `forecast_numeric_series`, `predict_numeric_regression`, `reconciled_sales_forecast`, `customer_lapse_risk_model`, `return_risk_model`, `scan_monthly_anomalies`, `product_sales_pareto`, or `data_quality_audit`. Use `execute_sql` for a bounded read-only query when needed.
4. Check row counts, nulls, date coverage, and aggregation math. For ratios, divide summed numerators by summed denominators; do not average row-level ratios.
5. Report the answer, evidence, assumptions, caveats, and a useful next question. Separate facts from interpretation and do not claim causation from descriptive patterns. Present every monetary amount only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions.

The data is synthetic and cannot establish real pharma market, clinical, or patient conclusions."""


@mcp.prompt()
def forecasting_simulation_workflow(
    forecast_question: str,
    target: str = "net sales",
    horizon: str = "6 months",
) -> str:
    """Directs agents to the matching forecasting, predictive-risk, or scenario-simulation tool and requires temporal validation."""
    return f"""Build an evidence-based forecast or scenario for this question: {forecast_question}
Target: {target}. Requested horizon: {horizon}.

1. Read `pharma://context/guide` and `pharma://schema/tables`; confirm a date column, target grain, unit, and complete date coverage. Use `profile_table` for unfamiliar fields.
2. Choose the method that matches the question:
   - Monthly net sales, gross profit, or invoice-line baseline: `forecast_sales`.
   - Choose among monthly methods on a shared holdout: `compare_forecast_models`; monitor the chosen method with `forecast_accuracy_monitor`.
   - Require segment totals to add up: `reconciled_sales_forecast`.
   - Apply explicit future price, discount, volume, and unit-cost assumptions: `sales_driver_scenario_forecast`; do not present those assumptions as a learned demand response.
   - Another numeric measure in any catalog table/view with a date or timestamp: `forecast_numeric_series`; choose aggregation, frequency, and optional segment deliberately.
   - A numeric outcome explained by known numeric inputs: `predict_numeric_regression`; exclude fields unavailable at prediction time and the target itself.
   - No-purchase-in-180-day customer ranking: `customer_lapse_risk_model`.
   - Historical invoice-line return ranking: `return_risk_model`.
   - Discount what-if with uncertain demand response: `monte_carlo_discount_simulation`. State the supplied elasticity assumptions.
   - Same-volume discount arithmetic: `discount_scenario_simulator`.
   - Descriptive SKU price/volume trajectory: `price_volume_elasticity`; its historical relationship is not causal.
3. Preserve the chronological holdout metrics and report the scored periods, MAE/MAPE/RMSE or classification metrics, and empirical error ranges where returned. Do not call a simulation a fitted forecast, or a risk score a confirmed outcome.
4. State assumptions and limitations, including incomplete periods, sparse history, synthetic provenance, and uncertainty. The dataset has no stock-on-hand or stockout history, so it cannot support an inventory availability forecast.
5. Show every human-facing monetary amount only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions. Label count measures separately and do not convert counts into money.

Separate observed history, model estimates, and caller-supplied scenario assumptions in the final {target} forecast."""


@mcp.prompt()
def visualization_prompt(
    chart_question: str,
    audience: str = "business leadership",
    target_format: str = "Vega-Lite",
    theme: str = "color_vision_safe",
) -> str:
    """Builds a format-aware chart generation prompt with accessibility and theme guidance."""
    return f"""Design a {target_format} visualization for {audience} answering: {chart_question}
Use theme `{theme}` from `pharma://design/color-themes` and data returned by the analytics tools.

Before writing the chart, identify the metric, denominator, unit, date grain, and category ordering. Present monetary axes/tooltips in ₹ Lakhs (L) or ₹ Crores (Cr) only, never millions/billions; use the `currency_unit` selected by `generate_chart_ready_dataset`. Prefer a time-series line for trends, sorted bars for category comparisons, and small multiples over a confusing dual axis. Preserve zero baselines for bars. Include descriptive titles, axis labels, readable tooltips, missing-value handling, and accessible non-color cues. Use stable category colors across views.

Return the chart spec in the exact requested format, then give a short takeaway tied to displayed values. If the data includes forecasts, distinguish observed and projected periods and display only the empirical bands returned by the model. If it includes simulations, label the scenario assumptions and ranges. If a chart switcher is supported, wire it to `chart_alternatives[].id` and reuse the same `rows` payload; mark incompatible alternatives as disabled. Do not infer causation, invent values, or present this synthetic dataset as real-world pharma, clinical, or patient evidence. State date coverage and explain any unit conversion."""


@mcp.prompt()
def data_quality_audit_workflow(table_name: str = "fact_sales") -> str:
    """Creates a repeatable profiling and data-quality audit workflow for a selected table or view."""
    return f"""Audit data quality for `{table_name}`.

1. Call the `data_quality_audit` tool for null rates, selected-key uniqueness, date bounds, and known join coverage.
2. Call `get_table_summary` and `profile_table` for schema, row count, cardinality, numeric distribution, and samples.
3. Look for impossible date ranges, negative or zero amounts where unexpected, duplicate business keys, unexpected category values, and outliers. Use `execute_sql` with bounded output.
4. Return a table of check, method, result, severity, and follow-up. Distinguish confirmed defects from review flags; do not assume an outlier is invalid without domain evidence. Present monetary amounts only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions."""

@mcp.prompt()
def margin_leakage_investigation(therapeutic_area: str = "") -> str:
    """Investigative prompt workflow for discovering profit margin erosion, discount abuse, and product returns."""
    area_filter = f"for '{therapeutic_area}'" if therapeutic_area else "across all therapeutic areas"
    return f"""Investigate profit margin leakage and discount trends {area_filter}.

### Recommended Investigative Sequence:
1. Call `analyze_margin_erosion` to identify products and channels where margins compressed between 2021 and 2026.
2. Call `analyze_discount_behavior` with `min_discount_pct=12.0` to locate customer segments receiving steep discounts.
3. Call `analyze_returns_and_spoilage` to see if near-expiry stock or damaged goods amplify profitability losses.
4. Call `generate_chart_ready_dataset` with `chart_type='line'` and `x_dimension='year'` to visually illustrate the margin decay.

### Color & Styling Recommendation:
When displaying charts for this investigation, highlight gross profit in `#0D9488` (Teal) and discount rate in `#E11D48` (Rose/Red) with dotted red threshold lines for discounts > 10%. Present monetary amounts only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions."""

@mcp.prompt()
def customer_churn_risk_analysis(region: str = "") -> str:
    """Investigative prompt workflow for discovering dormant accounts and prioritizing evidence-based follow-up."""
    reg_filter = f"in {region}" if region else "nationwide"
    return f"""Analyze customer dormancy and account retention risk {reg_filter}.

### Recommended Investigative Sequence:
1. Call `identify_dormant_customers` with `days_since_last_order=120` to discover valuable accounts that stopped ordering.
2. Call `get_customer_profile` on top dormant accounts by lifetime net sales.
3. Use `customer_cohort_retention` and `regional_growth_matrix` to compare repeat-order patterns and regional context.
4. Provide a ranked follow-up table with customer type, payment terms, dormancy, and lifetime value. Open receivables and credit balances are not available in these tools; do not infer them from payment terms. Present monetary amounts only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions."""

@mcp.prompt()
def sales_rep_performance_review(territory_code: str = "") -> str:
    """Workflow to benchmark sales rep quotas, margin delivery, and discount discipline."""
    terr = f"for territory {territory_code}" if territory_code else "across all territories"
    return f"""Perform an end-to-end performance audit of sales representatives {terr}.

### Recommended Steps:
1. Call `benchmark_sales_reps` to rank reps by net sales volume, average margin achieved, and average discount given.
2. Identify outliers who produce high sales volume but heavily discount, destroying gross margin.
3. Call `get_sales_by_dimension` grouped by 'channel' for underperforming territories to spot channel mix weaknesses.
4. Suggest coaching questions and review actions supported by the metrics. Do not invent quotas or recommend territory changes without quota and coverage data. Present monetary amounts only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions."""


# ==============================================================================
# TOOLS (Comprehensive Analytics, Inspections, & Data Viz Formatter)
# ==============================================================================

# Tool 1: execute_sql
@mcp.tool()
async def execute_sql(
    query: str = Field(..., description="One read-only analytical SQL statement beginning with SELECT, WITH, DESCRIBE, SHOW, or EXPLAIN."),
    max_rows: int = Field(500, description="Maximum result rows returned (1-2000). The query may match more rows; `truncated` reports when the cap was reached.", ge=1, le=2000),
    ctx: Context = None
) -> str:
    """Executes an arbitrary read-only SQL query on the DuckDB database and returns structured JSON rows.
    Tables available: fact_sales (10M rows), dim_customer (60k), dim_sku (84), dim_rep (160), dim_date (1822), README (12).
    Views available: sales_full (joined view), monthly_summary. External file access is disabled.
    """
    try:
        cleaned = _validate_readonly_query(query)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})

    if ctx:
        await ctx.info(f"Executing SQL query: {cleaned[:80]}...")

    try:
        results = _run_query(cleaned, max_rows=max_rows + 1)
        truncated = len(results) > max_rows
        data = results[:max_rows]
        return json.dumps({
            "returned_rows": len(data),
            "truncated": truncated,
            "data": data,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# Tool 2: get_table_summary
@mcp.tool()
async def get_table_summary(
    table_name: str = Field(..., description="Target table or view name (e.g. 'fact_sales', 'dim_customer', 'dim_sku', 'monthly_summary', 'sales_full').")
) -> str:
    """Inspects table structure, row count, column definitions, and returns 5 sample records."""
    con = get_connection(read_only=True)
    try:
        schema_name, actual_name, relation_type = _resolve_relation(con, table_name)
        qualified = _relation_sql(schema_name, actual_name)
        count = con.execute(f"SELECT count(*) FROM {qualified}").fetchone()[0]
        schema = con.execute(f"DESCRIBE {qualified}").fetchall()
        sample_rows = con.execute(f"SELECT * FROM {qualified} LIMIT 5").fetchall()
        columns = [item[0] for item in schema]
        return json.dumps({
            "table": f"{schema_name}.{actual_name}",
            "kind": relation_type,
            "total_rows": count,
            "columns": [{"name": s[0], "type": s[1], "nullable": s[2] == "YES"} for s in schema],
            "sample_rows": [
                {name: _sanitize_val(value) for name, value in zip(columns, row)}
                for row in sample_rows
            ],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        con.close()


@mcp.tool()
async def list_database_objects(
    include_columns: bool = Field(True, description="Include column names and data types for each table and view."),
    ctx: Context = None,
) -> str:
    """Discovers every table and view in the attached DuckDB database without scanning fact rows."""
    if ctx:
        await ctx.info("Reading DuckDB catalog metadata")
    con = get_connection(read_only=True)
    try:
        objects = con.execute(
            """SELECT table_schema, table_name, table_type
               FROM information_schema.tables
               WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
               ORDER BY table_schema, table_name"""
        ).fetchall()
        result = []
        for schema_name, table_name, relation_type in objects:
            item = {"schema": schema_name, "name": table_name, "kind": relation_type}
            if include_columns:
                columns = con.execute(
                    """SELECT column_name, data_type, is_nullable
                       FROM information_schema.columns
                       WHERE table_schema = ? AND table_name = ?
                       ORDER BY ordinal_position""",
                    [schema_name, table_name],
                ).fetchall()
                item["columns"] = [
                    {"name": name, "type": data_type, "nullable": nullable == "YES"}
                    for name, data_type, nullable in columns
                ]
            result.append(item)
        return json.dumps({"objects": result}, indent=2)
    finally:
        con.close()


@mcp.tool()
async def profile_table(
    table_name: str = Field(..., description="Catalog table or view, optionally schema-qualified (for example 'main.fact_sales')."),
    max_columns: int = Field(40, description="Maximum leading columns to profile (1-80).", ge=1, le=80),
    include_sample: bool = Field(True, description="Include up to three sample rows."),
    ctx: Context = None,
) -> str:
    """Profiles a table with row count, null rate, approximate cardinality, ranges, numeric distribution, and optional samples."""
    con = get_connection(read_only=True)
    try:
        schema_name, actual_name, relation_type = _resolve_relation(con, table_name)
        qualified = _relation_sql(schema_name, actual_name)
        columns = con.execute(f"DESCRIBE {qualified}").fetchall()
        selected = columns[:max_columns]
        expressions = ["count(*) AS row_count"]
        aliases = []
        numeric_types = ("tinyint", "smallint", "integer", "bigint", "hugeint", "utinyint", "usmallint", "uinteger", "ubigint", "uhugeint", "float", "double", "real", "decimal", "numeric")
        for index, column in enumerate(selected):
            name, data_type = column[0], column[1]
            quoted = _quote_identifier(name)
            prefix = f"c{index}"
            expressions.extend([
                f"count({quoted}) AS {prefix}_nonnull",
                f"approx_count_distinct(CAST({quoted} AS VARCHAR)) AS {prefix}_distinct",
                f"min(CAST({quoted} AS VARCHAR)) AS {prefix}_min",
                f"max(CAST({quoted} AS VARCHAR)) AS {prefix}_max",
            ])
            stats = {"name": name, "type": data_type, "aliases": prefix}
            if data_type.lower().startswith(numeric_types):
                expressions.extend([
                    f"avg(try_cast({quoted} AS DOUBLE)) AS {prefix}_avg",
                    f"stddev_samp(try_cast({quoted} AS DOUBLE)) AS {prefix}_stddev",
                    f"quantile_cont(try_cast({quoted} AS DOUBLE), 0.25) AS {prefix}_q25",
                    f"quantile_cont(try_cast({quoted} AS DOUBLE), 0.5) AS {prefix}_median",
                    f"quantile_cont(try_cast({quoted} AS DOUBLE), 0.75) AS {prefix}_q75",
                ])
                stats["numeric"] = True
            aliases.append(stats)

        if ctx:
            await ctx.info(f"Profiling {schema_name}.{actual_name} ({len(selected)} of {len(columns)} columns)")
        aggregate = con.execute(f"SELECT {', '.join(expressions)} FROM {qualified}").fetchone()
        row_count = aggregate[0]
        columns_profile = []
        value_index = 1
        for item in aliases:
            profile = {
                "name": item["name"],
                "type": item["type"],
                "null_count": row_count - aggregate[value_index],
                "null_pct": round(100 * (row_count - aggregate[value_index]) / row_count, 2) if row_count else None,
                "approx_distinct": aggregate[value_index + 1],
                "min": _sanitize_val(aggregate[value_index + 2]),
                "max": _sanitize_val(aggregate[value_index + 3]),
            }
            value_index += 4
            if item.get("numeric"):
                for stat_name in ("mean", "stddev", "q25", "median", "q75"):
                    profile[stat_name] = _sanitize_val(aggregate[value_index])
                    value_index += 1
            columns_profile.append(profile)

        samples = []
        if include_sample:
            sample_rows = con.execute(f"SELECT * FROM {qualified} LIMIT 3").fetchall()
            samples = [
                {column[0]: _sanitize_val(value) for column, value in zip(selected, row)}
                for row in sample_rows
            ]
        return json.dumps({
            "table": f"{schema_name}.{actual_name}",
            "kind": relation_type,
            "row_count": row_count,
            "columns_profiled": len(selected),
            "columns_total": len(columns),
            "column_profiles": columns_profile,
            "sample_rows": samples,
            "truncated_columns": len(columns) > len(selected),
            "notes": ["Approximate distinct counts are provided for speed.", "String min/max values use lexical ordering."],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        con.close()


@mcp.tool()
async def compare_sales_periods(
    baseline_start: str = Field(..., description="Baseline period start date (YYYY-MM-DD)."),
    baseline_end: str = Field(..., description="Baseline period end date (YYYY-MM-DD)."),
    comparison_start: str = Field(..., description="Comparison period start date (YYYY-MM-DD)."),
    comparison_end: str = Field(..., description="Comparison period end date (YYYY-MM-DD)."),
    group_by: str = Field("overall", description="Break results out by: overall, therapeutic_area, brand_name, channel, region, or fiscal_year."),
    limit: int = Field(50, description="Maximum categories to return (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Compares two non-overlapping sales periods across revenue, profit, margin, discounts, packs, and returns."""
    dates = []
    try:
        dates = [date.fromisoformat(value) for value in (baseline_start, baseline_end, comparison_start, comparison_end)]
    except ValueError:
        return json.dumps({"error": "All period dates must be valid ISO dates in YYYY-MM-DD format."})
    b_start, b_end, c_start, c_end = dates
    if b_start > b_end or c_start > c_end:
        return json.dumps({"error": "Each period start date must be on or before its end date."})
    if not (b_end < c_start or c_end < b_start):
        return json.dumps({"error": "Baseline and comparison periods must not overlap."})
    dimensions = {
        "overall": "'All sales'",
        "therapeutic_area": "therapeutic_area",
        "brand_name": "brand_name",
        "channel": "channel",
        "region": "region",
        "fiscal_year": "fiscal_year",
    }
    dimension = dimensions.get(group_by.lower())
    if not dimension:
        return json.dumps({"error": f"Invalid group_by. Choose one of: {', '.join(dimensions)}."})
    if ctx:
        await ctx.info("Comparing the requested sales periods")
    sql = f"""
    WITH scoped AS (
      SELECT {dimension} AS category, invoice_date, net_sales_inr, gross_profit_inr,
             gross_sales_inr, discount_inr, quantity_packs, returned, credit_note_inr
      FROM sales_full
      WHERE (invoice_date BETWEEN ? AND ?) OR (invoice_date BETWEEN ? AND ?)
    )
    SELECT category,
      count(*) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS baseline_invoice_lines,
      count(*) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS comparison_invoice_lines,
      sum(net_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS baseline_net_sales_inr,
      sum(net_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS comparison_net_sales_inr,
      sum(gross_profit_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS baseline_gross_profit_inr,
      sum(gross_profit_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS comparison_gross_profit_inr,
      sum(quantity_packs) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS baseline_packs,
      sum(quantity_packs) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS comparison_packs,
      100.0 * sum(gross_profit_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?)
        / nullif(sum(net_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS baseline_margin_pct,
      100.0 * sum(gross_profit_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?)
        / nullif(sum(net_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS comparison_margin_pct,
      100.0 * sum(discount_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?)
        / nullif(sum(gross_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS baseline_discount_pct,
      100.0 * sum(discount_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?)
        / nullif(sum(gross_sales_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS comparison_discount_pct,
      100.0 * count(*) FILTER (WHERE returned AND invoice_date BETWEEN ? AND ?)
        / nullif(count(*) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS baseline_return_rate_pct,
      100.0 * count(*) FILTER (WHERE returned AND invoice_date BETWEEN ? AND ?)
        / nullif(count(*) FILTER (WHERE invoice_date BETWEEN ? AND ?), 0) AS comparison_return_rate_pct,
      sum(credit_note_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS baseline_credit_notes_inr,
      sum(credit_note_inr) FILTER (WHERE invoice_date BETWEEN ? AND ?) AS comparison_credit_notes_inr
    FROM scoped GROUP BY category
    ORDER BY comparison_net_sales_inr DESC NULLS LAST
    LIMIT {limit}
    """
    pair = [baseline_start, baseline_end]
    comp = [comparison_start, comparison_end]
    params = pair + comp
    for _ in range(4):
        params.extend(pair + comp)
    params.extend(pair + pair + comp + comp)  # weighted gross margin numerators and denominators
    params.extend(pair + pair + comp + comp)  # weighted discount numerators and denominators
    params.extend(pair + pair + comp + comp)  # return rates
    params.extend(pair + comp)  # credit notes
    try:
        rows = _run_query(sql, params)
        for row in rows:
            for metric in ("net_sales_inr", "gross_profit_inr", "packs"):
                before, after = row.get(f"baseline_{metric}"), row.get(f"comparison_{metric}")
                row[f"{metric}_change_pct"] = round(100 * (after - before) / abs(before), 2) if before not in (None, 0) and after is not None else None
        return json.dumps({"baseline": [baseline_start, baseline_end], "comparison": [comparison_start, comparison_end], "group_by": group_by, "data": rows}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def scan_monthly_anomalies(
    metric: str = Field("net_sales_inr", description="Metric in monthly_summary: net_sales_inr, gross_profit_inr, gross_margin_pct, avg_discount_pct, or invoices."),
    lookback_months: int = Field(12, description="Number of preceding months used as the rolling baseline (3-24).", ge=3, le=24),
    z_threshold: float = Field(2.5, description="Absolute rolling z-score threshold for flagging (1.0-5.0).", ge=1.0, le=5.0),
    ctx: Context = None,
) -> str:
    """Flags unusual monthly sales values against the preceding rolling mean and sample standard deviation."""
    columns = {"net_sales_inr", "gross_profit_inr", "gross_margin_pct", "avg_discount_pct", "invoices"}
    metric = metric.lower()
    if metric not in columns:
        return json.dumps({"error": f"Invalid metric. Choose one of: {', '.join(sorted(columns))}."})
    if ctx:
        await ctx.info(f"Scanning monthly {metric} values against the prior {lookback_months} months")
    sql = f"""
    WITH baseline AS (
      SELECT month, {metric} AS observed,
        avg({metric}) OVER w AS baseline_mean,
        stddev_samp({metric}) OVER w AS baseline_stddev,
        count({metric}) OVER w AS baseline_months
      FROM monthly_summary
      WINDOW w AS (ORDER BY month ROWS BETWEEN {lookback_months} PRECEDING AND 1 PRECEDING)
    )
    SELECT month, observed, round(baseline_mean, 2) AS baseline_mean,
      round(baseline_stddev, 2) AS baseline_stddev, baseline_months,
      round((observed - baseline_mean) / nullif(baseline_stddev, 0), 2) AS rolling_z_score,
      abs((observed - baseline_mean) / nullif(baseline_stddev, 0)) >= ? AS is_anomaly
    FROM baseline
    WHERE baseline_months >= 3
    ORDER BY month
    """
    try:
        rows = _run_query(sql, [z_threshold])
        return json.dumps({
            "metric": metric,
            "baseline": f"previous {lookback_months} months; at least 3 observations required",
            "threshold": z_threshold,
            "anomalies": [row for row in rows if row["is_anomaly"]],
            "monthly_series": rows,
            "interpretation": "A rolling z-score is a screening signal, not a causal explanation or a forecast.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def customer_cohort_retention(
    cohort_start: str = Field("", description="Optional first cohort month (YYYY-MM-DD); cohorts are grouped by first-order calendar month."),
    cohort_end: str = Field("", description="Optional last cohort month (YYYY-MM-DD)."),
    months_to_track: int = Field(12, description="Maximum months since first order to return (1-36).", ge=1, le=36),
    ctx: Context = None,
) -> str:
    """Builds a monthly customer retention matrix from each customer's first invoice month."""
    cohort_params: list[str] = []
    cohort_filter = ""
    parsed_start = parsed_end = None
    try:
        if cohort_start:
            parsed_start = date.fromisoformat(cohort_start)
            cohort_filter += " AND cohort_month >= date_trunc('month', CAST(? AS DATE))"
            cohort_params.append(cohort_start)
        if cohort_end:
            parsed_end = date.fromisoformat(cohort_end)
            cohort_filter += " AND cohort_month <= date_trunc('month', CAST(? AS DATE))"
            cohort_params.append(cohort_end)
    except Exception:
        return json.dumps({"error": "Cohort range must use YYYY-MM-DD dates."})
    if parsed_start and parsed_end and parsed_start > parsed_end:
        return json.dumps({"error": "cohort_start must be on or before cohort_end."})
    if ctx:
        await ctx.info("Calculating first-order cohorts and monthly repeat activity")
    params: list[str | int] = cohort_params + [months_to_track]
    sql = f"""
    WITH first_order AS (
      SELECT customer_id, date_trunc('month', min(invoice_date))::DATE AS cohort_month
      FROM fact_sales GROUP BY customer_id
    ),
    cohort_sizes AS (
      SELECT cohort_month, count(*) AS cohort_size FROM first_order
      WHERE true {cohort_filter} GROUP BY cohort_month
    ),
    activity AS (
      SELECT DISTINCT customer_id, date_trunc('month', invoice_date)::DATE AS activity_month
      FROM fact_sales
    ),
    retained AS (
      SELECT f.cohort_month,
        date_diff('month', f.cohort_month, a.activity_month) AS month_number,
        count(DISTINCT a.customer_id) AS active_customers
      FROM first_order f JOIN activity a USING(customer_id)
      JOIN cohort_sizes s USING(cohort_month)
      WHERE a.activity_month >= f.cohort_month
        AND date_diff('month', f.cohort_month, a.activity_month) < ?
      GROUP BY f.cohort_month, month_number
    )
    SELECT r.cohort_month, r.month_number, s.cohort_size, r.active_customers,
      round(100.0 * r.active_customers / nullif(s.cohort_size, 0), 2) AS retention_pct
    FROM retained r JOIN cohort_sizes s USING(cohort_month)
    ORDER BY r.cohort_month, r.month_number
    """
    try:
        rows = _run_query(sql, params)
        return json.dumps({
            "definition": "Customers grouped by first invoice calendar month; retained means at least one invoice in the elapsed activity month.",
            "months_to_track": months_to_track,
            "data": rows,
            "note": "The final cohorts have fewer observable months because the dataset ends on 2026-10-27.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def product_sales_pareto(
    top_n: int = Field(50, description="Maximum ranked SKUs to return (1-100).", ge=1, le=100),
    fiscal_year: str = Field("", description="Optional fiscal year filter, for example FY25-26."),
    therapeutic_area: str = Field("", description="Optional therapeutic area filter."),
    ctx: Context = None,
) -> str:
    """Ranks SKUs by net sales and returns cumulative contribution plus A/B/C Pareto bands."""
    filters = []
    params = []
    if fiscal_year:
        filters.append("d.fiscal_year = ?")
        params.append(fiscal_year)
    if therapeutic_area:
        filters.append("s.therapeutic_area = ?")
        params.append(therapeutic_area)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    if ctx:
        await ctx.info("Ranking products and computing cumulative revenue contribution")
    sql = f"""
    WITH product_sales AS (
      SELECT s.sku_id, s.brand_name, s.generic_name, s.therapeutic_area,
        sum(f.net_sales_inr) AS net_sales_inr
      FROM fact_sales f JOIN dim_sku s USING(sku_id)
      JOIN dim_date d ON d.date = f.invoice_date
      {where}
      GROUP BY s.sku_id, s.brand_name, s.generic_name, s.therapeutic_area
    ), ranked AS (
      SELECT *, row_number() OVER (ORDER BY net_sales_inr DESC) AS sales_rank,
        sum(net_sales_inr) OVER (ORDER BY net_sales_inr DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / nullif(sum(net_sales_inr) OVER (), 0) AS cumulative_sales_share
      FROM product_sales
    )
    SELECT sku_id, brand_name, generic_name, therapeutic_area, sales_rank,
      round(net_sales_inr, 2) AS net_sales_inr,
      round(100 * net_sales_inr / nullif(sum(net_sales_inr) OVER (), 0), 2) AS sales_share_pct,
      round(100 * cumulative_sales_share, 2) AS cumulative_sales_share_pct,
      CASE WHEN cumulative_sales_share <= 0.80 THEN 'A'
           WHEN cumulative_sales_share <= 0.95 THEN 'B' ELSE 'C' END AS pareto_band
    FROM ranked ORDER BY sales_rank LIMIT {top_n}
    """
    try:
        return json.dumps({"ranking_basis": "Net sales, INR", "abc_cutoffs": {"A": "cumulative through 80%", "B": "80%-95%", "C": "above 95%"}, "data": _run_query(sql, params)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def customer_rfm_segments(
    reference_date: str = Field("", description="Optional valid reference date (YYYY-MM-DD); defaults to the last invoice date in the dataset."),
    region: str = Field("", description="Optional region filter."),
    customer_limit: int = Field(100, description="Maximum ranked customers to include (1-500).", ge=1, le=500),
    ctx: Context = None,
) -> str:
    """Scores customers on recency, distinct invoice frequency, and net sales, then summarizes practical RFM segments."""
    parsed_reference = None
    if reference_date:
        try:
            parsed_reference = date.fromisoformat(reference_date)
        except ValueError:
            return json.dumps({"error": "reference_date must be a valid YYYY-MM-DD date."})
        latest_date = _run_query("SELECT max(invoice_date) AS latest_invoice_date FROM fact_sales")
        latest_date = date.fromisoformat(latest_date[0]["latest_invoice_date"]) if latest_date and latest_date[0]["latest_invoice_date"] else None
        if latest_date and parsed_reference < latest_date:
            return json.dumps({"error": "reference_date cannot be earlier than the dataset's latest invoice date."})
    if ctx:
        await ctx.info("Scoring customer recency, frequency, and monetary value")
    region_filter = "AND c.region = ?" if region else ""
    params = [reference_date or None] + ([region] if region else [])
    sql = f"""
    WITH reference AS (
      SELECT coalesce(CAST(? AS DATE), max(invoice_date)) AS reference_date
      FROM fact_sales
    ), customer_metrics AS (
      SELECT c.customer_id, c.customer_type, c.region, c.channel, c.payment_terms_days,
        max(f.invoice_date) AS last_order_date,
        r.reference_date AS reference_date,
        date_diff('day', max(f.invoice_date), r.reference_date) AS recency_days,
        count(DISTINCT f.invoice_no) AS invoice_frequency,
        round(sum(f.net_sales_inr), 2) AS monetary_value_inr
      FROM fact_sales f
      JOIN dim_customer c USING(customer_id)
      CROSS JOIN reference r
      WHERE coalesce(f.returned, false) = false {region_filter}
      GROUP BY c.customer_id, c.customer_type, c.region, c.channel, c.payment_terms_days, r.reference_date
    ), scored AS (
      SELECT *,
        ntile(5) OVER (ORDER BY recency_days DESC, customer_id) AS r_score,
        ntile(5) OVER (ORDER BY invoice_frequency ASC, customer_id) AS f_score,
        ntile(5) OVER (ORDER BY monetary_value_inr ASC, customer_id) AS m_score
      FROM customer_metrics
    ), segmented AS (
      SELECT *, CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN f_score >= 4 AND m_score >= 4 THEN 'Loyal high value'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'New or promising'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'At risk'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'Lapsed'
        ELSE 'Develop'
      END AS segment
      FROM scored
    )
    SELECT * FROM segmented
    ORDER BY r_score + f_score + m_score DESC, monetary_value_inr DESC
    """
    try:
        customers = _run_query(sql, params)
        segment_summary: dict[str, dict[str, Any]] = {}
        for customer in customers:
            name = customer["segment"]
            summary = segment_summary.setdefault(name, {"segment": name, "customers": 0, "sales_inr": 0.0, "recency_days_total": 0, "frequency_total": 0})
            summary["customers"] += 1
            summary["sales_inr"] += customer["monetary_value_inr"] or 0
            summary["recency_days_total"] += customer["recency_days"] or 0
            summary["frequency_total"] += customer["invoice_frequency"] or 0
        for summary in segment_summary.values():
            count = summary["customers"]
            summary["sales_inr"] = round(summary["sales_inr"], 2)
            summary["avg_recency_days"] = round(summary.pop("recency_days_total") / count, 1)
            summary["avg_invoice_frequency"] = round(summary.pop("frequency_total") / count, 1)
        return json.dumps({
            "reference_date": reference_date or (customers[0]["reference_date"] if customers else None),
            "scoring": {"scale": "1-5 quintiles within the selected population", "recency": "5 is most recent", "frequency": "5 is most frequent", "monetary": "5 has highest net sales"},
            "segments": sorted(segment_summary.values(), key=lambda item: item["sales_inr"], reverse=True),
            "priority_customers": customers[:customer_limit],
            "customer_count": len(customers),
            "caveat": "RFM is a descriptive prioritization score; it does not predict churn or explain why a customer changed behavior. Returned invoice lines are excluded from sales and frequency.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def forecast_sales(
    metric: str = Field("net_sales_inr", description="Monthly metric: net_sales_inr, gross_profit_inr, or invoice_lines (the monthly_summary count of fact rows)."),
    horizon_months: int = Field(6, description="Number of future months to project (1-12).", ge=1, le=12),
    history_months: int = Field(60, description="Number of recent complete months to use (24-60).", ge=24, le=60),
    ctx: Context = None,
) -> str:
    """Produces a seasonal-naive monthly forecast with rolling 12-month holdout scores and empirical error bands."""
    metric = metric.lower()
    if metric == "invoices":
        metric = "invoice_lines"
    metric_columns = {"net_sales_inr": "net_sales_inr", "gross_profit_inr": "gross_profit_inr", "invoice_lines": "invoices"}
    if metric not in metric_columns:
        return json.dumps({"error": f"metric must be one of: {', '.join(metric_columns)}."})
    if ctx:
        await ctx.info(f"Preparing {horizon_months}-month seasonal-naive forecast for {metric}")
    try:
        cutoff = _run_query("SELECT date_trunc('month', max(invoice_date))::DATE AS partial_month FROM fact_sales")
        partial_month = cutoff[0]["partial_month"] if cutoff else None
        rows = _run_query(
            f"SELECT month, {metric_columns[metric]} AS value FROM monthly_summary WHERE (CAST(? AS DATE) IS NULL OR month < CAST(? AS DATE)) ORDER BY month DESC LIMIT ?",
            [partial_month, partial_month, history_months],
        )
        history = list(reversed(rows))
        if len(history) < 24:
            return json.dumps({"error": "At least 24 complete monthly observations are required for a 12-month seasonal backtest."})
        values_by_month = {row["month"][:7]: float(row["value"]) for row in history if row["value"] is not None}
        absolute_errors = []
        percentage_errors = []
        signed_errors = []
        backtest_rows = []
        backtest_start = max(12, len(history) - 12)
        for index in range(backtest_start, len(history)):
            target = date.fromisoformat(history[index]["month"])
            prior_year = date(target.year - 1, target.month, 1)
            prediction = values_by_month.get(prior_year.strftime("%Y-%m"))
            actual = float(history[index]["value"])
            if prediction is None:
                continue
            error = actual - prediction
            absolute_errors.append(abs(error))
            signed_errors.append(error)
            if actual != 0:
                percentage_errors.append(abs(error / actual) * 100)
            backtest_rows.append({"month": target.isoformat(), "actual": actual, "forecast": prediction, "absolute_error": abs(error)})
        if not backtest_rows:
            return json.dumps({"error": "Not enough matching prior-year months to backtest the seasonal model."})
        sorted_errors = sorted(absolute_errors)
        p90_error = sorted_errors[max(0, math.ceil(0.9 * len(sorted_errors)) - 1)]
        latest_month = date.fromisoformat(history[-1]["month"])
        forecasts = []
        for step in range(1, horizon_months + 1):
            month_index = latest_month.year * 12 + latest_month.month - 1 + step
            year, month_number = divmod(month_index, 12)
            month_number += 1
            target = date(year, month_number, 1)
            source = date(year - 1, month_number, 1)
            estimate = values_by_month.get(source.strftime("%Y-%m"))
            if estimate is None:
                continue
            forecasts.append({
                "month": target.isoformat(),
                "seasonal_reference_month": source.isoformat(),
                "forecast": round(estimate, 2),
                "lower_empirical_band": round(max(0.0, estimate - p90_error), 2),
                "upper_empirical_band": round(estimate + p90_error, 2),
            })
        return json.dumps({
            "metric": metric,
            "history_months_used": len(history),
            "last_complete_month": history[-1]["month"],
            "method": "Seasonal naive: each future month equals the observed value from the same calendar month one year earlier.",
            "forecast": forecasts,
            "backtest": {
                "method": "Rolling holdout over up to the latest 12 complete months, each forecast from its prior-year month.",
                "months_scored": len(backtest_rows),
                "mae": round(sum(absolute_errors) / len(absolute_errors), 2),
                "mape_pct": round(sum(percentage_errors) / len(percentage_errors), 2) if percentage_errors else None,
                "bias_actual_minus_forecast": round(sum(signed_errors) / len(signed_errors), 2),
                "p90_absolute_error": round(p90_error, 2),
            },
            "uncertainty_note": "Bands use the historical 90th percentile absolute backtest error; they are empirical ranges, not formal confidence intervals.",
            "data_note": "The latest invoice month is excluded because it may be incomplete.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def compare_forecast_models(
    metric: str = Field("net_sales_inr", description="Monthly metric: net_sales_inr, gross_profit_inr, or invoice_lines."),
    horizon_months: int = Field(6, description="Number of future months (1-12).", ge=1, le=12),
    history_months: int = Field(60, description="Number of recent complete months to use (24-60).", ge=24, le=60),
    ctx: Context = None,
) -> str:
    """Compares five monthly forecast alternatives on the same rolling holdout and returns switchable forecasts for the same future months."""
    metric = metric.lower().strip()
    if metric == "invoices":
        metric = "invoice_lines"
    metric_columns = {"net_sales_inr": "net_sales_inr", "gross_profit_inr": "gross_profit_inr", "invoice_lines": "invoices"}
    if metric not in metric_columns:
        return json.dumps({"error": f"metric must be one of: {', '.join(metric_columns)}."})
    if ctx:
        await ctx.info(f"Backtesting forecast alternatives for {metric}")
    try:
        latest = _run_query("SELECT date_trunc('month', max(invoice_date))::DATE AS partial_month FROM fact_sales")
        partial_month = latest[0]["partial_month"] if latest else None
        rows = _run_query(
            f"SELECT month, {metric_columns[metric]} AS value FROM monthly_summary WHERE (CAST(? AS DATE) IS NULL OR month < CAST(? AS DATE)) ORDER BY month DESC LIMIT ?",
            [partial_month, partial_month, history_months],
        )
        observations = [
            (date.fromisoformat(str(row["month"])[:10]), float(row["value"]))
            for row in reversed(rows) if row["value"] is not None
        ]
        if len(observations) < 24:
            return json.dumps({"error": "At least 24 complete monthly observations are needed to compare seasonal models."})
        holdout_start = max(12, len(observations) - 12)
        model_results = []
        for model_id, label in MONTHLY_FORECAST_MODELS.items():
            errors = []
            actuals = []
            scored_periods = []
            for index in range(holdout_start, len(observations)):
                period, actual = observations[index]
                predicted = _monthly_model_prediction(model_id, observations[:index], period)
                if predicted is None:
                    continue
                errors.append(actual - predicted)
                actuals.append(actual)
                scored_periods.append({"month": period.isoformat(), "actual": round(actual, 2), "predicted": round(predicted, 2)})
            if not errors:
                continue
            absolute_errors = [abs(error) for error in errors]
            p90_error = _percentile(absolute_errors, 0.9)
            wape = 100 * sum(absolute_errors) / sum(abs(actual) for actual in actuals) if sum(abs(actual) for actual in actuals) else None
            mape_values = [abs(error / actual) * 100 for error, actual in zip(errors, actuals) if actual != 0]
            future_rows = []
            for step in range(1, horizon_months + 1):
                target = _advance_month(observations[-1][0], step)
                predicted = _monthly_model_prediction(model_id, observations, target)
                if predicted is None:
                    continue
                future_rows.append({
                    "month": target.isoformat(),
                    "forecast": round(predicted, 2),
                    "lower_empirical_band": round(predicted - p90_error, 2),
                    "upper_empirical_band": round(predicted + p90_error, 2),
                })
            model_results.append({
                "id": model_id,
                "label": label,
                "metrics": {
                    "holdout_months": len(errors),
                    "mae": round(sum(absolute_errors) / len(errors), 2),
                    "wape_pct": round(wape, 3) if wape is not None else None,
                    "mape_pct": round(sum(mape_values) / len(mape_values), 3) if mape_values else None,
                    "bias_actual_minus_forecast": round(sum(errors) / len(errors), 2),
                    "p90_absolute_error": round(p90_error, 2),
                },
                "holdout_rows": scored_periods,
                "forecast_rows": future_rows,
            })
        if not model_results:
            return json.dumps({"error": "No forecast model could produce a scored holdout."})
        best_model = min(
            model_results,
            key=lambda result: result["metrics"]["wape_pct"] if result["metrics"]["wape_pct"] is not None else float("inf"),
        )
        shared_rows = []
        for step in range(1, horizon_months + 1):
            target = _advance_month(observations[-1][0], step)
            shared_rows.append({
                "month": target.isoformat(),
                "forecasts": {
                    result["id"]: next((row["forecast"] for row in result["forecast_rows"] if row["month"] == target.isoformat()), None)
                    for result in model_results
                },
            })
        return json.dumps({
            "metric": metric,
            "history_months_used": len(observations),
            "last_complete_month": observations[-1][0].isoformat(),
            "backtest_window": {"start": observations[holdout_start][0].isoformat(), "end": observations[-1][0].isoformat(), "method": "One-step rolling-origin; each forecast uses only earlier months."},
            "selection_rule": "Lowest WAPE on the shared rolling holdout; this is a model-selection aid, not a guarantee of future accuracy.",
            "selected_model_id": best_model["id"],
            "forecast_alternatives": model_results,
            "shared_forecast_rows": shared_rows,
            "interaction": {"selector": "forecast_alternatives[].id", "behavior": "Switch models over the same forecast months in shared_forecast_rows; do not change the data aggregation."},
            "limitations": [
                "All alternatives use only the selected metric's history; they do not include price, promotion, inventory, or market covariates.",
                "Empirical bands use each model's holdout absolute errors and are not formal confidence intervals.",
                "Values are exact INR for monetary metrics; present human-facing amounts in ₹ Lakhs (L) or ₹ Crores (Cr).",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def forecast_accuracy_monitor(
    metric: str = Field("net_sales_inr", description="Monthly metric: net_sales_inr, gross_profit_inr, or invoice_lines."),
    model_id: str = Field("auto", description="Model to monitor, or auto to choose the lowest-WAPE model on the earlier calibration window."),
    monitor_months: int = Field(6, description="Number of most recent complete months to monitor (3-12).", ge=3, le=12),
    interval_target_pct: int = Field(90, description="Historical absolute-error coverage target used to set an empirical band width (50-99).", ge=50, le=99),
    history_months: int = Field(60, description="Number of recent complete months to evaluate (36-60).", ge=36, le=60),
    ctx: Context = None,
) -> str:
    """Monitors rolling forecast error, empirical interval coverage, and recent accuracy drift for a selected monthly model."""
    metric = metric.lower().strip()
    if metric == "invoices":
        metric = "invoice_lines"
    metric_columns = {"net_sales_inr": "net_sales_inr", "gross_profit_inr": "gross_profit_inr", "invoice_lines": "invoices"}
    if metric not in metric_columns:
        return json.dumps({"error": f"metric must be one of: {', '.join(metric_columns)}."})
    model_id = model_id.lower().strip()
    if model_id != "auto" and model_id not in MONTHLY_FORECAST_MODELS:
        return json.dumps({"error": f"model_id must be auto or one of: {', '.join(MONTHLY_FORECAST_MODELS)}."})
    if ctx:
        await ctx.info("Scoring recent one-step forecasts against an earlier calibration window")
    try:
        latest = _run_query("SELECT date_trunc('month', max(invoice_date))::DATE AS partial_month FROM fact_sales")
        partial_month = latest[0]["partial_month"] if latest else None
        rows = _run_query(
            f"SELECT month, {metric_columns[metric]} AS value FROM monthly_summary WHERE (CAST(? AS DATE) IS NULL OR month < CAST(? AS DATE)) ORDER BY month DESC LIMIT ?",
            [partial_month, partial_month, history_months],
        )
        history = [
            (date.fromisoformat(str(row["month"])[:10]), float(row["value"]))
            for row in reversed(rows) if row["value"] is not None
        ]
        if len(history) < 36:
            return json.dumps({"error": "At least 36 complete months are required to select a model, calibrate a band, and monitor recent coverage."})
        monitor_start = max(12, len(history) - monitor_months)
        calibration_by_model: dict[str, list[tuple[float, float, date]]] = {name: [] for name in MONTHLY_FORECAST_MODELS}
        for index in range(12, monitor_start):
            period, actual = history[index]
            for candidate in MONTHLY_FORECAST_MODELS:
                predicted = _monthly_model_prediction(candidate, history[:index], period)
                if predicted is not None:
                    calibration_by_model[candidate].append((actual, predicted, period))
        if model_id == "auto":
            def calibration_wape(candidate: str) -> float:
                scored = calibration_by_model[candidate]
                denominator = sum(abs(actual) for actual, _, _ in scored)
                return sum(abs(actual - predicted) for actual, predicted, _ in scored) / denominator if denominator else float("inf")
            selected_model = min(MONTHLY_FORECAST_MODELS, key=calibration_wape)
        else:
            selected_model = model_id
        calibration = calibration_by_model[selected_model]
        if len(calibration) < 6:
            return json.dumps({"error": "The selected model has fewer than six calibration forecasts; provide more complete monthly history."})
        calibration_errors = [actual - predicted for actual, predicted, _ in calibration]
        interval_half_width = _percentile([abs(error) for error in calibration_errors], interval_target_pct / 100.0)
        monitored = []
        for index in range(monitor_start, len(history)):
            period, actual = history[index]
            predicted = _monthly_model_prediction(selected_model, history[:index], period)
            if predicted is None:
                continue
            error = actual - predicted
            monitored.append({
                "month": period.isoformat(),
                "actual": round(actual, 2),
                "forecast": round(predicted, 2),
                "lower_empirical_band": round(predicted - interval_half_width, 2),
                "upper_empirical_band": round(predicted + interval_half_width, 2),
                "absolute_error": round(abs(error), 2),
                "inside_empirical_band": abs(error) <= interval_half_width,
            })
        if len(monitored) < 3:
            return json.dumps({"error": "Fewer than three recent monitor forecasts could be scored."})
        recent_mae = sum(row["absolute_error"] for row in monitored) / len(monitored)
        calibration_mae = sum(abs(error) for error in calibration_errors) / len(calibration_errors)
        drift_ratio = recent_mae / calibration_mae if calibration_mae else (float("inf") if recent_mae else 1.0)
        coverage_pct = 100 * sum(row["inside_empirical_band"] for row in monitored) / len(monitored)
        drift_flag = drift_ratio > 1.5
        coverage_flag = coverage_pct < max(0, interval_target_pct - 10)
        return json.dumps({
            "metric": metric,
            "model_id": selected_model,
            "model_label": MONTHLY_FORECAST_MODELS[selected_model],
            "selection": "Lowest WAPE on the earlier calibration window." if model_id == "auto" else "Caller-selected model.",
            "calibration_periods": len(calibration),
            "calibration_end_month": calibration[-1][2].isoformat(),
            "monitor_periods": len(monitored),
            "monitor_start_month": monitored[0]["month"],
            "monitor_end_month": monitored[-1]["month"],
            "calibration_mae": round(calibration_mae, 2),
            "recent_mae": round(recent_mae, 2),
            "recent_to_calibration_mae_ratio": round(drift_ratio, 3) if math.isfinite(drift_ratio) else None,
            "interval_target_pct": interval_target_pct,
            "calibrated_absolute_error_half_width": round(interval_half_width, 2),
            "recent_interval_coverage_pct": round(coverage_pct, 2),
            "drift_flag": drift_flag,
            "coverage_flag": coverage_flag,
            "status": "attention" if drift_flag or coverage_flag else "within_monitor_thresholds",
            "monitor_rows": monitored,
            "interpretation": [
                "Coverage is measured on a recent rolling holdout against an absolute-error band calibrated on earlier one-step errors; the band is not a formal confidence interval.",
                "Drift is flagged when recent MAE exceeds 1.5 times calibration MAE; coverage is flagged when it falls more than 10 percentage points below the requested target.",
                "A monitor flag indicates changed forecast performance and should trigger data and model review; it does not explain the cause.",
                "The data is synthetic; present monetary values only in ₹ Lakhs (L) or ₹ Crores (Cr).",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def forecast_numeric_series(
    table_name: str = Field("fact_sales", description="Catalog table or view containing the date and numeric measure, for example fact_sales or monthly_summary."),
    date_column: str = Field("invoice_date", description="Date or timestamp column that defines the time axis."),
    value_column: str = Field("net_sales_inr", description="Numeric measure to aggregate and forecast."),
    frequency: str = Field("monthly", description="Aggregation period: daily, weekly, monthly, quarterly, or yearly."),
    aggregation: str = Field("sum", description="Aggregate source rows within each period using sum or average."),
    group_by: str = Field("", description="Optional catalog column to forecast separate groups; at most 20 groups are selected by absolute aggregate size."),
    horizon: int = Field(6, description="Number of periods to forecast (1-24).", ge=1, le=24),
    ctx: Context = None,
) -> str:
    """Forecasts any catalog-backed numeric measure over time with optional segmentation, seasonal trend regression, and a chronological holdout."""
    frequency = frequency.lower().strip()
    aggregation = aggregation.lower().strip()
    truncation = {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year"}
    if frequency not in truncation:
        return json.dumps({"error": "frequency must be daily, weekly, monthly, quarterly, or yearly."})
    if aggregation not in {"sum", "avg"}:
        return json.dumps({"error": "aggregation must be sum or avg."})
    season_cycles = {"daily": 7, "weekly": 52, "monthly": 12, "quarterly": 4}
    cycle = season_cycles.get(frequency)
    if ctx:
        await ctx.info(f"Checking catalog columns and preparing a {frequency} forecast")
    con = get_connection(read_only=True)
    try:
        schema_name, relation_name, _ = _resolve_relation(con, table_name)
        relation = _relation_sql(schema_name, relation_name)
        columns = con.execute(f"DESCRIBE {relation}").fetchall()
        column_types = {str(item[0]).lower(): (str(item[0]), str(item[1]).upper()) for item in columns}
        requested = [date_column, value_column] + ([group_by] if group_by else [])
        missing = [name for name in requested if name.lower() not in column_types]
        if missing:
            return json.dumps({"error": f"Unknown column(s) for {table_name}: {', '.join(missing)}."})
        actual_date, date_type = column_types[date_column.lower()]
        actual_value, value_type = column_types[value_column.lower()]
        if not (date_type.startswith("DATE") or date_type.startswith("TIMESTAMP")):
            return json.dumps({"error": f"{actual_date} must be a DATE or TIMESTAMP column; found {date_type}."})
        if not re.match(r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|REAL|DECIMAL|NUMERIC)", value_type):
            return json.dumps({"error": f"{actual_value} must be numeric; found {value_type}."})
        actual_group = column_types[group_by.lower()][0] if group_by else ""
        if actual_group and actual_group.lower() in {actual_date.lower(), actual_value.lower()}:
            return json.dumps({"error": "group_by must be a different column from the date and measure."})
        date_sql = f'r.{_quote_identifier(actual_date)}'
        value_sql = f'try_cast(r.{_quote_identifier(actual_value)} AS DOUBLE)'
        period_sql = f"date_trunc('{truncation[frequency]}', {date_sql})::DATE"
        latest_period = con.execute(
            f"SELECT date_trunc('{truncation[frequency]}', max({_quote_identifier(actual_date)}))::DATE FROM {relation}"
        ).fetchone()[0]
        if latest_period is None:
            return json.dumps({"error": "No rows with a date were found in the selected table or view."})
        if actual_group:
            group_sql = f"coalesce(cast(r.{_quote_identifier(actual_group)} AS VARCHAR), '(missing)')"
            query = f"""
            WITH popular AS (
              SELECT {group_sql} AS series
              FROM {relation} AS r
              WHERE {date_sql} IS NOT NULL AND {period_sql} < ? AND {value_sql} IS NOT NULL
              GROUP BY series
              ORDER BY abs(sum({value_sql})) DESC NULLS LAST
              LIMIT 20
            )
            SELECT {period_sql} AS period, {group_sql} AS series,
              {aggregation}({value_sql}) AS value, count(*) AS source_rows
            FROM {relation} AS r
            JOIN popular p ON p.series = {group_sql}
            WHERE {date_sql} IS NOT NULL AND {period_sql} < ? AND {value_sql} IS NOT NULL
            GROUP BY period, series
            ORDER BY series, period
            LIMIT 10000
            """
        else:
            query = f"""
            SELECT {period_sql} AS period, '(all)' AS series,
              {aggregation}({value_sql}) AS value, count(*) AS source_rows
            FROM {relation} AS r
            WHERE {date_sql} IS NOT NULL AND {period_sql} < ? AND {value_sql} IS NOT NULL
            GROUP BY period
            ORDER BY period
            LIMIT 500
            """
        rows = con.execute(query, [latest_period, latest_period] if actual_group else [latest_period]).fetchall()
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        con.close()

    def parse_period(value: Any) -> date:
        return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])

    def ordinal(period: date) -> int:
        if frequency == "daily":
            return period.toordinal()
        if frequency == "weekly":
            return period.toordinal() // 7
        if frequency == "monthly":
            return period.year * 12 + period.month - 1
        if frequency == "quarterly":
            return period.year * 4 + (period.month - 1) // 3
        return period.year

    def phase(period: date) -> int:
        if frequency == "daily":
            return period.weekday()
        if frequency == "weekly":
            return period.isocalendar().week - 1
        if frequency == "monthly":
            return period.month - 1
        if frequency == "quarterly":
            return (period.month - 1) // 3
        return 0

    def advance(period: date) -> date:
        if frequency == "daily":
            from datetime import timedelta
            return period + timedelta(days=1)
        if frequency == "weekly":
            from datetime import timedelta
            return period + timedelta(days=7)
        if frequency == "monthly":
            month_index = period.year * 12 + period.month
            year, month = divmod(month_index, 12)
            return date(year, month + 1, 1)
        if frequency == "quarterly":
            month_index = period.year * 12 + period.month - 1 + 3
            year, month = divmod(month_index, 12)
            return date(year, month + 1, 1)
        return date(period.year + 1, 1, 1)

    def features(period: date, origin: int, span: int, use_season: bool) -> list[float]:
        result = [1.0, (ordinal(period) - origin) / max(1, span)]
        if use_season and cycle:
            angle = 2.0 * math.pi * phase(period) / cycle
            result.extend([math.sin(angle), math.cos(angle)])
        return result

    groups: dict[str, list[tuple[date, float, int]]] = {}
    for row in rows:
        period = parse_period(row[0])
        groups.setdefault(str(row[1]), []).append((period, float(row[2]), int(row[3])))
    if not groups:
        return json.dumps({"error": "No non-null date and numeric observations were found."})
    series_output = []
    for series_name, observations in groups.items():
        observations.sort(key=lambda item: item[0])
        if len(observations) < 12:
            continue
        holdout_size = max(3, math.ceil(0.2 * len(observations)))
        train = observations[:-holdout_size]
        holdout = observations[-holdout_size:]
        if len(train) < 6:
            continue
        train_origin = min(ordinal(item[0]) for item in train)
        train_span = max(1, max(ordinal(item[0]) for item in train) - train_origin)
        train_season = bool(cycle and len(train) >= 2 * cycle)
        coefficients = _fit_ridge_regression(
            [features(item[0], train_origin, train_span, train_season) for item in train],
            [item[1] for item in train],
        )
        holdout_predictions = [
            sum(a * b for a, b in zip(features(item[0], train_origin, train_span, train_season), coefficients))
            for item in holdout
        ]
        errors = [actual - predicted for actual, predicted in zip((item[1] for item in holdout), holdout_predictions)]
        abs_errors = sorted(abs(error) for error in errors)
        p90_error = abs_errors[max(0, math.ceil(0.9 * len(abs_errors)) - 1)]
        actuals = [item[1] for item in holdout]
        mae = sum(abs(error) for error in errors) / len(errors)
        mape_values = [abs(error / actual) * 100 for error, actual in zip(errors, actuals) if actual != 0]

        full_origin = min(ordinal(item[0]) for item in observations)
        full_span = max(1, max(ordinal(item[0]) for item in observations) - full_origin)
        full_season = bool(cycle and len(observations) >= 2 * cycle)
        full_coefficients = _fit_ridge_regression(
            [features(item[0], full_origin, full_span, full_season) for item in observations],
            [item[1] for item in observations],
        )
        forecasts = []
        future_period = advance(observations[-1][0])
        for _ in range(horizon):
            predicted = sum(a * b for a, b in zip(features(future_period, full_origin, full_span, full_season), full_coefficients))
            forecasts.append({
                "period": future_period.isoformat(),
                "forecast": round(predicted, 4),
                "lower_empirical_band": round(predicted - p90_error, 4),
                "upper_empirical_band": round(predicted + p90_error, 4),
            })
            future_period = advance(future_period)
        series_output.append({
            "series": series_name,
            "observations": len(observations),
            "last_observed_period": observations[-1][0].isoformat(),
            "latest_period_source_rows": observations[-1][2],
            "model": "Ridge trend regression" + (" with annual/weekly seasonality terms" if full_season else " without seasonal terms"),
            "forecast": forecasts,
            "backtest": {
                "holdout_periods": len(holdout),
                "mae": round(mae, 4),
                "mape_pct": round(sum(mape_values) / len(mape_values), 3) if mape_values else None,
                "p90_absolute_error": round(p90_error, 4),
            },
        })
    if not series_output:
        return json.dumps({"error": "Each forecast series needs at least 12 observations and 6 training observations."})
    return json.dumps({
        "table": table_name,
        "date_column": date_column,
        "value_column": value_column,
        "aggregation": aggregation,
        "frequency": frequency,
        "unit": f"same unit as {value_column}; values are unscaled",
        "excluded_latest_source_period": latest_period.isoformat(),
        "groups_returned": len(series_output),
        "series": series_output,
        "method_notes": [
            "The last 20% of observed periods (at least three) is held out chronologically; no random split is used.",
            "The latest period present in the source table is excluded because it may be incomplete.",
            "The model is a small trend regression with one seasonal harmonic when at least two full cycles are available.",
            "Empirical bands use the 90th percentile absolute holdout error. They are not formal confidence or prediction intervals.",
            "Forecasts are extrapolations of historical patterns, not causal effects, clinical predictions, or inventory availability forecasts.",
            "Grouped forecasts return at most the 20 groups with the largest absolute aggregate measure; groups with fewer than 12 periods are skipped.",
        ],
    }, indent=2)


@mcp.tool()
async def predict_numeric_regression(
    predictor_columns: list[str] = Field(..., description="One to eight numeric predictor columns known at prediction time."),
    table_name: str = Field("fact_sales", description="Catalog table or view containing a numeric outcome, numeric predictors, and a date column."),
    target_column: str = Field("gross_profit_inr", description="Numeric outcome to predict on later-dated holdout rows."),
    date_column: str = Field("invoice_date", description="DATE or TIMESTAMP used to split earlier training rows from later holdout rows."),
    sample_per_month: int = Field(500, description="Maximum deterministic rows sampled per month (100-2000).", ge=100, le=2000),
    top_errors: int = Field(20, description="Number of largest absolute holdout errors to return (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Fits a regularized numeric regression on catalog-validated predictors and scores it on a later chronological holdout."""
    if not 1 <= len(predictor_columns) <= 8:
        return json.dumps({"error": "predictor_columns must contain between 1 and 8 numeric column names."})
    if len({column.lower() for column in predictor_columns}) != len(predictor_columns):
        return json.dumps({"error": "predictor_columns must be unique."})
    if ctx:
        await ctx.info("Checking numeric regression columns and sampling rows by month")
    con = get_connection(read_only=True)
    try:
        schema_name, relation_name, _ = _resolve_relation(con, table_name)
        relation = _relation_sql(schema_name, relation_name)
        columns = con.execute(f"DESCRIBE {relation}").fetchall()
        column_types = {str(item[0]).lower(): (str(item[0]), str(item[1]).upper()) for item in columns}
        names = [target_column, date_column] + predictor_columns
        missing = [name for name in names if name.lower() not in column_types]
        if missing:
            return json.dumps({"error": f"Unknown column(s) for {table_name}: {', '.join(missing)}."})
        actual_target, target_type = column_types[target_column.lower()]
        actual_date, date_type = column_types[date_column.lower()]
        actual_predictors = [column_types[name.lower()][0] for name in predictor_columns]
        numeric_pattern = r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|UTINYINT|USMALLINT|UINTEGER|UBIGINT|FLOAT|DOUBLE|REAL|DECIMAL|NUMERIC)"
        for name, data_type in [(actual_target, target_type)] + [(column_types[item.lower()][0], column_types[item.lower()][1]) for item in predictor_columns]:
            if not re.match(numeric_pattern, data_type):
                return json.dumps({"error": f"{name} must be numeric; found {data_type}."})
        if not (date_type.startswith("DATE") or date_type.startswith("TIMESTAMP")):
            return json.dumps({"error": f"{actual_date} must be a DATE or TIMESTAMP column; found {date_type}."})
        predictor_names_lower = {item.lower() for item in actual_predictors}
        if actual_date.lower() == actual_target.lower() or actual_date.lower() in predictor_names_lower:
            return json.dumps({"error": "The date column must be distinct from the target and predictor columns."})
        if actual_target.lower() in {item.lower() for item in actual_predictors}:
            return json.dumps({"error": "The target column cannot also be a predictor."})
        date_sql = f'r.{_quote_identifier(actual_date)}'
        target_sql = f'try_cast(r.{_quote_identifier(actual_target)} AS DOUBLE)'
        predictor_sql = [f'try_cast(r.{_quote_identifier(name)} AS DOUBLE)' for name in actual_predictors]
        complete_predicate = " AND ".join([f"{item} IS NOT NULL AND isfinite({item})" for item in [target_sql] + predictor_sql])
        hash_parts = [f"cast({date_sql} AS VARCHAR)", f"cast({target_sql} AS VARCHAR)"] + [f"cast({item} AS VARCHAR)" for item in predictor_sql]
        hash_sql = "hash(concat(" + ", '|', ".join(hash_parts) + "))"
        selected = [f"{date_sql}::DATE AS event_date", f"{target_sql} AS target"] + [f"{item} AS x{index}" for index, item in enumerate(predictor_sql)]
        sql = f"""
        SELECT {', '.join(selected)}
        FROM {relation} AS r
        WHERE {date_sql} IS NOT NULL AND {complete_predicate}
        QUALIFY row_number() OVER (
          PARTITION BY date_trunc('month', {date_sql}) ORDER BY {hash_sql}
        ) <= ?
        ORDER BY event_date
        LIMIT 200000
        """
        cursor = con.execute(sql, [sample_per_month])
        result_columns = [item[0] for item in cursor.description]
        records = [dict(zip(result_columns, row)) for row in cursor.fetchall()]
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        con.close()
    if len(records) < 30:
        return json.dumps({"error": "At least 30 complete rows are required for regression and time-ordered validation."})

    def event_day(row: dict[str, Any]) -> date:
        value = row["event_date"]
        return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])

    unique_dates = sorted({event_day(row) for row in records})
    if len(unique_dates) < 2:
        return json.dumps({"error": "At least two distinct dates are required for a chronological holdout."})
    split_date = unique_dates[min(len(unique_dates) - 1, max(1, math.ceil(len(unique_dates) * 0.8)))]
    train = [row for row in records if event_day(row) < split_date]
    holdout = [row for row in records if event_day(row) >= split_date]
    if len(train) < 20 or len(holdout) < 5:
        return json.dumps({"error": "The chronological split needs at least 20 training rows and 5 later holdout rows; increase sample_per_month or use a wider date range."})
    predictor_count = len(actual_predictors)
    means = [sum(float(row[f"x{index}"]) for row in train) / len(train) for index in range(predictor_count)]
    scales = []
    for index, mean in enumerate(means):
        variance = sum((float(row[f"x{index}"]) - mean) ** 2 for row in train) / len(train)
        scales.append(math.sqrt(variance) or 1.0)

    def model_features(row: dict[str, Any]) -> list[float]:
        return [1.0] + [(float(row[f"x{index}"]) - means[index]) / scales[index] for index in range(predictor_count)]

    try:
        coefficients = _fit_ridge_regression(
            [model_features(row) for row in train],
            [float(row["target"]) for row in train],
            ridge=0.1,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    actuals = [float(row["target"]) for row in holdout]
    predictions = [sum(a * b for a, b in zip(model_features(row), coefficients)) for row in holdout]
    residuals = [actual - predicted for actual, predicted in zip(actuals, predictions)]
    mean_actual = sum(actuals) / len(actuals)
    ss_total = sum((actual - mean_actual) ** 2 for actual in actuals)
    ss_residual = sum(error ** 2 for error in residuals)
    nonzero_actuals = [(error, actual) for error, actual in zip(residuals, actuals) if actual != 0]
    mape_pct = (
        round(sum(abs(error / actual) * 100 for error, actual in nonzero_actuals) / len(nonzero_actuals), 4)
        if nonzero_actuals else None
    )
    scored = []
    for row, actual, predicted, error in zip(holdout, actuals, predictions, residuals):
        scored.append({
            "date": event_day(row).isoformat(),
            "actual": round(actual, 4),
            "predicted": round(predicted, 4),
            "absolute_error": round(abs(error), 4),
        })
    scored.sort(key=lambda row: row["absolute_error"], reverse=True)
    return json.dumps({
        "table": table_name,
        "target": actual_target,
        "predictors": actual_predictors,
        "date_column": actual_date,
        "training_rows": len(train),
        "holdout_rows": len(holdout),
        "sampled_rows": len(records),
        "sample_per_month_cap": sample_per_month,
        "holdout_start_date": split_date.isoformat(),
        "model": "L2-regularized linear regression; predictors are standardized using training rows only.",
        "holdout_metrics": {
            "mae": round(sum(abs(error) for error in residuals) / len(residuals), 5),
            "rmse": round(math.sqrt(ss_residual / len(residuals)), 5),
            "r_squared": round(1 - ss_residual / ss_total, 5) if ss_total else None,
            "mape_pct": mape_pct,
            "mean_error_actual_minus_predicted": round(sum(residuals) / len(residuals), 5),
        },
        "coefficients_per_predictor_standard_deviation": dict(zip(["intercept_at_training_means"] + actual_predictors, [round(value, 6) for value in coefficients])),
        "largest_holdout_errors": scored[:top_errors],
        "limitations": [
            "This is a configurable numeric baseline for later-dated rows, not an automatic feature-selection system; choose predictors available at prediction time and avoid target leakage.",
            "Categorical predictors are not encoded; only numeric columns are supported. Rows with missing or non-finite target/features are excluded.",
            "Holdout accuracy is historical and does not guarantee future performance. The capped deterministic monthly sample may underrepresent rare cases.",
            "The tool scores existing holdout rows; for future values of a single measure over time, use forecast_numeric_series.",
            "Present monetary predictions only in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions.",
        ],
    }, indent=2)


@mcp.tool()
async def discount_scenario_simulator(
    discount_change_pp: float = Field(2.0, description="Change the weighted discount rate by this many percentage points; positive increases discounts, negative reduces them.", ge=-50.0, le=50.0),
    group_by: str = Field("therapeutic_area", description="Scenario breakdown: overall, therapeutic_area, brand_name, channel, region, or fiscal_year."),
    start_date: str = Field("", description="Optional start date (YYYY-MM-DD)."),
    end_date: str = Field("", description="Optional end date (YYYY-MM-DD)."),
    limit: int = Field(50, description="Maximum groups to return (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Estimates same-volume sales and gross-profit changes under a weighted discount-rate scenario."""
    dimensions = {
        "overall": "'All sales'",
        "therapeutic_area": "therapeutic_area",
        "brand_name": "brand_name",
        "channel": "channel",
        "region": "region",
        "fiscal_year": "fiscal_year",
    }
    dimension = dimensions.get(group_by.lower())
    if not dimension:
        return json.dumps({"error": f"Invalid group_by. Choose one of: {', '.join(dimensions)}."})
    filters = ["coalesce(returned, false) = false"]
    params: list[str] = []
    try:
        if start_date:
            date.fromisoformat(start_date)
            filters.append("invoice_date >= ?")
            params.append(start_date)
        if end_date:
            date.fromisoformat(end_date)
            filters.append("invoice_date <= ?")
            params.append(end_date)
    except ValueError:
        return json.dumps({"error": "Date filters must be valid ISO dates in YYYY-MM-DD format."})
    if start_date and end_date and start_date > end_date:
        return json.dumps({"error": "start_date must be on or before end_date."})
    if ctx:
        await ctx.info("Calculating same-volume discount scenario by selected segment")
    sql = f"""
    SELECT {dimension} AS segment,
      count(*) AS invoice_lines,
      sum(quantity_packs) AS packs,
      sum(gross_sales_inr) AS gross_sales_inr,
      sum(discount_inr) AS discount_inr,
      sum(net_sales_inr) AS baseline_net_sales_inr,
      sum(cogs_inr) AS cogs_inr,
      sum(gross_profit_inr) AS baseline_gross_profit_inr
    FROM sales_full
    WHERE {' AND '.join(filters)}
    GROUP BY segment
    ORDER BY gross_sales_inr DESC NULLS LAST
    LIMIT {limit}
    """
    try:
        rows = _run_query(sql, params)
        for row in rows:
            gross_sales = row["gross_sales_inr"] or 0
            baseline_discount = row["discount_inr"] or 0
            cogs = row["cogs_inr"] or 0
            baseline_profit = row["baseline_gross_profit_inr"] or 0
            current_rate = 100 * baseline_discount / gross_sales if gross_sales else 0.0
            scenario_rate = min(100.0, max(0.0, current_rate + discount_change_pp))
            scenario_discount = gross_sales * scenario_rate / 100
            scenario_net_sales = gross_sales - scenario_discount
            scenario_profit = scenario_net_sales - cogs
            volume_change_needed = 100 * (baseline_profit / scenario_profit - 1) if baseline_profit > 0 and scenario_profit > 0 else None
            row.update({
                "baseline_discount_pct": round(current_rate, 2),
                "scenario_discount_pct": round(scenario_rate, 2),
                "scenario_discount_inr": round(scenario_discount, 2),
                "scenario_net_sales_inr": round(scenario_net_sales, 2),
                "net_sales_change_inr": round(scenario_net_sales - (row["baseline_net_sales_inr"] or 0), 2),
                "scenario_gross_profit_inr": round(scenario_profit, 2),
                "gross_profit_change_inr": round(scenario_profit - baseline_profit, 2),
                "scenario_gross_margin_pct": round(100 * scenario_profit / scenario_net_sales, 2) if scenario_net_sales else None,
                "volume_change_needed_pct_to_match_baseline_profit": round(volume_change_needed, 2) if volume_change_needed is not None else None,
            })
        return json.dumps({
            "discount_change_pp_requested": discount_change_pp,
            "group_by": group_by,
            "data": rows,
            "assumptions": [
                "Returned invoice lines are excluded.",
                "Gross sales, COGS per unit, product mix, and sales volume stay fixed for the same-volume calculation.",
                "The volume change needed is arithmetic to restore baseline gross profit under unchanged per-unit economics; it is not a demand forecast.",
                "The discount rate is clipped to 0%-100%.",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def monte_carlo_discount_simulation(
    discount_change_pp: float = Field(2.0, description="Change the weighted discount rate by percentage points; a positive value increases the discount.", ge=-50.0, le=50.0),
    elasticity_mean: float = Field(-1.2, description="Assumed mean price elasticity: quantity percentage response per 1% price change; commonly negative, but set from a defensible scenario assumption.", ge=-5.0, le=5.0),
    elasticity_std: float = Field(0.3, description="Assumed standard deviation of elasticity draws; zero makes the volume response deterministic.", ge=0.0, le=3.0),
    group_by: str = Field("therapeutic_area", description="Scenario breakdown: overall, therapeutic_area, brand_name, channel, region, or fiscal_year."),
    draws: int = Field(2000, description="Monte Carlo draws per group (100-10000).", ge=100, le=10000),
    seed: int = Field(2026, description="Repeatable random seed (0-2147483647).", ge=0, le=2147483647),
    start_date: str = Field("", description="Optional start date (YYYY-MM-DD)."),
    end_date: str = Field("", description="Optional end date (YYYY-MM-DD)."),
    ctx: Context = None,
) -> str:
    """Simulates profit and sales distributions under a discount change and an explicit uncertain price-elasticity assumption."""
    dimensions = {
        "overall": "'All sales'", "therapeutic_area": "therapeutic_area", "brand_name": "brand_name",
        "channel": "channel", "region": "region", "fiscal_year": "fiscal_year",
    }
    dimension = dimensions.get(group_by.lower())
    if not dimension:
        return json.dumps({"error": f"Invalid group_by. Choose one of: {', '.join(dimensions)}."})
    filters = ["coalesce(returned, false) = false"]
    params: list[Any] = []
    try:
        if start_date:
            date.fromisoformat(start_date)
            filters.append("invoice_date >= ?")
            params.append(start_date)
        if end_date:
            date.fromisoformat(end_date)
            filters.append("invoice_date <= ?")
            params.append(end_date)
    except ValueError:
        return json.dumps({"error": "Date filters must be valid ISO dates in YYYY-MM-DD format."})
    if start_date and end_date and start_date > end_date:
        return json.dumps({"error": "start_date must be on or before end_date."})
    if ctx:
        await ctx.info(f"Simulating {draws} discount-response outcomes for each selected group")
    sql = f"""
    SELECT {dimension} AS segment, count(*) AS invoice_lines,
      sum(gross_sales_inr) AS gross_sales_inr,
      sum(discount_inr) AS discount_inr,
      sum(net_sales_inr) AS baseline_net_sales_inr,
      sum(cogs_inr) AS baseline_cogs_inr,
      sum(gross_profit_inr) AS baseline_gross_profit_inr
    FROM sales_full
    WHERE {' AND '.join(filters)}
    GROUP BY segment
    ORDER BY abs(gross_sales_inr) DESC NULLS LAST
    LIMIT 50
    """
    try:
        rows = _run_query(sql, params)
        rng = random.Random(seed)

        def percentile(sorted_values: list[float], fraction: float) -> float:
            position = (len(sorted_values) - 1) * fraction
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return sorted_values[lower]
            return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)

        simulated_groups = []
        for row in rows:
            gross = float(row["gross_sales_inr"] or 0.0)
            discount = float(row["discount_inr"] or 0.0)
            baseline_net = float(row["baseline_net_sales_inr"] or 0.0)
            baseline_cogs = float(row["baseline_cogs_inr"] or 0.0)
            baseline_profit = float(row["baseline_gross_profit_inr"] or 0.0)
            if gross <= 0 or baseline_net <= 0:
                continue
            current_rate = discount / gross
            scenario_rate = min(0.9999, max(0.0, current_rate + discount_change_pp / 100.0))
            realized_price_ratio = max(0.0001, (1.0 - scenario_rate) / max(1e-6, 1.0 - current_rate))
            simulated_sales = []
            simulated_profits = []
            simulated_volume_change = []
            better_count = 0
            for _ in range(draws):
                elasticity = rng.gauss(elasticity_mean, elasticity_std) if elasticity_std else elasticity_mean
                elasticity = min(5.0, max(-5.0, elasticity))
                volume_factor = realized_price_ratio ** elasticity
                scenario_sales = baseline_net * realized_price_ratio * volume_factor
                scenario_cogs = baseline_cogs * volume_factor
                scenario_profit = scenario_sales - scenario_cogs
                simulated_sales.append(scenario_sales)
                simulated_profits.append(scenario_profit)
                simulated_volume_change.append((volume_factor - 1.0) * 100.0)
                better_count += scenario_profit > baseline_profit
            sales_sorted = sorted(simulated_sales)
            profit_sorted = sorted(simulated_profits)
            volume_sorted = sorted(simulated_volume_change)
            simulated_groups.append({
                "segment": row["segment"],
                "invoice_lines": row["invoice_lines"],
                "baseline_discount_pct": round(current_rate * 100, 3),
                "scenario_discount_pct": round(scenario_rate * 100, 3),
                "baseline_net_sales_inr": round(baseline_net, 2),
                "baseline_gross_profit_inr": round(baseline_profit, 2),
                "simulated_net_sales_inr_p10_p50_p90": [round(percentile(sales_sorted, q), 2) for q in (0.1, 0.5, 0.9)],
                "simulated_gross_profit_inr_p10_p50_p90": [round(percentile(profit_sorted, q), 2) for q in (0.1, 0.5, 0.9)],
                "simulated_volume_change_pct_p10_p50_p90": [round(percentile(volume_sorted, q), 3) for q in (0.1, 0.5, 0.9)],
                "probability_profit_exceeds_baseline_pct": round(100 * better_count / draws, 2),
            })
        return json.dumps({
            "group_by": group_by,
            "discount_change_pp": discount_change_pp,
            "draws_per_group": draws,
            "seed": seed,
            "elasticity_assumption": {"distribution": "Normal, clipped to [-5, 5]", "mean": elasticity_mean, "standard_deviation": elasticity_std},
            "data": simulated_groups,
            "assumptions": [
                "The scenario changes realized price in proportion to the weighted discount-rate change; all other mix and gross prices stay fixed.",
                "Quantity responds as (scenario price / baseline price) raised to the sampled elasticity; COGS scales with quantity at the historical average unit cost.",
                "Elasticity is supplied by the caller and is not estimated or validated by this simulation. Outcomes are conditional scenarios, not forecasts or causal estimates.",
                "Returned lines are excluded; scenario discount is clipped to 0%-99.99% to keep the price ratio defined.",
                "Values are exact INR for computation; present monetary results only in ₹ Lakhs (L) or ₹ Crores (Cr).",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def sales_driver_scenario_forecast(
    group_by: str = Field("therapeutic_area", description="Forecast overall or by therapeutic_area, brand_name, channel, or region."),
    horizon_months: int = Field(6, description="Number of future months to scenario (1-12).", ge=1, le=12),
    price_change_pct: float = Field(0.0, description="Assumed change in gross realized price, percent, applied to the seasonal baseline (-90% to +100%).", ge=-90.0, le=100.0),
    discount_change_pp: float = Field(0.0, description="Assumed change to weighted discount rate, in percentage points (-50 to +50).", ge=-50.0, le=50.0),
    volume_change_pct: float = Field(0.0, description="Base assumed change in packs sold; this can represent a promotion's assumed demand response (-90% to +300%).", ge=-90.0, le=300.0),
    volume_sensitivity_pp: float = Field(5.0, description="Half-width of low/high volume sensitivity cases, in percentage points (0-100).", ge=0.0, le=100.0),
    unit_cogs_change_pct: float = Field(0.0, description="Assumed change to COGS per pack, percent (-90% to +300%).", ge=-90.0, le=300.0),
    top_n: int = Field(20, description="Largest groups to forecast separately; remaining groups are combined into Other (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Projects low/base/high sales and gross-profit scenarios from caller-supplied price, discount, volume, and unit-cost changes."""
    try:
        group_by = group_by.lower().strip()
        if ctx:
            await ctx.info("Preparing grouped seasonal baselines and driver scenarios")
        latest_partial, rows = _monthly_sales_segment_rows(group_by)
        if not rows:
            return json.dumps({"error": "No complete monthly sales observations were found."})
        monthly: dict[tuple[str, date], dict[str, Any]] = {}
        for row in rows:
            period = row["period"] if isinstance(row["period"], date) else date.fromisoformat(str(row["period"])[:10])
            monthly[(str(row["segment"]), period)] = row
        complete_periods = sorted({period for _, period in monthly})
        if len(complete_periods) < 24:
            return json.dumps({"error": "At least 24 complete months are required for seasonal driver scenarios."})
        latest_complete = complete_periods[-1]
        trailing_start = _advance_month(latest_complete, -11)
        group_totals: dict[str, float] = {}
        for (group, period), row in monthly.items():
            if period >= trailing_start:
                group_totals[group] = group_totals.get(group, 0.0) + abs(float(row["net_sales_inr"] or 0.0))
        all_groups = sorted(group_totals, key=group_totals.get, reverse=True)
        if group_by == "overall":
            chosen_groups = all_groups[:1]
        else:
            chosen_groups = all_groups[:top_n]
        low_volume = max(-99.9, volume_change_pct - volume_sensitivity_pp)
        high_volume = min(500.0, volume_change_pct + volume_sensitivity_pp)
        volume_cases = {"low": low_volume, "base": volume_change_pct, "high": high_volume}
        forecasts = []
        missing_references = 0
        price_factor = 1.0 + price_change_pct / 100.0
        cogs_factor = 1.0 + unit_cogs_change_pct / 100.0
        for step in range(1, horizon_months + 1):
            target_period = _advance_month(latest_complete, step)
            reference_period = date(target_period.year - 1, target_period.month, 1)
            for group in chosen_groups:
                baseline = monthly.get((group, reference_period))
                if not baseline:
                    missing_references += 1
                    continue
                gross = float(baseline["gross_sales_inr"] or 0.0)
                discount = float(baseline["discount_inr"] or 0.0)
                baseline_net = float(baseline["net_sales_inr"] or 0.0)
                baseline_cogs = float(baseline["cogs_inr"] or 0.0)
                baseline_profit = float(baseline["gross_profit_inr"] or 0.0)
                discount_rate = discount / gross if gross else 0.0
                scenario_discount_rate = min(0.9999, max(0.0, discount_rate + discount_change_pp / 100.0))
                cases = {}
                for case_name, volume_pct in volume_cases.items():
                    volume_factor = max(0.001, 1.0 + volume_pct / 100.0)
                    scenario_gross = gross * price_factor * volume_factor
                    scenario_sales = scenario_gross * (1.0 - scenario_discount_rate)
                    scenario_cogs = baseline_cogs * volume_factor * cogs_factor
                    scenario_profit = scenario_sales - scenario_cogs
                    cases[case_name] = {
                        "assumed_volume_change_pct": round(volume_pct, 3),
                        "net_sales_inr": round(scenario_sales, 2),
                        "gross_profit_inr": round(scenario_profit, 2),
                        "gross_margin_pct": round(100 * scenario_profit / scenario_sales, 3) if scenario_sales else None,
                    }
                forecasts.append({
                    "month": target_period.isoformat(),
                    "baseline_reference_month": reference_period.isoformat(),
                    "group": group,
                    "baseline_net_sales_inr": round(baseline_net, 2),
                    "baseline_gross_profit_inr": round(baseline_profit, 2),
                    "baseline_discount_pct": round(discount_rate * 100, 3),
                    "scenario_discount_pct": round(scenario_discount_rate * 100, 3),
                    "scenarios": cases,
                })
        return json.dumps({
            "group_by": group_by,
            "groups_returned": len(chosen_groups),
            "last_complete_month": latest_complete.isoformat(),
            "excluded_latest_source_month": latest_partial.isoformat(),
            "driver_assumptions": {
                "gross_price_change_pct": price_change_pct,
                "discount_change_pp": discount_change_pp,
                "base_volume_change_pct": volume_change_pct,
                "low_high_volume_sensitivity_pp": volume_sensitivity_pp,
                "unit_cogs_change_pct": unit_cogs_change_pct,
            },
            "forecast_rows": forecasts,
            "missing_group_reference_periods": missing_references,
            "method": "Seasonal-naive baseline from the same month in the prior year, adjusted algebraically by the supplied driver assumptions.",
            "limitations": [
                "Low/base/high cases vary the assumed volume response only; they are sensitivity scenarios, not probability intervals.",
                "The volume assumption may stand in for a promotion response, but the dataset has no promotion field and this tool does not estimate an uplift.",
                "Product mix and all other per-pack economics stay at the seasonal baseline; no inventory availability or capacity constraint is modeled.",
                "All monetary inputs and outputs are exact INR; present human-facing amounts only in ₹ Lakhs (L) or ₹ Crores (Cr).",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def reconciled_sales_forecast(
    group_by: str = Field("therapeutic_area", description="Bottom-level forecast groups: therapeutic_area, brand_name, channel, or region."),
    metric: str = Field("net_sales_inr", description="Additive measure: net_sales_inr or gross_profit_inr."),
    horizon_months: int = Field(6, description="Number of future months to forecast (1-12).", ge=1, le=12),
    top_n: int = Field(20, description="Return the largest groups separately; combine remaining groups as Other (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Builds bottom-up monthly forecasts whose returned group totals sum exactly to the parent total."""
    metric = metric.lower().strip()
    group_by = group_by.lower().strip()
    if metric not in {"net_sales_inr", "gross_profit_inr"}:
        return json.dumps({"error": "metric must be net_sales_inr or gross_profit_inr."})
    if group_by == "overall":
        return json.dumps({"error": "Choose a dimension so the forecast can reconcile child groups to their parent."})
    try:
        if ctx:
            await ctx.info(f"Building bottom-up {metric} forecasts by {group_by}")
        latest_partial, rows = _monthly_sales_segment_rows(group_by)
        if not rows:
            return json.dumps({"error": "No complete monthly sales observations were found."})
        monthly: dict[tuple[str, date], float] = {}
        groups: set[str] = set()
        periods: set[date] = set()
        for row in rows:
            period = row["period"] if isinstance(row["period"], date) else date.fromisoformat(str(row["period"])[:10])
            group = str(row["segment"])
            groups.add(group)
            periods.add(period)
            monthly[(group, period)] = float(row[metric] or 0.0)
        ordered_periods = sorted(periods)
        if len(ordered_periods) < 12:
            return json.dumps({"error": "At least 12 complete months are required to reconcile seasonal forecasts."})
        latest_complete = ordered_periods[-1]
        trailing_start = _advance_month(latest_complete, -11)
        ranking = {group: 0.0 for group in groups}
        for (group, period), value in monthly.items():
            if period >= trailing_start:
                ranking[group] += abs(value)
        ordered_groups = sorted(groups, key=lambda group: ranking[group], reverse=True)
        selected_groups = ordered_groups[:top_n]
        remaining_groups = set(ordered_groups[top_n:])
        forecast_rows = []
        for step in range(1, horizon_months + 1):
            target_period = _advance_month(latest_complete, step)
            source_period = date(target_period.year - 1, target_period.month, 1)
            child_values = {
                group: monthly.get((group, source_period), 0.0)
                for group in ordered_groups
            }
            parent_total = sum(child_values.values())
            children = [
                {"group": group, "forecast_inr": round(child_values[group], 2)}
                for group in selected_groups
            ]
            if remaining_groups:
                children.append({
                    "group": "Other (all remaining)",
                    "forecast_inr": round(sum(child_values[group] for group in remaining_groups), 2),
                })
            child_sum = round(sum(child["forecast_inr"] for child in children), 2)
            parent_rounded = round(parent_total, 2)
            rounding_adjustment = round(parent_rounded - child_sum, 2)
            if rounding_adjustment:
                children[-1]["forecast_inr"] = round(children[-1]["forecast_inr"] + rounding_adjustment, 2)
                children[-1]["rounding_adjustment_inr"] = rounding_adjustment
                child_sum = round(sum(child["forecast_inr"] for child in children), 2)
            forecast_rows.append({
                "month": target_period.isoformat(),
                "baseline_reference_month": source_period.isoformat(),
                "parent_forecast_inr": parent_rounded,
                "children_forecast_sum_inr": child_sum,
                "reconciliation_gap_inr": round(child_sum - parent_rounded, 2),
                "children": children,
            })
        return json.dumps({
            "group_by": group_by,
            "metric": metric,
            "method": "Bottom-up seasonal naive: each child group uses its same-month-prior-year value; the parent is the sum of all child forecasts.",
            "last_complete_month": latest_complete.isoformat(),
            "excluded_latest_source_month": latest_partial.isoformat(),
            "top_groups_returned": len(selected_groups),
            "remaining_group_count": len(remaining_groups),
            "forecast_rows": forecast_rows,
            "additivity_guarantee": "Children include an Other bucket for all unlisted groups; children sum to the parent at currency precision.",
            "caveats": [
                "This is a coherent seasonal baseline, not a separately optimized forecast for every level of the hierarchy.",
                "A group without a same-month-prior-year observation receives zero; new groups may need a different baseline.",
                "The source is synthetic. Format monetary amounts in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions.",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def customer_lapse_risk_model(
    top_n: int = Field(50, description="Number of highest-risk customers to return (1-200).", ge=1, le=200),
    ctx: Context = None,
) -> str:
    """Fits a time-ordered logistic model for no repurchase in 180 days using customer recency, frequency, sales, and discount history."""
    if ctx:
        await ctx.info("Building historical customer snapshots and a 180-day lapse holdout")
    sql = """
    WITH latest AS (
      SELECT max(invoice_date)::DATE AS latest_date FROM fact_sales
    ), snapshots AS (
      SELECT 'train' AS split, (latest_date - INTERVAL 360 DAY)::DATE AS cutoff FROM latest
      UNION ALL
      SELECT 'test' AS split, (latest_date - INTERVAL 180 DAY)::DATE AS cutoff FROM latest
    ), features AS (
      SELECT s.split, s.cutoff, f.customer_id,
        date_diff('day', max(f.invoice_date), s.cutoff) AS recency_days,
        count(DISTINCT f.invoice_no) AS invoice_frequency,
        sum(f.net_sales_inr) / 100000.0 AS window_sales_lakhs,
        avg(f.trade_discount_pct) AS avg_discount_pct
      FROM snapshots s
      JOIN fact_sales f
        ON f.invoice_date > s.cutoff - INTERVAL 180 DAY
       AND f.invoice_date <= s.cutoff
       AND coalesce(f.returned, false) = false
      GROUP BY s.split, s.cutoff, f.customer_id
    ), outcomes AS (
      SELECT s.split, s.cutoff, f.customer_id, count(*) AS future_invoice_lines
      FROM snapshots s
      JOIN fact_sales f
        ON f.invoice_date > s.cutoff
       AND f.invoice_date <= s.cutoff + INTERVAL 180 DAY
       AND coalesce(f.returned, false) = false
      GROUP BY s.split, s.cutoff, f.customer_id
    )
    SELECT x.split, x.cutoff, x.customer_id, x.recency_days,
      x.invoice_frequency, x.window_sales_lakhs, x.avg_discount_pct,
      CASE WHEN coalesce(o.future_invoice_lines, 0) > 0 THEN 0 ELSE 1 END AS lapsed,
      c.customer_type, c.region, c.channel, c.payment_terms_days
    FROM features x
    LEFT JOIN outcomes o USING(split, cutoff, customer_id)
    LEFT JOIN dim_customer c USING(customer_id)
    ORDER BY x.split, x.customer_id
    """
    try:
        rows = _run_query(sql)
        train = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        if len(train) < 100 or len(test) < 100:
            return json.dumps({"error": "The selected 180-day windows produced too few customer snapshots; at least 100 training and test examples are required."})

        def feature_row(row: dict[str, Any]) -> list[float]:
            return [
                1.0,
                float(row["recency_days"] or 0) / 180.0,
                math.log1p(max(0, int(row["invoice_frequency"] or 0))),
                math.log1p(max(0.0, float(row["window_sales_lakhs"] or 0.0))),
                float(row["avg_discount_pct"] or 0.0) / 100.0,
            ]

        train_labels = [int(row["lapsed"]) for row in train]
        coefficients = _fit_logistic_regression([feature_row(row) for row in train], train_labels)
        test_probabilities = [_predict_logistic_probability(feature_row(row), coefficients) for row in test]
        test_labels = [int(row["lapsed"]) for row in test]
        scored = []
        for row, probability in zip(test, test_probabilities):
            scored.append({
                "customer_id": row["customer_id"],
                "customer_type": row["customer_type"],
                "region": row["region"],
                "channel": row["channel"],
                "payment_terms_days": row["payment_terms_days"],
                "last_window_recency_days": row["recency_days"],
                "last_window_invoice_frequency": row["invoice_frequency"],
                "last_window_net_sales_inr": round(float(row["window_sales_lakhs"] or 0) * 100000, 2),
                "lapse_probability_pct": round(probability * 100, 2),
                "actual_lapsed_180d": bool(row["lapsed"]),
            })
        scored.sort(key=lambda item: item["lapse_probability_pct"], reverse=True)
        return json.dumps({
            "target": "No non-returned purchase in the 180 days after each snapshot cutoff.",
            "training_cutoff": train[0]["cutoff"],
            "test_cutoff": test[0]["cutoff"],
            "feature_window_days": 180,
            "model": "L2-regularized logistic regression trained on the earlier snapshot; later snapshot is a chronological holdout.",
            "feature_transforms": ["recency_days / 180", "log(1 + invoice_frequency)", "log(1 + net_sales_in_lakhs)", "average_discount_pct / 100"],
            "training_rows": len(train),
            "training_lapse_rate_pct": round(100 * sum(train_labels) / len(train_labels), 3),
            "holdout_metrics": _binary_model_metrics(test_labels, test_probabilities),
            "coefficients": dict(zip(["intercept", "recency_scaled", "log_frequency", "log_sales_lakhs", "discount_fraction"], [round(value, 5) for value in coefficients])),
            "highest_risk_test_customers": scored[:top_n],
            "caveats": [
                "This is a historical prioritization score based on synthetic sales records; it does not explain customer intent or prove an intervention will change behavior.",
                "Risk probabilities are estimated from one chronological holdout and may be miscalibrated; check the reported Brier score and prevalence before operational use.",
                "A no-purchase label means no observed invoice in the selected 180-day window, not a confirmed lost account.",
                "Format all human-facing monetary amounts in ₹ Lakhs (L) or ₹ Crores (Cr), never millions or billions.",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def return_risk_model(
    sample_per_month: int = Field(500, description="Maximum deterministic invoice lines sampled per month for training and holdout (100-2000).", ge=100, le=2000),
    ctx: Context = None,
) -> str:
    """Fits a temporal logistic model for invoice-line return risk using only sale-time features and a deterministic monthly sample."""
    try:
        latest_rows = _run_query("SELECT max(invoice_date)::DATE AS latest_date FROM fact_sales")
        latest_date = latest_rows[0]["latest_date"] if latest_rows else None
        if not latest_date:
            return json.dumps({"error": "No invoice dates are available."})
        if not isinstance(latest_date, date):
            latest_date = date.fromisoformat(str(latest_date)[:10])
        cutoff = date.fromordinal(latest_date.toordinal() - 180)
        if ctx:
            await ctx.info(f"Sampling invoice lines for a return-risk holdout ending {latest_date.isoformat()}")
        sql = """
        SELECT f.invoice_no, f.sku_id, f.invoice_date, f.expiry_date,
          f.quantity_packs, f.trade_discount_pct, f.returned,
          coalesce(s.cold_chain, 0) AS cold_chain,
          date_diff('day', f.invoice_date, f.expiry_date) AS days_to_expiry
        FROM fact_sales f
        JOIN dim_sku s USING(sku_id)
        WHERE f.invoice_date IS NOT NULL AND f.returned IS NOT NULL
        QUALIFY row_number() OVER (
          PARTITION BY date_trunc('month', f.invoice_date)
          ORDER BY hash(concat(coalesce(cast(f.invoice_no AS VARCHAR), ''), '|', cast(f.sku_id AS VARCHAR), '|', cast(f.invoice_date AS VARCHAR)))
        ) <= ?
        ORDER BY f.invoice_date
        """
        rows = _run_query(sql, [sample_per_month])
        train = [row for row in rows if str(row["invoice_date"])[:10] <= cutoff.isoformat()]
        holdout = [row for row in rows if str(row["invoice_date"])[:10] > cutoff.isoformat()]
        if len(train) < 100 or len(holdout) < 50:
            return json.dumps({"error": "The sampled date windows produced too few invoice lines; adjust the sample size or check date coverage."})

        def feature_row(row: dict[str, Any]) -> list[float]:
            expiry = row["days_to_expiry"]
            if expiry is None:
                expiry_scaled = 0.0
                expiry_missing = 1.0
            else:
                expiry_scaled = max(-3.0, min(3.0, float(expiry) / 365.0))
                expiry_missing = 0.0
            invoice_date = date.fromisoformat(str(row["invoice_date"])[:10])
            angle = 2 * math.pi * (invoice_date.month - 1) / 12
            return [
                1.0,
                expiry_scaled,
                expiry_missing,
                float(row["trade_discount_pct"] or 0.0) / 100.0,
                math.log1p(max(0.0, float(row["quantity_packs"] or 0.0))),
                1.0 if row["cold_chain"] else 0.0,
                math.sin(angle),
                math.cos(angle),
            ]

        train_labels = [1 if row["returned"] else 0 for row in train]
        holdout_labels = [1 if row["returned"] else 0 for row in holdout]
        coefficients = _fit_logistic_regression([feature_row(row) for row in train], train_labels)
        probabilities = [_predict_logistic_probability(feature_row(row), coefficients) for row in holdout]
        scored = []
        for row, probability in zip(holdout, probabilities):
            scored.append({
                "invoice_no": row["invoice_no"],
                "sku_id": row["sku_id"],
                "invoice_date": row["invoice_date"],
                "days_to_expiry_at_sale": row["days_to_expiry"],
                "discount_pct": row["trade_discount_pct"],
                "cold_chain": bool(row["cold_chain"]),
                "return_probability_pct": round(100 * probability, 2),
                "actual_returned": bool(row["returned"]),
            })
        scored.sort(key=lambda item: item["return_probability_pct"], reverse=True)
        return json.dumps({
            "target": "Whether a sampled invoice line is marked returned in the source data.",
            "latest_invoice_date": latest_date.isoformat(),
            "holdout_start_after": cutoff.isoformat(),
            "sample_per_month_cap": sample_per_month,
            "training_rows": len(train),
            "training_return_rate_pct": round(100 * sum(train_labels) / len(train_labels), 3),
            "model": "L2-regularized logistic regression; the most recent 180 days are a chronological holdout.",
            "sale_time_features_only": ["days to expiry at invoice", "expiry-date-missing flag", "discount percent", "log(1 + packs)", "cold-chain flag", "invoice-month sine/cosine"],
            "holdout_metrics": _binary_model_metrics(holdout_labels, probabilities),
            "coefficients": dict(zip(["intercept", "days_to_expiry_scaled", "expiry_missing", "discount_fraction", "log_packs", "cold_chain", "month_sin", "month_cos"], [round(value, 5) for value in coefficients])),
            "highest_risk_holdout_lines": scored[:50],
            "caveats": [
                "The monthly hash sample is deterministic but may miss rare patterns; metrics apply to that sampled holdout.",
                "The result is a historical risk ranking, not an operational or causal claim. Verify the model on later, representative data before decisions.",
                "Only fields known at invoice time are used as predictors; the return outcome is shown for holdout evaluation only.",
            ],
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def basket_affinity_analysis(
    min_pair_invoices: int = Field(25, description="Minimum distinct invoices containing both products (1-10000).", ge=1, le=10000),
    min_confidence: float = Field(0.2, description="Minimum directional confidence to show a pair (0.0-1.0).", ge=0.0, le=1.0),
    therapeutic_area: str = Field("", description="Optional restriction so both products in a pair belong to this therapeutic area."),
    start_date: str = Field("", description="Optional start date (YYYY-MM-DD)."),
    end_date: str = Field("", description="Optional end date (YYYY-MM-DD)."),
    limit: int = Field(30, description="Maximum product pairs to return (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Finds SKU pairs bought on the same invoice and reports support, directional confidence, and lift."""
    filters = ["invoice_no IS NOT NULL", "coalesce(returned, false) = false"]
    params: list[Any] = []
    try:
        if start_date:
            date.fromisoformat(start_date)
            filters.append("invoice_date >= ?")
            params.append(start_date)
        if end_date:
            date.fromisoformat(end_date)
            filters.append("invoice_date <= ?")
            params.append(end_date)
    except ValueError:
        return json.dumps({"error": "Date filters must be valid ISO dates in YYYY-MM-DD format."})
    if start_date and end_date and start_date > end_date:
        return json.dumps({"error": "start_date must be on or before end_date."})
    if therapeutic_area:
        filters.append("sku_id IN (SELECT sku_id FROM dim_sku WHERE therapeutic_area = ?)")
        params.append(therapeutic_area)
    if ctx:
        await ctx.info("Building distinct invoice-SKU baskets and ranking association rules")
    sql = f"""
    WITH invoice_items AS (
      SELECT DISTINCT invoice_no, customer_id, invoice_date, sku_id
      FROM fact_sales
      WHERE {' AND '.join(filters)}
    ), sku_frequency AS (
      SELECT sku_id, count(*) AS invoice_count FROM invoice_items GROUP BY sku_id
    ), total_baskets AS (
      SELECT count(*) AS invoice_count
      FROM (SELECT invoice_no, customer_id, invoice_date FROM invoice_items GROUP BY ALL) grouped_baskets
    ), pairs AS (
      SELECT a.sku_id AS sku_a, b.sku_id AS sku_b, count(*) AS pair_invoice_count
      FROM invoice_items a
      JOIN invoice_items b USING(invoice_no, customer_id, invoice_date)
      WHERE a.sku_id < b.sku_id
      GROUP BY a.sku_id, b.sku_id
    )
    SELECT p.sku_a, sa.brand_name AS product_a, sa.therapeutic_area AS area_a,
      p.sku_b, sb.brand_name AS product_b, sb.therapeutic_area AS area_b,
      p.pair_invoice_count,
      round(100.0 * p.pair_invoice_count / nullif(t.invoice_count, 0), 3) AS support_pct,
      round(p.pair_invoice_count::DOUBLE / nullif(fa.invoice_count, 0), 3) AS confidence_a_to_b,
      round(p.pair_invoice_count::DOUBLE / nullif(fb.invoice_count, 0), 3) AS confidence_b_to_a,
      round((p.pair_invoice_count::DOUBLE * t.invoice_count) / nullif(fa.invoice_count * fb.invoice_count, 0), 3) AS lift
    FROM pairs p
    JOIN sku_frequency fa ON fa.sku_id = p.sku_a
    JOIN sku_frequency fb ON fb.sku_id = p.sku_b
    JOIN total_baskets t ON true
    JOIN dim_sku sa ON sa.sku_id = p.sku_a
    JOIN dim_sku sb ON sb.sku_id = p.sku_b
    WHERE p.pair_invoice_count >= ?
      AND greatest(p.pair_invoice_count::DOUBLE / nullif(fa.invoice_count, 0),
                   p.pair_invoice_count::DOUBLE / nullif(fb.invoice_count, 0)) >= ?
    ORDER BY lift DESC, pair_invoice_count DESC
    LIMIT {limit}
    """
    try:
        rows = _run_query(sql, params + [min_pair_invoices, min_confidence])
        return json.dumps({
            "basket_key": "invoice_no + customer_id + invoice_date",
            "association_metrics": {"support_pct": "pair invoices / all invoices", "confidence_a_to_b": "pair invoices / invoices containing product A", "lift": "observed co-purchase rate divided by the rate expected if independent"},
            "min_pair_invoices": min_pair_invoices,
            "min_confidence": min_confidence,
            "data": rows,
            "caveat": "Co-purchase is descriptive association, not a causal recommendation. The tool counts distinct SKU presence per invoice and does not account for product availability or promotions.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def data_quality_audit(
    table_name: str = Field("fact_sales", description="Catalog table or view to inspect."),
    key_columns: str = Field("", description="Optional comma-separated candidate key columns to check for duplicate rows (for example 'invoice_no, sku_id')."),
    date_column: str = Field("", description="Optional date/timestamp column to summarize; defaults to a conventional date column when available."),
    max_columns: int = Field(60, description="Maximum leading columns to check (1-80).", ge=1, le=80),
    ctx: Context = None,
) -> str:
    """Audits null rates, negative numeric values, selected-key uniqueness, date bounds, and known fact-table join coverage."""
    con = get_connection(read_only=True)
    try:
        schema_name, actual_name, relation_type = _resolve_relation(con, table_name)
        qualified = _relation_sql(schema_name, actual_name)
        columns = con.execute(f"DESCRIBE {qualified}").fetchall()
        selected = columns[:max_columns]
        expressions = ["count(*) AS row_count"]
        aliases = []
        numeric_types = ("tinyint", "smallint", "integer", "bigint", "hugeint", "utinyint", "usmallint", "uinteger", "ubigint", "uhugeint", "float", "double", "real", "decimal", "numeric")
        for index, column in enumerate(selected):
            name, data_type = column[0], column[1]
            quoted = _quote_identifier(name)
            prefix = f"q{index}"
            expressions.append(f"count({quoted}) AS {prefix}_nonnull")
            is_numeric = data_type.lower().startswith(numeric_types)
            if is_numeric:
                expressions.append(f"count(*) FILTER (WHERE try_cast({quoted} AS DOUBLE) < 0) AS {prefix}_negative")
            aliases.append({"name": name, "type": data_type, "prefix": prefix, "numeric": is_numeric})
        if ctx:
            await ctx.info(f"Auditing {schema_name}.{actual_name} across {len(selected)} of {len(columns)} columns")
        aggregate = con.execute(f"SELECT {', '.join(expressions)} FROM {qualified}").fetchone()
        row_count = aggregate[0]
        cursor = 1
        checks = []
        review_flags = []
        for column in aliases:
            non_null = aggregate[cursor]
            null_count = row_count - non_null
            item = {
                "column": column["name"],
                "type": column["type"],
                "null_count": null_count,
                "null_pct": round(100 * null_count / row_count, 2) if row_count else None,
            }
            cursor += 1
            if column["numeric"]:
                item["negative_value_count"] = aggregate[cursor]
                if aggregate[cursor]:
                    review_flags.append(f"{column['name']} contains {aggregate[cursor]} negative numeric values; confirm whether these are valid returns or credits.")
                cursor += 1
            if null_count:
                review_flags.append(f"{column['name']} contains {null_count} null values.")
            checks.append(item)

        available = {column[0].lower(): column for column in columns}
        requested_keys = [value.strip() for value in key_columns.split(",") if value.strip()]
        if not requested_keys:
            default_keys = {
                "dim_customer": ["customer_id"], "dim_sku": ["sku_id"],
                "dim_rep": ["sales_rep_id"], "dim_date": ["date"], "readme": ["item"],
            }
            requested_keys = default_keys.get(actual_name.lower(), [])
        resolved_keys = []
        for key in requested_keys:
            actual_column = available.get(key.lower())
            if not actual_column:
                return json.dumps({"error": f"Key column '{key}' is not in {schema_name}.{actual_name}."})
            resolved_keys.append(actual_column[0])

        key_uniqueness = {"columns": resolved_keys, "checked": bool(resolved_keys)}
        if resolved_keys:
            group_keys = ", ".join(_quote_identifier(key) for key in resolved_keys)
            duplicate_sql = f"""
            SELECT count(*) AS duplicate_groups, coalesce(sum(group_size - 1), 0) AS duplicate_extra_rows
            FROM (SELECT count(*) AS group_size FROM {qualified} GROUP BY {group_keys} HAVING count(*) > 1) duplicates
            """
            duplicate_groups, duplicate_rows = con.execute(duplicate_sql).fetchone()
            key_uniqueness.update({"duplicate_groups": duplicate_groups, "duplicate_extra_rows": duplicate_rows})
            if duplicate_rows:
                review_flags.append(f"Selected key {', '.join(resolved_keys)} has {duplicate_rows} duplicate extra rows.")
        else:
            key_uniqueness["note"] = "No declared key was inferred for this object; pass key_columns to check uniqueness."

        if date_column:
            candidate = available.get(date_column.lower())
            if not candidate:
                return json.dumps({"error": f"Date column '{date_column}' is not in {schema_name}.{actual_name}."})
        else:
            candidate = next((available[name] for name in ("invoice_date", "date", "first_order_date", "created_at", "timestamp") if name in available), None)
        date_summary = None
        if candidate:
            type_name = candidate[1].upper()
            if "DATE" not in type_name and "TIME" not in type_name:
                return json.dumps({"error": f"Column '{candidate[0]}' is not a date or timestamp type."})
            quoted_date = _quote_identifier(candidate[0])
            minimum, maximum, null_dates = con.execute(
                f"SELECT min({quoted_date}), max({quoted_date}), count(*) FILTER (WHERE {quoted_date} IS NULL) FROM {qualified}"
            ).fetchone()
            date_summary = {"column": candidate[0], "min": _sanitize_val(minimum), "max": _sanitize_val(maximum), "null_count": null_dates}

        join_coverage = None
        if schema_name.lower() == "main" and actual_name.lower() == "fact_sales":
            coverage = con.execute("""
              SELECT count(*) AS fact_rows,
                count(*) FILTER (WHERE c.customer_id IS NULL) AS missing_customer_matches,
                count(*) FILTER (WHERE s.sku_id IS NULL) AS missing_sku_matches,
                count(*) FILTER (WHERE d.date IS NULL) AS missing_date_matches
              FROM fact_sales f
              LEFT JOIN dim_customer c USING(customer_id)
              LEFT JOIN dim_sku s USING(sku_id)
              LEFT JOIN dim_date d ON d.date = f.invoice_date
            """).fetchone()
            join_coverage = {"fact_rows": coverage[0], "missing_customer_matches": coverage[1], "missing_sku_matches": coverage[2], "missing_date_matches": coverage[3]}
            for key, count in join_coverage.items():
                if key.startswith("missing_") and count:
                    review_flags.append(f"{key.replace('_', ' ')}: {count}.")

        return json.dumps({
            "table": f"{schema_name}.{actual_name}",
            "kind": relation_type,
            "row_count": row_count,
            "columns_checked": len(selected),
            "columns_total": len(columns),
            "column_checks": checks,
            "key_uniqueness": key_uniqueness,
            "date_range": date_summary,
            "known_join_coverage": join_coverage,
            "review_flags": review_flags,
            "interpretation": "Flags identify items to review; negative values and nulls are not automatically data defects.",
            "truncated_columns": len(columns) > len(selected),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
    finally:
        con.close()


@mcp.tool()
async def sales_driver_decomposition(
    baseline_start: str = Field(..., description="Baseline period start date (YYYY-MM-DD)."),
    baseline_end: str = Field(..., description="Baseline period end date (YYYY-MM-DD)."),
    comparison_start: str = Field(..., description="Comparison period start date (YYYY-MM-DD)."),
    comparison_end: str = Field(..., description="Comparison period end date (YYYY-MM-DD)."),
    group_by: str = Field("overall", description="Return decomposition for overall, therapeutic_area, brand_name, channel, or region."),
    limit: int = Field(30, description="Maximum segment rows to return (1-100).", ge=1, le=100),
    ctx: Context = None,
) -> str:
    """Splits net sales change into price, total-volume, and SKU-mix effects using an order-independent Shapley decomposition."""
    try:
        parsed = [date.fromisoformat(value) for value in (baseline_start, baseline_end, comparison_start, comparison_end)]
    except ValueError:
        return json.dumps({"error": "All period dates must be valid YYYY-MM-DD dates."})
    b_start, b_end, c_start, c_end = parsed
    if b_start > b_end or c_start > c_end:
        return json.dumps({"error": "Each period start must be on or before its end."})
    if not (b_end < c_start or c_end < b_start):
        return json.dumps({"error": "Baseline and comparison periods must not overlap."})
    dimensions = {
        "overall": "'All sales'", "therapeutic_area": "therapeutic_area",
        "brand_name": "brand_name", "channel": "channel", "region": "region",
    }
    dimension = dimensions.get(group_by.lower())
    if not dimension:
        return json.dumps({"error": f"Invalid group_by. Choose one of: {', '.join(dimensions)}."})
    if ctx:
        await ctx.info("Building SKU-level price, volume, and mix decomposition")
    sql = f"""
    WITH scoped AS (
      SELECT {dimension} AS segment, sku_id, invoice_date, net_sales_inr, quantity_packs,
        CASE WHEN invoice_date BETWEEN ? AND ? THEN 'baseline' ELSE 'comparison' END AS period
      FROM sales_full
      WHERE ((invoice_date BETWEEN ? AND ?) OR (invoice_date BETWEEN ? AND ?))
        AND coalesce(returned, false) = false
    )
    SELECT segment, sku_id,
      sum(net_sales_inr) FILTER (WHERE period = 'baseline') AS baseline_sales_inr,
      sum(quantity_packs) FILTER (WHERE period = 'baseline') AS baseline_packs,
      sum(net_sales_inr) FILTER (WHERE period = 'comparison') AS comparison_sales_inr,
      sum(quantity_packs) FILTER (WHERE period = 'comparison') AS comparison_packs
    FROM scoped GROUP BY segment, sku_id
    """
    try:
        rows = _run_query(sql, [baseline_start, baseline_end, baseline_start, baseline_end, comparison_start, comparison_end])
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["segment"]), []).append(row)

        def decompose(items: list[dict[str, Any]]) -> dict[str, Any]:
            product_data = {}
            total_packs = {"baseline": 0.0, "comparison": 0.0}
            total_sales = {"baseline": 0.0, "comparison": 0.0}
            for item in items:
                q0 = float(item["baseline_packs"] or 0)
                q1 = float(item["comparison_packs"] or 0)
                r0 = float(item["baseline_sales_inr"] or 0)
                r1 = float(item["comparison_sales_inr"] or 0)
                p0 = r0 / q0 if q0 else None
                p1 = r1 / q1 if q1 else None
                if p0 is None:
                    p0 = p1 or 0.0
                if p1 is None:
                    p1 = p0
                product_data[item["sku_id"]] = {"q0": q0, "q1": q1, "p0": p0, "p1": p1}
                total_packs["baseline"] += q0
                total_packs["comparison"] += q1
                total_sales["baseline"] += r0
                total_sales["comparison"] += r1
            q0, q1 = total_packs["baseline"], total_packs["comparison"]
            if not q0 and not q1:
                return {"baseline_net_sales_inr": 0.0, "comparison_net_sales_inr": 0.0, "net_sales_change_inr": 0.0, "price_effect_inr": 0.0, "volume_effect_inr": 0.0, "mix_effect_inr": 0.0, "reconciliation_gap_inr": 0.0, "baseline_packs": 0, "comparison_packs": 0}
            state = {"volume": 0, "mix": 0, "price": 0}

            def modeled_revenue() -> float:
                volume = q1 if state["volume"] else q0
                total = 0.0
                for value in product_data.values():
                    quantity = value["q1"] if state["mix"] else value["q0"]
                    mix_share = quantity / (q1 if state["mix"] else q0) if (q1 if state["mix"] else q0) else 0.0
                    price = value["p1"] if state["price"] else value["p0"]
                    total += mix_share * price
                return volume * total

            effects = {"price": 0.0, "volume": 0.0, "mix": 0.0}
            permutations = [
                ("volume", "mix", "price"), ("volume", "price", "mix"),
                ("mix", "volume", "price"), ("mix", "price", "volume"),
                ("price", "volume", "mix"), ("price", "mix", "volume"),
            ]
            for ordering in permutations:
                state = {"volume": 0, "mix": 0, "price": 0}
                previous_value = modeled_revenue()
                for factor in ordering:
                    state[factor] = 1
                    next_value = modeled_revenue()
                    effects[factor] += (next_value - previous_value) / len(permutations)
                    previous_value = next_value
            actual_delta = total_sales["comparison"] - total_sales["baseline"]
            effect_total = effects["price"] + effects["volume"] + effects["mix"]
            return {
                "baseline_net_sales_inr": round(total_sales["baseline"], 2),
                "comparison_net_sales_inr": round(total_sales["comparison"], 2),
                "net_sales_change_inr": round(actual_delta, 2),
                "price_effect_inr": round(effects["price"], 2),
                "volume_effect_inr": round(effects["volume"], 2),
                "mix_effect_inr": round(effects["mix"], 2),
                "effect_share_of_change_pct": {key: round(100 * value / actual_delta, 2) if actual_delta else None for key, value in effects.items()},
                "reconciliation_gap_inr": round(actual_delta - effect_total, 2),
                "baseline_packs": round(q0, 2),
                "comparison_packs": round(q1, 2),
            }

        all_results = [{"segment": segment, **decompose(items)} for segment, items in groups.items()]
        total = decompose(rows)
        all_results.sort(key=lambda item: abs(item["net_sales_change_inr"]), reverse=True)
        return json.dumps({
            "baseline": [baseline_start, baseline_end],
            "comparison": [comparison_start, comparison_end],
            "group_by": group_by,
            "total": total,
            "segments": all_results[:limit],
            "segments_returned": min(limit, len(all_results)),
            "segments_total": len(all_results),
            "method": "Three-factor Shapley decomposition of net sales = total packs × weighted unit price, with SKU pack shares as mix. Average contribution across all six factor orderings; effects reconcile to the overall change up to rounding.",
            "scope": "Non-returned invoice lines only. New/discontinued SKU prices are carried from the active period so entry and exit are attributed to mix.",
            "caveat": "Decomposition is an accounting identity, not a causal estimate of what would happen under an intervention.",
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

# Tool 3: get_sales_by_dimension
@mcp.tool()
async def get_sales_by_dimension(
    dimension: str = Field(
        ...,
        description="Dimension to group by. One of: 'therapeutic_area', 'brand_name', 'channel', 'customer_type', 'region', 'state', 'city_tier', 'year', 'fiscal_year'."
    ),
    metric: str = Field(
        "net_sales_inr",
        description="Sorting metric: 'net_sales_inr', 'gross_profit_inr', 'total_packs', 'gross_margin_pct', or 'avg_discount_pct'."
    ),
    start_date: str = Field("", description="Optional ISO start date (YYYY-MM-DD), e.g. '2023-01-01'."),
    end_date: str = Field("", description="Optional ISO end date (YYYY-MM-DD), e.g. '2025-12-31'."),
    therapeutic_area: str = Field("", description="Optional therapeutic area filter (e.g. 'Oncology', 'Cardiology', 'Diabetes')."),
    limit: int = Field(20, description="Max rows to return (1-100).", ge=1, le=100)
) -> str:
    """Aggregates sales performance, revenue, gross margin, pack volume, and discount percentage grouped by any business dimension."""
    valid_dims = {
        'therapeutic_area': 's.therapeutic_area',
        'brand_name': 's.brand_name',
        'channel': 'c.channel',
        'customer_type': 'c.customer_type',
        'region': 'c.region',
        'state': 'c.state',
        'city_tier': 'c.city_tier',
        'year': 'd.year',
        'fiscal_year': 'd.fiscal_year'
    }
    dimension_key = dimension.lower()
    dim_expr = valid_dims.get(dimension_key)
    if not dim_expr:
        return json.dumps({"error": f"Invalid dimension '{dimension}'. Available options: {list(valid_dims.keys())}"})

    dimension = dimension_key
    metric = metric.lower()
    if metric not in {"net_sales_inr", "gross_profit_inr", "total_packs", "gross_margin_pct", "avg_discount_pct"}:
        return json.dumps({"error": "Invalid metric. Choose net_sales_inr, gross_profit_inr, total_packs, gross_margin_pct, or avg_discount_pct."})

    try:
        if start_date:
            date.fromisoformat(start_date)
        if end_date:
            date.fromisoformat(end_date)
    except ValueError:
        return json.dumps({"error": "Date filters must be valid ISO dates in YYYY-MM-DD format."})
    if start_date and end_date and start_date > end_date:
        return json.dumps({"error": "start_date must be on or before end_date."})

    filters = []
    params = []
    if start_date:
        filters.append("f.invoice_date >= ?")
        params.append(start_date)
    if end_date:
        filters.append("f.invoice_date <= ?")
        params.append(end_date)
    if therapeutic_area:
        filters.append("s.therapeutic_area = ?")
        params.append(therapeutic_area)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    sort_mapping = {
        'net_sales_inr': 'net_sales_inr DESC',
        'gross_profit_inr': 'gross_profit_inr DESC',
        'total_packs': 'total_packs DESC',
        'gross_margin_pct': 'gross_margin_pct DESC',
        'avg_discount_pct': 'avg_discount_pct DESC'
    }
    order_clause = sort_mapping[metric]

    sql = f"""
    SELECT 
        {dim_expr} AS {dimension},
        count(*) AS invoice_lines,
        sum(f.quantity_packs) AS total_packs,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS gross_margin_pct,
        round(100 * sum(f.discount_inr) / nullif(sum(f.gross_sales_inr), 0), 2) AS avg_discount_pct
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_customer c USING(customer_id)
    JOIN dim_date d ON d.date = f.invoice_date
    {where_clause}
    GROUP BY 1
    ORDER BY {order_clause}
    LIMIT {limit}
    """
    return json.dumps(_run_query(sql, params), indent=2)

# Tool 4: get_top_products
@mcp.tool()
async def get_top_products(
    top_n: int = Field(10, description="Number of top products to return (1-50).", ge=1, le=50),
    by: str = Field("net_sales_inr", description="Ranking metric: 'net_sales_inr', 'gross_profit_inr', or 'quantity_packs'."),
    therapeutic_area: str = Field("", description="Optional filter by therapeutic area (e.g. 'Cardiology', 'Oncology')."),
    fiscal_year: str = Field("", description="Optional fiscal year filter (e.g. 'FY24-25', 'FY23-24').")
) -> str:
    """Identifies the highest-performing pharmaceutical products and brand variants ranked by sales revenue, gross margin, or pack volume."""
    filters = []
    params = []
    if therapeutic_area:
        filters.append("s.therapeutic_area = ?")
        params.append(therapeutic_area)
    if fiscal_year:
        filters.append("d.fiscal_year = ?")
        params.append(fiscal_year)
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    sort_columns = {
        "net_sales_inr": "net_sales_inr",
        "gross_profit_inr": "gross_profit_inr",
        "quantity_packs": "quantity_packs",
    }
    sort_col = sort_columns.get(by.lower())
    if not sort_col:
        return json.dumps({"error": "by must be net_sales_inr, gross_profit_inr, or quantity_packs."})

    sql = f"""
    SELECT 
        s.sku_id,
        s.brand_name,
        s.generic_name,
        s.therapeutic_area,
        s.dosage_form,
        s.strength,
        s.brand_type,
        sum(f.quantity_packs) AS quantity_packs,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS gross_margin_pct
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_date d ON d.date = f.invoice_date
    {where_clause}
    GROUP BY s.sku_id, s.brand_name, s.generic_name, s.therapeutic_area, s.dosage_form, s.strength, s.brand_type
    ORDER BY {sort_col} DESC
    LIMIT {top_n}
    """
    return json.dumps(_run_query(sql, params), indent=2)

# Tool 5: analyze_margin_erosion
@mcp.tool()
async def analyze_margin_erosion(
    therapeutic_area: str = Field("", description="Optional therapeutic area filter (e.g. 'Oncology', 'Anti-infectives')."),
    channel: str = Field("", description="Optional channel filter ('Hospital', 'Wholesale', 'Retail', 'Online', 'Institutional').")
) -> str:
    """Tracks year-over-year gross profit margin % erosion and average trade discount % expansion across drug classes."""
    filters = []
    params = []
    if therapeutic_area:
        filters.append("s.therapeutic_area = ?")
        params.append(therapeutic_area)
    if channel:
        filters.append("c.channel = ?")
        params.append(channel)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    sql = f"""
    SELECT 
        d.year,
        s.therapeutic_area,
        count(*) AS total_invoices,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS margin_pct,
        round(100 * sum(f.discount_inr) / nullif(sum(f.gross_sales_inr), 0), 2) AS avg_discount_pct
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_customer c USING(customer_id)
    JOIN dim_date d ON d.date = f.invoice_date
    {where}
    GROUP BY d.year, s.therapeutic_area
    ORDER BY s.therapeutic_area, d.year
    """
    return json.dumps(_run_query(sql, params), indent=2)

# Tool 6: analyze_discount_behavior
@mcp.tool()
async def analyze_discount_behavior(
    min_discount_pct: float = Field(10.0, description="Minimum trade discount percentage threshold (e.g. 10.0 or 15.0).", ge=0.0, le=50.0),
    limit: int = Field(50, description="Maximum number of customer/segment rows to return.", ge=1, le=100)
) -> str:
    """Discovers which customer channels and therapeutic areas are receiving unusually deep trade discounts (> threshold %)."""
    sql = f"""
    SELECT 
        c.customer_type,
        c.channel,
        s.therapeutic_area,
        s.brand_name,
        count(*) as discounted_invoices,
        round(avg(f.trade_discount_pct), 2) AS avg_trade_discount_pct,
        round(max(f.trade_discount_pct), 2) AS max_trade_discount_pct,
        round(sum(f.discount_inr), 2) AS total_discount_given_inr,
        round(sum(f.net_sales_inr), 2) AS net_sales_realized_inr
    FROM fact_sales f
    JOIN dim_customer c USING(customer_id)
    JOIN dim_sku s USING(sku_id)
    WHERE f.trade_discount_pct >= ?
    GROUP BY c.customer_type, c.channel, s.therapeutic_area, s.brand_name
    ORDER BY total_discount_given_inr DESC
    LIMIT {limit}
    """
    return json.dumps(_run_query(sql, [min_discount_pct]), indent=2)

# Tool 7: analyze_returns_and_spoilage
@mcp.tool()
async def analyze_returns_and_spoilage(
    group_by: str = Field(
        "return_reason",
        description="Dimension to group returns: 'return_reason', 'therapeutic_area', 'channel', or 'region'."
    )
) -> str:
    """Audits returned drug shipments, total credit notes issued, and calculates correlation with proximity to expiry date."""
    valid = {
        "return_reason": "f.return_reason",
        "therapeutic_area": "s.therapeutic_area",
        "channel": "c.channel",
        "region": "c.region"
    }
    col = valid.get(group_by.lower())
    if not col:
        return json.dumps({"error": f"Invalid group_by. Choose one of: {', '.join(valid)}."})
    
    sql = f"""
    SELECT 
        {col} AS category,
        count(*) AS return_invoices,
        round(sum(f.credit_note_inr), 2) AS total_credit_note_inr,
        round(avg(date_diff('day', f.invoice_date, f.expiry_date)), 1) AS avg_days_to_expiry_at_sale
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_customer c USING(customer_id)
    WHERE f.returned = true
    GROUP BY 1
    ORDER BY total_credit_note_inr DESC
    """
    return json.dumps(_run_query(sql), indent=2)

# Tool 8: expiry_risk_profile
@mcp.tool()
async def expiry_risk_profile(
    max_days_to_expiry: int = Field(90, description="Threshold of days between invoice date and batch expiry (e.g. 60, 90, 180).", ge=1, le=365)
) -> str:
    """Flags batch sales invoiced uncomfortably close to drug expiry date and checks return incidence rates."""
    sql = """
    SELECT 
        s.brand_name,
        s.therapeutic_area,
        count(*) AS near_expiry_sales_count,
        round(avg(date_diff('day', f.invoice_date, f.expiry_date)), 1) AS avg_days_left,
        round(sum(f.net_sales_inr), 2) AS at_risk_sales_inr,
        sum(CASE WHEN f.returned = true THEN 1 ELSE 0 END) AS returned_count,
        round(100.0 * sum(CASE WHEN f.returned = true THEN 1 ELSE 0 END) / count(*), 2) AS return_rate_pct
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    WHERE date_diff('day', f.invoice_date, f.expiry_date) <= ?
    GROUP BY s.brand_name, s.therapeutic_area
    ORDER BY near_expiry_sales_count DESC
    LIMIT 25
    """
    return json.dumps(_run_query(sql, [max_days_to_expiry]), indent=2)

# Tool 9: get_customer_profile
@mcp.tool()
async def get_customer_profile(
    customer_id: str = Field(..., description="Unique customer ID (e.g. 'CUST000001', 'CUST000042').")
) -> str:
    """Fetches end-to-end purchasing telemetry for a single pharmacy, hospital, or wholesale distributor."""
    cust = _run_query("SELECT * FROM dim_customer WHERE customer_id = ?", [customer_id])
    if not cust:
        return json.dumps({"error": f"Customer '{customer_id}' not found in database."})
        
    stats = _run_query("""
    SELECT 
        count(*) as total_orders,
        min(invoice_date) as first_order_date,
        max(invoice_date) as last_order_date,
        round(sum(net_sales_inr), 2) as lifetime_net_sales_inr,
        round(sum(gross_profit_inr), 2) as lifetime_gross_profit_inr,
        round(avg(trade_discount_pct), 2) as avg_discount_pct,
        sum(CASE WHEN returned = true THEN 1 ELSE 0 END) as returned_orders
    FROM fact_sales
    WHERE customer_id = ?
    """, [customer_id])
    
    top_skus = _run_query("""
    SELECT s.brand_name, sum(f.quantity_packs) as packs, round(sum(f.net_sales_inr), 2) as sales_inr
    FROM fact_sales f JOIN dim_sku s USING(sku_id)
    WHERE f.customer_id = ?
    GROUP BY s.brand_name ORDER BY sales_inr DESC LIMIT 5
    """, [customer_id])
    
    return json.dumps({
        "customer": cust[0],
        "lifetime_stats": stats[0] if stats else {},
        "top_purchased_brands": top_skus
    }, indent=2)

# Tool 10: identify_dormant_customers
@mcp.tool()
async def identify_dormant_customers(
    days_since_last_order: int = Field(180, description="Dormancy cutoff in elapsed days without placing an invoice.", ge=30, le=1000),
    reference_date: str = Field("2026-10-27", description="Reference date for calculating dormancy (defaults to end of dataset: 2026-10-27)."),
    limit: int = Field(25, description="Number of dormant accounts to return.", ge=1, le=100)
) -> str:
    """Discovers high lifetime-value customers who have lapsed or stopped ordering for more than X days."""
    sql = """
    SELECT 
        c.customer_id,
        c.customer_type,
        c.region,
        c.state,
        c.channel,
        max(f.invoice_date) AS last_order_date,
        date_diff('day', max(f.invoice_date), CAST(? AS DATE)) AS days_inactive,
        round(sum(f.net_sales_inr), 2) AS lifetime_net_sales_inr
    FROM fact_sales f
    JOIN dim_customer c USING(customer_id)
    GROUP BY c.customer_id, c.customer_type, c.region, c.state, c.channel
    HAVING date_diff('day', max(f.invoice_date), CAST(? AS DATE)) >= ?
    ORDER BY lifetime_net_sales_inr DESC
    LIMIT ?
    """
    return json.dumps(_run_query(sql, [reference_date, reference_date, days_since_last_order, limit]), indent=2)

# Tool 11: benchmark_sales_reps
@mcp.tool()
async def benchmark_sales_reps(
    region: str = Field("", description="Optional geographical region filter ('North', 'South', 'East', 'West', 'Central')."),
    limit: int = Field(25, description="Number of reps to benchmark (1-100).", ge=1, le=100)
) -> str:
    """Ranks and benchmarks sales reps on gross revenue, margin delivery, and discount adherence."""
    filters = []
    params = []
    if region:
        filters.append("r.region = ?")
        params.append(region)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    
    sql = f"""
    SELECT 
        r.sales_rep_id,
        r.region,
        r.state,
        r.rep_grade,
        count(distinct f.customer_id) AS active_customers,
        count(*) AS total_transactions,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS margin_pct,
        round(100 * sum(f.discount_inr) / nullif(sum(f.gross_sales_inr), 0), 2) AS avg_discount_pct
    FROM fact_sales f
    JOIN dim_customer c USING(customer_id)
    JOIN dim_rep r ON r.sales_rep_id = c.sales_rep_id
    {where}
    GROUP BY r.sales_rep_id, r.region, r.state, r.rep_grade
    ORDER BY net_sales_inr DESC
    LIMIT {limit}
    """
    return json.dumps(_run_query(sql, params), indent=2)

# Tool 12: regional_growth_matrix
@mcp.tool()
async def regional_growth_matrix() -> str:
    """Returns a cross-tabulated matrix of sales volume and gross margin across all 5 Indian regions by calendar year."""
    sql = """
    SELECT 
        c.region,
        d.year,
        round(sum(f.net_sales_inr), 2) as net_sales_inr,
        round(sum(f.gross_profit_inr), 2) as gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) as gross_margin_pct
    FROM fact_sales f
    JOIN dim_customer c USING(customer_id)
    JOIN dim_date d ON d.date = f.invoice_date
    GROUP BY c.region, d.year
    ORDER BY c.region, d.year
    """
    return json.dumps(_run_query(sql), indent=2)

# Tool 13: cold_chain_compliance_analysis
@mcp.tool()
async def cold_chain_compliance_analysis() -> str:
    """Compares revenue, return rates, and credit notes between temperature-controlled biologics (cold chain) and ambient drugs."""
    sql = """
    SELECT 
        CASE WHEN s.cold_chain = 1 THEN 'Cold Chain (Temp Controlled)' ELSE 'Ambient' END AS storage_condition,
        count(*) AS total_shipments,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS gross_margin_pct,
        sum(CASE WHEN f.returned = true THEN 1 ELSE 0 END) AS returns_count,
        round(100.0 * sum(CASE WHEN f.returned = true THEN 1 ELSE 0 END) / count(*), 2) AS return_rate_pct,
        round(sum(f.credit_note_inr), 2) AS credit_notes_inr
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    GROUP BY 1
    """
    return json.dumps(_run_query(sql), indent=2)

# Tool 14: seasonal_trends_forecast
@mcp.tool()
async def seasonal_trends_forecast(
    therapeutic_area: str = Field("Anti-infectives", description="Therapeutic area to analyze (e.g. 'Anti-infectives', 'Pain & Respiratory').")
) -> str:
    """Analyzes 5-year historical monthly seasonality curves and volume swings to support supply-chain forecasting."""
    sql = """
    SELECT 
        month,
        month_name,
        round(avg(monthly_sales), 2) as avg_monthly_sales_inr,
        round(min(monthly_sales), 2) as min_monthly_sales_inr,
        round(max(monthly_sales), 2) as max_monthly_sales_inr
    FROM (
        SELECT 
            d.year AS year,
            d.month AS month,
            d.month_name AS month_name,
            sum(f.net_sales_inr) as monthly_sales
        FROM fact_sales f
        JOIN dim_sku s USING(sku_id)
        JOIN dim_date d ON d.date = f.invoice_date
        WHERE s.therapeutic_area = ?
        GROUP BY d.year, d.month, d.month_name
    ) sub
    GROUP BY month, month_name
    ORDER BY month
    """
    return json.dumps(_run_query(sql, [therapeutic_area]), indent=2)

# Tool 15: price_volume_elasticity
@mcp.tool()
async def price_volume_elasticity(
    sku_id: str = Field(..., description="Unique product SKU ID (e.g. 'SKU001', 'SKU010').")
) -> str:
    """Estimates year-over-year price-volume elasticity from annual realized price and pack volume changes."""
    sku_info = _run_query("SELECT * FROM dim_sku WHERE sku_id = ?", [sku_id])
    if not sku_info:
        return json.dumps({"error": f"SKU '{sku_id}' not found in database."})
        
    sql = """
    SELECT 
        d.year,
        round(sum(f.net_sales_inr) / nullif(sum(f.quantity_packs), 0), 2) AS realized_price_per_pack,
        round(avg(f.trade_discount_pct), 2) AS avg_discount_pct,
        sum(f.quantity_packs) AS total_packs_sold,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr
    FROM fact_sales f
    JOIN dim_date d ON d.date = f.invoice_date
    WHERE f.sku_id = ?
    GROUP BY d.year
    ORDER BY d.year
    """
    trajectory = _run_query(sql, [sku_id])
    for index, row in enumerate(trajectory):
        if index == 0:
            row["price_change_pct"] = None
            row["volume_change_pct"] = None
            row["elasticity_estimate"] = None
            continue
        previous = trajectory[index - 1]
        previous_price = previous["realized_price_per_pack"]
        previous_volume = previous["total_packs_sold"]
        price_change = 100 * (row["realized_price_per_pack"] - previous_price) / abs(previous_price) if previous_price else None
        volume_change = 100 * (row["total_packs_sold"] - previous_volume) / abs(previous_volume) if previous_volume else None
        row["price_change_pct"] = round(price_change, 2) if price_change is not None else None
        row["volume_change_pct"] = round(volume_change, 2) if volume_change is not None else None
        row["elasticity_estimate"] = round(volume_change / price_change, 3) if price_change not in (None, 0) and volume_change is not None else None
    return json.dumps({
        "sku": sku_info[0],
        "yearly_trajectory": trajectory,
        "method": "Year-over-year percentage change in packs divided by percentage change in realized net sales per pack (net sales divided by packs).",
        "caveat": "This is a descriptive estimate from observational annual data, not a causal demand elasticity; mix, discounts, availability, and seasonality may confound it.",
    }, indent=2)

# Tool 16: generate_chart_ready_dataset
@mcp.tool()
async def generate_chart_ready_dataset(
    chart_type: str = Field(
        "bar",
        description="Initial view to select: 'bar', 'line', or 'pie'. A table and compatible chart alternatives are returned for the same data."
    ),
    x_dimension: str = Field(
        "therapeutic_area",
        description="Dimension for X-axis / categories ('therapeutic_area', 'channel', 'region', 'year', 'month', 'fiscal_year')."
    ),
    y_metric: str = Field(
        "net_sales_inr",
        description="Metric for Y-axis ('net_sales_inr', 'gross_profit_inr', 'gross_margin_pct', 'avg_discount_pct', 'total_packs')."
    ),
    secondary_metric: str = Field(
        "",
        description="Optional secondary metric for dual-axis charts (e.g. 'gross_margin_pct')."
    ),
    filter_dimension: str = Field("", description="Optional filter column (e.g. 'therapeutic_area' or 'region')."),
    filter_value: str = Field("", description="Optional filter value (e.g. 'Oncology' or 'North')."),
    top_n: int = Field(12, description="Max categories to return.", ge=1, le=50),
    target_format: str = Field("echarts", description="Chart specification format: echarts, vega-lite, or plotly."),
    theme: str = Field("clinical_light", description="Theme from pharma://design/color-themes: clinical_light, executive_dark, or color_vision_safe."),
    currency_unit: str = Field("crore", description="Human-facing scale for INR chart values: crore or lakh. Million/billion scales are not available."),
    ctx: Context = None,
) -> str:
    """Returns chart data plus a view switcher with bar, line, pie, and table alternatives built from the same rows."""
    valid_dims = {
        'therapeutic_area': 's.therapeutic_area',
        'channel': 'c.channel',
        'region': 'c.region',
        'year': 'd.year',
        'month': 'd.month_name',
        'fiscal_year': 'd.fiscal_year'
    }
    valid_metrics = {'net_sales_inr', 'gross_profit_inr', 'gross_margin_pct', 'avg_discount_pct', 'total_packs'}
    x_key = x_dimension.lower()
    metric_key = y_metric.lower()
    secondary_key = secondary_metric.lower() if secondary_metric else ""
    filter_key = filter_dimension.lower() if filter_dimension else ""
    chart_key = chart_type.lower()
    format_key = target_format.lower()
    theme_key = theme.lower()
    currency_key = currency_unit.lower()
    if x_key not in valid_dims:
        return json.dumps({"error": f"Invalid x_dimension. Choose one of: {', '.join(valid_dims)}."})
    if metric_key not in valid_metrics:
        return json.dumps({"error": f"Invalid y_metric. Choose one of: {', '.join(sorted(valid_metrics))}."})
    if secondary_key and secondary_key not in valid_metrics:
        return json.dumps({"error": f"Invalid secondary_metric. Choose one of: {', '.join(sorted(valid_metrics))}."})
    if secondary_key == metric_key:
        return json.dumps({"error": "secondary_metric must differ from y_metric."})
    if chart_key not in {"bar", "line", "pie"}:
        return json.dumps({"error": "chart_type must be bar, line, or pie."})
    if format_key not in {"echarts", "vega-lite", "plotly"}:
        return json.dumps({"error": "target_format must be echarts, vega-lite, or plotly."})
    if currency_key not in {"lakh", "crore"}:
        return json.dumps({"error": "currency_unit must be lakh or crore."})
    theme_tokens = {
        "clinical_light": {"background": "#FFFFFF", "text": "#1E293B", "grid": "#E2E8F0", "colors": ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00", "#332288", "#999933"]},
        "executive_dark": {"background": "#0F172A", "text": "#F8FAFC", "grid": "#334155", "colors": ["#56B4E9", "#00C08B", "#F28E2B", "#E78AC3", "#8DD3C7", "#FFD166", "#A78BFA", "#F87171"]},
        "color_vision_safe": {"background": "#FFFFFF", "text": "#1E293B", "grid": "#CBD5E1", "colors": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]},
    }
    if theme_key not in theme_tokens:
        return json.dumps({"error": f"Invalid theme. Choose one of: {', '.join(theme_tokens)}."})
    if filter_key and filter_key not in valid_dims:
        return json.dumps({"error": f"Invalid filter_dimension. Choose one of: {', '.join(valid_dims)}."})
    if bool(filter_key) != bool(filter_value):
        return json.dumps({"error": "Provide both filter_dimension and filter_value, or leave both empty."})
    if chart_key == "pie" and secondary_key:
        return json.dumps({"error": "Pie charts support one metric; remove secondary_metric or select bar/line."})

    dim_col = valid_dims[x_key]
    where_clause = ""
    params = []
    if filter_key:
        where_clause = f"WHERE {valid_dims[filter_key]} = ?"
        params.append(filter_value)
    sort_clause = "month_order ASC" if x_key == "month" else ("category ASC" if x_key in {"year", "fiscal_year"} else f"{metric_key} DESC NULLS LAST")

    sql = f"""
    SELECT 
        CAST({dim_col} AS VARCHAR) AS category,
        min(d.month) AS month_order,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS gross_margin_pct,
        round(100 * sum(f.discount_inr) / nullif(sum(f.gross_sales_inr), 0), 2) AS avg_discount_pct,
        sum(f.quantity_packs) AS total_packs,
        count(*) OVER () AS _category_count
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_customer c USING(customer_id)
    JOIN dim_date d ON d.date = f.invoice_date
    {where_clause}
    GROUP BY 1
    ORDER BY {sort_clause}
    LIMIT {top_n}
    """
    if ctx:
        await ctx.info(f"Building {format_key} alternatives for {x_key} grouped data")
    raw_rows = _run_query(sql, params)
    total_categories = raw_rows[0]["_category_count"] if raw_rows else 0
    for row in raw_rows:
        row.pop("month_order", None)
        row.pop("_category_count", None)
    categories = [str(row["category"]) for row in raw_rows]
    tokens = theme_tokens[theme_key]
    category_colors = {
        "therapeutic_area": {"Oncology": "#7C3AED", "Anti-infectives": "#0D9488", "Cardiology": "#E11D48", "Diabetes": "#0284C7", "Dermatology": "#D97706", "Pain & Respiratory": "#059669", "Nutraceuticals": "#647A00"},
        "channel": {"Hospital": "#4338CA", "Wholesale": "#0284C7", "Retail": "#0D9488", "Online": "#B45309", "Institutional": "#64748B"},
    }.get(x_key, {})
    colors = [category_colors.get(category, tokens["colors"][index % len(tokens["colors"])]) for index, category in enumerate(categories)]
    currency_divisor = 100_000 if currency_key == "lakh" else 10_000_000
    metric_fields = {
        "net_sales_inr": f"net_sales_{currency_key}",
        "gross_profit_inr": f"gross_profit_{currency_key}",
        "gross_margin_pct": "gross_margin_pct",
        "avg_discount_pct": "avg_discount_pct",
        "total_packs": "total_packs",
    }
    unit_label = "₹ Lakhs (L)" if currency_key == "lakh" else "₹ Crores (Cr)"
    display_rows = []
    for row in raw_rows:
        display_row = {"category": row["category"]}
        for metric, field_name in metric_fields.items():
            value = row.get(metric)
            display_row[field_name] = round(value / currency_divisor, 3) if metric.endswith("_inr") and value is not None else value
        display_rows.append(display_row)
    values = {metric: [row.get(metric_fields[metric]) for row in display_rows] for metric in valid_metrics}
    metrics = [metric_key] + ([secondary_key] if secondary_key else [])
    metric_labels = {
        "net_sales_inr": f"Net sales ({'₹ L' if currency_key == 'lakh' else '₹ Cr'})",
        "gross_profit_inr": f"Gross profit ({'₹ L' if currency_key == 'lakh' else '₹ Cr'})",
        "gross_margin_pct": "Gross margin (%)",
        "avg_discount_pct": "Weighted discount (%)",
        "total_packs": "Packs sold",
    }
    selected_kind = chart_key

    def format_spec(kind: str) -> dict[str, Any]:
        if format_key == "echarts":
            if kind == "pie":
                pie_data = [{"name": category, "value": values[metric_key][index], "itemStyle": {"color": colors[index]}} for index, category in enumerate(categories)]
                return {"title": {"text": f"{metric_labels[metric_key]} by {x_key.replace('_', ' ').title()}", "left": "center"}, "backgroundColor": tokens["background"], "textStyle": {"color": tokens["text"], "fontFamily": "Inter, system-ui, sans-serif"}, "tooltip": {"trigger": "item"}, "legend": {"type": "scroll", "bottom": 0}, "series": [{"name": metric_labels[metric_key], "type": "pie", "radius": ["38%", "68%"], "avoidLabelOverlap": True, "data": pie_data}]}
            chart_series = []
            for index, metric in enumerate(metrics):
                item = {"name": metric_labels[metric], "type": kind, "data": values[metric], "itemStyle": {"color": tokens["colors"][index % len(tokens["colors"])]}, "yAxisIndex": index}
                if kind == "line":
                    item.update({"smooth": False, "showSymbol": len(categories) <= 24, "symbolSize": 7, "lineStyle": {"width": 3}, "areaStyle": {"opacity": 0.08}})
                chart_series.append(item)
            axes = [{"type": "value", "name": metric_labels[metrics[0]], "splitLine": {"lineStyle": {"color": tokens["grid"]}}}]
            for axis, metric in zip(axes, metrics):
                if metric.endswith("_pct"):
                    axis["axisLabel"] = {"formatter": "{value}%"}
                elif metric.endswith("_inr"):
                    axis["axisLabel"] = {"formatter": "₹{value} L" if currency_key == "lakh" else "₹{value} Cr"}
            if secondary_key:
                axes.append({"type": "value", "name": metric_labels[secondary_key], "position": "right", "splitLine": {"show": False}})
                if secondary_key.endswith("_pct"):
                    axes[-1]["axisLabel"] = {"formatter": "{value}%"}
                elif secondary_key.endswith("_inr"):
                    axes[-1]["axisLabel"] = {"formatter": "₹{value} L" if currency_key == "lakh" else "₹{value} Cr"}
            return {"title": {"text": f"{metric_labels[metric_key]} by {x_key.replace('_', ' ').title()}", "left": "left"}, "backgroundColor": tokens["background"], "textStyle": {"color": tokens["text"], "fontFamily": "Inter, system-ui, sans-serif"}, "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow" if kind == "bar" else "line"}}, "legend": {"top": 30}, "grid": {"left": 70, "right": 75 if secondary_key else 30, "bottom": 80, "containLabel": True}, "xAxis": {"type": "category", "data": categories, "axisLabel": {"rotate": 30 if any(len(category) > 12 for category in categories) else 0, "hideOverlap": True}}, "yAxis": axes, "series": chart_series}

        if format_key == "vega-lite":
            title = f"{metric_labels[metric_key]} by {x_key.replace('_', ' ').title()}"
            if kind == "pie":
                return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "title": title, "data": {"values": display_rows}, "mark": {"type": "arc", "innerRadius": 52, "tooltip": True}, "encoding": {"theta": {"field": metric_fields[metric_key], "type": "quantitative", "title": metric_labels[metric_key]}, "color": {"field": "category", "type": "nominal", "scale": {"domain": categories, "range": colors}}, "tooltip": [{"field": "category", "type": "nominal"}, {"field": metric_fields[metric_key], "type": "quantitative", "title": metric_labels[metric_key]}]}, "config": {"background": tokens["background"], "view": {"stroke": None}, "axis": {"labelColor": tokens["text"], "titleColor": tokens["text"], "gridColor": tokens["grid"]}}}
            layers = []
            for index, metric in enumerate(metrics):
                y_axis = {"orient": "left" if index == 0 else "right"}
                if metric.endswith("_pct"):
                    y_axis["labelExpr"] = "datum.value + '%'"
                field_name = metric_fields[metric]
                layers.append({"mark": {"type": kind, "point": kind == "line", "tooltip": True}, "encoding": {"x": {"field": "category", "type": "ordinal", "sort": categories, "title": x_key.replace("_", " ").title()}, "y": {"field": field_name, "type": "quantitative", "title": metric_labels[metric], "axis": y_axis}, "color": {"value": tokens["colors"][index % len(tokens["colors"])]}, "tooltip": [{"field": "category", "type": "nominal"}, {"field": field_name, "type": "quantitative", "title": metric_labels[metric]}]}})
            return {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "title": title, "data": {"values": display_rows}, "layer": layers, "resolve": {"scale": {"y": "independent"}} if secondary_key else {}, "config": {"background": tokens["background"], "view": {"stroke": None}, "axis": {"labelColor": tokens["text"], "titleColor": tokens["text"], "gridColor": tokens["grid"]}}}

        if kind == "pie":
            value_format = "%{value:.1f}%" if metric_key in {"gross_margin_pct", "avg_discount_pct"} else ("%{value:,.0f} packs" if metric_key == "total_packs" else f"₹%{{value:,.2f}} {'L' if currency_key == 'lakh' else 'Cr'}")
            data = [{"type": "pie", "labels": categories, "values": values[metric_key], "hole": 0.42, "marker": {"colors": colors}, "textinfo": "label+percent", "hovertemplate": f"%{{label}}<br>{value_format}<extra></extra>"}]
        else:
            data = []
            for index, metric in enumerate(metrics):
                trace = {"type": "scatter" if kind == "line" else "bar", "name": metric_labels[metric], "x": categories, "y": values[metric], "marker": {"color": tokens["colors"][index % len(tokens["colors"])]}, "yaxis": "y" if index == 0 else "y2"}
                if kind == "line":
                    trace.update({"mode": "lines+markers", "line": {"width": 3}})
                data.append(trace)
        y_axis = {"title": metric_labels[metric_key], "gridcolor": tokens["grid"]}
        if metric_key.endswith("_pct"):
            y_axis["ticksuffix"] = "%"
        elif metric_key.endswith("_inr"):
            y_axis["ticksuffix"] = " L" if currency_key == "lakh" else " Cr"
        layout = {"title": f"{metric_labels[metric_key]} by {x_key.replace('_', ' ').title()}", "paper_bgcolor": tokens["background"], "plot_bgcolor": tokens["background"], "font": {"family": "Inter, system-ui, sans-serif", "color": tokens["text"]}, "xaxis": {"title": x_key.replace("_", " ").title(), "categoryorder": "array", "categoryarray": categories, "gridcolor": tokens["grid"]}, "yaxis": y_axis, "legend": {"orientation": "h", "y": -0.2}}
        if secondary_key:
            second_axis = {"title": metric_labels[secondary_key], "overlaying": "y", "side": "right"}
            if secondary_key.endswith("_pct"):
                second_axis["ticksuffix"] = "%"
            elif secondary_key.endswith("_inr"):
                second_axis["ticksuffix"] = " L" if currency_key == "lakh" else " Cr"
            layout["yaxis2"] = second_axis
        return {"data": data, "layout": layout}

    is_time = x_key in {"year", "month", "fiscal_year"}
    pie_compatible = bool(raw_rows) and len(raw_rows) == total_categories and not is_time and total_categories <= 5 and not secondary_key and metric_key in {"net_sales_inr", "gross_profit_inr", "total_packs"}
    recommended = "line" if is_time else "bar"
    requested_chart_type = chart_key
    if (chart_key == "pie" and not pie_compatible) or (chart_key == "line" and not is_time):
        selected_kind = recommended
    alternatives = []
    for kind in ("bar", "line", "pie", "table"):
        compatible = (kind != "pie" or pie_compatible) and (kind != "line" or is_time)
        if kind == "line" and not is_time:
            reason = "Line charts connect categories without a natural order; use bar for clearer comparison."
        elif kind == "pie" and not pie_compatible:
            reason = "Pie is suitable only for a small, non-temporal part-to-whole view with one additive metric and at most five categories."
        elif kind == "table":
            reason = "Exact values and all returned measures; best for lookup or export."
        else:
            reason = "Recommended for chronological categories." if kind == "line" and is_time else ("Recommended for category comparison." if kind == "bar" and not is_time else "Suitable for a small part-to-whole comparison.")
        alternatives.append({
            "id": kind,
            "label": {"bar": "Bar", "line": "Line", "pie": "Pie", "table": "Table"}[kind],
            "compatible": compatible,
            "recommended": kind == recommended,
            "selected": kind == selected_kind,
            "reason": reason,
            "spec": {"columns": list(display_rows[0].keys()) if display_rows else [], "rows": display_rows} if kind == "table" else format_spec(kind),
        })
    return json.dumps({
        "dataset_id": f"{x_key}:{metric_key}:{secondary_key}:{filter_key}:{filter_value}:{top_n}:{currency_key}",
        "requested_chart_type": requested_chart_type,
        "selected_chart_type": selected_kind,
        "recommended_chart_type": recommended,
        "target_format": format_key,
        "theme": theme_key,
        "currency_unit": currency_key,
        "currency_unit_label": unit_label,
        "x_dimension": x_key,
        "primary_metric": metric_key,
        "secondary_metric": secondary_key or None,
        "categories": categories,
        "series": [{"name": metric_labels[metric], "metric": metric, "display_field": metric_fields[metric], "data": values[metric]} for metric in metrics],
        "metric_units": {metric: unit_label if metric.endswith("_inr") else ("packs" if metric == "total_packs" else "percent") for metric in metrics},
        "rows": raw_rows,
        "display_rows": display_rows,
        "selected_spec": format_spec(selected_kind),
        "design_tokens": {"background": tokens["background"], "text": tokens["text"], "grid": tokens["grid"], "category_colors": dict(zip(categories, colors))},
        "chart_alternatives": alternatives,
        "interaction": {"control": "Use chart_alternatives[].id as the selector value; all views use this response's same rows and metric values."},
        "data_notes": ["Raw rows retain exact INR for computation; display_rows and chart specs convert amounts to the selected lakh/crore unit.", "Do not relabel lakh/crore values as millions. Gross margin and discount percentages are weighted at the grouped level.", "Pie views are marked incompatible when the data is temporal, has too many categories, or uses a non-additive/multiple metric.", "Line views are marked incompatible without a chronological x dimension.", *([f"The requested {requested_chart_type} chart was unsuitable, so {selected_kind} is selected."] if requested_chart_type != selected_kind else [])],
    }, indent=2)


def main():
    """Runs the FastMCP server via standard IO."""
    mcp.run()

if __name__ == "__main__":
    main()
