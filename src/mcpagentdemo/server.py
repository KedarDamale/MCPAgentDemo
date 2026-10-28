"""PharmaSales FastMCP Server with comprehensive analytical tools, resources, and visualization-guided prompts."""

import json
from datetime import date
from decimal import Decimal
from typing import Any, List
from pydantic import Field

from fastmcp import FastMCP, Context
from mcpagentdemo.db import get_connection, get_db_path

mcp = FastMCP(
    name="PharmaSales Analytics MCP",
    instructions="""This server provides comprehensive analytical, operational, and data visualization capabilities over 5 years of Indian Pharmaceutical Sales data stored in DuckDB.
Access 10,000,000 sales transactions (2021-2026), 60,000 customers, 84 SKUs, 160 sales reps across 5 regions.
Every tool includes precise typing, field descriptions, and input validation.
Use the resources for schema and catalog inspection.
Use the prompts for structured analytical investigations and ready-to-render data visualization configurations (ECharts, Vega-Lite, Mermaid, and Plotly) with standardized medical/pharma color themes."""
)

def _sanitize_val(val: Any) -> Any:
    if isinstance(val, (date,)):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    return val

def _run_query(sql: str, params: list = None) -> list[dict[str, Any]]:
    con = get_connection(read_only=True)
    try:
        if params:
            cur = con.execute(sql, params)
        else:
            cur = con.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [
            {col: _sanitize_val(val) for col, val in zip(cols, row)}
            for row in rows
        ]
    finally:
        con.close()


# ==============================================================================
# RESOURCES
# ==============================================================================

@mcp.resource("pharma://schema/tables")
def get_tables_schema() -> str:
    """Returns the full database schema for all DuckDB tables and views, including column names, data types, and nullability."""
    con = get_connection(read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        schema_dict = {}
        for (table_name,) in tables:
            cols = con.execute(f"DESCRIBE {table_name}").fetchall()
            schema_dict[table_name] = [
                {"column": c[0], "type": c[1], "nullable": c[2]} for c in cols
            ]
        return json.dumps(schema_dict, indent=2)
    finally:
        con.close()

@mcp.resource("pharma://metadata/readme")
def get_dataset_metadata() -> str:
    """Returns domain business rules, GST tax reform cutoffs (Sept 2025), shelf-life formulas, and data caveats from README."""
    con = get_connection(read_only=True)
    try:
        rows = con.execute("SELECT * FROM README").fetchall()
        return json.dumps({k: v for k, v in rows}, indent=2)
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
    """Returns the standardized pharmaceutical data visualization design system, color palettes, and theme tokens."""
    design_system = {
        "design_philosophy": "Clean, high-contrast, accessible clinical aesthetic tailored for executive pharma reporting.",
        "primary_palette": {
            "navy_slate": "#1E293B",
            "teal_clinical": "#0D9488",
            "sky_blue": "#0284C7",
            "emerald_growth": "#059669",
            "rose_negative": "#E11D48",
            "amber_warning": "#D97706",
            "violet_innovative": "#7C3AED",
            "indigo_deep": "#4338CA"
        },
        "therapeutic_area_colors": {
            "Oncology": "#7C3AED",
            "Anti-infectives": "#0D9488",
            "Cardiology": "#E11D48",
            "Diabetes": "#0284C7",
            "Dermatology": "#F59E0B",
            "Pain & Respiratory": "#10B981",
            "Nutraceuticals": "#84CC16"
        },
        "channel_colors": {
            "Hospital": "#4338CA",
            "Wholesale": "#0284C7",
            "Retail": "#0D9488",
            "Online": "#F59E0B",
            "Institutional": "#6B7280"
        },
        "status_and_risk_colors": {
            "safe_high_margin": "#059669",
            "neutral_margin": "#0284C7",
            "margin_warning": "#F59E0B",
            "margin_danger": "#E11D48",
            "expired_or_returned": "#BE123C",
            "ambient_storage": "#64748B",
            "cold_chain_storage": "#0284C7"
        },
        "font_family": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "chart_background": "#FFFFFF",
        "grid_line_color": "#F1F5F9"
    }
    return json.dumps(design_system, indent=2)


# ==============================================================================
# PROMPTS (Guided Agent Workflows & Data Viz Generation)
# ==============================================================================

@mcp.prompt()
def data_visualization_dashboard(
    dashboard_topic: str = "Executive Sales & Margin Overview",
    target_format: str = "echarts"
) -> str:
    """Comprehensive design and implementation prompt that instructs the agent on creating rich charts (ECharts, Vega-Lite, Plotly, or Mermaid) with clinical color themes."""
    return f"""You are tasked with generating high-impact data visualizations for: "{dashboard_topic}".
Target Format: {target_format} (e.g., Apache ECharts, Vega-Lite, Plotly, or Mermaid.js).

### Pharma Color Theme Guidelines:
Always apply the standardized Pharma Design Palette:
- Primary Metric (Revenue / Net Sales): `#0284C7` (Sky Blue) or `#1E293B` (Slate)
- Secondary Metric (Gross Margin / Profit): `#0D9488` (Teal Clinical)
- Margin Leakage / Returns / Negative Variance: `#E11D48` (Rose Negative)
- Approaching Expiry / Credit Risk Warning: `#D97706` (Amber Warning)
- Cold Chain vs Ambient: `#0284C7` (Cold Chain) vs `#64748B` (Ambient)

### Therapeutic Area Color Coding:
- Oncology: `#7C3AED` (Violet)
- Anti-infectives: `#0D9488` (Teal)
- Cardiology: `#E11D48` (Crimson)
- Diabetes: `#0284C7` (Blue)
- Dermatology: `#F59E0B` (Amber)
- Pain & Respiratory: `#10B981` (Emerald)
- Nutraceuticals: `#84CC16` (Lime)

### Execution Steps:
1. Query necessary aggregations using `generate_chart_ready_dataset` or `get_sales_by_dimension`.
2. Format numbers clearly (INR in Lakhs ₹L or Crores ₹Cr, percentages with 1-2 decimal places).
3. If generating ECharts: output a valid, beautiful `option = {{ ... }}` JavaScript/JSON configuration with tooltip, legends, and smooth area gradients.
4. If generating Mermaid: provide valid `xychart-beta` or `flowchart TD` code blocks.
5. Provide a concise analytical commentary highlighting the key takeaways shown in the chart."""

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
When displaying charts for this investigation, highlight gross profit in `#0D9488` (Teal) and discount rate in `#E11D48` (Rose/Red) with dotted red threshold lines for discounts > 10%."""

@mcp.prompt()
def customer_churn_risk_analysis(region: str = "") -> str:
    """Investigative prompt workflow for discovering dormant pharmacy accounts and outstanding credit exposure."""
    reg_filter = f"in {region}" if region else "nationwide"
    return f"""Analyze customer churn risk, order dormancy, and outstanding credit exposure {reg_filter}.

### Recommended Investigative Sequence:
1. Call `identify_dormant_customers` with `days_since_last_order=120` to discover valuable accounts that stopped ordering.
2. Call `get_customer_profile` on top dormant accounts by lifetime net sales.
3. Call `regional_growth_matrix` to determine whether specific regional branches are driving higher dormancy.
4. Provide a ranked prioritization table with customer tier, payment terms, and recommended win-back incentives."""

@mcp.prompt()
def sales_rep_performance_review(territory_code: str = "") -> str:
    """Workflow to benchmark sales rep quotas, margin delivery, and discount discipline."""
    terr = f"for territory {territory_code}" if territory_code else "across all territories"
    return f"""Perform an end-to-end performance audit of sales representatives {terr}.

### Recommended Steps:
1. Call `benchmark_sales_reps` to rank reps by net sales volume, average margin achieved, and average discount given.
2. Identify outliers who produce high sales volume but heavily discount, destroying gross margin.
3. Call `get_sales_by_dimension` grouped by 'channel' for underperforming territories to spot channel mix weaknesses.
4. Suggest concrete coaching points, quota adjustments, and territory reassignments."""


# ==============================================================================
# TOOLS (Comprehensive Analytics, Inspections, & Data Viz Formatter)
# ==============================================================================

# Tool 1: execute_sql
@mcp.tool()
async def execute_sql(
    query: str = Field(..., description="Read-only SQL query to execute on DuckDB. Must start with SELECT, WITH, DESCRIBE, SHOW, or EXPLAIN."),
    max_rows: int = Field(500, description="Maximum number of rows to return (1-2000). Defaults to 500.", ge=1, le=2000),
    ctx: Context = None
) -> str:
    """Executes an arbitrary read-only SQL query on the DuckDB database and returns structured JSON rows.
    Tables available: fact_sales (10M rows), dim_customer (60k), dim_sku (84), dim_rep (160), dim_date (1822), README (12).
    Views available: sales_full (joined view with 47 columns), monthly_summary (60 rows).
    """
    cleaned = query.strip()
    if not cleaned.lower().startswith(("select", "with", "describe", "show", "explain")):
        return json.dumps({"error": "Security check: Only read-only queries (SELECT, WITH, DESCRIBE, SHOW, EXPLAIN) are permitted."})
    
    if ctx:
        await ctx.info(f"Executing SQL query: {cleaned[:80]}...")
        
    try:
        results = _run_query(cleaned)
        truncated = len(results) > max_rows
        return json.dumps({
            "total_matches": len(results),
            "returned_rows": min(len(results), max_rows),
            "truncated": truncated,
            "data": results[:max_rows]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# Tool 2: get_table_summary
@mcp.tool()
async def get_table_summary(
    table_name: str = Field(..., description="Target table or view name (e.g. 'fact_sales', 'dim_customer', 'dim_sku', 'monthly_summary', 'sales_full').")
) -> str:
    """Inspects table structure, row count, column definitions, and returns 5 sample records."""
    try:
        con = get_connection(read_only=True)
        count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
        schema = con.execute(f"DESCRIBE {table_name}").fetchall()
        con.close()
        sample = _run_query(f"SELECT * FROM {table_name} LIMIT 5")
        return json.dumps({
            "table": table_name,
            "total_rows": count,
            "columns": [{"name": s[0], "type": s[1]} for s in schema],
            "sample_rows": sample
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

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
    dim_expr = valid_dims.get(dimension.lower())
    if not dim_expr:
        return json.dumps({"error": f"Invalid dimension '{dimension}'. Available options: {list(valid_dims.keys())}"})

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
    order_clause = sort_mapping.get(metric, 'net_sales_inr DESC')

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

    sort_col = "net_sales_inr" if by == "net_sales_inr" else ("gross_profit_inr" if by == "gross_profit_inr" else "quantity_packs")

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
    col = valid.get(group_by.lower(), "f.return_reason")
    
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
        d.month,
        d.month_name,
        round(avg(monthly_sales), 2) as avg_monthly_sales_inr,
        round(min(monthly_sales), 2) as min_monthly_sales_inr,
        round(max(monthly_sales), 2) as max_monthly_sales_inr
    FROM (
        SELECT 
            d.year,
            d.month,
            d.month_name,
            sum(f.net_sales_inr) as monthly_sales
        FROM fact_sales f
        JOIN dim_sku s USING(sku_id)
        JOIN dim_date d ON d.date = f.invoice_date
        WHERE s.therapeutic_area = ?
        GROUP BY d.year, d.month, d.month_name
    ) sub
    GROUP BY d.month, d.month_name
    ORDER BY d.month
    """
    return json.dumps(_run_query(sql, [therapeutic_area]), indent=2)

# Tool 15: price_volume_elasticity
@mcp.tool()
async def price_volume_elasticity(
    sku_id: str = Field(..., description="Unique product SKU ID (e.g. 'SKU001', 'SKU010').")
) -> str:
    """Analyzes price elasticity by tracking annual changes in realized selling price versus pack unit volume."""
    sku_info = _run_query("SELECT * FROM dim_sku WHERE sku_id = ?", [sku_id])
    if not sku_info:
        return json.dumps({"error": f"SKU '{sku_id}' not found in database."})
        
    sql = """
    SELECT 
        d.year,
        round(avg(f.selling_price_per_pack_inr), 2) AS avg_selling_price,
        round(avg(f.trade_discount_pct), 2) AS avg_discount_pct,
        sum(f.quantity_packs) AS total_packs_sold,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr
    FROM fact_sales f
    JOIN dim_date d ON d.date = f.invoice_date
    WHERE f.sku_id = ?
    GROUP BY d.year
    ORDER BY d.year
    """
    return json.dumps({
        "sku": sku_info[0],
        "yearly_trajectory": _run_query(sql, [sku_id])
    }, indent=2)

# Tool 16: generate_chart_ready_dataset
@mcp.tool()
async def generate_chart_ready_dataset(
    chart_type: str = Field(
        "bar",
        description="Desired chart visualization: 'bar', 'line', 'pie', 'scatter', or 'stacked_bar'."
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
    top_n: int = Field(12, description="Max categories to return.", ge=1, le=50)
) -> str:
    """Produces structured chart payloads including categories, series, suggested color codes, and chart configuration snippets."""
    valid_dims = {
        'therapeutic_area': 's.therapeutic_area',
        'channel': 'c.channel',
        'region': 'c.region',
        'year': 'd.year',
        'month': 'd.month_name',
        'fiscal_year': 'd.fiscal_year'
    }
    dim_col = valid_dims.get(x_dimension.lower(), 's.therapeutic_area')

    where_clause = ""
    params = []
    if filter_dimension and filter_value and filter_dimension in valid_dims:
        where_clause = f"WHERE {valid_dims[filter_dimension]} = ?"
        params.append(filter_value)

    sql = f"""
    SELECT 
        {dim_col} AS category,
        round(sum(f.net_sales_inr), 2) AS net_sales_inr,
        round(sum(f.gross_profit_inr), 2) AS gross_profit_inr,
        round(100 * sum(f.gross_profit_inr) / nullif(sum(f.net_sales_inr), 0), 2) AS gross_margin_pct,
        round(100 * sum(f.discount_inr) / nullif(sum(f.gross_sales_inr), 0), 2) AS avg_discount_pct,
        sum(f.quantity_packs) AS total_packs
    FROM fact_sales f
    JOIN dim_sku s USING(sku_id)
    JOIN dim_customer c USING(customer_id)
    JOIN dim_date d ON d.date = f.invoice_date
    {where_clause}
    GROUP BY 1
    ORDER BY {y_metric if y_metric in ['net_sales_inr', 'gross_profit_inr', 'gross_margin_pct', 'avg_discount_pct', 'total_packs'] else 'net_sales_inr'} DESC
    LIMIT {top_n}
    """
    raw_rows = _run_query(sql, params)

    categories = [str(r["category"]) for r in raw_rows]
    primary_series = [r.get(y_metric, 0) for r in raw_rows]
    secondary_series = [r.get(secondary_metric, 0) for r in raw_rows] if secondary_metric else []

    theme_colors = {
        "net_sales_inr": "#0284C7",
        "gross_profit_inr": "#0D9488",
        "gross_margin_pct": "#10B981",
        "avg_discount_pct": "#E11D48",
        "total_packs": "#6366F1"
    }

    result = {
        "chart_type": chart_type,
        "x_dimension": x_dimension,
        "y_metric": y_metric,
        "categories": categories,
        "series": [
            {
                "name": y_metric.replace("_", " ").title(),
                "color": theme_colors.get(y_metric, "#0284C7"),
                "data": primary_series
            }
        ],
        "design_tokens": {
            "font_family": "Inter, sans-serif",
            "background": "#FFFFFF",
            "grid_color": "#F1F5F9"
        }
    }
    if secondary_metric:
        result["series"].append({
            "name": secondary_metric.replace("_", " ").title(),
            "color": theme_colors.get(secondary_metric, "#E11D48"),
            "data": secondary_series,
            "yAxisIndex": 1
        })
    return json.dumps(result, indent=2)


def main():
    """Runs the FastMCP server via standard IO."""
    mcp.run()

if __name__ == "__main__":
    main()
