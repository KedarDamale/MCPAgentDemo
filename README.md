# PharmaSales Analytics MCP

A FastMCP server that gives connected agents read-only access to a synthetic Indian pharmaceutical sales dataset in DuckDB. It exposes the database schema and business context as MCP resources, reusable analytics and visualization prompts, and tools for exploration, profiling, sales analysis, cohort retention, anomaly screening, and chart generation.

The dataset is synthetic. It contains no real company, customer, or patient records and must not be used to support clinical or real-world market claims.

## Run locally

Requires Python 3.12 or newer and `uv`.

```bash
uv sync
uv run mcpagentdemo
```

The command above starts the MCP server over stdio for local MCP clients. To run the hosted HTTP app locally:

```bash
uv run uvicorn mcpagentdemo.app:app --host 127.0.0.1 --port 8000
```

The Streamable HTTP endpoint is `/mcp`. For a deployed Vercel domain, connect using `https://<your-deployment-domain>/mcp`.

## MCP resources

| URI | Purpose |
| --- | --- |
| `pharma://context/guide` | Grain, join keys, metric definitions, assumptions, and safe-analysis guidance for agents |
| `pharma://schema/tables` | Tables, views, columns, types, and nullability |
| `pharma://table/{table_name}` | Schema, row count, and sample rows for a selected table or view |
| `pharma://metadata/readme` | Dataset provenance, business rules, and caveats |
| `pharma://catalog/skus` | SKU product catalog |
| `pharma://summary/monthly` | Monthly KPI summary |
| `pharma://design/color-themes` | Clinical light, executive dark, color-vision-safe themes, semantic colors, and chart rules |

## Analytics tools

**Explore and validate data:** `list_database_objects`, `get_table_summary`, `profile_table`, `data_quality_audit`, `execute_sql`.

**Sales and operations:** `get_sales_by_dimension`, `get_top_products`, `compare_sales_periods`, `sales_driver_decomposition`, `product_sales_pareto`, `discount_scenario_simulator`, `basket_affinity_analysis`, `analyze_margin_erosion`, `analyze_discount_behavior`, `analyze_returns_and_spoilage`, `expiry_risk_profile`, `cold_chain_compliance_analysis`, `regional_growth_matrix`, `seasonal_trends_forecast`, `price_volume_elasticity`.

**Customer and team:** `get_customer_profile`, `identify_dormant_customers`, `customer_cohort_retention`, `customer_rfm_segments`, `benchmark_sales_reps`.

**Forecasting, prediction, and simulation:** `forecast_sales` (seasonal-naive baseline), `compare_forecast_models` (five alternatives on a shared rolling holdout), `forecast_accuracy_monitor` (empirical interval coverage and error drift), `forecast_numeric_series` (any catalog-backed date/numeric series, with optional group forecasts), `predict_numeric_regression` (numeric outcome plus selected numeric features and time holdout), `reconciled_sales_forecast` (group forecasts that add up to the parent), `sales_driver_scenario_forecast` (low/base/high cases from supplied price, discount, volume, and unit-cost assumptions), `customer_lapse_risk_model` (180-day inactivity ranking), `return_risk_model` (invoice-line return ranking), `price_volume_elasticity` (descriptive SKU price/volume trajectory), `discount_scenario_simulator` (same-volume arithmetic), and `monte_carlo_discount_simulation` (ranges under supplied elasticity assumptions).

**Time series and visualization:** `scan_monthly_anomalies`, `generate_chart_ready_dataset`.

`execute_sql` accepts one read-only `SELECT`, `WITH`, `DESCRIBE`, `SHOW`, or `EXPLAIN` statement and caps returned rows. DuckDB runs with external file access disabled so SQL cannot use file-reading table functions to access files outside the database.

## Prompts

- `advanced_analytics_workflow` — dataset discovery through evidence-based reporting.
- `forecasting_simulation_workflow` — choose among model comparison, monitoring, reconciled forecasts, predictive scores, regression, and driver scenarios; report validation and uncertainty.
- `data_quality_audit_workflow` — use the quality audit tool and investigate flagged fields.
- `data_visualization_dashboard` — dashboard brief with selectable chart format and theme.
- `visualization_prompt` — focused accessible chart prompt for an audience and question.
- `margin_leakage_investigation`, `customer_churn_risk_analysis`, and `sales_rep_performance_review` — domain workflows.

The chart tool returns a bar, line, pie, and table view selector built from the same aggregated rows. It marks unsuitable chart alternatives and includes an ECharts, Vega-Lite, or Plotly specification for chart views. Its `currency_unit` option is `lakh` or `crore`; chart labels and values use ₹L or ₹Cr. Raw tool fields remain exact INR for calculations. Present monetary values only in Lakhs or Crores, never millions or billions (1 L = ₹100,000; 1 Cr = ₹10,000,000).

The generic series and regression tools can use any table or view already present in the connected DuckDB catalog. The forecasting tools distinguish time-series forecasts, later-period regression scores, historical risk rankings, and caller-assumption simulations. They report chronological holdout metrics where applicable. Empirical error bands are not formal confidence intervals; predictive results are not causal evidence. The data is synthetic and has no stock-on-hand or stockout history, so it cannot support an inventory-availability forecast. New external sources must first be made available as DuckDB tables or views; no external provider credentials are configured here. Use `pharma://context/guide` for method-specific caveats.

## Data and deployment configuration

Set `DUCKDB_PATH` to use a different DuckDB file. If no local database is present, the current serverless fallback downloads the configured database artifact to `/tmp`.

The GitHub Actions workflow runs the project checks on pushes and pull requests to `main`. Vercel deployment is configured through `vercel.json`; the deployed connector URL depends on the Vercel project domain.
