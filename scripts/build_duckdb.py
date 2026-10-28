"""Rebuild pharma_sales.duckdb from the Parquet files.
Usage:
    python scripts/build_duckdb.py [data_dir] [output_duckdb_path]
Default:
    data_dir = data
    output = data/pharma_sales.duckdb
"""
import glob
import os
import sys
import time
import duckdb

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

if len(sys.argv) > 1:
    data_dir = sys.argv[1]
elif os.path.exists(os.path.join(root, "data", "dim_sku.parquet")):
    data_dir = os.path.join(root, "data")
elif os.path.exists(os.path.join(here, "dim_sku.parquet")):
    data_dir = here
else:
    data_dir = os.path.join(root, "data")

if len(sys.argv) > 2:
    db_path = sys.argv[2]
else:
    db_path = os.path.join(data_dir, "pharma_sales.duckdb")

facts = sorted(glob.glob(os.path.join(data_dir, "fact_sales_*.parquet")))
assert facts, f"No fact_sales_*.parquet files found in {data_dir}"

if os.path.exists(db_path):
    os.remove(db_path)

print(f"Building DuckDB database at: {db_path}")
print(f"Reading from: {data_dir} ({len(facts)} fact files)")

start_time = time.time()
con = duckdb.connect(db_path)

p = lambda n: os.path.join(data_dir, n + ".parquet").replace("\\", "/")

print("Loading fact_sales...")
con.execute(
    "CREATE TABLE fact_sales AS SELECT * FROM read_parquet(?) ORDER BY invoice_date",
    [[f.replace("\\", "/") for f in facts]],
)

print("Loading dimension tables...")
for t in ["dim_sku", "dim_customer", "dim_rep", "dim_date", "README"]:
    con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{p(t)}')")

print("Creating views...")
con.execute(
    """CREATE VIEW sales_full AS SELECT f.*, date_diff('day',f.invoice_date,f.expiry_date) AS days_to_expiry_at_sale,
 round(100*f.gross_profit_inr/nullif(f.net_sales_inr,0),2) AS gross_margin_pct,
 s.brand_name,s.generic_name,s.therapeutic_area,s.dosage_form,s.strength,s.pack_size,s.brand_type,s.drug_schedule,s.cold_chain,
 c.customer_type,c.channel,c.region,c.state,c.city_tier,c.territory_code,c.sales_rep_id,c.payment_terms_days,
 d.fiscal_year,d.fiscal_quarter,d.year,d.quarter,d.month
 FROM fact_sales f JOIN dim_sku s USING(sku_id) JOIN dim_customer c USING(customer_id) JOIN dim_date d ON d.date=f.invoice_date"""
)

con.execute(
    """CREATE VIEW monthly_summary AS SELECT date_trunc('month',invoice_date)::DATE AS month, count(*) AS invoices, sum(net_sales_inr) AS net_sales_inr,
 sum(gross_profit_inr) AS gross_profit_inr, round(100*sum(gross_profit_inr)/sum(net_sales_inr),2) AS gross_margin_pct,
 round(sum(discount_inr)*100/sum(gross_sales_inr),2) AS avg_discount_pct FROM fact_sales GROUP BY 1 ORDER BY 1"""
)

row_count = con.execute("SELECT count(*) FROM fact_sales").fetchone()[0]
elapsed = time.time() - start_time
print(f"Loaded {row_count:,} rows into {db_path} in {elapsed:.2f}s")
con.close()

