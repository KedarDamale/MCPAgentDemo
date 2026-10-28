import pytest
import json
from fastmcp import Client
from starlette.testclient import TestClient
from mcpagentdemo.app import app
from mcpagentdemo.server import mcp

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mcp_endpoint"] == "/mcp"

def test_mcp_initialize(client):
    response = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-runner", "version": "1.0"}
        }
    })
    assert response.status_code == 200

def test_tools_registered():
    import asyncio
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) >= 16
    tool_names = [t.name for t in tools]
    assert "execute_sql" in tool_names
    assert "get_table_summary" in tool_names
    assert "get_sales_by_dimension" in tool_names
    assert "get_top_products" in tool_names
    assert "analyze_margin_erosion" in tool_names
    assert "analyze_discount_behavior" in tool_names
    assert "analyze_returns_and_spoilage" in tool_names
    assert "expiry_risk_profile" in tool_names
    assert "get_customer_profile" in tool_names
    assert "identify_dormant_customers" in tool_names
    assert "benchmark_sales_reps" in tool_names
    assert "regional_growth_matrix" in tool_names
    assert "cold_chain_compliance_analysis" in tool_names
    assert "seasonal_trends_forecast" in tool_names
    assert "price_volume_elasticity" in tool_names
    assert "generate_chart_ready_dataset" in tool_names

def test_resources_registered():
    import asyncio
    resources = asyncio.run(mcp.list_resources())
    assert len(resources) >= 5
    uris = [str(r.uri) for r in resources]
    assert "pharma://schema/tables" in uris
    assert "pharma://metadata/readme" in uris
    assert "pharma://catalog/skus" in uris
    assert "pharma://summary/monthly" in uris
    assert "pharma://design/color-themes" in uris

def test_numeric_regression_tool_schema():
    import asyncio
    tools = asyncio.run(mcp.list_tools())
    tool = next(tool for tool in tools if tool.name == "predict_numeric_regression")
    schema = tool.parameters
    assert schema["required"] == ["predictor_columns"]
    assert schema["properties"]["predictor_columns"]["type"] == "array"
    assert schema["properties"]["table_name"]["default"] == "fact_sales"
    assert schema["properties"]["target_column"]["default"] == "gross_profit_inr"

def test_prompts_registered():
    import asyncio
    prompts = asyncio.run(mcp.list_prompts())
    assert len(prompts) >= 4
    names = [p.name for p in prompts]
    assert "data_visualization_dashboard" in names
    assert "margin_leakage_investigation" in names
    assert "customer_churn_risk_analysis" in names
    assert "sales_rep_performance_review" in names

def test_generate_chart_ready_dataset():
    import asyncio
    async def generate_chart():
        async with Client(mcp) as client:
            return await client.call_tool("generate_chart_ready_dataset", {
                "chart_type": "bar",
                "x_dimension": "therapeutic_area",
                "y_metric": "net_sales_inr"
            })
    res = asyncio.run(generate_chart())
    assert not res.is_error
    data = json.loads(res.content[0].text)
    assert data["selected_chart_type"] == "bar"
    assert data["categories"]
    assert len(data["categories"]) == len(data["series"][0]["data"])
