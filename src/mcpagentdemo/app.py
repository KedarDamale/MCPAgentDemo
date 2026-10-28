"""FastAPI/Starlette application entrypoint for Vercel Serverless and local uvicorn hosting."""

import os
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount
from starlette.middleware.cors import CORSMiddleware
from mcpagentdemo.server import mcp

# Create the FastMCP HTTP app with Streamable HTTP & JSON support
mcp_app = mcp.http_app(path="/mcp")

async def root_handler(request):
    """Health check and server info."""
    return JSONResponse({
        "status": "healthy",
        "name": "PharmaSales Analytics FastMCP Server",
        "protocol": "Model Context Protocol (MCP)",
        "mcp_endpoint": "/mcp",
        "docs": "Connect to /mcp with any MCP-compatible client or agent."
    })

# Main Starlette application wrapping MCP with lifespan
app = Starlette(
    routes=[
        Route("/", endpoint=root_handler, methods=["GET"]),
        Mount("", app=mcp_app)
    ],
    lifespan=mcp_app.lifespan
)

# Enable CORS for browser agents / web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
