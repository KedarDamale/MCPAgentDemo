"""FastAPI/Starlette application entrypoint for Vercel Serverless and local uvicorn hosting."""

import os
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from mcpagentdemo.server import mcp

# Create the FastMCP HTTP app configured at root path '/'
mcp_app = mcp.http_app(path="/")

mcp_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UniversalMCPMiddleware:
    """Normalize Vercel rewrites and prefixes so requests to /, /mcp, /api/index.py, etc. all reach FastMCP."""
    def __init__(self, inner: ASGIApp):
        self.inner = inner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            # Strip Vercel function filename prefixes
            for prefix in ("/api/index.py", "/api/index"):
                if path.startswith(prefix):
                    path = path[len(prefix):]
                    break
            # Strip /mcp mount prefix
            if path.startswith("/mcp"):
                path = path[4:]
            if not path.startswith("/"):
                path = "/" + path
            scope["path"] = path

        await self.inner(scope, receive, send)

app = UniversalMCPMiddleware(mcp_app)
