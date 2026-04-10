#!/usr/bin/env python
"""Run the Factory Layout WebUI server."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "group_placement.webui.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["group_placement/webui"],
    )
