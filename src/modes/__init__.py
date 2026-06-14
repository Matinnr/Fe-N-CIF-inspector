"""Mode packages — one per top-level workflow in the dashboard.

Each mode exposes a `render()` function that draws its UI inside the
main pane. The router in app.py picks which one to call based on the
sidebar mode switcher.

The split makes it trivial to add a new mode (drop a file here, wire
it into MODE_REGISTRY in app.py) and equally trivial to retire one.
"""
