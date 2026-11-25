"""Small shared infrastructure; scientific policies remain in their domain packages.

Import the specific module needed by a caller. In particular, importing file or
timestamp utilities must not import NumPy, models, configuration, or API clients.
"""
