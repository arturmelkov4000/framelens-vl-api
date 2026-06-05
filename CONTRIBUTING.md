# Contributing

FrameLens VL API is intentionally small. Contributions should keep the HTTP API stable, avoid committing private media or tokens, and include focused tests for parsing, configuration, or request validation changes.

## Local Checks

```bash
python -m unittest discover -s tests
```

For backend changes, document the model/server used for manual verification and include any limitations in the pull request.
