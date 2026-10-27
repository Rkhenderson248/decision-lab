# Recordings

Real model runs for the AI engineering pages (analyst agent, LLM eval harness, document intelligence), recorded
once and replayed. The pages never call a model.

Record (about 15 to 25 minutes; a few dollars at list prices):

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
python -m recorder.record
git add recordings && git commit -m "Record model runs" && git push
```

Options: `--models claude-sonnet-5-5,claude-haiku-5-5` (default), `--only analyst|docs`, `--cases L1,M2`,
`--repeats 3` (adds a consistency measure to the eval page). Re-running skips anything already recorded; delete a
file to record it again. The key is read from the environment and never written to disk.

Layout: `analyst/<model>/<case>.json` (one trace per question), `docs/<model>.json` (one extraction per invoice),
`manifest.json` (when, SDK version, models, prompt hashes).
