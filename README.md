# Glean Model Sync

Self-contained helper script to fetch Glean's model catalog and merge supported models the config files of different harnesses, e.g. Oh My Pi (OMP) and Deepseek Harness (DSH)

> [!NOTE]
> This is a community repository. It is not endorsed, sponsored, managed, or supported by Glean. You use the contents at your own risk.

## What this does

By default, targets OMP and updates `~/.omp/agent/models.yml`. Existing provider settings and
models not present in the fetched catalog are retained, and the previous `models.yml` is backed up in the same directory.

`--format dsh` writes a copy-ready `cordis.patch.yml` beside this script without touching `~/.dsh`.

Run with `--dry-run` to preview the YAML diff, or `--input-json` to test against a saved response.

## Requirements

* uv
* Python 3.11+

Required environment variables:
* `GLEAN_API_KEY` - Bearer token for the Glean model catalog endpoint. Must be an LLM Gateway/Proxy key, NOT a Client API key.

Optional environment variables:
* `GLEAN_INSTANCE_ID` - The tenant ID of your Glean environment. This is the prefix before the `-be` in the subdomain of your Glean server URL. You can get this from the **Glean Admin Console -> About Glean -> Instance name**. E.g. If your server instance is `https://tenant-prod-be.glean.com` then your instance ID will be `tenant-prod`.

If you don't set `GLEAN_INSTANCE_ID`, you will need to explicitly pass the instance ID via the `--instance` argument when running the script.

## Usage

```bash
# Set environment variables
export GLEAN_INSTANCE_ID=tenant-id  # from the Glean server URL of your tenant
export GLEAN_API_KEY=...            # Requires an LLM Gateway/Proxy key

# Preview OMP config
uv run sync_glean_models.py --dry-run

# Update the OMP config
uv run sync_glean_models.py

# Alternative to populating GLEAN_INSTANCE_ID
uv run sync_glean_models.py --instance tenant-id

# Create DSH config
uv run sync_glean_models.py --format dsh

# Preview the DSH config
uv run sync_glean_models.py --format dsh --dry-run
```
