"""Download WearGait-PD (version 1) from Synapse into $PD_DATA_ROOT/gait/weargait.

Dataset: Kontson et al., Synapse, doi:10.7303/syn52540892, licensed CC BY 4.0.
Paper: Anderson et al., Sci Data 13, 440 (2026), doi:10.1038/s41597-026-06806-2.
Access needs a free Synapse account and agreement to the Synapse pledge.

Setup:
    export SYNAPSE_AUTH_TOKEN=<personal access token, view and download scopes>
    export PD_DATA_ROOT=/path/to/data      # the folder that holds gait/ and handwriting/
    python download_weargait_pd_v1.py

Never put the token in this file. The SYNAPSE_METADATA_MANIFEST.tsv files list
every expected file; their paths are relative to $PD_DATA_ROOT/gait/weargait.
"""
import os
from pathlib import Path

VERSION_FOLDER_ID = "syn55052683"  # WearGait-PD "Version 1" folder


def get_token(environ=os.environ):
    token = environ.get("SYNAPSE_AUTH_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "SYNAPSE_AUTH_TOKEN is not set. Create a personal access token with "
            "the view and download scopes in your Synapse account settings, then "
            "export it. Do not write it into this file."
        )
    return token


def get_download_dir(environ=os.environ):
    root = environ.get("PD_DATA_ROOT", "").strip()
    if not root:
        raise SystemExit(
            "PD_DATA_ROOT is not set. Point it at the folder that holds gait/ "
            "and handwriting/, for example: export PD_DATA_ROOT=$HOME/pd-data"
        )
    return Path(root).expanduser() / "gait" / "weargait"


def main():
    token = get_token()
    download_dir = get_download_dir()

    import synapseclient
    import synapseutils

    download_dir.mkdir(parents=True, exist_ok=True)

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    print(f"Downloading Synapse folder {VERSION_FOLDER_ID} into {download_dir} ...")
    files = synapseutils.syncFromSynapse(
        syn,
        VERSION_FOLDER_ID,
        path=str(download_dir),
    )

    print(f"Done. Downloaded/synced {len(files)} items.")


if __name__ == "__main__":
    main()
