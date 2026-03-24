import os
import synapseclient
import synapseutils

# ---- CONFIG ----
VERSION_FOLDER_ID = "syn55052683"   # Version 1 folder from your screenshot
DOWNLOAD_DIR = "/ocean/projects/med260006p/shared/biomedAI/project/replica/data/gait/weargait"
# ----------------

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    syn = synapseclient.Synapse()

    # Option 1: login interactively after running `synapse login`
    # Option 2: set SYNAPSE_AUTH_TOKEN as an environment variable
    syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3MjkxMjE4NSwiaWF0IjoxNzcyOTEyMTg1LCJqdGkiOiIzMzM3MSIsInN1YiI6IjM1NzcyMjYifQ.pXAmHysm4xhnOFsM_2nwLcxoYt4uiPPOQbm9bP8RiibU6vmVk5h-dDVP1Zriw2l8SqkuPjgRrG1hTke3nO9BXFom03GeZWsVt-BLuqGFBHtwlzseWHHIwHGZq_xsd31-VobQ1h_4PaxhzAd9k2h-rmvKV71ZReGvJ9sWfK9TTMBRdMNxJ97wsU_8MRH_j_6m-WUSY7SvLN5MiQAeKeY0CR-1RQiwQlly_l34sCIPwqXfW0aYwxtQz7nzpLc8Rhds6UAo3qDFq97inJfLARDOR4BpBLS63NQVnDLlYxV5lDohIss8WYhcHl5WNIdL62W1P4MKlJxH3K84SP_V9eozcw")

    print(f"Downloading Synapse folder {VERSION_FOLDER_ID} into {DOWNLOAD_DIR} ...")
    files = synapseutils.syncFromSynapse(
        syn,
        VERSION_FOLDER_ID,
        path=DOWNLOAD_DIR
    )

    print(f"Done. Downloaded/synced {len(files)} items.")

if __name__ == "__main__":
    main()