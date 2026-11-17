"""Download the public Apache-2.0 checkpoints of the final submission at pinned commits.

Only clean members are listed. Checkpoints of the same author trained on Phase 1 test labels or on Phase 2 audio
(linsna-distilled, linsna-soup6, seeds 43/44/46) are deliberately excluded (see DATASETS.md)."""
from huggingface_hub import snapshot_download

MEMBERS = {
    "anyantudre/waxal-w2vbert-lin-specialist": "c5ed34742e6150c5169e70981f40393fd13cb7c6",
    "anyantudre/waxal-w2vbert-sna-specialist": "2cb8644b065dd8690e166250df528d85927a084d",
    "anyantudre/waxal-w2vbert-linsna-afrivoicemix": "31de7fd5fc16eb53f3ea3e717f2e02d4b5d8605d",
    "anyantudre/waxal-w2vbert-linsna-soup5": "16c13b7381e666637cc291e25d57e27ff50616d6",
}

if __name__ == "__main__":
    for repo, revision in MEMBERS.items():
        for attempt in range(5):
            try:
                print(repo, snapshot_download(repo, revision=revision), flush=True); break
            except Exception as e:
                print("retry", repo, e, flush=True)
