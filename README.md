# BoneScanCEP

AI-Based Multi-Task Analysis of Bone Scan with a Streamlit UI for uploads and metrics.

## Run locally

```powershell
pip install -r requirements.txt
streamlit run UI.py
```

## Deploy (Streamlit Cloud)

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app.
3. Select this repo and set the main file to `UI.py`.
4. Deploy.

## Notes

- The Local path option is hidden on Streamlit Cloud because the server cannot access your local disk.
- Upload images in the UI to compute metrics.
