# Shape Recognizer — Vercel-deployable version

This is a rebuild of the Colab notebook's shape-recognizer as an actual
static site + Python serverless function, which is what Vercel can deploy.

## Structure

```
index.html          Static frontend (canvas drawing, voice commands, Desmos graph)
api/recognize.py     Vercel Python serverless function — does the curve fitting
requirements.txt     Python dependency (numpy) for the function
```

## Before deploying

Open `index.html` and replace `YOUR_API_KEY_HERE` in the `<script src="...desmos...">`
tag near the top with a free key from https://www.desmos.com/api?lang=en.

## Deploying

1. Replace the contents of your `Equation-Analyser-` GitHub repo with these
   three files/folders (delete the old `.ipynb`-based content, or put this
   in a fresh repo — either works).
2. Push to GitHub.
3. In Vercel, either let it auto-redeploy (if already connected to this repo)
   or re-import the repo. Framework Preset can stay "Other" — no build step
   is needed for a static `index.html` + `/api` function.
4. Once deployed, visiting the site root should load the page directly
   (no more 404), and drawing/voice commands will call `/api/recognize`.

## Notes

- The Python function only needs `numpy`; `requirements.txt` handles that.
- CORS headers are included in `api/recognize.py` even though they're not
  strictly needed for same-origin requests — harmless to leave in.
- If a POST request to `/api/recognize` fails, check the function's logs in
  the Vercel dashboard (Deployments → your deployment → Functions) — that's
  where Python errors (e.g. missing numpy, bad JSON) will show up.
