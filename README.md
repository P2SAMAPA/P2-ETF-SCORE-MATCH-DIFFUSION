# Score Match Diffusion Engine

Implements denoising score matching (Stein) to learn the score function ∇ log p(x) of ETF return distributions. The score at the current market state indicates the direction of highest probability density – a local momentum/mean‑reversion signal. Langevin dynamics can be used for sampling (not used for ranking). The absolute score magnitude ranks ETFs.

- **Model:** MLP with SiLU activations
- **Training:** Denoising score matching (Gaussian noise)
- **Score:** ∇ log p(x) (gradient of log density)
- **Ranking:** |score| (larger = stronger signal)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
