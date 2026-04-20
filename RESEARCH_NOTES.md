# Research notes

Per-paper notes on the literature underlying `IDEAS.md`. Each entry lists:
- **Access status**: what I actually fetched vs. summarized from prior
  knowledge.
- **Core claim**: the paper's thesis in one line.
- **What's usable**: the concrete technique or result to borrow.
- **Caveats**: what the paper does *not* prove or implications to verify.

Fetching was done 2026-04-20. Anything marked "prior knowledge" means I'm
summarizing from training data, not a fresh read - treat accordingly.

---

## Fractal / long-memory literature

### Mandelbrot & Hudson (2004) - *The Misbehavior of Markets*
- **Access:** Book. Not fetched.
- **Core claim:** Markets exhibit fractal scaling and fat tails; Gaussian
  models systematically underestimate risk.
- **Usable:** Conceptual framing, not methodology. Skim ch. 7-10 for the
  multifractal model of asset returns (MMAR) overview.
- **Caveats:** Popular book, not a methods reference. For actual formulas,
  read Calvet-Fisher (2002) MMAR papers instead.

### Peters (1994) - *Fractal Market Hypothesis*
- **Access:** Book. Not fetched.
- **Core claim:** Markets have different "investor horizons" that stabilize
  them; violations of the fractal market hypothesis cause crashes.
- **Usable:** The framing of scale-dependent trader behavior.
- **Caveats:** Methodology (classical R/S) is superseded. Lo (1991) showed
  Peters-style findings are often artifacts.

### Lo (1991), *Long-term memory in stock market prices* - Econometrica 59(5)
- **Access:** Paywalled. Not fetched. Content summarized from prior knowledge.
- **Core claim:** Classical R/S statistic is biased upward in presence of
  short-range autocorrelation. Modified R/S (with Newey-West style long-run
  variance) shows no significant long memory in US equities.
- **Usable:** The modified R/S test. Implement as a skeptical companion to
  DFA. Critical sanity check.
- **Caveats:** The paper is about equities, not crypto. Later work has found
  some long memory in crypto and EM FX that survives Lo's correction.
- **Implementation note:** Andrews' data-dependent bandwidth rule for the
  Newey-West window `q` matters; hard-coding `q = 5` will give misleading
  results.

### Di Matteo, Aste, Dacorogna (2005), *Scaling behaviors in differently
developed markets* - Physica A 324
- **Access:** Abstract fetched, full paper not.
- **Core claim:** Generalized Hurst exponent differs systematically between
  developed (e.g. S&P) and emerging (e.g. smaller stock indices) markets.
  Emerging markets show stronger persistence.
- **Usable:** Use scaling exponent as a market-maturity classifier. Crypto
  likely shows "emerging market" scaling patterns - relevant to Bariviera's
  later BTC work.
- **Caveats:** The generalized Hurst requires careful estimation; use
  MF-DFA-style methods rather than naive moment scaling.

### Willinger, Taqqu, Teverovsky (1999), *Stock market prices and
long-range dependence* - Finance and Stochastics
- **Access:** Not fetched. Summarized from prior knowledge.
- **Core claim:** Estimators of Hurst are biased; long-range dependence is
  very hard to distinguish from short-range dependence on finite samples.
- **Usable:** Motivates using *multiple* estimators (R/S, DFA, wavelet-based)
  and looking for agreement. Disagreement - common - means don't claim the
  effect.

### López de Prado (2018), *Advances in Financial Machine Learning*, ch. 5
- **Access:** Book. Ch. 1 freely on SSRN (abstract_id=3104847), ch. 5 is not.
  Summarized from Hudson & Thames blog post + prior knowledge.
- **Core claim:** Integer differencing destroys memory. Fractional
  differencing with `d` in (0, 1) produces a stationary series that still
  correlates strongly with the original. Use as an ML feature transform.
- **Usable:** Direct. See IDEAS.md A3 for algorithm and pseudocode.
- **Caveats:** The fixed-width window (FFD) variant is preferred over
  expanding window because FFD is drift-free and avoids weight-growth issues.
- **Third-party implementations to cross-check:** `mlfinlab` (Hudson & Thames,
  now commercial), `fracdiff` on PyPI, `github.com/eortizt/Fracdiff`.

### Hosking (1981), *Fractional differencing* - Biometrika 68(1)
- **Access:** Paywalled. Not fetched.
- **Core claim:** Original theoretical paper introducing the fractional
  differencing operator for ARFIMA models.
- **Usable:** Reference for derivation of the binomial-expansion weight
  formula. Not needed for implementation.

---

## Crypto-specific empirical work

### Bariviera (2017), *The inefficiency of Bitcoin revisited: a dynamic
approach* - Economics Letters (arXiv:1709.08090)
- **Access:** arXiv PDF fetched and summarized (binary extraction was partial;
  summary via WebFetch worked).
- **Core claim:** BTC's Hurst exponent is *time-varying*. Early period
  (2011-2013) shows strong persistence (H > 0.55); later period (2014-2017)
  drifts toward H = 0.5 as the market matures.
- **Usable:** This is the empirical anchor for treating Hurst as a regime
  indicator rather than a fixed market property. Rolling DFA is the right
  estimator per the paper.
- **Methodology summary:** DFA is more robust than R/S when trends are
  present. Paper reports both and notes R/S over-detects.
- **Replication target:** Extend the analysis through 2026 and see whether H
  has stayed near 0.5 or re-diverged. That's a direct first notebook.

### Bariviera et al. (2020), *Heterogeneity in cryptocurrencies' multifractal
profiles* - Finance Research Letters 39 (arXiv:2003.09720)
- **Access:** Abstract fetched; methodology "multi-scaling methodologies"
  confirmed. Full paper only as binary PDF.
- **Core claim:** Across 84 cryptocurrencies, multifractal profiles are
  highly heterogeneous. Some behave as monofractal fractional Gaussian
  noise; others are genuinely multifractal. Shuffling reduces but does not
  eliminate multifractality, implying fat tails contribute as much as long
  memory.
- **Usable:** Rank crypto pairs by multifractality width `delta h` and treat
  it as an intrinsic roughness measure. Strong-`delta h` assets are poor
  candidates for strategies assuming normal returns.
- **Caveats:** Not all multifractality implies predictability. The finding
  about shuffling not removing multifractality is the key methodological
  warning.

### Wątorek, Kwapień, Drożdż (2022), *Multifractal cross-correlations of BTC
and ETH trading characteristics* (arXiv:2208.01445)
- **Access:** Abstract fetched.
- **Core claim:** Using MFDCCA (multifractal detrended cross-correlation
  analysis), BTC and ETH show multifractal cross-correlations in price,
  volume, and trade frequency. Cross-correlations persist at long time scales
  even under time lags.
- **Usable:** Pair-trade BTC-ETH signal construction. Cross-correlations on
  specific timescales (from wavelet coherence or MFDCCA) can define the
  *timescale* on which to trade the pair.
- **Caveats:** "Post-COVID-19 time" window - regime may not persist. Verify
  on more recent data.
- **Note:** Author attribution correction - this is Wątorek et al., not
  Kristoufek as I originally cited. Kristoufek has related but separate
  papers.

### Kristoufek - body of work on crypto efficiency
- **Access:** Not individually fetched. Referenced in search results.
- **Core claim across papers:** Crypto markets are time-varying in their
  efficiency. Capital market efficiency can be ranked using long-memory +
  fractal dimension + approximate entropy (Kristoufek & Vosvrda 2014, EPJ B).
- **Usable:** The composite efficiency index from Kristoufek-Vosvrda is a
  cleaner single-number summary than Hurst alone. Worth implementing.

---

## Fourier / spectral literature

### Carr & Madan (1999), *Option Valuation Using the Fast Fourier Transform*
- Journal of Computational Finance 2(4), pp. 61-73
- **Access:** Multiple PDFs exist (NYU, Imperial College, ResearchGate).
  WebFetch returned binary-only; summarized from prior knowledge.
- **Core claim:** For any model with known characteristic function of log
  spot, the strip of European call prices across strikes can be computed
  via a single FFT with damping parameter `alpha`.
- **Usable:** Full algorithm in IDEAS.md B1. This is the single most rigorous
  Fourier application in quant finance.
- **Caveats:** `alpha` choice matters. Simpson rule weights in the
  integration reduce error by one order. Must use fine-grained `v` sampling
  for deep OTM.
- **Accessible PDF URLs to re-fetch:**
  - `engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf`
  - `ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/.../CarrMadan.pdf`

### Lewis (2001), *A simple option formula for general jump-diffusion and other
exponential Lévy processes*
- **Access:** Not fetched.
- **Core claim:** Alternative Fourier inversion formula to Carr-Madan. Single
  integral along a specific contour in the complex plane.
- **Usable:** Numerically more stable than Carr-Madan for some models (no
  damping needed). Worth implementing as a cross-check.

### Gençay, Selçuk, Whitcher (2001), *An Introduction to Wavelets and Other
Filtering Methods in Finance and Economics*
- **Access:** Book. Not fetched.
- **Core claim:** Textbook treatment of wavelets applied to financial and
  economic time series.
- **Usable:** Reference for MODWT (maximum overlap DWT) which is
  translation-invariant and better for financial data than plain DWT.

### Ramsey (1999), *The contribution of wavelets to the analysis of economic
and financial data*
- **Access:** Not fetched.
- **Core claim:** Survey. Lays out where wavelets add value vs Fourier in
  non-stationary financial data.

---

## What I deliberately excluded

- **Elliott Wave literature** - narrative, not falsifiable. Skip.
- **"Cycle detection" via FFT on price** - cottage industry of papers that
  don't reproduce out-of-sample. The informative counter-evidence is the
  repeated non-reproduction itself; no single paper to cite.
- **Technical "fractal" indicators (Bill Williams etc.)** - pattern matchers
  sold as fractal analysis.

---

## Papers I should still read

If/when the project goes beyond the first notebooks:

- **Calvet & Fisher (2002, 2004)** - MMAR proper derivation. The multifractal
  model of asset returns.
- **Peng et al. (1994)** - DFA original method paper (`Nature`). Short and
  worth a direct read.
- **Kantelhardt et al. (2002)** - MF-DFA original paper. The canonical
  reference for `h(q)` spectrum.
- **Andrews (1991)** - Data-dependent bandwidth for Newey-West. Needed for
  honest modified-R/S.
- **In & Kim (2013)** - Wavelet coherence for pair trading (specific enough
  to be actionable).
