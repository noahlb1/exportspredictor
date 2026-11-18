# Trade Agreement (RTA) effects in the gravity literature

For #62, the PPML benchmark of the Trade Agreement Scenario Effect (decision #33). These are the published estimates that `ppml.LITERATURE` holds, with where each number sits in its source. Every figure is a coefficient on an RTA/FTA dummy in log points. `exp(b) − 1` turns it into the proportional change in members' trade that the regression associates with an agreement.

## Estimates used

| Source | Specification | Coefficient | exp(b) − 1 |
|---|---|---|---|
| Yotov, Piermartini, Monteiro & Larch (2016) [1], ch. 1, Table 3, col. (2) | PPML, Exporter-year and Importer-year effects, international Flows only, no pair effects, 1986–2006 in 4-year intervals | 0.191 (s.e. 0.066) | +21% |
| Head & Mayer (2014) [2], Table 4 | Median of 108 "structural gravity" estimates (some form of country effects) | 0.28 | +32% |
| Head & Mayer (2014) [2], Table 4 | Mean of the same 108 | 0.36 | +43% |
| Head & Mayer (2014) [2], Table 4 | Median of 257 estimates, all gravity papers | 0.47 | +60% |
| Yotov et al. (2016) [1], Table 3, col. (4) | PPML with pair effects and intra-national Flows | 0.557 (s.e. 0.102) | +75% |
| Baier & Bergstrand (2007) [3], abstract | Panel with pair and country-year effects: "on average, an FTA approximately doubles two members' bilateral international trade after 10 years" | ≈ ln 2 = 0.69 | ≈ +100% |

So the literature range used in the notebook runs from **+21% to +100%**.

## Context for reading them

- **Pair effects raise the estimate.** In [1], adding pair effects moves the RTA coefficient from 0.409 to 0.557. Following Baier & Bergstrand [3], pair effects absorb the reason the countries signed (they already traded a lot), so estimates without them are biased down.
- **Intra-national Flows raise it too.** In [1], the coefficient rises from 0.191 to 0.409 when domestic sales are added, because an RTA diverts trade from domestic to international sales. BACI has no intra-national Flows, so our benchmark is closer to the international-only specifications. That would bias it down against [1]'s col. (4).
- **Phasing-in raises it.** With lags of 4, 8 and 12 years, [1] reports a total RTA effect of 0.992 (col. 6). With a globalisation control, the contemporaneous term drops to 0.116 and the total to 0.475 (col. 7). Our benchmark has only a contemporaneous dummy, observed every 3 years.
- **The meta-analysis is not causal.** Head & Mayer warn that the Table 4 numbers "should not be interpreted as preferred estimates of the causal effects", because most papers do not deal with the endogeneity of agreements. Cipollina & Salvatici (2010), cited in [2], report a mean of 0.59 and a median of 0.38 over 1,867 estimates.
- **3-way inference.** Weidner & Zylkin (2021) [4] show that 3-way PPML is consistent, but that its cluster-robust intervals are not exactly centred in finite samples. The benchmark's interval is an order of magnitude, not a precise test.
- **The estimator.** The benchmark uses IRLS with the fixed effects swept out by alternating projections, as `ppmlhdfe` does [5]. PPML itself, rather than OLS on log Flows, follows Santos Silva & Tenreyro (2006) [6], because it keeps the zeros and stays consistent under heteroskedasticity.

## Sources

1. Yotov, Y. V., Piermartini, R., Monteiro, J.-A. & Larch, M. (2016). *An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model*. WTO/UNCTAD. Ch. 1, Application 2 and Table 3 (pp. 50–52 of the PDF). https://www.wto.org/english/res_e/booksp_e/advancedwtounctad2016_e.pdf
2. Head, K. & Mayer, T. (2014). "Gravity Equations: Workhorse, Toolkit, and Cookbook". *Handbook of International Economics*, vol. 4, pp. 131–195. Working-paper version: CEPII WP 2013-27, Table 4 (p. 34). https://www.cepii.fr/PDF_PUB/wp/2013/wp2013-27.pdf
3. Baier, S. L. & Bergstrand, J. H. (2007). "Do free trade agreements actually increase members' international trade?" *Journal of International Economics* 71(1), 72–95. https://doi.org/10.1016/j.jinteco.2006.02.005
4. Weidner, M. & Zylkin, T. (2021). "Bias and consistency in three-way gravity models". *Journal of International Economics* 132, 103513. https://doi.org/10.1016/j.jinteco.2021.103513
5. Correia, S., Guimarães, P. & Zylkin, T. (2020). "Fast Poisson estimation with high-dimensional fixed effects". *Stata Journal* 20(1), 95–115. arXiv:1903.01690.
6. Santos Silva, J. M. C. & Tenreyro, S. (2006). "The Log of Gravity". *Review of Economics and Statistics* 88(4), 641–658.

The numbers from [1] and [2] were read from the PDFs above on 18 Nov 2026. The figure from [3] is the paper's abstract.
