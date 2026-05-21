# TODO

Open work for the dissertation, ordered roughly by what comes next. Items move forward as scope changes; resolved items are deleted rather than struck through.

---

## 10. Pre-submission checklist

Tick these off in order, just before submitting.

- **DOI / author / year spot-check** before submission. Highest-risk: recent preprints (`marchetti2025multidimensional`, `gorodkin2004comparing`).
- **Exact-page / theorem citation verification** — for each "Theorem X of [Y]" claim, find the exact page/theorem. Have Claude do a first pass; user verifies.
- **Dataset references.** Every dataset cites its OpenML / source entry. Example: volkert (41166): https://www.openml.org/d/4116

**Final review.**

- Add link to repo and state that everything is reproducible.
- In the main tex, refer to the repo, specially the suplement pdf.

# Depois de entregar:

## Gerais

- Are we using bold/italic correctly?
- **Rethink which person we want to use**. First person should be fine, but let's double check.
- Citar mais.
- **Third-party LLM review pass** — ChatGPT, Gemini 3.1, Claude Opus, Claude Sonnet.
- All image and tables must be refereced in the text.
- Legends sometimes are duplicate of prose and vice versa. Check.
- Remove : before equations.
- Double check all the notation table.
- Understand how everyplot was made. Have a small cheatsheet.
- Corrigir algumas imagens levemente problemáticas. Exemplo: variance scalling figure.
- Existe formula fechada pra non decomposable rules?
- Check if there are undefined-before-use notation or terms/definition. We have a script for that.
- Search for handwaved sections/statements.
- Re-read ALL proofs.
- Capital-case heading style. Apply consistently across chapters.
- FullMatrixSearch could normalize by C_max and then the range would be closer to what it samples.
- Alguma imagem vazando?
- Quebrar win rate por dataset.

## Code & repo hygiene (orthogonal)

Can happen in parallel with any of the above.

- **README + project documentation polish.**
- **`threshold_opt` public API docs / typing** — pair with PyPI release.
- **Comment cleanup pass** across `src/` and `experiments/` — remove silly comments, keep informative ones useful for human/LLM context.
- **Security tests** — assert no API keys / credentials accidentally checked in.
- **Gitignore audit** — verify what's ignored is what should be ignored. `tex/figures/*.png` is fine while PNG outputs exist as build artifacts; revisit as scripts are ported to TikZ in §3.
- **Publish to PyPi.**

## 9. Reading list (before presentation)

- **Foundational neural network models.**
- **Marchetti's SOL** (`marchetti2022sol`, cited in §Conclusion future-work; pairs with §8 "SOL method mention"). The original 2022 paper is binary; there is also a 2025 multiclass extension (arXiv 2511.22587 — "MultiSOL on the Simplex") worth scanning.

## Re-run related

- **Switch multiclass model AUC table to macro averaging.** Currently `roc_auc_score(multi_class="ovr", average="weighted")` in [experiments/common/pipeline.py:501](experiments/common/pipeline.py#L501); switch to `average="macro"` so the AUC table uses the same equal-weight convention as the macro $F_\beta$ / G-Mean optimization objectives of Chapter~7. Re-run the multiclass campaign (or recompute AUC offline from stored scores if available) and update the caption in [tex/content/07-multiclass-experiments.tex](tex/content/07-multiclass-experiments.tex) accordingly.

## Reler

- Charles Elkan. “The Foundations of Cost-Sensitive Learning”. In: Proceedings
of the 17th International Joint Conference on Artificial Intelligence (IJCAI).
2001, pp. 973–978