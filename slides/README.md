# Defense deck (HTML)

The defense slides as a web page: reveal.js for navigation, MathJax for the
formulas, plain JavaScript for the interactive figures. Everything it loads
is in this folder, so it runs with no network.

## Presenting

Open `index.html` in Chrome or Edge (double click).

| Key | Action |
|---|---|
| arrows, space, PgUp/PgDn | previous / next slide |
| `F` | full screen |
| `S` | speaker view in a second window: notes, next slide, timer |
| `O` or `Esc` | overview of all slides |
| `G` | go to a slide by number |

Links such as P4 or B10 jump to the backup slide; the browser's back
button (Alt + left arrow) returns, and so does the yellow "back to slide N"
button that appears at the bottom left, or the Backspace key. While a slider has the focus, the arrow
keys move the slider; click outside it to give them back to the deck.

Each interactive figure starts in the state of the original static figure.
Reloading the page (F5) resets all of them and stays on the current slide.

## Layout

```
index.html      built page (committed, so the folder works as it is)
build.py        src/*.html -> index.html
src/            one fragment per part of the talk, one <section> per slide;
                00-motivation.html holds the slides before the first roadmap
theme/          ime.css (IME-USP look), deck.js (runtime), fonts/ (Libertinus, OFL)
figures/static  SVG of every figure
figures/tex     TikZ sources of the figures made for the slides
figures/js      one script per interactive figure, common.js shared
figures/data    numbers of the result charts (generated)
vendor/         reveal.js 5.2.0, MathJax 3.2.2 (tex-svg-full), with their licenses
tools/          build_figures.py, export_data.py, check.py, shots.py
```

The fragments in `src/` are the source of the deck and are edited by hand.

## Make targets (from the repository root)

| Target | What it does |
|---|---|
| `make slides-html` | rebuild `index.html` after editing `src/` |
| `make slides-html-check` | check that every formula of the fragments is in the built page, that every backup link has a target, that no slide overflows and that MathJax reports no error |
| `make slides-html-shots` | screenshot every slide to `slides/.build/shots/`, with contact sheets |
| `make slides-html-pdf` | fallback PDF, one page per slide, from the screenshots |
| `make slides-html-figures` | recompile the static figures from `tex/` and `figures/tex/` (needs pdflatex and pdftocairo) |

`tools/export_data.py` regenerates `figures/data/` from the published
result files; it needs pandas (run it with the project environment).

A figure can be opened in a named state by the URL, which the screenshots
use: `index.html?simplex_cost_partition=hedge#/28`. `?static` shows the
static image of every figure.
