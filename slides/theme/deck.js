// Runtime of the deck: typeset the math, mount the interactive figures,
// draw the header map and the slide number, then start reveal.js.
//
// URL switches, used by slides/tools/check.py:
//   ?static     keep every figure as its static image
//   ?check      write an overflow and math-error report into #check-report
(function () {
  'use strict';

  var query = new URLSearchParams(window.location.search);
  var Deck = window.Deck = window.Deck || {};
  Deck.figures = Deck.figures || {};

  // Title bar: the title and, on a slide that presents one of the four
  // contributions, its name. Footer: author, title and subtitle of the
  // talk, the current part and the slide number.
  var CONTRIBUTIONS = { 1: 'Common framework', 2: 'Voronoi prototypes', 3: 'Benchmark', 4: 'Sensitivity to K', 5: 'Library' };

  function decorate() {
    document.querySelectorAll('.slides > section').forEach(function (s) {
      var h2 = s.querySelector('h2');
      if (!h2) return;
      var text = document.createElement('span');
      text.className = 'title-text';
      while (h2.firstChild) text.appendChild(h2.firstChild);
      h2.appendChild(text);
      if (s.dataset.contrib) {
        var tag = document.createElement('span');
        tag.className = 'contrib-tag';
        tag.textContent = 'Contribution: ' + CONTRIBUTIONS[s.dataset.contrib];
        h2.appendChild(tag);
      }
      var where = s.hasAttribute('data-backup') ? 'Backup'
        : (s.dataset.partName ? s.dataset.partName + '\u2002\u00b7\u2002' : '') + s.dataset.n + '/' + s.dataset.total;
      var foot = document.createElement('div');
      foot.className = 'slide-foot';
      [['Carlo Lemos\u2002\u00b7\u2002Argmax is not enough: Empirical threshold and partition optimization for binary and multiclass classification', 'l'],
       [where, 'r']].forEach(function (item) {
        var span = document.createElement('span');
        span.className = item[1];
        span.textContent = item[0];
        foot.appendChild(span);
      });
      s.appendChild(foot);
    });
  }

  // A figure becomes interactive when a script registered Deck.figures[name].
  // The static image stays in the page: it is what the print layout shows.
  function mountFigures() {
    document.querySelectorAll('figure.fig[data-fig]').forEach(function (fig) {
      // With ?static a figure keeps its static image, if it has one.
      if (query.has('static') && fig.querySelector('img, .static')) return;
      var make = Deck.figures[fig.dataset.fig];
      if (!make) return;
      var state = {};
      try { state = JSON.parse(fig.dataset.state || '{}'); } catch (e) { state = {}; }
      var preset = query.get(fig.dataset.fig);
      if (preset) state.preset = preset;
      var host = document.createElement('div');
      host.className = 'demo';
      fig.appendChild(host);
      fig.classList.add('live');
      make(host, state, fig);
    });
  }

  // A paragraph of the slide body that fits on one line is centred. Blocks,
  // lists, columns and figures keep their own alignment.
  function centreShortParagraphs() {
    document.querySelectorAll('.slides > section').forEach(function (s) {
      var shown = s.style.display, hidden = s.hidden;
      s.style.display = 'flex';
      s.hidden = false;
      s.querySelectorAll('.col').forEach(function (c) { if (c.offsetWidth < 760) c.classList.add('narrow'); });
      s.querySelectorAll('.body > p, .body > div[class^="fs-"] > p').forEach(function (p) {
        if (p.querySelector('.dm') || p.closest('.refs')) return;
        var line = parseFloat(getComputedStyle(p).lineHeight);
        if (p.offsetHeight < 1.6 * line) p.classList.add('one-line');
      });
      s.style.display = shown;
      s.hidden = hidden;
    });
  }

  // Following a link to a backup slide (B.., P..) remembers where it was
  // followed from. A button at the bottom left, and the Backspace key, go
  // back there; the button disappears once back.
  function backLinks() {
    var origin = null;
    var button = document.createElement('button');
    button.className = 'back-link';
    button.type = 'button';
    button.hidden = true;
    document.querySelector('.reveal').appendChild(button);
    var back = function () {
      if (!origin) return;
      var to = origin;
      origin = null;
      button.hidden = true;
      Reveal.slide(to.h, to.v);
    };
    document.querySelectorAll('.slides a.bk').forEach(function (a) {
      a.addEventListener('click', function () {
        if (origin) return;                     // keep the first origin through a chain of links
        origin = Reveal.getIndices();
        var s = Reveal.getCurrentSlide();
        var name = s.dataset.n ? 'slide ' + s.dataset.n : (s.id || 'the previous slide');
        button.textContent = '\u2190 back to ' + name;
        button.hidden = false;
      });
    });
    button.addEventListener('click', back);
    document.addEventListener('keydown', function (ev) {
      var t = ev.target;
      if (ev.key === 'Backspace' && origin && !(t && t.tagName === 'INPUT')) { ev.preventDefault(); back(); }
    });
    Reveal.on('slidechanged', function () {
      if (origin && Reveal.getIndices().h === origin.h) { origin = null; button.hidden = true; }
    });
  }

  // Slides whose content does not fit, and formulas MathJax could not read.
  function report() {
    var out = { overflow: [], mathErrors: 0, slides: 0, live: 0 };
    document.querySelectorAll('.slides > section').forEach(function (s, i) {
      out.slides += 1;
      var body = s.querySelector('.body');
      if (!body) return;
      // reveal.js hides the slides far from the current one; show it to measure.
      var shown = s.style.display, hidden = s.hidden;
      s.style.display = 'flex';
      s.hidden = false;
      var dy = body.scrollHeight - body.clientHeight;
      var dx = body.scrollWidth - body.clientWidth;
      s.style.display = shown;
      s.hidden = hidden;
      if (dy > 2 || dx > 2) {
        var h2 = s.querySelector('h2');
        out.overflow.push({ index: i, title: h2 ? h2.firstChild.textContent : '', dy: dy, dx: dx });
      }
    });
    out.mathErrors = document.querySelectorAll('mjx-merror, [data-mjx-error]').length;
    out.live = document.querySelectorAll('figure.fig.live').length;
    var pre = document.getElementById('check-report');
    pre.textContent = JSON.stringify(out);
  }

  function start() {
    decorate();
    mountFigures();
    return Reveal.initialize({
      width: 1600,
      height: 900,
      margin: 0,
      minScale: 0.1,
      maxScale: 4,
      center: false,
      display: 'flex',
      hash: true,
      controls: false,
      progress: true,
      slideNumber: false,
      transition: 'none',
      backgroundTransition: 'none',
      pdfSeparateFragments: false,
      // Arrow keys and the mouse belong to a focused slider, not to the deck.
      keyboardCondition: function (event) {
        var t = event.target;
        return !(t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'BUTTON'));
      },
      plugins: [RevealNotes]
    }).then(function () {
      centreShortParagraphs();
      backLinks();
      if (query.has('check')) report();
      document.documentElement.classList.add('deck-ready');
    });
  }

  MathJax.startup.promise
    .then(function () { return MathJax.typesetPromise(); })
    .then(start);
})();
