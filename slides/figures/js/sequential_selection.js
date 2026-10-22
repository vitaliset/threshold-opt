// The sequential cascade (SequentialSelection) for K = 3: the classes are
// tested in a priority order, and the first whose probability reaches its
// threshold is predicted; if none does, argmax decides. Each piece of a
// region is an intersection of half-planes with the simplex, so the pieces
// are clipped one by one; their union per class need not be convex, and it
// changes with the order. The argmax boundaries are dashed.
(function () {
  'use strict';
  var fx = Deck.fx;
  var TRI = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  var ARGMAX = fx.regions([0, 1, 2].map(function (k) { return function (p) { return -p[k]; }; }));

  // Pieces of each class region for the order `ord` and thresholds `tau`,
  // and, for each test, the thin strip around p_k = tau_k where it is decisive.
  function pieces(ord, tau) {
    var out = [[], [], []];
    out.strips = [];
    var rest = TRI;  // where no earlier test fired
    ord.forEach(function (k) {
      if (rest.length >= 3 && tau[k] > 0.001 && tau[k] < 0.999) {
        var strip = fx.clip(fx.clip(rest, function (p) { return p[k] - tau[k] - 0.004; }), function (p) { return tau[k] - 0.004 - p[k]; });
        if (strip.length >= 3) out.strips.push([k, strip]);
      }
      var fires = fx.clip(rest, function (p) { return tau[k] - p[k]; });       // p_k >= tau_k
      if (fires.length >= 3) out[k].push(fires);
      rest = rest.length ? fx.clip(rest, function (p) { return p[k] - tau[k]; }) : rest;  // p_k <= tau_k
    });
    if (rest.length >= 3) {
      // `rest` is convex: clip it by the half-planes of each argmax cell.
      [0, 1, 2].forEach(function (k) {
        var piece = rest;
        [0, 1, 2].forEach(function (j) {
          if (j !== k && piece.length) piece = fx.clip(piece, function (p) { return p[j] - p[k]; });
        });
        if (piece.length >= 3) out[k].push(piece);
      });
    }
    return out;
  }

  Deck.figures.sequential_selection = function (host, state) {
    var ord = [2, 1, 0], tau = [1, 0.4, 0.3];
    if (state.preset === 'argmax') tau = [1, 1, 1];
    fx.flex(host, 'row', 40);
    host.style.alignItems = 'center';
    var W = 520, H = 340;
    var svg = fx.svg(host, W, H);
    var S = fx.simplex(W / 2, 40, 330);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 24);
    var side = fx.html('div', 'ctl', host);
    side.style.flexDirection = 'column';
    side.style.alignItems = 'flex-start';
    side.style.gap = '10px';
    fx.html('div', 'readout', side, 'order of the tests');
    var orders = { '210': [2, 1, 0], '120': [1, 2, 0], '012': [0, 1, 2] };
    var row = fx.html('div', '', side);
    fx.choice(row, [['210', '2 → 1 → 0'], ['120', '1 → 2 → 0'], ['012', '0 → 1 → 2']], '210', function (key) { ord = orders[key]; draw(); });
    var sliders = fx.html('div', '', side);
    sliders.style.display = 'flex';
    sliders.style.flexDirection = 'column';
    sliders.style.gap = '4px';
    [0, 1, 2].forEach(function (k) {
      fx.slider(sliders, { label: 'τ<sub>' + k + '</sub>', min: 0, max: 1, step: 0.01, value: tau[k], width: 200 }, function (v) { tau[k] = v; draw(); });
    });

    function draw() {
      fx.clear(layer);
      var parts = pieces(ord, tau);
      parts.forEach(function (list, k) {
        list.forEach(function (poly) {
          fx.el('path', { d: S.path(poly), fill: fx.fill[k], stroke: fx.fill[k], 'stroke-width': 1, 'stroke-linejoin': 'round' }, layer);
        });
      });
      ARGMAX.forEach(function (poly) {
        fx.el('path', { d: S.path(poly), fill: 'none', stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: 0.5 }, layer);
      });
      // Each test's threshold, only where it decides (no earlier test fired).
      parts.strips.forEach(function (s) {
        fx.el('path', { d: S.path(s[1]), fill: fx.cls[s[0]], stroke: 'none', opacity: 0.9 }, layer);
      });
      S.outline(layer);
      parts.forEach(function (list, k) {
        var best = null, area = 0;
        list.forEach(function (poly) { var a = fx.area(poly); if (a > area) { area = a; best = poly; } });
        if (!best || area < 0.02) return;
        var c = S.xy(fx.centroid(best));
        fx.tex(layer, c[0], c[1], '\\mathcal{P}_' + k, { color: fx.cls[k], size: 26 });
      });
    }
    draw();
  };
})();
