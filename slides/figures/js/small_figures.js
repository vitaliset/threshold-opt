// Four small figures added to slides that had no picture.
//   capacity_bounds  the capacity values and bounds of Theorems 7.7.2 to
//                    7.7.5 against K.
//   hedge_simplex    the Bayes partition of the hedge matrix (K = 3).
//   plugin_scatter   the plug-in rule for K = 2: only the points where
//                    p_hat and eta fall on opposite sides of the threshold
//                    cost anything, and the excess risk is below the bound
//                    of Proposition 2.4.1.
(function () {
  'use strict';
  var fx = Deck.fx;

  function costScores(C) {
    return [0, 1, 2].map(function (i) { return function (p) { return C[0][i] * p[0] + C[1][i] * p[1] + C[2][i] * p[2]; }; });
  }

  // ---- Capacity of the hypothesis classes ----
  Deck.figures.capacity_bounds = function (host, state) {
    var K = state.preset === 'ten' ? 10 : 3;
    fx.flex(host, 'row', 50);
    var W = 860, H = 520, L = 100, R = 820, T = 20, B = 450;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var layer = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    side.style.width = '560px';
    var bar = fx.html('div', 'ctl', side);
    bar.style.justifyContent = 'flex-start';
    fx.slider(bar, { label: 'K', min: 2, max: 20, step: 1, value: K, digits: 0, width: 300 }, function (v) { K = v; draw(); });
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.45';
    var CURVES = [
      { f: function (k) { return k - 1; }, color: fx.blue, label: 'K − 1' },
      { f: function (k) { return k * k; }, color: fx.softblue, label: 'K²' }
    ];
    var X = function (k) { return L + (k - 2) / 18 * (R - L); };
    var Y = function (v) { return B - v / 430 * (B - T); };

    function draw() {
      fx.clear(layer);
      // K of the benchmark: 2 (binary) and 3 to 10 (multiclass).
      fx.el('rect', { x: X(2), y: T, width: X(10) - X(2), height: B - T, fill: fx.yellow, opacity: 0.16 }, layer);
      fx.text(layer, (X(2) + X(10)) / 2, T + 28, 'K of the benchmark', { 'font-size': 23, fill: '#7a5a00' });
      [0, 100, 200, 300, 400].forEach(function (t) {
        fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#e3e3e3', 'stroke-width': 1.5 }, layer);
        fx.text(layer, L - 12, Y(t) + 9, String(t), { 'font-size': 25, 'text-anchor': 'end' });
      });
      [2, 5, 10, 15, 20].forEach(function (k) { fx.text(layer, X(k), B + 30, String(k), { 'font-size': 25 }); });
      fx.tex(layer, (L + R) / 2, B + 58, 'K', { size: 28 });
      fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      CURVES.forEach(function (c) {
        var d = '';
        for (var k = 2; k <= 20; k += 0.25) d += (d ? 'L' : 'M') + X(k).toFixed(1) + ' ' + Y(c.f(k)).toFixed(1) + ' ';
        fx.el('path', { d: d, fill: 'none', stroke: c.color, 'stroke-width': 4 }, layer);
        fx.el('circle', { cx: X(K), cy: Y(c.f(K)), r: 8, fill: c.color, stroke: '#fff', 'stroke-width': 2 }, layer);
      });
      fx.el('line', { x1: X(K), y1: B, x2: X(K), y2: T, stroke: fx.ink, 'stroke-width': 2, 'stroke-dasharray': '6 6' }, layer);
      var dot = function (c) { return '<span style="color:' + c + '">●</span> '; };
      fx.math(readout,
        dot(fx.blue) + '$d_{Nat}(\\mathcal{H}_{PS}) = ' + (K - 1) + '$, &nbsp; $d_{Nat}(\\mathcal{H}_{Sc}) \\leq ' + (K - 1) + '$<br>' +
        dot(fx.softblue) + '$d_{Nat}(\\mathcal{H}_{Vor}) = d_{Nat}(\\mathcal{H}_{Full}) \\leq ' + K * K + '$<br>' +
        dot(fx.softblue) + '$d_{Nat}(\\mathcal{H}_{Vor}^{\\Delta}) \\leq ' + K * K + '$' +
        '<br><br>same gap term of the bound:<br><strong>' + fx.fmt(K * K / (K - 1), 1) + ' times</strong> the validation data under the<br>full-matrix bound, against the parallel shift');
    }
    draw();
  };

  // ---- Admissible cost matrices: the hedge matrix ----
  Deck.figures.hedge_simplex = function (host) {
    var C = [[0, 10, 2], [10, 0, 2], [3, 3, 5]];
    var W = 330, H = 268;
    var svg = fx.svg(host, W, H);
    var S = fx.simplex(W / 2, 26, 250);
    var regions = fx.regions(costScores(C));
    fx.drawRegions(svg, S, regions);
    S.outline(svg);
    S.labels(svg, 20);
    regions.forEach(function (poly, k) {
      if (fx.area(poly) < 0.01) return;
      var c = S.xy(fx.centroid(poly));
      fx.tex(svg, c[0], c[1], '\\mathcal{P}_' + k + '^*', { color: fx.cls[k], size: 22 });
    });
    var q = S.xy([.5, .5, 0]);
    fx.el('circle', { cx: q[0], cy: q[1], r: 6, fill: fx.ink }, svg);
  };

  // ---- The plug-in rule ----
  Deck.figures.plugin_scatter = function (host, state) {
    var thr = state.preset === 'half' ? 0.5 : 0.25, noise = state.preset === 'noisy' ? 0.3 : 0.12;
    var rng = fx.rng(8), base = [];
    for (var i = 0; i < 260; i++) base.push([rng(), rng.normal()]);
    fx.flex(host, 'row', 50);
    var W = 600, H = 520, L = 110, T = 16, side = 440, B = T + side;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var layer = fx.el('g', {}, svg);
    var panel = fx.html('div', '', host);
    panel.style.width = '760px';
    var bar = fx.html('div', 'ctl col', panel);
    fx.slider(bar, { label: 'threshold p*', min: 0.05, max: 0.95, step: 0.01, value: thr, width: 200 }, function (v) { thr = v; draw(); });
    fx.slider(bar, { label: 'estimation error', min: 0, max: 0.4, step: 0.01, value: noise, width: 200 }, function (v) { noise = v; draw(); });
    var readout = fx.html('div', 'readout', panel);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.4';
    var X = function (v) { return L + v * side; }, Y = function (v) { return B - v * side; };

    function draw() {
      fx.clear(layer);
      // Costs with zero diagonal and C01 + C10 = 1, so that p* = C01.
      var cmax = Math.max(thr, 1 - thr), excess = 0, l1 = 0, wrong = 0;
      [[0, thr, thr, 1], [thr, 1, 0, thr]].forEach(function (r) {
        fx.el('rect', { x: X(r[0]), y: Y(r[3]), width: X(r[1]) - X(r[0]), height: Y(r[2]) - Y(r[3]), fill: fx.red, opacity: 0.13 }, layer);
      });
      fx.el('rect', { x: L, y: T, width: side, height: side, fill: 'none', stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: '#aaa', 'stroke-width': 1.5, 'stroke-dasharray': '5 5' }, layer);
      fx.el('line', { x1: X(thr), y1: B, x2: X(thr), y2: T, stroke: fx.ink, 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, layer);
      fx.el('line', { x1: L, y1: Y(thr), x2: L + side, y2: Y(thr), stroke: fx.ink, 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, layer);
      base.forEach(function (b) {
        var eta = b[0], p = Math.min(1, Math.max(0, eta + noise * b[1]));
        var differs = (eta >= thr) !== (p >= thr);
        l1 += 2 * Math.abs(p - eta);
        if (differs) { wrong += 1; excess += Math.abs(eta - thr); }
        fx.el('circle', { cx: X(eta), cy: Y(p), r: differs ? 7 : 5, fill: differs ? fx.red : '#777', opacity: differs ? 1 : 0.6 }, layer);
      });
      fx.tex(layer, L + side / 2, B + 48, '\\eta(x)', { size: 26 });
      fx.tex(layer, L - 60, T + side / 2, '\\hat{p}(x)', { size: 26 });
      fx.tex(layer, X(thr), B + 16, 'p^*', { size: 22 });
      var n = base.length;
      fx.math(readout, 'shaded: $\\hat{h}(x) \\neq h^*(x)$, <strong>' + Math.round(100 * wrong / n) + '%</strong> of the points<br>' +
        'excess risk $R(\\hat{h}) - R(h^*) = ' + fx.fmt(excess / n, 3) + '$<br>' +
        'bound $C_{max} \\, \\mathbb{E}\\|\\hat{\\mathbf{p}} - \\boldsymbol{\\eta}\\|_1 = ' + fx.fmt(cmax * l1 / n, 3) + '$');
    }
    draw();
  };
})();
