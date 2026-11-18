// Lemma 9.2.3 by simulation. The perturbation eps is uniform on the sphere
// of radius r in V = {v : sum v_k = 0}, of dimension K - 1, and
//     phi(p + eps) - phi(p) = g' eps + M |eps|^2
// with g in V, |g| = 1 (the test functional is quadratic, so the remainder
// bound of the lemma holds with constant M). Then, exactly,
//     E[(phi(p + eps) - phi(p))^2] = r^2 / (K - 1) + M^2 r^4,
// whose first term is the rate of the lemma. The chart divides by r^2: the
// leading term is 1 / (K - 1), dots are Monte Carlo means over N draws.
(function () {
  'use strict';
  var fx = Deck.fx;
  var KS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20];

  // Mean of (g' eps + M r^2)^2 / r^2 over n draws, g = (e_0 - e_1) / sqrt 2.
  function simulate(rng, K, r, M, n) {
    var total = 0;
    for (var i = 0; i < n; i++) {
      var z = [], mean = 0, k;
      for (k = 0; k < K; k++) { z.push(rng.normal()); mean += z[k] / K; }
      var norm = 0;
      for (k = 0; k < K; k++) { z[k] -= mean; norm += z[k] * z[k]; }
      norm = Math.sqrt(norm);
      var ge = r * (z[0] - z[1]) / norm / Math.SQRT2;
      var d = ge + M * r * r;
      total += d * d;
    }
    return total / n / (r * r);
  }

  Deck.figures.dimensional_dilution = function (host, state) {
    var r = 0.05, M = 0, e = 9, seed = 3;
    if (state.preset === 'remainder') { r = 0.3; M = 2; }
    if (state.preset === 'few') { e = 5; }
    fx.flex(host, 'column', 6);
    var W = 1320, H = 450, L = 120, R = 1270, T = 20, B = 362;
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    fx.slider(bar, { label: 'radius r', min: 0.01, max: 0.4, step: 0.01, value: r, width: 220 }, function (v) { r = v; draw(); });
    fx.slider(bar, { label: 'curvature M', min: 0, max: 4, step: 0.1, value: M, digits: 1, width: 220 }, function (v) { M = v; draw(); });
    fx.slider(bar, { label: 'draws N', min: 4, max: 13, step: 1, value: e, width: 220,
      format: function (v) { return String(Math.pow(2, v)); } }, function (v) { e = v; draw(); });
    fx.button(bar, 'draw again', function () { seed += 1; draw(); });
    var readout = fx.html('div', 'readout', host);
    var X = function (K) { return L + (K - 2) / 18 * (R - L - 40) + 20; };
    var ymax = 1.25;
    var Y = function (v) { return B - Math.min(v, ymax) / ymax * (B - T); };

    function draw() {
      fx.clear(layer);
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
        fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#e3e3e3', 'stroke-width': 1.5 }, layer);
        fx.text(layer, L - 12, Y(t) + 9, String(t), { 'font-size': 26, 'text-anchor': 'end' });
      });
      fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: L, y1: B, x2: L, y2: T, stroke: fx.grey, 'stroke-width': 2 }, layer);
      [2, 4, 6, 8, 10, 12, 14, 16, 18, 20].forEach(function (K) { fx.text(layer, X(K), B + 32, String(K), { 'font-size': 26 }); });
      fx.tex(layer, (L + R) / 2, B + 68, '\\text{number of classes } K', { size: 28 });
      fx.tex(layer, R - 20, T + 80, '\\dfrac{\\mathbb{E}[(\\phi(\\mathbf{p}+\\boldsymbol{\\epsilon})-\\phi(\\mathbf{p}))^2]}{r^2\\,\\|\\mathbf{g}\\|^2}', { size: 26, anchor: 'end' });
      var lead = '', exact = '';
      for (var K = 2; K <= 20; K += 0.1) {
        lead += (K > 2 ? 'L' : 'M') + X(K).toFixed(1) + ' ' + Y(1 / (K - 1)).toFixed(1) + ' ';
        exact += (K > 2 ? 'L' : 'M') + X(K).toFixed(1) + ' ' + Y(1 / (K - 1) + M * M * r * r).toFixed(1) + ' ';
      }
      if (M > 0) fx.el('path', { d: exact, fill: 'none', stroke: fx.red, 'stroke-width': 3, 'stroke-dasharray': '9 7' }, layer);
      fx.el('path', { d: lead, fill: 'none', stroke: fx.ink, 'stroke-width': 4 }, layer);
      var rng = fx.rng(seed * 104729 + e), n = Math.pow(2, e);
      KS.forEach(function (K) {
        var v = simulate(rng, K, r, M, n);
        fx.el('circle', { cx: X(K), cy: Y(v), r: 9, fill: fx.softblue, stroke: '#fff', 'stroke-width': 2 }, layer);
      });
      fx.tex(layer, X(4.4), Y(1 / 3.4) - 30, '\\dfrac{1}{K-1}', { size: 26, anchor: 'start' });
      if (M > 0) fx.tex(layer, X(11), Y(1 / 10 + M * M * r * r) - 34, '\\dfrac{1}{K-1} + M^2 r^2', { size: 28, anchor: 'start', color: fx.red });
      readout.innerHTML = 'dots: Monte Carlo means over ' + n + ' draws per K' +
        (M > 0 ? '; remainder M²r⁴, divided by r²: ' + fx.fmt(M * M * r * r, 4) + ' (within the O(r³) of the lemma)' : '');
    }
    draw();
  };
})();
