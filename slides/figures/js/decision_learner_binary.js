// The decision learner for K = 2: the validation sample V as points
// (z_i, y_i), z_i = p-hat(x_i), one row per label, and the empirical risk
//     R-hat(tau) = (1/m) sum_i C_{y_i, delta_tau(z_i)}
// of the threshold rule, against tau, with tau on a slider. Dark, outlined
// points are the errors at the current tau. The sample is drawn once (m = 30)
// from the miscalibrated estimate of the empirical-threshold slide,
// P(Y = 1 | z) = 0.42 + 0.5 z, with C01 = 4 and C10 = 2.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C01 = 4, C10 = 2, M = 30, GRID = 301;

  Deck.figures.decision_learner_binary = function (host, state) {
    var rng = fx.rng(115);   // a draw whose best threshold (0.48) errs on both labels, near the population optimum 0.49
    var pts = [];
    for (var i = 0; i < M; i++) {
      var z = rng();
      pts.push({ z: z, y: rng() < 0.42 + 0.5 * z ? 1 : 0, dy: (rng() - 0.5) * 18 });
    }
    var risk = function (t) {
      var s = 0;
      pts.forEach(function (p) { var d = p.z >= t ? 1 : 0; if (d !== p.y) s += p.y === 1 ? C10 : C01; });
      return s / M;
    };
    var grid = [], best = 0;
    for (var g = 0; g < GRID; g++) { var t = g / (GRID - 1); grid.push(risk(t)); if (grid[g] < grid[best]) best = g; }
    var tauEmp = best / (GRID - 1);
    var tau = state.preset === 'best' ? tauEmp : 0.7;

    fx.flex(host, 'column', 4);
    var W = 1340, H = 246, L = 140, R = 1290;
    var svg = fx.svg(host, W, H);
    var base = fx.el('g', {}, svg);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    var slider = fx.slider(bar, { label: 'τ', min: 0, max: 1, step: 1 / (GRID - 1), value: tau, width: 360 }, function (v) { tau = v; draw(); });
    fx.button(bar, 'best τ', function () { tau = tauEmp; slider.set(tau); draw(); });
    var readout = fx.html('div', 'readout', host);
    var X = function (v) { return L + v * (R - L); };
    var ROW = { 1: 22, 0: 58 };
    var T = 82, B = 212, top = Math.max.apply(null, grid) * 1.05;
    var Y = function (v) { return B - v / top * (B - T); };

    fx.text(base, L - 20, ROW[1] + 8, 'y = 1', { 'font-size': 24, 'text-anchor': 'end', fill: fx.red });
    fx.text(base, L - 20, ROW[0] + 8, 'y = 0', { 'font-size': 24, 'text-anchor': 'end', fill: fx.blue });
    fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, base);
    fx.el('line', { x1: L, y1: B, x2: L, y2: T, stroke: fx.grey, 'stroke-width': 2 }, base);
    fx.tex(base, L - 20, (T + B) / 2, '\\hat{R}(\\tau)', { size: 26, anchor: 'end' });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) { fx.text(base, X(v), B + 30, String(v), { 'font-size': 22, fill: fx.grey }); });
    fx.tex(base, R + 30, B + 22, '\\tau', { size: 24 });
    var d = grid.map(function (v, g) { return (g ? 'L' : 'M') + X(g / (GRID - 1)).toFixed(1) + ' ' + Y(v).toFixed(1); }).join(' ');
    fx.el('path', { d: d, fill: 'none', stroke: fx.ink, 'stroke-width': 2.5 }, base);
    fx.el('circle', { cx: X(tauEmp), cy: Y(grid[best]), r: 8, fill: fx.yellow, stroke: fx.ink, 'stroke-width': 2 }, base);
    fx.tex(base, X(tauEmp), Y(grid[best]) + 28, '\\tau_{emp}', { size: 22 });

    function draw() {
      fx.clear(layer);
      var x = X(tau);
      fx.el('line', { x1: x, y1: ROW[1] - 26, x2: x, y2: B, stroke: fx.yellow, 'stroke-width': 3, 'stroke-dasharray': '8 6' }, layer);
      var fp = 0, fn = 0;
      pts.forEach(function (p) {
        var cx = X(p.z), cy = ROW[p.y] + p.dy, wrong = (p.z >= tau ? 1 : 0) !== p.y;
        if (wrong) { if (p.y === 1) fn += 1; else fp += 1; }
        fx.el('circle', { cx: cx, cy: cy, r: wrong ? 11 : 8, fill: p.y ? fx.red : fx.blue, opacity: wrong ? 1 : 0.35, stroke: wrong ? fx.ink : 'none', 'stroke-width': 2.5 }, layer);
      });
      fx.el('circle', { cx: x, cy: Y(risk(tau)), r: 9, fill: fx.ink }, layer);
      readout.innerHTML = 'τ = ' + fx.fmt(tau, 2) + ': ' + fp + ' false positives (cost ' + C01 + '), ' + fn + ' false negatives (cost ' + C10 +
        '), R̂(τ) = ' + fx.fmt(risk(tau), 2) + '; τ<sub>emp</sub> = ' + fx.fmt(tauEmp, 2);
    }
    draw();
  };
})();
