// Calibrate, then decide, with Platt scaling (K = 2). The calibrator is
//     g(q) = sigma(a logit(q) + b),   a > 0,
// drawn over the diagonal (no correction). The rule predicts 1 when
// g(q) >= p*; since g increases, that is q >= g^{-1}(p*), a threshold on the
// raw estimate (black, labelled "calibrated"). The grey line is p* itself,
// where the analytical rule would cut the uncalibrated estimate (labelled
// "no calibrator"). Default: a = 0.55, b = 0.7 as in the four-calibrator
// backup, and p* = 2/3.
(function () {
  'use strict';
  var fx = Deck.fx;
  var sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
  var logit = function (q) { return Math.log(q / (1 - q)); };

  Deck.figures.platt_calibration = function (host, state) {
    var a = 0.55, b = 0.7, ps = 2 / 3;
    if (state.preset === 'identity') { a = 1; b = 0; }
    fx.flex(host, 'row', 50);
    var W = 600, H = 500, L = 80, T = 40, side = 330, B = T + side;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var layer = fx.el('g', {}, svg);
    var right = fx.html('div', '', host);
    right.style.width = '520px';
    var bar = fx.html('div', 'ctl col', right);
    fx.slider(bar, { label: 'slope a', min: 0.2, max: 3, step: 0.01, value: a, width: 260 }, function (v) { a = v; draw(); });
    fx.slider(bar, { label: 'intercept b', min: -2, max: 2, step: 0.01, value: b, width: 260 }, function (v) { b = v; draw(); });
    fx.slider(bar, { label: 'p*', min: 0.05, max: 0.95, step: 0.01, value: ps, width: 260 }, function (v) { ps = v; draw(); });
    var readout = fx.html('div', 'readout', right);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.45';
    var X = function (q) { return L + q * side; };
    var Y = function (v) { return B - v * side; };

    function draw() {
      fx.clear(layer);
      var cut = sigmoid((logit(ps) - b) / a);
      fx.el('rect', { x: X(cut), y: T, width: X(1) - X(cut), height: side, fill: fx.red, opacity: 0.08 }, layer);
      fx.el('rect', { x: L, y: T, width: side, height: side, fill: 'none', stroke: '#bbb', 'stroke-width': 1.5 }, layer);
      fx.el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: '#9a9a9a', 'stroke-width': 2, 'stroke-dasharray': '7 6' }, layer);
      var d = '';
      for (var i = 0; i <= 200; i++) {
        var q = 0.001 + 0.998 * i / 200;
        d += (i ? 'L' : 'M') + X(q).toFixed(1) + ' ' + Y(sigmoid(a * logit(q) + b)).toFixed(1) + ' ';
      }
      fx.el('path', { d: d, fill: 'none', stroke: fx.red, 'stroke-width': 4 }, layer);
      fx.el('line', { x1: L, y1: Y(ps), x2: X(1), y2: Y(ps), stroke: fx.yellow, 'stroke-width': 3, 'stroke-dasharray': '8 6' }, layer);
      // The two cuts on the raw estimate, labelled under the axis on two rows.
      fx.el('line', { x1: X(ps), y1: T, x2: X(ps), y2: B + 80, stroke: '#9a9a9a', 'stroke-width': 2.5, 'stroke-dasharray': '6 5' }, layer);
      fx.el('line', { x1: X(cut), y1: Y(ps), x2: X(cut), y2: B + 22, stroke: fx.ink, 'stroke-width': 3 }, layer);
      fx.tex(layer, X(cut), B + 36, 'g^{-1}(p^*)', { size: 22 });
      fx.text(layer, X(cut), B + 66, 'calibrated', { 'font-size': 20 });
      fx.tex(layer, X(ps), B + 96, 'p^*', { size: 22, color: fx.grey });
      fx.text(layer, X(ps), B + 124, 'no calibrator', { 'font-size': 20, fill: fx.grey });
      fx.tex(layer, L - 12, Y(ps), 'p^*', { size: 24, anchor: 'end' });
      fx.text(layer, L - 12, Y(0) + 8, '0', { 'font-size': 22, 'text-anchor': 'end', fill: fx.grey });
      fx.text(layer, L - 12, Y(1) + 8, '1', { 'font-size': 22, 'text-anchor': 'end', fill: fx.grey });
      fx.tex(layer, X(1) + 26, B, '\\hat{p}', { size: 24 });
      fx.tex(layer, L + 8, T - 20, 'g(\\hat{p})', { size: 24, anchor: 'start' });
      fx.math(readout, 'calibrated: predict $1$ when $g(\\hat{p}) \\geq p^*$,<br>that is $\\hat{p} \\geq ' + fx.fmt(cut, 2) + '$<br>' +
        'no calibrator: $\\hat{p} \\geq p^* = ' + fx.fmt(ps, 2) + '$');
    }
    draw();
  };
})();
