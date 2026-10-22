// Schematic reliability diagram: P(Y = 1 | p_hat = q) against q. The
// diagonal is perfect calibration. The distorted curve is the Bezier curve
// of the static figure, blended with the diagonal by the distortion slider
// (1 = the static figure, 0 = calibrated). A threshold on the estimate is
// read on the curve: at 0.09 it selects points of true frequency about 0.22.
(function () {
  'use strict';
  var fx = Deck.fx;

  // The curve of tex/figures/slides/calibration_sketch.tex:
  // (0, 0.18) .. controls (0.35, 0.30) and (0.65, 0.70) .. (1, 0.82).
  function bezier(q) {
    var lo = 0, hi = 1, t = 0.5;
    for (var i = 0; i < 40; i++) {
      t = (lo + hi) / 2;
      var x = 3 * (1 - t) * (1 - t) * t * 0.35 + 3 * (1 - t) * t * t * 0.65 + t * t * t;
      if (x < q) lo = t; else hi = t;
    }
    var u = 1 - t;
    return u * u * u * 0.18 + 3 * u * u * t * 0.30 + 3 * u * t * t * 0.70 + t * t * t * 0.82;
  }

  Deck.figures.calibration_sketch = function (host, state) {
    var d = 1, thr = 0.09;
    if (state.preset === 'calibrated') d = 0;
    if (state.preset === 'half') thr = 0.5;
    fx.flex(host, 'column', 4);
    var W = 520, H = 420, L = 86, B = 350, side = 300;
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl col', host);
    bar.style.gap = '2px';
    fx.slider(bar, { label: 'distortion', min: 0, max: 1.3, step: 0.01, value: d, width: 200 }, function (v) { d = v; draw(); });
    fx.slider(bar, { label: 'threshold', min: 0.01, max: 0.99, step: 0.01, value: thr, width: 200 }, function (v) { thr = v; draw(); });

    function freq(q) { return Math.min(1, Math.max(0, q + d * (bezier(q) - q))); }
    function X(q) { return L + q * side; }
    function Y(v) { return B - v * side; }

    function draw() {
      fx.clear(layer);
      fx.el('line', { x1: X(0), y1: B, x2: X(1) + 26, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: X(0), y1: B, x2: X(0), y2: Y(1) - 22, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: fx.blue, 'stroke-width': 3, 'stroke-dasharray': '9 7' }, layer);
      var path = '';
      for (var i = 0; i <= 100; i++) path += (i ? 'L' : 'M') + X(i / 100).toFixed(1) + ' ' + Y(freq(i / 100)).toFixed(1) + ' ';
      fx.el('path', { d: path, fill: 'none', stroke: fx.red, 'stroke-width': 5 }, layer);
      var f = freq(thr);
      fx.el('path', { d: 'M' + X(thr) + ' ' + B + ' L' + X(thr) + ' ' + Y(f) + ' L' + X(0) + ' ' + Y(f), fill: 'none', stroke: fx.ink, 'stroke-width': 2, 'stroke-dasharray': '3 5' }, layer);
      fx.el('circle', { cx: X(thr), cy: Y(thr), r: 6, fill: fx.blue }, layer);
      fx.el('circle', { cx: X(thr), cy: Y(f), r: 7, fill: fx.red }, layer);
      fx.tex(layer, X(0) - 10, Y(1) - 46, '\\mathbb{P}(Y = 1 \\mid \\hat{p}(X) = q)', { size: 23, anchor: 'start' });
      fx.tex(layer, X(1) + 40, B, 'q', { size: 24 });
      fx.tex(layer, X(1), B + 24, '1', { size: 22 });
      fx.tex(layer, X(0) - 16, Y(1), '1', { size: 22 });
      fx.tex(layer, X(thr), B + 24, fx.fmt(thr, 2), { size: 22 });
      fx.tex(layer, X(0) - 12, Y(f), fx.fmt(f, 2), { size: 22, anchor: 'end' });
      fx.text(layer, X(.80), Y(.80) - 16, 'calibrated', { fill: fx.blue, 'font-size': 22, transform: 'rotate(-45 ' + X(.80) + ' ' + (Y(.80) - 16) + ')' });
    }
    draw();
  };
})();
