// Share of sampled configurations that satisfy the DLC, against K, for the
// samplers that can break it (left panel of the dissertation figure
// sampler_metrics_vs_k: means over 20 trials of 300 draws). Sequential d^2
// and Shrink satisfy it on every draw by construction, drawn as the dashed
// line at 1.
(function () {
  'use strict';
  var fx = Deck.fx;
  var KS = [3, 4, 5, 6, 8, 12, 16, 20, 30, 40, 50, 75, 100];
  var CURVES = [
    { name: 'Swap-max', color: '#2CA02C', v: [0.999, 0.996, 0.99, 0.988, 0.981, 0.96, 0.939, 0.916, 0.875, 0.829, 0.789, 0.688, 0.614] },
    { name: 'Concentrated Dir', color: '#9467BD', v: [0.973, 0.925, 0.858, 0.771, 0.577, 0.237, 0.065, 0.013, 0, 0, 0, 0, 0] },
    { name: 'Hungarian', color: '#FF7F0E', v: [0.653, 0.409, 0.263, 0.165, 0.06, 0.006, 0.001, 0, 0, 0, 0, 0, 0] }
  ];

  Deck.figures.sampler_dlc_vs_k = function (host, state) {
    fx.flex(host, 'column', 4);
    var W = 620, H = 420, L = 80, R = 600, T = 60, B = 340;
    var svg = fx.svg(host, W, H);
    var X = function (k) { return L + (Math.log(k) - Math.log(3)) / (Math.log(100) - Math.log(3)) * (R - L); };
    var Y = function (v) { return B - v * (B - T); };
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#e3e3e3', 'stroke-width': 1.5 }, svg);
      fx.text(svg, L - 10, Y(t) + 8, String(t), { 'font-size': 22, 'text-anchor': 'end' });
    });
    [3, 10, 30, 100].forEach(function (k) { fx.text(svg, X(k), B + 30, String(k), { 'font-size': 22 }); });
    fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, svg);
    fx.tex(svg, (L + R) / 2, B + 64, '\\text{number of classes } K', { size: 24 });
    fx.text(svg, L, 22, 'share of draws that satisfy the DLC', { 'font-size': 22, 'text-anchor': 'start', fill: fx.grey });
    fx.el('line', { x1: L, y1: Y(1), x2: R, y2: Y(1), stroke: fx.ink, 'stroke-width': 3, 'stroke-dasharray': '9 6' }, svg);
    fx.text(svg, R, Y(1) - 10, 'sequential d², shrink: always', { 'font-size': 21, 'text-anchor': 'end' });
    CURVES.forEach(function (c, n) {
      var d = c.v.map(function (v, i) { return (i ? 'L' : 'M') + X(KS[i]).toFixed(1) + ' ' + Y(v).toFixed(1); }).join(' ');
      fx.el('path', { d: d, fill: 'none', stroke: c.color, 'stroke-width': 3.5 }, svg);
      c.v.forEach(function (v, i) { fx.el('circle', { cx: X(KS[i]), cy: Y(v), r: 4.5, fill: c.color }, svg); });
    });
    // Labels next to their curves.
    fx.text(svg, X(40), Y(0.829) + 34, 'swap-max', { 'font-size': 22, fill: '#2CA02C' });
    fx.text(svg, X(9.5), Y(0.42), 'concentrated Dir', { 'font-size': 22, fill: '#9467BD', 'text-anchor': 'start' });
    fx.text(svg, X(4.3), Y(0.2), 'Hungarian', { 'font-size': 22, fill: '#FF7F0E', 'text-anchor': 'start' });
  };
})();
