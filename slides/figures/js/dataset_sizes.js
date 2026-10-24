// The datasets of the two benchmarks: one row each, the sample count on a
// log axis. The binary chart also gives the imbalance ratio (majority over
// minority class size), the multiclass chart the number of classes K. The
// numbers are those of the dissertation's dataset tables
// (figures/data/datasets.js, written by slides/tools/export_data.py).
(function () {
  'use strict';
  var fx = Deck.fx;

  function chart(scope, extraHead, extraOf) {
    return function (host) {
      var rows = Deck.data.datasets[scope];
      var rh = 36, L = 250, R = 480, T = 36, W = 740, H = T + rows.length * rh + 44;
      var svg = fx.svg(host, W, H);
      var lo = Math.log(100), hi = Math.log(100000);
      var X = function (n) { return L + (Math.log(n) - lo) / (hi - lo) * (R - L); };
      [100, 1000, 10000, 100000].forEach(function (t) {
        fx.el('line', { x1: X(t), y1: T, x2: X(t), y2: T + rows.length * rh, stroke: '#dcdcdc', 'stroke-width': 1.5 }, svg);
        fx.tex(svg, X(t), T + rows.length * rh + 22, '10^{' + Math.round(Math.log10(t)) + '}', { size: 22 });
      });
      fx.text(svg, (L + R) / 2, T - 10, 'samples', { 'font-size': 26, fill: fx.grey });
      fx.text(svg, R + 160, T - 10, extraHead, { 'font-size': 26, fill: fx.grey });
      rows.forEach(function (row, i) {
        var y = T + (i + 0.5) * rh;
        fx.text(svg, L - 14, y + 9, row.name, { 'font-size': 28, 'text-anchor': 'end' });
        fx.el('line', { x1: X(100), y1: y, x2: X(row.n), y2: y, stroke: fx.softblue, 'stroke-width': 4 }, svg);
        fx.el('circle', { cx: X(row.n), cy: y, r: 8, fill: fx.blue }, svg);
        fx.text(svg, X(row.n) + 14, y + 8, row.n.toLocaleString('en-US'), { 'font-size': 24, 'text-anchor': 'start', fill: fx.grey });
        fx.text(svg, R + 160, y + 9, extraOf(row), { 'font-size': 28 });
      });
    };
  }

  Deck.figures.dataset_sizes_binary = chart('binary', 'imbalance ratio', function (r) { return r.ir.toFixed(1); });
  Deck.figures.dataset_sizes_multiclass = chart('multiclass', 'classes K', function (r) { return String(r.K); });
})();
