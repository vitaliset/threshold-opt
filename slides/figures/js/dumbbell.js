// Paired win-or-tie rates: two dots per item joined by a line, in place of
// a table of pairs. The numbers come from figures/data/win_rates.js and
// equal those of the tables they replace (kept in the page as the static
// version). Whiskers are 95% bootstrap intervals over cells.
//   dumbbell_models:   Analytical against EmpiricalCV, per base classifier
//                      (binary cost cells).
//   dumbbell_split_cv: the 10% split against 5-fold CV for three tuned
//                      rules (multiclass cost cells), with the untuned
//                      Analytical rate as a reference line.
(function () {
  'use strict';
  var fx = Deck.fx;
  var SHORT = { 'LGBMClassifier()': 'LGBM', 'XGBoost': 'XGB', 'CatBoost': 'Cat', 'RandomForest': 'RF', 'ExtraTrees': 'ET',
    'LogisticRegression()': 'LogReg', 'GaussianNB': 'NB' };
  var GOLD = '#E6AF3C', DARK = '#142D69', LIGHT = '#7FB6E6';

  function legend(host, items) {
    var top = fx.html('div', 'ctl', host);
    top.style.marginTop = '0';
    items.forEach(function (it) {
      fx.html('span', '', top).innerHTML = '<span class="swatch" style="height:18px;width:18px;border-radius:50%;background:' + it[0] + '"></span>' + it[1];
    });
  }

  // One classifier per column, rate on the vertical axis.
  Deck.figures.dumbbell_models = function (host) {
    var data = Deck.data.win_rates.binary_cost;
    fx.flex(host, 'column', 2);
    legend(host, [[GOLD, 'Analytical'], [DARK, 'EmpiricalCV']]);
    var W = 1320, H = 318, L = 96, R = 1290, T = 26, B = 262;
    var svg = fx.svg(host, W, H);
    var Y = function (v) { return B - v / 100 * (B - T); };
    [0, 25, 50, 75, 100].forEach(function (t) {
      fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#dcdcdc', 'stroke-width': 1.5 }, svg);
      fx.text(svg, L - 12, Y(t) + 9, String(t), { 'font-size': 26, 'text-anchor': 'end' });
    });
    fx.text(svg, 26, (T + B) / 2, 'cells won or tied (%)', { 'font-size': 24, transform: 'rotate(-90 26 ' + (T + B) / 2 + ')' });
    data.models.forEach(function (m, i) {
      var x = L + (i + 0.5) * (R - L) / data.models.length;
      var s = data.slices['model:' + m], a = s.Analytical, c = s.EmpiricalCV;
      fx.el('line', { x1: x, y1: Y(a[0]), x2: x, y2: Y(c[0]), stroke: '#8a8a8a', 'stroke-width': 5 }, svg);
      [[a, GOLD, -1], [c, DARK, 1]].forEach(function (d) {
        var v = d[0], side = Math.abs(a[0] - c[0]) < 0.5 ? d[2] : 0;
        fx.el('line', { x1: x + side * 9, y1: Y(Math.max(v[0] - v[1], 0)), x2: x + side * 9, y2: Y(Math.min(v[0] + v[1], 100)), stroke: d[1], 'stroke-width': 2, opacity: 0.55 }, svg);
        fx.el('circle', { cx: x + side * 9, cy: Y(v[0]), r: 11, fill: d[1], stroke: '#fff', 'stroke-width': 2 }, svg);
      });
      // Value labels beside the dots, pushed apart when the two rates are close.
      var same = Math.abs(a[0] - c[0]) < 0.5, close = Math.abs(a[0] - c[0]) < 14;
      var up = c[0] >= a[0] ? 1 : -1;
      fx.text(svg, x + 22, Y(c[0]) + 9 - (close ? up * 12 : 0), fx.fmt(c[0], 0), { 'font-size': 25, 'text-anchor': 'start', fill: DARK });
      if (!same) fx.text(svg, x + 22, Y(a[0]) + 9 + (close ? up * 12 : 0), fx.fmt(a[0], 0), { 'font-size': 25, 'text-anchor': 'start', fill: '#9a6d00' });
      fx.text(svg, x, B + 40, SHORT[m] || m, { 'font-size': 27 });
    });
  };

  // One tuned rule per row, rate on the horizontal axis.
  Deck.figures.dumbbell_split_cv = function (host) {
    var all = Deck.data.win_rates.multiclass_cost.slices.all;
    var rows = ['RiskPartitioning', 'LinearScaling', 'FullMatrixPartitioning'];
    fx.flex(host, 'column', 2);
    legend(host, [[LIGHT, 'tuned on the 10% split'], [DARK, 'tuned by 5-fold CV'], [GOLD, 'Analytical, untuned']]);
    var W = 1320, rh = 70, L = 400, R = 1200, T = 14, B = T + rows.length * rh;
    var svg = fx.svg(host, W, B + 44);
    var X = function (v) { return L + v / 100 * (R - L); };
    [0, 25, 50, 75, 100].forEach(function (t) {
      fx.el('line', { x1: X(t), y1: T, x2: X(t), y2: B, stroke: '#dcdcdc', 'stroke-width': 1.5 }, svg);
      fx.text(svg, X(t), B + 30, String(t), { 'font-size': 26 });
    });
    fx.text(svg, R + 40, B + 30, '% of cost cells', { 'font-size': 24, 'text-anchor': 'start' });
    var ref = all.Analytical[0];
    fx.el('line', { x1: X(ref), y1: T - 6, x2: X(ref), y2: B, stroke: GOLD, 'stroke-width': 5 }, svg);
    rows.forEach(function (s, i) {
      var y = T + (i + 0.5) * rh + 10, a = all[s], c = all[s + 'CV'];
      fx.text(svg, L - 18, y + 9, s, { 'font-size': 28, 'text-anchor': 'end' });
      fx.el('line', { x1: X(a[0]), y1: y, x2: X(c[0]), y2: y, stroke: '#8a8a8a', 'stroke-width': 5 }, svg);
      [[a, LIGHT], [c, DARK]].forEach(function (d) {
        var v = d[0];
        fx.el('line', { x1: X(v[0] - v[1]), y1: y, x2: X(v[0] + v[1]), y2: y, stroke: d[1], 'stroke-width': 2 }, svg);
        fx.el('circle', { cx: X(v[0]), cy: y, r: 12, fill: d[1], stroke: '#fff', 'stroke-width': 2 }, svg);
      });
      fx.text(svg, X(a[0]), y - 20, fx.fmt(a[0], 0) + '%', { 'font-size': 27 });
      fx.text(svg, X(c[0]), y - 20, fx.fmt(c[0], 0) + '%', { 'font-size': 27 });
    });
  };
})();
