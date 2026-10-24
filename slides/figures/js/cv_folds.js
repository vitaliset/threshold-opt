// Where each binary strategy spends its labels (Chapter 4). Top: the single
// split, with the estimator on 70% of the data, the threshold on a 10%
// validation split of m labels, and 20% for the test. Bottom: EmpiricalCV
// on the 80% pool, five folds; in fold k the estimator is fitted on the
// other four and scored on fold k, and the threshold minimizes the average
// of the five per-fold risks. It is chosen on 8m labels; the final
// estimator is refitted on the whole pool.
(function () {
  'use strict';
  var fx = Deck.fx;
  var FIT = '#9fb3d9', TUNE = '#E6AF3C', TEST = '#d5d5d5';

  Deck.figures.cv_folds = function (host, state) {
    fx.flex(host, 'column', 4);
    var W = 1400, H = 430, L = 260, R = 1360;
    var svg = fx.svg(host, W, H);
    var X = function (f) { return L + f * (R - L); };
    var seg = function (y, h, from, to, color, label) {
      fx.el('rect', { x: X(from), y: y, width: X(to) - X(from), height: h, fill: color, stroke: '#fff', 'stroke-width': 2 }, svg);
      if (label) fx.text(svg, (X(from) + X(to)) / 2, y + h / 2 + 8, label, { 'font-size': 22, fill: fx.ink });
    };

    fx.text(svg, L - 20, 38, 'single split', { 'font-size': 24, 'text-anchor': 'end' });
    seg(12, 40, 0, 0.7, FIT, 'fit the estimator (70%)');
    seg(12, 40, 0.7, 0.8, TUNE, 'tune τ');
    seg(12, 40, 0.8, 1, TEST, 'test (20%)');

    fx.text(svg, X(0) , 92, 'EmpiricalCV, five folds of the 80% pool', { 'font-size': 24, 'text-anchor': 'start' });
    for (var k = 0; k < 5; k++) {
      var y = 106 + k * 46;
      fx.text(svg, L - 20, y + 26, 'fold ' + (k + 1), { 'font-size': 22, 'text-anchor': 'end', fill: fx.grey });
      for (var j = 0; j < 5; j++) seg(y, 36, j * 0.16, (j + 1) * 0.16, j === k ? TUNE : FIT, j === k ? 'score' : '');
      seg(y, 36, 0.8, 1, TEST, k === 0 ? 'test (20%)' : '');
    }
    var yb = 106 + 5 * 46 + 20;
    fx.text(svg, X(0.5), yb + 22, 'single split: τ is tuned on m labels (10%); EmpiricalCV: on 8m labels (80%), and the estimator is refitted on the pool', { 'font-size': 22, fill: fx.grey });
  };
})();
