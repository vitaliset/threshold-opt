// The four calibrators as maps from the raw estimate to the calibrated one
// (K = 2). One panel per calibrator: a few members of its family in light
// strokes and one in red, over the diagonal (no correction). The sliders
// under a panel are the parameters of its red curve. With sigma the
// logistic sigmoid and q the raw estimate:
//   temperature  sigma(logit(q) / T)                       1 parameter
//   Platt        sigma(a s + b), drawn with the score
//                s = logit(q)                               2 parameters
//   Dirichlet    sigma(w1 log q - w0 log(1 - q) + b), the
//                binary form of softmax(W log p + b)        3 parameters
//   isotonic     a monotone step function                   nonparametric
// The curves are examples chosen for the drawing, not fitted to data.
(function () {
  'use strict';
  var fx = Deck.fx;
  var sigmoid = function (x) { return 1 / (1 + Math.exp(-x)); };
  var logit = function (q) { return Math.log(q / (1 - q)); };

  function steps(levels) {
    return function (q) { return levels[Math.min(levels.length - 1, Math.floor(q * levels.length))]; };
  }

  // Each panel: the light curves of its family, the red curve as a function
  // of its parameters, and those parameters as [label, min, max, start].
  var PANELS = [
    { name: 'Temperature scaling',
      maps: [0.5, 0.75, 1.5, 3].map(function (T) { return function (q) { return sigmoid(logit(q) / T); }; }),
      params: [['T', 0.3, 4, 2]],
      main: function (v) { return function (q) { return sigmoid(logit(q) / v[0]); }; } },
    { name: 'Platt scaling',
      maps: [[0.5, -1], [1, 1], [2, 0.8], [1.6, -1.2]].map(function (p) { return function (q) { return sigmoid(p[0] * logit(q) + p[1]); }; }),
      params: [['a', 0.2, 3, 0.55], ['b', -2, 2, 0.7]],
      main: function (v) { return function (q) { return sigmoid(v[0] * logit(q) + v[1]); }; } },
    { name: 'Dirichlet calibration',
      maps: [[0.4, 1.4, 0.3], [1.6, 0.5, 0.2], [0.5, 0.5, -0.8], [1.5, 1.5, 0.9]].map(function (p) {
        return function (q) { return sigmoid(p[0] * Math.log(q) - p[1] * Math.log(1 - q) + p[2]); };
      }),
      params: [['w<sub>1</sub>', 0.2, 3, 1.3], ['w<sub>0</sub>', 0.2, 3, 0.45], ['b', -2, 2, 1.0]],
      main: function (v) { return function (q) { return sigmoid(v[0] * Math.log(q) - v[1] * Math.log(1 - q) + v[2]); }; } },
    { name: 'Isotonic regression',
      maps: [steps([0.02, 0.05, 0.2, 0.2, 0.55, 0.7, 0.95]), steps([0.1, 0.3, 0.35, 0.6, 0.62, 0.9])],
      params: [],
      main: function () { return steps([0.04, 0.12, 0.12, 0.3, 0.45, 0.45, 0.68, 0.8, 0.93, 0.97]); }, step: true }
  ];

  Deck.figures.calibrator_maps = function (host) {
    var side = 250, gap = 98, L = 60, T = 50, W = L + 4 * side + 3 * gap + 40, H = T + side + 42;
    var svg = fx.svg(host, W, H);
    var row = fx.html('div', '', host);
    row.style.display = 'flex';
    row.style.width = W + 'px';
    row.style.margin = '0 auto';
    row.style.alignItems = 'flex-start';
    PANELS.forEach(function (panel, k) {
      var x0 = L + k * (side + gap), B = T + side;
      var X = function (q) { return x0 + q * side; }, Y = function (v) { return B - v * side; };
      var path = function (f) {
        var d = '', n = panel.step ? 700 : 160;
        for (var i = 0; i <= n; i++) { var q = 0.003 + i / n * 0.994; d += (i ? 'L' : 'M') + X(q).toFixed(1) + ' ' + Y(f(q)).toFixed(1) + ' '; }
        return d;
      };
      fx.text(svg, x0 + side / 2, T - 18, panel.name, { 'font-size': 30, 'font-weight': 700, fill: fx.blue });
      fx.el('rect', { x: x0, y: T, width: side, height: side, fill: '#fff', stroke: '#9a9a9a', 'stroke-width': 1.5 }, svg);
      fx.el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: 0.6 }, svg);
      panel.maps.forEach(function (f) { fx.el('path', { d: path(f), fill: 'none', stroke: '#B9C3D9', 'stroke-width': 3 }, svg); });
      var values = panel.params.map(function (p) { return p[3]; });
      var red = fx.el('path', { d: path(panel.main(values)), fill: 'none', stroke: fx.red, 'stroke-width': 5, 'stroke-linejoin': 'round' }, svg);
      fx.text(svg, X(0), B + 26, '0', { 'font-size': 22 });
      fx.text(svg, X(1), B + 26, '1', { 'font-size': 22 });
      fx.text(svg, x0 - 12, Y(1) + 8, '1', { 'font-size': 22, 'text-anchor': 'end' });
      fx.tex(svg, x0 + side / 2, B + 26, '\\hat{p}', { size: 24 });
      if (k === 0) fx.tex(svg, x0 - 30, T + side / 2, 'g(\\hat{p})', { size: 24 });

      // Under the panel, aligned with it: one slider per parameter of the red curve.
      var box = fx.html('div', 'ctl col', row);
      box.style.width = side + 'px';
      box.style.marginTop = '0';
      box.style.marginLeft = (k === 0 ? L : gap) + 'px';
      box.style.gap = '0';
      box.style.alignItems = 'center';
      panel.params.forEach(function (p, j) {
        fx.slider(box, { label: p[0], min: p[1], max: p[2], step: 0.01, value: p[3], width: 130 }, function (v) {
          values[j] = v;
          red.setAttribute('d', path(panel.main(values)));
        });
      });
      if (!panel.params.length) fx.html('span', '', box, 'monotone steps');
    });
  };
})();
