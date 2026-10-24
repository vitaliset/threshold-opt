// Relative excess risk against the variance multiplier lambda, one curve
// per K, on log (left) and linear (right) vertical scales, as in the static
// figure. The data are the coordinates of tex/figures/scaling_variance.tex
// (figures/data/scaling_variance.js). The buttons single out one K, with
// its 95% band; the readout compares it with K = 2 at the largest lambda.
(function () {
  'use strict';
  var fx = Deck.fx;
  // One colour per K, from K = 2 (dark blue) to K = 20 (red), in order.
  var COLORS = ['#142D69', '#0064A0', '#2A9D8F', '#6BA547', '#C9A227', '#E08A2B', '#C8502E', '#82143C'];

  Deck.figures.scaling_variance = function (host, state) {
    var data = Deck.data.scaling_variance, curves = data.curves;
    var sel = state.preset === 'all' ? 0 : Number(state.preset) || 0;   // 0 = all curves
    fx.flex(host, 'column', 4);
    var W = 1440, H = 380;
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    bar.style.gap = '6px 8px';
    fx.html('span', '', bar, 'K =');
    var choices = [['0', 'all']].concat(curves.map(function (c) { return [String(c.K), String(c.K)]; }));
    fx.choice(bar, choices, String(sel), function (k) { sel = Number(k); draw(); });
    var readout = fx.html('div', 'readout', host);

    var xs = curves[0].lambda, xmin = Math.log(1), xmax = Math.log(xs[xs.length - 1] * 1.08);
    var ymaxLin = Math.max.apply(null, curves.map(function (c) { return Math.max.apply(null, c.hi); })) * 1.05;
    var ymin = 1e-3, ymaxLog = 3;

    function panel(x0, log) {
      var L = x0 + 96, R = x0 + 620, T = 16, B = 310;
      var X = function (v) { return L + (Math.log(v) - xmin) / (xmax - xmin) * (R - L); };
      var Y = log
        ? function (v) { return B - (Math.log(Math.max(v, ymin)) - Math.log(ymin)) / (Math.log(ymaxLog) - Math.log(ymin)) * (B - T); }
        : function (v) { return B - v / ymaxLin * (B - T); };
      fx.el('rect', { x: L, y: T, width: R - L, height: B - T, fill: '#fff', stroke: '#bbb', 'stroke-width': 1.5 }, layer);
      [1, 2, 4, 8, 16].forEach(function (t) {
        fx.el('line', { x1: X(t), y1: T, x2: X(t), y2: B, stroke: '#e3e3e3', 'stroke-width': 1.5 }, layer);
        fx.text(layer, X(t), B + 30, String(t), { 'font-size': 26 });
      });
      var ticks = log ? [1e-3, 1e-2, 1e-1, 1] : [0, 0.5, 1, 1.5, 2, 2.5].filter(function (t) { return t <= ymaxLin; });
      ticks.forEach(function (t) {
        fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#e3e3e3', 'stroke-width': 1.5 }, layer);
        if (log) fx.tex(layer, L - 10, Y(t), '10^{' + Math.round(Math.log10(t)) + '}', { size: 24, anchor: 'end' });
        else fx.text(layer, L - 10, Y(t) + 9, String(t), { 'font-size': 26, 'text-anchor': 'end' });
      });
      fx.tex(layer, (L + R) / 2, B + 62, '\\text{variance multiplier } \\lambda', { size: 26 });
      fx.text(layer, log ? R - 12 : L + 14, log ? B - 14 : T + 32, log ? 'log scale' : 'linear scale', { 'font-size': 24, 'text-anchor': log ? 'end' : 'start', fill: fx.grey });
      var placed = [];
      // Draw the extreme curves first, so that their labels are always placed; the selected one goes on top.
      var order = curves.map(function (c, i) { return i; }).sort(function (a, b) {
        var rank = function (i) { return curves[i].K === sel ? 2 : (i === 0 || i === curves.length - 1) ? 0 : 1; };
        return rank(a) - rank(b);
      });
      order.forEach(function (i) {
        var c = curves[i], on = !sel || c.K === sel;
        var line = function (ys) { return ys.map(function (v, j) { return (j ? 'L' : 'M') + X(c.lambda[j]).toFixed(1) + ' ' + Y(v).toFixed(1); }).join(' '); };
        if (sel && on) {
          var band = line(c.hi) + ' ' + c.lo.map(function (v, j) { var k = c.lo.length - 1 - j; return 'L' + X(c.lambda[k]).toFixed(1) + ' ' + Y(c.lo[k]).toFixed(1); }).join(' ') + ' Z';
          fx.el('path', { d: band, fill: COLORS[i], opacity: 0.18 }, layer);
        }
        fx.el('path', { d: line(c.mean), fill: 'none', stroke: on ? COLORS[i] : '#c4c4c4', 'stroke-width': sel && on ? 5 : on ? 3 : 2 }, layer);
        // Curve labels at the right end, skipping those that would overlap.
        var last = c.mean.length - 1, ly = Y(c.mean[last]) + 8;
        if (on && (sel || placed.every(function (y) { return Math.abs(ly - y) > 24; }))) {
          fx.text(layer, R + 8, ly, 'K = ' + c.K, { 'font-size': 23, 'text-anchor': 'start', fill: COLORS[i] });
          placed.push(ly);
        }
      });
    }

    function draw() {
      fx.clear(layer);
      panel(0, true);
      panel(720, false);
      var last = xs.length - 1, base = curves[0];
      if (sel) {
        var c = curves.filter(function (k) { return k.K === sel; })[0];
        readout.innerHTML = 'at λ = ' + xs[last] + ': relative excess risk ' + fx.fmt(c.mean[last], 3) + ' for K = ' + c.K +
          (c.K === base.K ? '' : ', against ' + fx.fmt(base.mean[last], 3) + ' for K = ' + base.K + ' (' + fx.fmt(base.mean[last] / c.mean[last], 1) + ' times lower)');
      } else {
        var far = curves[curves.length - 1];
        readout.innerHTML = 'at λ = ' + xs[last] + ': ' + fx.fmt(base.mean[last], 3) + ' for K = ' + base.K + ' and ' + fx.fmt(far.mean[last], 3) + ' for K = ' + far.K;
      }
    }
    draw();
  };
})();
