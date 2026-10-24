// The threshold rule delta_tau(z) = 1 if z >= tau and 0 if z < tau, as a
// step function of the estimate z in [0, 1], with tau on a slider. The strip
// under the axis is the same rule read as a split of [0, 1]. Default tau = 0.3.
(function () {
  'use strict';
  var fx = Deck.fx;

  Deck.figures.threshold_step = function (host, state) {
    var tau = 0.3;
    if (state.preset === 'half') tau = 0.5;
    fx.flex(host, 'column', 4);
    var W = 1100, H = 250, L = 150, R = 1040, Y0 = 135, Y1 = 40, S = 165;
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    fx.slider(bar, { label: 'τ', min: 0, max: 1, step: 0.01, value: tau, width: 360 }, function (v) { tau = v; draw(); });
    var X = function (z) { return L + z * (R - L); };
    fx.el('line', { x1: L, y1: Y0, x2: R, y2: Y0, stroke: fx.grey, 'stroke-width': 2 }, svg);
    fx.el('line', { x1: L, y1: Y0, x2: L, y2: Y1 - 20, stroke: fx.grey, 'stroke-width': 2 }, svg);
    fx.tex(svg, L - 14, Y0, '0', { size: 24, anchor: 'end' });
    fx.tex(svg, L - 14, Y1, '1', { size: 24, anchor: 'end' });
    fx.tex(svg, L - 70, (Y0 + Y1) / 2, '\\delta_\\tau(z)', { size: 26, anchor: 'end' });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (z) { fx.text(svg, X(z), S + 66, String(z), { 'font-size': 22, fill: fx.grey }); });
    fx.tex(svg, R + 40, S + 58, 'z', { size: 26 });

    function draw() {
      fx.clear(layer);
      var x = X(tau);
      fx.el('rect', { x: L, y: S, width: x - L, height: 34, fill: fx.blue, opacity: 0.85 }, layer);
      fx.el('rect', { x: x, y: S, width: R - x, height: 34, fill: fx.red, opacity: 0.85 }, layer);
      if (x - L > 150) fx.text(layer, (L + x) / 2, S + 25, 'predict 0', { 'font-size': 22, fill: '#fff' });
      if (R - x > 150) fx.text(layer, (x + R) / 2, S + 25, 'predict 1', { 'font-size': 22, fill: '#fff' });
      fx.el('line', { x1: L, y1: Y0, x2: x, y2: Y0, stroke: fx.blue, 'stroke-width': 6 }, layer);
      fx.el('line', { x1: x, y1: Y1, x2: R, y2: Y1, stroke: fx.red, 'stroke-width': 6 }, layer);
      fx.el('line', { x1: x, y1: Y1 - 24, x2: x, y2: S + 44, stroke: fx.yellow, 'stroke-width': 4, 'stroke-dasharray': '8 6' }, layer);
      fx.el('circle', { cx: x, cy: Y0, r: 8, fill: '#fff', stroke: fx.blue, 'stroke-width': 3 }, layer);
      fx.el('circle', { cx: x, cy: Y1, r: 8, fill: fx.red }, layer);
      fx.tex(layer, x + 22, Y1 - 22, '\\tau', { size: 26 });
    }
    draw();
  };
})();
