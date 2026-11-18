// Conditional risks of the two actions for K = 2 with a zero diagonal:
// R(0 | x) = C10 eta_1 and R(1 | x) = C01 (1 - eta_1), crossing at
// C01 / (C01 + C10). Argmax keeps the threshold at 0.5 whatever the costs.
// Default: C01 = 1, C10 = 10, panel (b) of the static figure. As there, the
// risk axis is cut at 1.15 times the smaller cost.
(function () {
  'use strict';
  var fx = Deck.fx;

  Deck.figures.risk_lines = function (host, state) {
    var c01 = 1, c10 = 10;
    if (state.preset === 'equal') { c10 = 1; }
    if (state.preset === 'fp') { c01 = 10; c10 = 1; }

    fx.flex(host, 'row', 44);
    var W = 800, H = 330, L = 60, R = 720, B = 262, T = 46;
    var svg = fx.svg(host, W, H);
    var clip = fx.el('clipPath', { id: 'risk-lines-clip' }, fx.el('defs', {}, svg));
    fx.el('rect', { x: L, y: T, width: R - L, height: B - T }, clip);
    var layer = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    var controls = fx.html('div', 'ctl col', side);
    fx.math(fx.html('div', '', controls), '<span class="swatch" style="background:' + fx.blue + '"></span>$\\mathcal{R}(0 \\mid x) = C_{10}\\,\\eta_1(x)$');
    fx.math(fx.html('div', '', controls), '<span class="swatch" style="background:' + fx.red + '"></span>$\\mathcal{R}(1 \\mid x) = C_{01}\\,(1 - \\eta_1(x))$');
    fx.slider(controls, { label: 'C<sub>01</sub>, false positive', min: 1, max: 20, step: 0.5, value: c01, digits: 1 }, function (v) { c01 = v; draw(); });
    fx.slider(controls, { label: 'C<sub>10</sub>, false negative', min: 1, max: 20, step: 0.5, value: c10, digits: 1 }, function (v) { c10 = v; draw(); });
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';

    function draw() {
      fx.clear(layer);
      var top = 1.15 * Math.min(c01, c10);
      var X = function (e) { return L + e * (R - L); };
      var Y = function (r) { return B - (r / top) * (B - T); };
      var p = c01 / (c01 + c10);
      // Regions where each action has the smaller risk.
      fx.el('rect', { x: X(0), y: T, width: X(p) - X(0), height: B - T, fill: fx.blue, opacity: 0.09 }, layer);
      fx.el('rect', { x: X(p), y: T, width: X(1) - X(p), height: B - T, fill: fx.red, opacity: 0.09 }, layer);
      fx.el('line', { x1: X(0), y1: B, x2: X(1) + 22, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: X(0), y1: B, x2: X(0), y2: T - 14, stroke: fx.grey, 'stroke-width': 2 }, layer);
      var lines = fx.el('g', { 'clip-path': 'url(#risk-lines-clip)' }, layer);
      fx.el('line', { x1: X(0), y1: Y(0), x2: X(1), y2: Y(c10), stroke: fx.blue, 'stroke-width': 5 }, lines);
      fx.el('line', { x1: X(0), y1: Y(c01), x2: X(1), y2: Y(0), stroke: fx.red, 'stroke-width': 5 }, lines);
      // The argmax threshold, which does not move, and the Bayes threshold.
      fx.el('line', { x1: X(.5), y1: B, x2: X(.5), y2: T, stroke: '#9a9a9a', 'stroke-width': 2, 'stroke-dasharray': '3 6' }, layer);
      fx.el('line', { x1: X(p), y1: B, x2: X(p), y2: T, stroke: fx.ink, 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, layer);
      fx.el('circle', { cx: X(p), cy: Y(c10 * p), r: 7, fill: fx.ink }, layer);
      fx.tex(layer, X(0), B + 24, '0', { size: 24 });
      fx.tex(layer, X(1), B + 24, '1', { size: 24 });
      fx.tex(layer, X(1) + 30, B, '\\eta_1(x)', { size: 24, anchor: 'start' });
      fx.text(layer, X(.5), B + 32, '0.5 (argmax)', { fill: '#7a7a7a', 'font-size': 22 });
      var anchor = p < .12 ? 'start' : p > .88 ? 'end' : 'middle';
      fx.tex(layer, X(p), T - 22, fx.fmt(p, 3), { size: 26, anchor: anchor });
      fx.text(layer, 18, (T + B) / 2, 'conditional risk', { 'font-size': 22, transform: 'rotate(-90 18 ' + (T + B) / 2 + ')' });
      readout.innerHTML = 'the risks cross at C<sub>01</sub> / (C<sub>01</sub> + C<sub>10</sub>) = <strong>' + fx.fmt(p, 3) + '</strong>';
    }
    draw();
  };
})();
