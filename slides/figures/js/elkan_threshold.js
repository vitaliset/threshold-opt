// The threshold rule of Lemma 3.2.1 for a general 2 x 2 cost matrix.
// The conditional risks of the two actions are, with eta = P(Y = 1 | x),
//     R(0 | eta) = C00 (1 - eta) + C10 eta,
//     R(1 | eta) = C01 (1 - eta) + C11 eta.
// Predicting 1 is optimal where g(eta) = R(0 | eta) - R(1 | eta) >= 0, and
// g(eta) = eta m - c with c = C01 - C00 and m = c + (C10 - C11). Under
// reasonable costs (C01 > C00 and C10 > C11) both are positive, m > c, and
// the rule is the threshold p* = c / m in (0, 1). The strip under the plot
// is the decision rule; when a slider breaks the condition it shows what
// the Bayes rule becomes. Default: the Non-Zero Diagonal matrix of the
// binary study, [[0.8, 2.0], [1.0, 0.5]], with p* = 1.2 / 1.7.
(function () {
  'use strict';
  var fx = Deck.fx;
  var PRESETS = {
    zeroone: [[0, 1], [1, 0]],
    asym: [[0, 10], [1, 0]],
    nonzero: [[0.8, 2.0], [1.0, 0.5]],
    broken: [[0.8, 2.0], [0.3, 0.5]]
  };

  Deck.figures.elkan_threshold = function (host, state) {
    var key = PRESETS[state.preset] ? state.preset : 'nonzero';
    var C = PRESETS[key].map(function (r) { return r.slice(); });
    fx.flex(host, 'row', 56);
    var W = 820, H = 470, L = 80, R = 780, T = 24, B = 300;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var clip = fx.el('clipPath', { id: 'elkan-clip' }, fx.el('defs', {}, svg));
    fx.el('rect', { x: L, y: T, width: R - L, height: B - T }, clip);
    var layer = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    side.style.width = '560px';

    var legend = fx.html('div', 'ctl col', side);
    fx.math(fx.html('div', '', legend), '<span class="swatch" style="background:' + fx.blue + '"></span>$\\mathcal{R}(0 \\mid \\eta) = C_{00}(1 - \\eta) + C_{10}\\,\\eta$');
    fx.math(fx.html('div', '', legend), '<span class="swatch" style="background:' + fx.red + '"></span>$\\mathcal{R}(1 \\mid \\eta) = C_{01}(1 - \\eta) + C_{11}\\,\\eta$');
    var grid = fx.html('div', '', side);
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'auto 210px 210px';
    grid.style.gap = '6px 18px';
    grid.style.alignItems = 'center';
    grid.style.marginTop = '14px';
    fx.html('span', '', grid, '');
    fx.html('span', '', grid, 'predict 0').style.textAlign = 'center';
    fx.html('span', '', grid, 'predict 1').style.textAlign = 'center';
    var inputs = [];
    C.forEach(function (row, i) {
      fx.html('span', '', grid, 'true ' + i);
      inputs.push(row.map(function (_, k) {
        return fx.slider(fx.html('div', '', grid), { label: 'C<sub>' + i + k + '</sub>', min: 0, max: 10, step: 0.1, value: C[i][k], width: 96, digits: 1 }, function (v) {
          C[i][k] = v; picker.mark(''); draw();
        });
      }));
    });
    var bar = fx.html('div', 'ctl', side);
    bar.style.justifyContent = 'flex-start';
    bar.style.gap = '8px';
    var picker = fx.choice(bar, [['zeroone', '0/1 loss'], ['asym', 'Asymmetric 10×'], ['nonzero', 'Non-Zero Diagonal']], key, function (k) {
      C = PRESETS[k].map(function (r) { return r.slice(); });
      inputs.forEach(function (row, i) { row.forEach(function (input, c) { input.set(C[i][c]); }); });
      draw();
    });
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.4';

    var X = function (e) { return L + e * (R - L); };

    function draw() {
      fx.clear(layer);
      var top = Math.max(C[0][0], C[0][1], C[1][0], C[1][1], 0.5) * 1.1;
      var Y = function (r) { return B - r / top * (B - T); };
      var c = C[0][1] - C[0][0], m = c + (C[1][0] - C[1][1]);
      var g = function (e) { return e * m - c; };          // >= 0 where predicting 1 is optimal
      var reasonable = C[0][1] > C[0][0] && C[1][0] > C[1][1];
      var root = Math.abs(m) > 1e-12 ? c / m : null;
      var inside = root !== null && root > 0 && root < 1;

      // Intervals of [0, 1] by optimal action; ties go to the smaller index, as in the Bayes rule.
      var cuts = inside ? [0, root, 1] : [0, 1];
      var parts = [];
      for (var i = 0; i + 1 < cuts.length; i++) {
        var mid = (cuts[i] + cuts[i + 1]) / 2;
        parts.push({ a: cuts[i], b: cuts[i + 1], act: g(mid) > 0 ? 1 : 0 });
      }
      parts.forEach(function (p) {
        var col = p.act ? fx.red : fx.blue;
        fx.el('rect', { x: X(p.a), y: T, width: X(p.b) - X(p.a), height: B - T, fill: col, opacity: 0.08 }, layer);
        fx.el('rect', { x: X(p.a), y: B + 64, width: X(p.b) - X(p.a), height: 52, fill: col, opacity: 0.85 }, layer);
        if (X(p.b) - X(p.a) > 130) fx.text(layer, (X(p.a) + X(p.b)) / 2, B + 99, 'predict ' + p.act, { 'font-size': 26, fill: '#fff' });
      });
      fx.el('line', { x1: X(0), y1: B, x2: X(1) + 22, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: X(0), y1: B, x2: X(0), y2: T - 10, stroke: fx.grey, 'stroke-width': 2 }, layer);
      var lines = fx.el('g', { 'clip-path': 'url(#elkan-clip)' }, layer);
      fx.el('line', { x1: X(0), y1: Y(C[0][0]), x2: X(1), y2: Y(C[1][0]), stroke: fx.blue, 'stroke-width': 5 }, lines);
      fx.el('line', { x1: X(0), y1: Y(C[0][1]), x2: X(1), y2: Y(C[1][1]), stroke: fx.red, 'stroke-width': 5 }, lines);
      fx.tex(layer, X(0), B + 26, '0', { size: 24 });
      fx.tex(layer, X(1), B + 26, '1', { size: 24 });
      fx.tex(layer, X(1) + 30, B, '\\eta', { size: 26, anchor: 'start' });
      fx.text(layer, 24, (T + B) / 2, 'conditional risk', { 'font-size': 24, transform: 'rotate(-90 24 ' + (T + B) / 2 + ')' });
      fx.tex(layer, X(0) - 12, B + 90, '\\delta_{\\tau}', { size: 28, anchor: 'end' });
      if (inside) {
        fx.el('line', { x1: X(root), y1: B + 116, x2: X(root), y2: T, stroke: fx.ink, 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, layer);
        fx.el('circle', { cx: X(root), cy: Y(C[0][0] * (1 - root) + C[1][0] * root), r: 8, fill: fx.ink }, layer);
        fx.tex(layer, X(root), B + 142, (reasonable ? 'p^* = ' : '') + fx.fmt(root, 3), { size: 28, anchor: root < 0.12 ? 'start' : root > 0.88 ? 'end' : 'middle' });
      }

      var text;
      if (reasonable) {
        text = 'reasonable costs: <strong>p* = ' + fx.fmt(root, 3) + '</strong><br>= (C<sub>01</sub> − C<sub>00</sub>) / ((C<sub>01</sub> − C<sub>00</sub>) + (C<sub>10</sub> − C<sub>11</sub>))';
      } else if (inside) {
        text = '<span class="bad">not reasonable</span>: the rule is reversed,<br>predict 1 below ' + fx.fmt(root, 3) + ' and 0 above';
      } else {
        text = '<span class="bad">not reasonable</span>: predicting ' + parts[0].act + ' is optimal<br>for every η, no threshold in (0, 1)';
      }
      readout.innerHTML = text;
    }
    draw();
  };
})();
