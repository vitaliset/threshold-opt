// The Bayes rule at one point of the simplex (K = 3). Left: the cost matrix,
// with its columns in the colours of the classes. Middle: the simplex, where
// the point p can be dragged; the shaded regions are the Bayes regions.
// Right: the three risks L_i(p) = c_i' p, with c_i the column i of C, and the
// smallest one decides. The matrix is the example of the next slide:
// C_20 = 5, C_21 = 3, every other error 1. Default: p = (0.5, 0.2, 0.3),
// where argmax says 0 and the Bayes rule 2. Every part has a fixed size, so
// that the figure does not move while p is dragged.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C = [[0, 1, 1], [1, 0, 1], [5, 3, 0]];

  function risk(i, p) { return C[0][i] * p[0] + C[1][i] * p[1] + C[2][i] * p[2]; }

  Deck.figures.risk_bars_simplex = function (host, state) {
    var p = [0.5, 0.2, 0.3];
    if (state.preset === 'vertex') p = [0.9, 0.05, 0.05];
    fx.flex(host, 'row', 40);

    // The matrix, drawn once.
    var MW = 300, MH = 250;
    var m = fx.svg(fx.html('div', '', host), MW, MH);
    var cols = [140, 200, 260], rows = [120, 172, 224];
    fx.tex(m, 34, 40, '\\mathbf{C}', { size: 30 });
    fx.text(m, 200, 26, 'predict', { 'font-size': 22, fill: fx.grey });
    cols.forEach(function (x, j) { fx.text(m, x, 62, String(j), { 'font-size': 24, fill: fx.cls[j] }); });
    rows.forEach(function (y, i) { fx.text(m, 96, y + 8, 'true ' + i, { 'font-size': 22, 'text-anchor': 'end' }); });
    var t = 86, b = 252;
    fx.el('path', { d: 'M118 ' + t + ' L106 ' + t + ' L106 ' + (b - 4) + ' L118 ' + (b - 4), fill: 'none', stroke: fx.ink, 'stroke-width': 3 }, m);
    fx.el('path', { d: 'M282 ' + t + ' L294 ' + t + ' L294 ' + (b - 4) + ' L282 ' + (b - 4), fill: 'none', stroke: fx.ink, 'stroke-width': 3 }, m);
    C.forEach(function (row, i) {
      row.forEach(function (v, j) { fx.text(m, cols[j], rows[i] + 10, String(v), { 'font-size': 28, fill: fx.cls[j] }); });
    });

    var W = 390, H = 340;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var S = fx.simplex(W / 2, 30, 330);
    var scores = [0, 1, 2].map(function (i) { return function (q) { return risk(i, q); }; });
    var regions = fx.regions(scores);
    fx.drawRegions(svg, S, regions);
    S.outline(svg);
    S.labels(svg, 26);
    regions.forEach(function (poly, k) {
      var c = S.xy(fx.centroid(poly));
      fx.tex(svg, c[0], c[1], '\\mathcal{P}_' + k + '^*', { color: fx.cls[k], size: 26 });
    });
    var dot = fx.el('circle', { r: 14, fill: fx.ink, stroke: '#fff', 'stroke-width': 3 }, svg);
    fx.drag(dot, svg, function (x, y) { p = S.clamp(S.bary(x, y)); draw(); });

    var right = fx.html('div', '', host);
    fx.flex(right, 'column', 6);
    right.style.width = '480px';
    var BW = 470, BH = 258;
    var bars = fx.svg(right, BW, BH);
    var layer = fx.el('g', {}, bars);
    var readout = fx.html('div', 'readout', right);
    readout.style.width = '480px';
    readout.style.whiteSpace = 'nowrap';
    var L = 30, R = BW - 10, T = 24, B = 192, top = 5;
    var Y = function (v) { return B - v / top * (B - T); };
    fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, bars);
    var slot = (R - L) / 3;
    [0, 1, 2].forEach(function (i) {
      fx.text(bars, L + slot * (i + 0.5), B + 34, 'predict ' + i, { 'font-size': 24, fill: fx.cls[i] });
      fx.tex(bars, L + slot * (i + 0.5), B + 64, '\\mathcal{L}_' + i + '(\\mathbf{p})', { size: 24, color: fx.cls[i] });
    });

    function draw() {
      var q = S.xy(p);
      dot.setAttribute('cx', q[0]);
      dot.setAttribute('cy', q[1]);
      fx.clear(layer);
      var r = [0, 1, 2].map(function (i) { return risk(i, p); });
      var best = fx.argmin(r);
      r.forEach(function (v, i) {
        var x = L + slot * i + 22, w = slot - 44;
        fx.el('rect', { x: x, y: Y(v), width: w, height: B - Y(v), fill: fx.cls[i], 'fill-opacity': i === best ? 0.9 : 0.3, stroke: i === best ? fx.ink : 'none', 'stroke-width': 3 }, layer);
        fx.text(layer, x + w / 2, Y(v) - 10, fx.fmt(v, 2), { 'font-size': 24 });
      });
      var am = p.indexOf(Math.max.apply(null, p));
      // Always two lines, so that the figure keeps its height.
      readout.innerHTML = 'p = (' + p.map(function (x) { return fx.fmt(x, 2); }).join(', ') + ')<br>Bayes rule: ' + best + ', argmax: ' + am;
    }
    draw();
  };
})();
