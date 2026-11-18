// The conditional risk at one input, for the medical example (K = 2, class
// 1 = disease). The cost matrix is fixed and drawn as a matrix, row = true
// class, column = prediction: a treatment costs 1 whether or not the patient
// is sick, and a missed disease costs 10,
//     C = [[0, 1], [10, 1]].
// With eta = P(Y = 1 | x) on the only slider,
//     R(0 | x) = C00 (1 - eta) + C10 eta      (predict healthy)
//     R(1 | x) = C01 (1 - eta) + C11 eta      (predict sick)
// and the smaller bar is the prediction of least risk; the two are equal at
// eta = 1/10. The readout has a fixed size, so that the slide does not move
// as its text changes.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C = [[0, 1], [10, 1]];

  Deck.figures.risk_patient = function (host, state) {
    var eta = state.preset === 'half' ? 0.5 : 0.2;
    fx.flex(host, 'row', 40);

    // The matrix, drawn once.
    var left = fx.html('div', '', host);
    fx.flex(left, 'column', 6);
    var MW = 520, MH = 220;
    var m = fx.svg(left, MW, MH);
    var cols = [330, 440], rows = [112, 180];
    fx.tex(m, 150, 40, '\\mathbf{C}', { size: 30 });
    fx.text(m, cols[0], 46, 'predict 0', { 'font-size': 22, fill: fx.blue });
    fx.text(m, cols[1], 46, 'predict 1', { 'font-size': 22, fill: fx.red });
    fx.text(m, 248, rows[0] + 8, 'true 0 (healthy)', { 'font-size': 22, 'text-anchor': 'end' });
    fx.text(m, 248, rows[1] + 8, 'true 1 (sick)', { 'font-size': 22, 'text-anchor': 'end' });
    var top = 74, bottom = 214;
    fx.el('path', { d: 'M282 ' + top + ' L268 ' + top + ' L268 ' + bottom + ' L282 ' + bottom, fill: 'none', stroke: fx.ink, 'stroke-width': 3 }, m);
    fx.el('path', { d: 'M488 ' + top + ' L502 ' + top + ' L502 ' + bottom + ' L488 ' + bottom, fill: 'none', stroke: fx.ink, 'stroke-width': 3 }, m);
    C.forEach(function (row, i) {
      row.forEach(function (v, j) { fx.text(m, cols[j], rows[i] + 11, String(v), { 'font-size': 32 }); });
    });
    var bar = fx.html('div', 'ctl', left);
    fx.slider(bar, { label: 'η = P(Y = 1 | x)', min: 0, max: 1, step: 0.01, value: eta, width: 240 }, function (v) { eta = v; draw(); });

    var right = fx.html('div', '', host);
    fx.flex(right, 'column', 4);
    right.style.width = '760px';
    var W = 760, H = 210, L = 270, R = 680;
    var svg = fx.svg(right, W, H);
    var layer = fx.el('g', {}, svg);
    var readout = fx.html('div', 'readout', right);
    readout.style.width = '760px';
    readout.style.whiteSpace = 'nowrap';
    var lines = [
      { y: 18, label: 'predict 0', tex: 'C_{00}(1 - \\eta) + C_{10}\\,\\eta', color: fx.blue },
      { y: 112, label: 'predict 1', tex: 'C_{01}(1 - \\eta) + C_{11}\\,\\eta', color: fx.red }
    ];
    lines.forEach(function (r) {
      fx.text(svg, L - 16, r.y + 28, r.label, { 'font-size': 26, 'text-anchor': 'end', fill: r.color });
      fx.tex(svg, L - 16, r.y + 62, r.tex, { size: 21, anchor: 'end', color: r.color });
    });
    fx.el('line', { x1: L, y1: 8, x2: L, y2: 196, stroke: fx.grey, 'stroke-width': 2 }, svg);
    var cross = (C[0][1] - C[0][0]) / ((C[0][1] - C[0][0]) + (C[1][0] - C[1][1]));

    function draw() {
      fx.clear(layer);
      var risks = [C[0][0] * (1 - eta) + C[1][0] * eta, C[0][1] * (1 - eta) + C[1][1] * eta];
      var best = risks[1] < risks[0] ? 1 : 0;
      lines.forEach(function (r, i) {
        var w = risks[i] / 10 * (R - L);
        fx.el('rect', { x: L, y: r.y, width: Math.max(w, 1), height: 64, fill: r.color, 'fill-opacity': i === best ? 0.9 : 0.3, stroke: i === best ? fx.ink : 'none', 'stroke-width': 3 }, layer);
        fx.text(layer, L + w + 12, r.y + 42, fx.fmt(risks[i], 2), { 'font-size': 26, 'text-anchor': 'start' });
      });
      // Always two lines, so that the figure keeps its height.
      readout.innerHTML = 'η = ' + fx.fmt(eta, 2) + ': predict ' + best + ' has the smaller risk' +
        (best === 1 && eta < 0.5 ? ', though η < 0.5' : '') + '<br>the risks are equal at η = ' + fx.fmt(cross, 2);
    }
    draw();
  };
})();
