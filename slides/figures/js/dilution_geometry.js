// Geometry of Lemma 9.2.3. A perturbation eps of size r lies in V, of
// dimension K - 1, and changes a cost with gradient g by g' eps. For K = 2,
// V is a line and eps = +-r along g: all of it counts. For K = 3, V is a
// plane and eps is on a circle: only the component along g counts, and the
// mean of cos^2 over the circle is 1/2. In general the mean of
// (g' eps)^2 / (r^2 |g|^2) is 1 / (K - 1), drawn as bars on the right.
(function () {
  'use strict';
  var fx = Deck.fx;

  Deck.figures.dilution_geometry = function (host, state) {
    var theta = 50;
    if (state.preset === 'aligned') theta = 0;
    if (state.preset === 'orthogonal') theta = 90;
    fx.flex(host, 'column', 6);
    var W = 1440, H = 400;
    var svg = fx.svg(host, W, H);
    var base = fx.el('g', {}, svg);
    var live = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    var slider = fx.slider(bar, { label: 'direction of ε (K = 3)', min: 0, max: 360, step: 1, value: theta, digits: 0, width: 300,
      format: function (v) { return v + '°'; } }, function (v) { theta = v; drawLive(); });
    var readout = fx.html('div', 'readout', host);
    var gold = '#C9871A';

    // Panel 1, K = 2: the line V with p in the middle and eps = +-r.
    var l1 = 40, r1 = 380, y1 = 200, c1 = (l1 + r1) / 2, rad = 120;
    fx.tex(base, c1, 40, 'K = 2', { size: 30 });
    fx.el('line', { x1: l1, y1: y1, x2: r1, y2: y1, stroke: fx.grey, 'stroke-width': 2.5 }, base);
    fx.tex(base, r1 - 6, y1 - 26, 'V', { size: 26, anchor: 'end' });
    fx.el('line', { x1: c1 - rad, y1: y1, x2: c1 + rad, y2: y1, stroke: fx.blue, 'stroke-width': 9, 'stroke-linecap': 'round', opacity: 0.85 }, base);
    fx.arrow(base, c1, y1, 1, 0, 80, gold);
    fx.tex(base, c1 + 48, y1 - 28, '\\mathbf{g}', { size: 26, color: gold });
    [-1, 1].forEach(function (s) {
      fx.el('circle', { cx: c1 + s * rad, cy: y1, r: 10, fill: fx.softblue, stroke: '#fff', 'stroke-width': 2 }, base);
    });
    fx.tex(base, c1 - rad, y1 + 36, '-r', { size: 24 });
    fx.tex(base, c1 + rad, y1 + 36, '+r', { size: 24 });
    fx.el('circle', { cx: c1, cy: y1, r: 6, fill: fx.ink }, base);
    fx.tex(base, c1, 320, '(\\mathbf{g}^\\top \\boldsymbol{\\epsilon})^2 = r^2 \\|\\mathbf{g}\\|^2', { size: 26 });
    fx.text(base, c1, 362, 'the whole error counts', { 'font-size': 24, fill: fx.grey });

    // Panel 2, K = 3: the circle of radius r in the plane V.
    var c2 = 690, y2 = 190;
    fx.tex(base, c2, 40, 'K = 3', { size: 30 });
    fx.el('circle', { cx: c2, cy: y2, r: rad, fill: '#f4f6fb', stroke: fx.softblue, 'stroke-width': 3 }, base);
    fx.el('line', { x1: c2 - rad - 40, y1: y2, x2: c2 + rad + 40, y2: y2, stroke: '#c8c8c8', 'stroke-width': 2, 'stroke-dasharray': '6 6' }, base);
    fx.arrow(base, c2, y2, 1, 0, 80, gold);
    fx.tex(base, c2 + 48, y2 + 28, '\\mathbf{g}', { size: 26, color: gold });
    fx.tex(base, c2 + rad + 34, y2 - rad + 6, 'V', { size: 26 });
    fx.tex(base, c2, 362, '\\text{mean of } (\\mathbf{g}^\\top \\boldsymbol{\\epsilon})^2 = \\tfrac{1}{2} \\, r^2 \\|\\mathbf{g}\\|^2', { size: 26 });

    // Panel 3: 1 / (K - 1) for K = 2, ..., 10.
    var L = 1040, R = 1410, T = 70, B = 290;
    var KS = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    var bw = (R - L) / KS.length;
    fx.tex(base, (L + R) / 2, 40, '\\text{mean of } (\\mathbf{g}^\\top \\boldsymbol{\\epsilon})^2 / (r^2 \\|\\mathbf{g}\\|^2)', { size: 24 });
    fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, base);
    [0, 0.5, 1].forEach(function (t) {
      var y = B - t * (B - T);
      fx.el('line', { x1: L, y1: y, x2: R, y2: y, stroke: '#e3e3e3', 'stroke-width': 1.5 }, base);
      fx.text(base, L - 10, y + 8, String(t), { 'font-size': 22, 'text-anchor': 'end' });
    });
    KS.forEach(function (K, i) {
      var v = 1 / (K - 1), x = L + i * bw + 6, h = v * (B - T);
      fx.el('rect', { x: x, y: B - h, width: bw - 12, height: h, fill: K <= 3 ? fx.blue : '#9fb3d9' }, base);
      fx.text(base, x + (bw - 12) / 2, B + 28, String(K), { 'font-size': 22 });
    });
    fx.tex(base, (L + R) / 2, B + 62, 'K', { size: 24 });
    fx.tex(base, L + 3.2 * bw, T + 46, '\\dfrac{1}{K-1}', { size: 26, anchor: 'start' });

    // The movable eps on the circle, its projection on g, and the readout.
    // The handle is created once, so that a drag keeps its pointer capture.
    var handle = fx.el('circle', { r: 13, fill: fx.softblue, stroke: '#fff', 'stroke-width': 3 }, svg);
    fx.drag(handle, svg, function (x, y) {
      var d = Math.round(Math.atan2(y2 - y, x - c2) * 180 / Math.PI);
      theta = (d + 360) % 360;
      slider.set(theta);
      drawLive();
    });
    function drawLive() {
      fx.clear(live);
      var a = theta * Math.PI / 180, ex = c2 + rad * Math.cos(a), ey = y2 - rad * Math.sin(a);
      fx.el('line', { x1: c2, y1: y2, x2: ex, y2: ey, stroke: fx.softblue, 'stroke-width': 2.5 }, live);
      fx.el('line', { x1: ex, y1: ey, x2: ex, y2: y2, stroke: fx.grey, 'stroke-width': 2, 'stroke-dasharray': '5 5' }, live);
      fx.el('line', { x1: c2, y1: y2, x2: ex, y2: y2, stroke: fx.blue, 'stroke-width': 9, 'stroke-linecap': 'round', opacity: 0.85 }, live);
      fx.el('circle', { cx: c2, cy: y2, r: 6, fill: fx.ink }, live);
      handle.setAttribute('cx', ex);
      handle.setAttribute('cy', ey);
      var c = Math.cos(a);
      readout.innerHTML = 'For K = 3, the part of ε along g (thick) changes the cost: (gᵀε)² / (r²‖g‖²) = cos²(' + theta + '°) = ' +
        fx.fmt(c * c, 2) + '. Its mean over the circle is 1/2.';
    }
    drawLive();
  };
})();
