// The probability triangle in space (K = 3), for the motivation slide. The
// vectors (p0, p1, p2) with p_k >= 0 and p0 + p1 + p2 = 1 form the triangle
// with corners on the three axes; it is shaded by the argmax regions. The
// view swings slowly about the vertical axis while its slide is shown, so
// that the plane reads as a plane in space; dragging turns it by hand and
// pauses the swing for 2.5 s. World axes: p0 up, p1 and p2 on the floor.
(function () {
  'use strict';
  var fx = Deck.fx;

  Deck.figures.simplex_3d = function (host, state) {
    var W = 520, H = 330, cx = 255, cy = 236, s = 172, phi = 24 * Math.PI / 180;
    var base0 = 45, swing = 18, az = base0, spin = !state.still, held = -1e9;
    fx.flex(host, 'column', 0);
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);

    // A point (p0, p1, p2) on the screen; world x = p1, y = p2, z = p0.
    function project(p) {
      var th = az * Math.PI / 180, x = p[1], y = p[2], z = p[0];
      var u = x * Math.cos(th) - y * Math.sin(th);
      var d = x * Math.sin(th) + y * Math.cos(th);
      return [cx + s * u, cy - s * (z * Math.cos(phi) - d * Math.sin(phi))];
    }
    function path(pts) {
      return pts.map(function (p, i) { var q = project(p); return (i ? 'L' : 'M') + q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join(' ') + ' Z';
    }
    function line(a, b, attrs) {
      var qa = project(a), qb = project(b);
      return fx.el('line', Object.assign({ x1: qa[0], y1: qa[1], x2: qb[0], y2: qb[1] }, attrs), layer);
    }
    // The argmax regions: from each vertex to the two edge midpoints and the barycenter.
    var c = [1 / 3, 1 / 3, 1 / 3];
    var E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    var mid = function (a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2]; };
    var regions = [0, 1, 2].map(function (k) {
      var i = (k + 1) % 3, j = (k + 2) % 3;
      return [E[k], mid(E[k], E[i]), c, mid(E[k], E[j])];
    });
    var AX = ['p_0', 'p_1', 'p_2'];

    function draw() {
      fx.clear(layer);
      var O = [0, 0, 0];
      // Axes up to the corners, under the triangle.
      E.forEach(function (e) { line(O, e, { stroke: fx.grey, 'stroke-width': 2.5 }); });
      var o = project(O);
      fx.el('circle', { cx: o[0], cy: o[1], r: 4, fill: fx.grey }, layer);
      regions.forEach(function (poly, k) {
        fx.el('path', { d: path(poly), fill: fx.fill[k], 'fill-opacity': 0.92, stroke: '#8a8a8a', 'stroke-width': 2, 'stroke-linejoin': 'round' }, layer);
      });
      fx.el('path', { d: path(E), fill: 'none', stroke: fx.ink, 'stroke-width': 3, 'stroke-linejoin': 'round' }, layer);
      // The axes go on past the corners, with their names.
      E.forEach(function (e, k) {
        var far = e.map(function (v) { return v * 1.32; });
        line(e, far, { stroke: fx.grey, 'stroke-width': 2.5 });
        var q = project(far), qe = project(e);
        fx.arrow(layer, qe[0], qe[1], q[0] - qe[0], q[1] - qe[1], Math.hypot(q[0] - qe[0], q[1] - qe[1]), fx.grey);
        var lab = project(e.map(function (v) { return v * 1.45; }));
        fx.tex(layer, lab[0], lab[1], AX[k], { size: 24, color: fx.grey });
        fx.el('circle', { cx: qe[0], cy: qe[1], r: 5, fill: fx.ink }, layer);
      });
      regions.forEach(function (poly, k) {
        var g = [0, 1, 2].map(function (a) { return (poly[0][a] * 2 + poly[1][a] + poly[2][a] + poly[3][a]) / 5; });
        var q = project(g);
        fx.text(layer, q[0], q[1] + 7, 'class ' + k, { 'font-size': 20, fill: fx.cls[k] });
      });
    }
    draw();

    // Drag to turn by hand.
    var last = null;
    svg.addEventListener('pointerdown', function (ev) { held = performance.now(); last = ev.clientX; svg.setPointerCapture(ev.pointerId); });
    svg.addEventListener('pointermove', function (ev) {
      if (last === null) return;
      held = performance.now();
      az = Math.max(5, Math.min(85, az + (ev.clientX - last) * 0.4));
      last = ev.clientX;
      draw();
    });
    svg.addEventListener('pointerup', function () { last = null; });
    svg.style.cursor = 'grab';

    // A slow swing of +-18 degrees, about 12 s a period, at most 20 frames a second.
    var section = host.closest('section'), t0 = null, before = 0;
    function tick(now) {
      if (!spin) return;
      if (section && section.classList.contains('present')) {
        // After a drag, wait 2.5 s and swing again around the angle it left.
        if (now - held < 2500) { t0 = null; requestAnimationFrame(tick); return; }
        if (t0 === null) { t0 = now; base0 = az; }
        if (now - before > 50) {
          before = now;
          az = base0 + swing * Math.sin((now - t0) / 12000 * 2 * Math.PI);
          draw();
        }
      } else {
        t0 = null;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  };
})();
