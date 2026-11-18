// A Voronoi rule under three distances (K = 3): each point of the simplex
// goes to the nearest of three draggable prototypes. With the Euclidean
// distance, the one used in the dissertation, the boundaries are straight
// bisectors; with the Manhattan (L1) distance they bend, and with the
// Hellinger distance they curve. (Chebyshev is not offered: for K = 3 a
// difference d of two points of the simplex has sum 0, so |d|_1 = 2 |d|_inf
// and its cells are the Manhattan ones.) At the vertices every one gives argmax. The cells are painted on a canvas under the drawing, point by point.
(function () {
  'use strict';
  var fx = Deck.fx;
  var DIST = {
    euclid: function (a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); },
    l1: function (a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]); },
    hell: function (a, b) { return Math.hypot(Math.sqrt(a[0]) - Math.sqrt(b[0]), Math.sqrt(a[1]) - Math.sqrt(b[1]), Math.sqrt(a[2]) - Math.sqrt(b[2])); }
  };
  var NAMES = { euclid: 'Euclidean', l1: 'Manhattan (L1)', hell: 'Hellinger' };
  var FILL = ['#D9F2D9', '#D9D9FF', '#FFD9D9'];

  Deck.figures.voronoi_metrics = function (host, state) {
    var metric = DIST[state.preset] ? state.preset : 'euclid';
    var VERT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    var mus = state.preset === 'moved' ? [[.7, .2, .1], [.35, .45, .2], [.2, .25, .55]] : VERT.map(function (v) { return v.slice(); });
    fx.flex(host, 'column', 6);
    var W = 480, H = 410;
    var box = fx.html('div', '', host);
    box.style.position = 'relative';
    box.style.width = W + 'px';
    box.style.height = H + 'px';
    var canvas = fx.html('canvas', '', box);
    canvas.width = W; canvas.height = H;
    canvas.style.position = 'absolute'; canvas.style.left = '0'; canvas.style.top = '0';
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    var svg = fx.svg(box, W, H);
    svg.style.position = 'absolute'; svg.style.left = '0'; svg.style.top = '0';
    var S = fx.simplex(W / 2, 36, 400);
    var layer = fx.el('g', {}, svg);
    S.outline(svg);
    S.labels(svg, 26);
    var handles = mus.map(function (_, k) {
      var g = fx.el('g', {}, svg);
      fx.el('circle', { r: 26, fill: 'transparent' }, g);
      fx.el('circle', { r: 13, fill: fx.cls[k], stroke: '#fff', 'stroke-width': 3 }, g);
      fx.drag(g, svg, function (x, y) { mus[k] = S.clamp(S.bary(x, y)); schedule(); });
      return g;
    });
    var bar = fx.html('div', 'ctl', host);
    fx.choice(bar, [['euclid', NAMES.euclid], ['l1', NAMES.l1], ['hell', NAMES.hell]], metric, function (m) { metric = m; draw(); });
    fx.button(bar, 'at the vertices', function () { mus = VERT.map(function (v) { return v.slice(); }); draw(); });
    var readout = fx.html('div', 'readout', host);
    readout.style.minHeight = '1.5em';
    var ctx = canvas.getContext('2d');
    var pending = false;
    function schedule() { if (!pending) { pending = true; requestAnimationFrame(function () { pending = false; draw(); }); } }

    var exact = fx.el('g', {}, svg);
    svg.insertBefore(exact, svg.firstChild);   // under the outline and the prototypes

    function draw() {
      ctx.clearRect(0, 0, W, H);
      fx.clear(exact);
      // Euclidean cells are polygons, drawn exactly; the others are painted point by point.
      if (metric === 'euclid') fx.drawRegions(exact, S, fx.regions(fx.voronoi(mus)), { fill: FILL });
      var d = DIST[metric], n = metric === 'euclid' ? 0 : 160, cell = S.side / Math.max(n, 1);
      for (var a = 0; n && a <= n; a++) {
        for (var b = 0; a + b <= n; b++) {
          var p = [a / n, b / n, (n - a - b) / n], best = 0, bd = d(p, mus[0]);
          for (var k = 1; k < 3; k++) { var v = d(p, mus[k]); if (v < bd) { bd = v; best = k; } }
          var q = S.xy(p);
          ctx.fillStyle = FILL[best];
          ctx.fillRect(q[0] - cell * 0.6, q[1] - cell * 0.6, cell * 1.2, cell * 1.2);
        }
      }
      // Keep the paint inside the triangle.
      ctx.save();
      ctx.globalCompositeOperation = 'destination-in';
      ctx.beginPath();
      S.v.forEach(function (v, i) { if (i) ctx.lineTo(v[0], v[1]); else ctx.moveTo(v[0], v[1]); });
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      fx.clear(layer);
      var mid = S.xy([1 / 3, 1 / 3, 1 / 3]);
      mus.forEach(function (m, k) {
        var q = S.xy(m);
        handles[k].setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')');
        // The label sits on the side of the centre, so it stays inside the triangle at a vertex.
        var dx = mid[0] - q[0], dy = mid[1] - q[1], u = Math.hypot(dx, dy);
        var lx = u > 60 ? q[0] + dx / u * 40 : q[0] + 26, ly = u > 60 ? q[1] + dy / u * 40 : q[1] - 22;
        fx.tex(layer, lx, ly, '\\boldsymbol{\\mu}_' + k, { size: 24, color: fx.cls[k] });
      });
      readout.textContent = metric === 'euclid' ? 'Euclidean: straight boundaries, the bisectors of the prototypes' : metric === 'l1' ? 'Manhattan (L1): the boundaries bend' : 'Hellinger: the boundaries curve';
    }
    draw();
  };
})();
