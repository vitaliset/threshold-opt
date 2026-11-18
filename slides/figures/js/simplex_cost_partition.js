// Bayes partition of the simplex as the cost matrix changes (K = 3), in
// two linked views.
//   Left: the risk of each action, L_i(p) = c_i' p with c_i the column i of
//   C, drawn as a plane over the simplex. The Bayes rule takes the lowest
//   plane at each p; the lower envelope is drawn solid and its shadow on
//   the floor is the partition. This is the three-dimensional reading of
//   the thesis figure simplex_cost_partition_3d.
//   Right: the partition itself, with the normals n_ij = c_i - c_j of the
//   boundaries projected on the simplex, and the 0/1 boundaries dashed.
// Default: the example of the slide, C_20 = 5 and C_21 = 3.
(function () {
  'use strict';
  var fx = Deck.fx;
  var TRI = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  var PRESETS = {
    zeroone: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    example: [[0, 1, 1], [1, 0, 1], [5, 3, 0]],
    hedge: [[0, 10, 2], [10, 0, 2], [3, 3, 5]]
  };

  Deck.figures.simplex_cost_partition = function (host, state) {
    var key = PRESETS[state.preset] ? state.preset : 'example';
    var C = PRESETS[key].map(function (r) { return r.slice(); });
    var azimuth = 200;

    fx.flex(host, 'row', 26);
    var left = fx.html('div', '', host);
    var mid = fx.html('div', '', host);
    var right = fx.html('div', '', host);
    var W3 = 520, H3 = 470;
    var svg3 = fx.svg(left, W3, H3);
    var layer3 = fx.el('g', {}, svg3);
    var bar3 = fx.html('div', 'ctl', left);
    bar3.style.marginTop = '0';
    var held = -1e9;   // turns by itself while its slide is shown; a touch pauses it for 2.5 s
    var rot = fx.slider(bar3, { label: 'rotate', min: 0, max: 360, step: 1, value: azimuth, digits: 0, width: 220, format: function (v) { return v + '°'; } }, function (v) { held = performance.now(); azimuth = v; draw(); });
    svg3.addEventListener('pointerdown', function () { held = performance.now(); });
    var W = 440, H = 400;
    var svg = fx.svg(mid, W, H);
    var S = fx.simplex(W / 2, 36, 372);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 24);

    // The matrix: one small slider per entry, row = true class, column = prediction.
    var grid = fx.html('div', '', right);
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'auto repeat(3, 118px)';
    grid.style.gap = '4px 10px';
    grid.style.alignItems = 'center';
    fx.html('span', '', grid, '');
    for (var j = 0; j < 3; j++) fx.html('span', '', grid, 'pred. ' + j).style.textAlign = 'center';
    var inputs = [];
    C.forEach(function (row, i) {
      fx.html('span', '', grid, 'true ' + i);
      inputs.push(row.map(function (_, j) {
        var cell = fx.html('div', '', grid);
        return fx.slider(cell, { label: '', min: 0, max: 10, step: 0.5, value: C[i][j], width: 56, digits: 1 }, function (v) {
          C[i][j] = v;
          picker.mark('');
          draw();
        });
      }));
    });
    var bar = fx.html('div', 'ctl', right);
    bar.style.gap = '8px';
    var picker = fx.choice(bar, [['zeroone', '0/1'], ['example', 'C₂₀ = 5, C₂₁ = 3']], key, function (k) {
      C = PRESETS[k].map(function (r) { return r.slice(); });
      inputs.forEach(function (row, i) { row.forEach(function (input, j) { input.set(C[i][j]); }); });
      draw();
    });
    var readout = fx.html('div', 'readout', right);
    readout.style.fontSize = '26px';

    function score(i) {
      return function (p) { return C[0][i] * p[0] + C[1][i] * p[1] + C[2][i] * p[2]; };
    }

    // Point of the plane sum p = 1 where the three risks are equal.
    function meeting() {
      var a = [0, 1, 2].map(function (k) { return C[k][0] - C[k][1]; });
      var b = [0, 1, 2].map(function (k) { return C[k][0] - C[k][2]; });
      var det = function (m) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
      };
      var M = [a, b, [1, 1, 1]], d = det(M);
      if (Math.abs(d) < 1e-9) return null;
      var rhs = [0, 0, 1];
      return [0, 1, 2].map(function (c) {
        return det(M.map(function (row, r) { return row.map(function (x, cc) { return cc === c ? rhs[r] : x; }); })) / d;
      });
    }

    // ---- the three risk planes over the simplex ----
    function draw3(regions, scores) {
      fx.clear(layer3);
      var th = azimuth * Math.PI / 180, phi = 32 * Math.PI / 180;
      var top = Math.max(1, Math.max.apply(null, C.map(function (r) { return Math.max.apply(null, r); })));
      var base = 300, zs = 120 / top, gap = 120;   // the planes float `gap` px above the floor
      // Floor coordinates of a barycentric point, centred at the barycenter.
      var floor = function (p) { return [(p[1] - p[2]) * base / 2, (p[0] - 1 / 3) * base * Math.sqrt(3) / 2]; };
      var project = function (p, z) {
        var f = floor(p);
        var x = f[0] * Math.cos(th) - f[1] * Math.sin(th), d = f[0] * Math.sin(th) + f[1] * Math.cos(th);
        return { x: W3 / 2 + x, y: 360 + d * Math.sin(phi) - z, t: d * Math.cos(phi) + z * Math.sin(phi) };
      };
      var path = function (pts) { return pts.map(function (q, k) { return (k ? 'L' : 'M') + q.x.toFixed(1) + ' ' + q.y.toFixed(1); }).join(' ') + ' Z'; };

      // The floor: the partition, as the shadow of the lower envelope.
      regions.forEach(function (poly, k) {
        if (poly.length > 2) fx.el('path', { d: path(poly.map(function (p) { return project(p, 0); })), fill: fx.fill[k], stroke: '#8a8a8a', 'stroke-width': 2 }, layer3);
      });
      fx.el('path', { d: path(TRI.map(function (p) { return project(p, 0); })), fill: 'none', stroke: fx.ink, 'stroke-width': 2 }, layer3);

      // Each plane is cut along the two lines where it meets the others, so
      // that no piece crosses another plane; the pieces are then painted
      // from the farthest to the nearest.
      var pieces = [];
      [0, 1, 2].forEach(function (i) {
        var others = [0, 1, 2].filter(function (k) { return k !== i; });
        [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach(function (signs) {
          var poly = TRI;
          others.forEach(function (k, n) {
            if (poly.length) poly = fx.clip(poly, function (p) { return signs[n] * (scores[k](p) - scores[i](p)); });
          });
          if (poly.length < 3 || fx.area(poly) < 1e-7) return;
          var pts = poly.map(function (p) { return project(p, gap + zs * scores[i](p)); });
          var depth = pts.reduce(function (acc, q) { return acc + q.t; }, 0) / pts.length;
          // The clip keeps signs * (L_k - L_i) <= 0, so signs (-1, -1) is where
          // plane i lies below both others: its part of the lower envelope.
          pieces.push({ pts: pts, t: depth, i: i, low: signs[0] === -1 && signs[1] === -1 });
        });
      });
      pieces.sort(function (a, b) { return a.t - b.t; });
      pieces.forEach(function (pc) {
        fx.el('path', { d: path(pc.pts), fill: fx.cls[pc.i], 'fill-opacity': pc.low ? 0.72 : 0.13, stroke: fx.cls[pc.i], 'stroke-width': pc.low ? 2.5 : 1.5, 'stroke-opacity': pc.low ? 1 : 0.6, 'stroke-linejoin': 'round' }, layer3);
      });
      // Plane labels at their highest corner; vertex labels on the floor.
      [0, 1, 2].forEach(function (i) {
        var hi = 0;
        for (var k = 1; k < 3; k++) if (C[k][i] > C[hi][i]) hi = k;
        var q = project(TRI[hi], gap + zs * C[hi][i]);
        fx.tex(layer3, q.x + (q.x < W3 / 2 ? -28 : 28), q.y - 14 - 18 * i, '\\mathcal{L}_' + i, { size: 26, color: fx.cls[i] });
        var e = project(TRI[i].map(function (v) { return v * 1.16 - 0.16 / 3; }), 0);
        fx.tex(layer3, e.x, e.y + 6, '\\mathbf{e}_' + i, { size: 22 });
      });
    }

    function draw() {
      fx.clear(layer);
      var scores = [score(0), score(1), score(2)];
      var regions = fx.regions(scores);
      draw3(regions, scores);
      // 0/1 boundaries as a dashed reference.
      [[[1, 0, 0], [0, .5, .5]], [[0, 1, 0], [.5, 0, .5]], [[0, 0, 1], [.5, .5, 0]]].forEach(function (seg) {
        var a = S.xy(seg[0]), b = S.xy(seg[1]);
        fx.el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: '#b5b5b5', 'stroke-width': 1.5, 'stroke-dasharray': '7 6' }, layer);
      });
      fx.drawRegions(layer, S, regions);
      S.outline(layer);
      regions.forEach(function (poly, k) {
        if (fx.area(poly) < 0.012) return;
        var c = S.xy(fx.centroid(poly));
        fx.tex(layer, c[0], c[1], '\\mathcal{P}_' + k + '^*', { color: fx.cls[k], size: 26 });
      });
      var m = meeting();
      var inside = m && m.every(function (x) { return x > 1e-6; });
      if (inside) {
        var q = S.xy(m);
        fx.el('circle', { cx: q[0], cy: q[1], r: 7, fill: fx.ink }, layer);
      }
      readout.innerHTML = 'planes: the risks 𝓛<sub>i</sub>(p) = c<sub>i</sub>ᵀp; solid: the lowest<br>' +
        (inside ? 'they meet at (' + m.map(function (x) { return fx.fmt(x, 3); }).join(', ') + ')' : 'no common point inside the simplex');
      last = { regions: regions, scores: scores };
    }
    var last = null;
    draw();

    // About 12 degrees a second, redrawn at most 20 times a second.
    var section = host.closest('section'), before = null;
    function tick(now) {
      if (section && section.classList.contains('present') && last) {
        if (before === null || now - held < 2500) before = now;
        else if (now - before > 50) {
          azimuth = (azimuth + (now - before) * 0.012) % 360;
          before = now;
          rot.set(Math.round(azimuth));
          draw3(last.regions, last.scores);
        }
      } else {
        before = null;
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  };
})();
