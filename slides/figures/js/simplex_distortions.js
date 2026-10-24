// Miscalibration in the simplex (K = 3). The Bayes regions are those of a
// fixed cost matrix and do not move; a cloud of true probability vectors
// eta_i, coloured by class, is distorted by one slider.
//   Oversmoothing  (s > 0): p = (1 - s) eta + s u, a contraction toward the
//     barycenter u = (1/3, 1/3, 1/3), as in chapter 4 of the dissertation.
//   Overconfidence (s < 0): p = softmax(log(eta) / T), T = 1 - 0.85 |s|,
//     which pushes the estimates to the vertices.
// The cost matrix makes an error on true class 0 or 1 cost 3 and an error
// on true class 2 cost 1, so the Bayes region of class 2 is a small area
// at its vertex that does not contain the barycenter. Contracted
// estimates then stop reaching it and class 2 loses its recall, although
// the boundaries are where the cost matrix puts them. The matrix is
// element-wise reasonable, so at the vertices the plug-in rule is argmax.
// The matrix and the cloud are choices of this schematic.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C = [[0, 3, 3], [3, 0, 3], [1, 1, 0]];
  var SCORES = [0, 1, 2].map(function (i) { return function (p) { return C[0][i] * p[0] + C[1][i] * p[1] + C[2][i] * p[2]; }; });

  function decide(p) { return fx.argmin(SCORES.map(function (s) { return s(p); })); }

  function softmax(z) {
    var m = Math.max.apply(null, z);
    var e = z.map(function (v) { return Math.exp(v - m); }), sum = e[0] + e[1] + e[2];
    return e.map(function (v) { return v / sum; });
  }

  // True probability vectors, kept when the Bayes rule gives them their class.
  function cloud() {
    var rng = fx.rng(7), pts = [];
    var centres = [[.66, .18, .16], [.18, .66, .16], [.1, .1, .8]];
    var counts = [36, 36, 16];
    centres.forEach(function (c, y) {
      var kept = 0;
      while (kept < counts[y]) {
        var eta = softmax(c.map(function (v) { return Math.log(v) + 0.35 * rng.normal(); }));
        if (decide(eta) === y) { pts.push({ eta: eta, y: y }); kept += 1; }
      }
    });
    return pts;
  }

  Deck.figures.simplex_distortions = function (host, state) {
    var s = state.preset === 'smooth' ? 0.8 : state.preset === 'sharp' ? -0.9 : state.preset === 'none' ? 0 : 0.55;
    var pts = cloud();
    fx.flex(host, 'row', 60);
    var W = 560, H = 400;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var S = fx.simplex(W / 2, 34, 390);
    fx.drawRegions(svg, S, fx.regions(SCORES));
    S.outline(svg);
    S.labels(svg, 24);
    var u = S.xy([1 / 3, 1 / 3, 1 / 3]);
    fx.el('path', { d: 'M' + (u[0] - 9) + ' ' + u[1] + ' L' + (u[0] + 9) + ' ' + u[1] + ' M' + u[0] + ' ' + (u[1] - 9) + ' L' + u[0] + ' ' + (u[1] + 9), stroke: fx.ink, 'stroke-width': 2.5 }, svg);
    var layer = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    side.style.width = '640px';
    var bar = fx.html('div', 'ctl col', side);
    fx.slider(bar, { label: 'overconfident', min: -1, max: 1, step: 0.01, value: s, width: 300,
      format: function () { return 'oversmoothed'; } }, function (v) { s = v; draw(); });
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.45';

    function distort(eta) {
      if (s >= 0) return eta.map(function (v) { return (1 - s) * v + s / 3; });
      return softmax(eta.map(function (v) { return Math.log(v) / (1 - 0.85 * (-s)); }));
    }

    function draw() {
      fx.clear(layer);
      var hit = [0, 0, 0], n = [0, 0, 0];
      pts.forEach(function (pt) {
        var p = distort(pt.eta);
        n[pt.y] += 1;
        if (decide(p) === pt.y) hit[pt.y] += 1;
        var q = S.xy(p);
        fx.el('circle', { cx: q[0], cy: q[1], r: 6, fill: fx.cls[pt.y], stroke: '#fff', 'stroke-width': 1.2 }, layer);
      });
      var what = s > 0.02 ? 'contracted toward the barycenter (+)'
        : s < -0.02 ? 'sharpened, temperature ' + fx.fmt(1 - 0.85 * (-s), 2) : 'undistorted';
      readout.innerHTML = 'estimates ' + what + '<br>recall of the plug-in rule: ' + [0, 1, 2].map(function (k) {
        var r = hit[k] / n[k];
        return '<span style="color:' + fx.cls[k] + '">class ' + k + '</span> ' + (r === 0 ? '<span class="bad">0%</span>' : Math.round(100 * r) + '%');
      }).join(', ') + '<br>costs: an error on true class 0 or 1 costs 3, on true class 2 costs 1';
    }
    draw();
  };
})();
