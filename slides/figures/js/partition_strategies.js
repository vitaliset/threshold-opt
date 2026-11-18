// What a parallel shift can do and what prototypes add (K = 3, 0/1 loss),
// in one figure with two modes, after the two panels of the thesis figure.
//   Parallel shift, argmax_k (p_k - alpha_k): K - 1 = 2 parameters. They
//     are the position of the point where the three boundaries meet, which
//     can be dragged; every boundary keeps the direction it has under
//     argmax, so its rotation is 0 degrees.
//   Voronoi rule: K (K - 1) = 6 parameters, the prototypes, which can be
//     dragged. Each boundary is the perpendicular bisector of the segment
//     joining two prototypes (drawn thin), so it turns with that segment.
// The readout gives the rotation of each boundary relative to its argmax
// direction, and whether the dominant label constraint (DLC) holds: every
// vertex e_k strictly nearest to its own prototype mu_k.
// The argmax boundaries are dashed.
(function () {
  'use strict';
  var fx = Deck.fx;
  var E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  var PROTOTYPES = [[.7, .2, .1], [.35, .45, .2], [.2, .25, .55]];
  var MEETING = [.5, .3, .2];

  function dist2(a, b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
  }

  // For each vertex, whether its own prototype is the strict nearest one.
  Deck.fx.dlc = function (mus) {
    return E.map(function (v, k) {
      return mus.every(function (m, j) { return j === k || dist2(v, mus[k]) < dist2(v, m) - 1e-12; });
    });
  };

  Deck.figures.partition_strategies = function (host, state) {
    var mode = state.preset === 'shift' ? 'shift' : 'voronoi';
    var mus = (state.preset === 'broken' ? [[.3, .6, .1], [.2, .3, .5], [.6, .1, .3]] : PROTOTYPES).map(function (m) { return m.slice(); });
    var meet = MEETING.slice();
    fx.flex(host, 'row', 50);
    var W = 660, H = 540;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var S = fx.simplex(W / 2, 40, 540);
    var layer = fx.el('g', {}, svg);
    S.labels(svg);
    var handles = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    side.style.width = '640px';
    var bar = fx.html('div', 'ctl', side);
    bar.style.justifyContent = 'flex-start';
    bar.style.gap = '8px';
    fx.choice(bar, [['shift', 'parallel shift'], ['voronoi', 'prototypes']], mode, function (k) { mode = k; sync(); draw(); });
    fx.button(bar, 'reset to argmax', function () {
      mus = E.map(function (m) { return m.slice(); });
      meet = [1 / 3, 1 / 3, 1 / 3];
      draw();
    });
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.45';

    var dots = mus.map(function (_, k) {
      var g = fx.el('g', {}, handles);
      fx.el('circle', { r: 26, fill: 'transparent' }, g);
      fx.el('circle', { r: 13, fill: fx.cls[k], stroke: '#fff', 'stroke-width': 3 }, g);
      fx.drag(g, svg, function (x, y) { mus[k] = S.clamp(S.bary(x, y)); draw(); });
      return g;
    });
    var knot = fx.el('g', {}, handles);
    fx.el('circle', { r: 28, fill: 'transparent' }, knot);
    fx.el('rect', { x: -11, y: -11, width: 22, height: 22, fill: fx.ink, stroke: '#fff', 'stroke-width': 3 }, knot);
    fx.drag(knot, svg, function (x, y) {
      var p = S.clamp(S.bary(x, y));
      // Keep the meeting point strictly inside, so that all three regions remain.
      meet = p.map(function (v) { return 0.04 + 0.88 * v; });
      draw();
    });

    function sync() {
      dots.forEach(function (g) { g.style.display = mode === 'voronoi' ? '' : 'none'; });
      knot.style.display = mode === 'shift' ? '' : 'none';
    }

    // Signed angle, in degrees, between the boundary of classes i and j and
    // its direction under argmax; the boundary is normal to n.
    function rotation(n, i, j) {
      var screen = function (v) {
        return [v[0] * S.v[0][0] + v[1] * S.v[1][0] + v[2] * S.v[2][0], v[0] * S.v[0][1] + v[1] * S.v[1][1] + v[2] * S.v[2][1]];
      };
      var a = screen(n), b = screen(E[i].map(function (x, k) { return x - E[j][k]; }));
      var deg = Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) * 180 / Math.PI;
      return Math.round(deg);
    }

    function draw() {
      fx.clear(layer);
      var scores;
      if (mode === 'voronoi') {
        scores = fx.voronoi(mus);
      } else {
        // The three scores are equal at the meeting point: alpha = meet - barycenter.
        scores = [0, 1, 2].map(function (k) { return function (p) { return -(p[k] - (meet[k] - 1 / 3)); }; });
      }
      fx.drawRegions(layer, S, fx.regions(scores));
      [[0, .5, .5], [.5, 0, .5], [.5, .5, 0]].forEach(function (mid) {
        var a = S.xy([1 / 3, 1 / 3, 1 / 3]), b = S.xy(mid);
        fx.el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: .5 }, layer);
      });
      S.outline(layer);
      var lines = [];
      if (mode === 'voronoi') {
        // The segments whose perpendicular bisectors are the boundaries.
        [[0, 1], [0, 2], [1, 2]].forEach(function (pair) {
          var a = S.xy(mus[pair[0]]), b = S.xy(mus[pair[1]]);
          fx.el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: fx.ink, 'stroke-width': 1.5, opacity: 0.45 }, layer);
        });
        dots.forEach(function (g, k) {
          var q = S.xy(mus[k]);
          g.setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')');
        });
        var angles = [[0, 1], [0, 2], [1, 2]].map(function (pair) {
          var n = mus[pair[0]].map(function (x, k) { return x - mus[pair[1]][k]; });
          return pair[0] + '|' + pair[1] + ': ' + (dist2(mus[pair[0]], mus[pair[1]]) < 1e-9 ? '–' : rotation(n, pair[0], pair[1]) + '°');
        });
        var ok = fx.dlc(mus);
        var broken = ok.map(function (v, k) { return v ? null : 'e<sub>' + k + '</sub>'; }).filter(Boolean);
        lines.push('<strong>K(K − 1) = 6 parameters</strong>: the prototypes (drag them)');
        lines.push('each boundary bisects the segment between two prototypes');
        lines.push('rotation of the boundaries:<br><strong>' + angles.join(', &nbsp;') + '</strong>');
        lines.push(broken.length ? '<span class="bad">DLC violated at ' + broken.join(', ') + '</span>' : 'DLC holds');
      } else {
        var q = S.xy(meet);
        knot.setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')');
        lines.push('<strong>K − 1 = 2 parameters</strong>: the meeting point (drag it)');
        lines.push('every boundary keeps its argmax direction');
        lines.push('rotation of the boundaries:<br><strong>0|1: 0°, &nbsp;0|2: 0°, &nbsp;1|2: 0°</strong>');
      }
      readout.innerHTML = lines.join('<br>');
    }
    sync(); draw();
  };
})();
