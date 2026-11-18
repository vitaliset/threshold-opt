// How the sequential d^2 sampler is built, in three modes (K = 3).
//   uniform     each prototype mu_k uniform on the simplex;
//   swap        the same draws, with the largest coordinate of the k-th one
//               moved to position k (the first step of the sampler);
//   sequential  the sampler itself, one prototype at a time: a draw p for
//               mu_i is kept only if, for every earlier k,
//                 d^2(e_k, p) > d^2(e_k, mu_k)   (outside the dashed disc)
//                 d^2(e_i, p) < d^2(e_i, mu_k)   (inside the dotted disc).
// The figure draws the Voronoi partition of the prototypes and a line from
// each vertex to its nearest prototype; a vertex ringed in red goes to
// another class, so the DLC fails. In the sequential mode the shaded area is
// where the next prototype can be accepted and crosses are rejected draws.
// state.modes lists the modes the figure offers. Seeds: 1 for the two simple
// modes (the uniform draw fails at all three vertices, the swapped one holds),
// 180 for the sampler (one rejected draw for mu_1). The readout has a fixed
// size, so that the slide does not move as its text changes.
(function () {
  'use strict';
  var fx = Deck.fx;
  var E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  var count = 0;

  function d2(a, b) { return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]); }
  function uniform(rng) {
    var g = [0, 1, 2].map(function () { return -Math.log(Math.max(rng(), 1e-12)); });
    var s = g[0] + g[1] + g[2];
    return g.map(function (v) { return v / s; });
  }
  function swapTo(p, i) {
    var q = p.slice(), j = 0;
    for (var k = 1; k < 3; k++) if (q[k] > q[j]) j = k;
    var t = q[i]; q[i] = q[j]; q[j] = t;
    return q;
  }
  function nearest(mus, x) {
    var best = 0;
    for (var j = 1; j < mus.length; j++) if (d2(x, mus[j]) < d2(x, mus[best])) best = j;
    return best;
  }
  function accepted(p, i, mus) {
    for (var k = 0; k < i; k++) if (!(d2(E[k], p) > d2(E[k], mus[k]) && d2(E[i], p) < d2(E[i], mus[k]))) return false;
    return true;
  }

  Deck.figures.sampler_story = function (host, state) {
    var modes = state.modes || ['uniform', 'swap'];
    var mode = modes[0], uid = ++count;
    var NAMES = { uniform: 'uniform', swap: 'swap', sequential: 'sequential d²' };
    var tally = { uniform: [0, 0], swap: [0, 0] };   // [draws, failures]
    var seed = { uniform: 1, swap: 1, sequential: 180 };
    var mus = [], raws = [], rejected = [], rng = null, nextI = 0, lastRejected = 0;

    fx.flex(host, 'column', 6);
    var W = 470, H = 360;
    var svg = fx.svg(host, W, H);
    var S = fx.simplex(W / 2, 30, 356);
    var px = S.side / Math.SQRT2;
    var clip = fx.el('clipPath', { id: 'story-tri-' + uid }, fx.el('defs', {}, svg));
    fx.el('path', { d: S.path(E) }, clip);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 24);
    var bar = fx.html('div', 'ctl', host);
    bar.style.gap = '8px';
    if (modes.length > 1) fx.choice(bar, modes.map(function (m) { return [m, NAMES[m]]; }), mode, function (m) { mode = m; reset(); });
    var act = fx.button(bar, mode === 'sequential' ? 'next prototype' : 'new draw', function () { step(); });
    if (modes.indexOf('sequential') >= 0) fx.button(bar, 'new configuration', function () { seed.sequential += 1; reset(); });
    var readout = fx.html('div', 'readout', host);
    readout.style.width = W + 'px';
    readout.style.minHeight = modes.indexOf('sequential') >= 0 ? '4.4em' : '2.9em';
    readout.style.fontSize = '24px';
    readout.style.lineHeight = '1.45';

    function reset() {
      act.textContent = mode === 'sequential' ? 'next prototype' : 'new draw';
      rng = fx.rng(seed[mode]);
      if (mode === 'sequential') {
        mus = [swapTo(uniform(rng), 0)];   // nothing to compare with: the first draw is kept
        nextI = 1; rejected = []; lastRejected = 0;
      } else {
        tally[mode] = [0, 0];   // the count starts again when the mode is chosen
        drawSimple();
      }
      draw();
    }
    function drawSimple() {
      raws = [0, 1, 2].map(function () { return uniform(rng); });
      mus = mode === 'swap' ? raws.map(function (p, k) { return swapTo(p, k); }) : raws.slice();
      var t = tally[mode];
      t[0] += 1;
      if ([0, 1, 2].some(function (k) { return nearest(mus, E[k]) !== k; })) t[1] += 1;
    }
    function step() {
      if (mode !== 'sequential') { seed[mode] += 1; rng = fx.rng(seed[mode]); drawSimple(); draw(); return; }
      if (nextI >= 3) { seed.sequential += 1; reset(); return; }
      rejected = [];
      for (var tries = 0; tries < 500; tries++) {
        var p = swapTo(uniform(rng), nextI);
        if (accepted(p, nextI, mus)) { mus.push(p); break; }
        rejected.push(p);
      }
      lastRejected = rejected.length;
      nextI += 1;
      draw();
    }

    function dot(p, r, attrs) { var q = S.xy(p); return fx.el('circle', Object.assign({ cx: q[0], cy: q[1], r: r }, attrs), layer); }

    function draw() {
      fx.clear(layer);
      var full = mus.length === 3;
      if (full) fx.drawRegions(layer, S, fx.regions(fx.voronoi(mus)), { fill: ['#EEF8EE', '#EEEEFF', '#FFEEEE'] });
      else fx.el('path', { d: S.path(E), fill: '#fafafa' }, layer);
      // Sequential: where the next prototype can go, and the two discs of each earlier one.
      if (mode === 'sequential' && !full) {
        var i = nextI, g = fx.el('g', { 'clip-path': 'url(#story-tri-' + uid + ')' }, layer), h = 1 / 70, cell = S.side * h * 0.95;
        for (var a = 0; a <= 1 + 1e-9; a += h) {
          for (var b = 0; a + b <= 1 + 1e-9; b += h) {
            var p = [a, b, Math.max(0, 1 - a - b)];
            if (p[i] < Math.max(p[0], p[1], p[2])) continue;   // outside the argmax cell of class i
            if (!accepted(p, i, mus)) continue;
            var q = S.xy(p);
            fx.el('rect', { x: q[0] - cell / 2, y: q[1] - cell / 2, width: cell, height: cell, fill: fx.yellow, opacity: 0.45 }, g);
          }
        }
        mus.forEach(function (m, k) {
          var ek = S.xy(E[k]), ei = S.xy(E[i]);
          fx.el('circle', { cx: ek[0], cy: ek[1], r: Math.sqrt(d2(E[k], m)) * px, fill: 'none', stroke: fx.cls[k], 'stroke-width': 2.5, 'stroke-dasharray': '9 6' }, g);
          fx.el('circle', { cx: ei[0], cy: ei[1], r: Math.sqrt(d2(E[i], m)) * px, fill: 'none', stroke: fx.cls[i], 'stroke-width': 2.5, 'stroke-dasharray': '2 5' }, g);
        });
      }
      S.outline(layer);
      // Swap: each raw draw (hollow) and where the swap takes it.
      if (mode === 'swap') {
        raws.forEach(function (p, k) {
          if (d2(p, mus[k]) < 1e-12) return;
          var a0 = S.xy(p), a1 = S.xy(mus[k]);
          fx.el('line', { x1: a0[0], y1: a0[1], x2: a1[0], y2: a1[1], stroke: fx.grey, 'stroke-width': 2, 'stroke-dasharray': '5 5' }, layer);
          dot(p, 8, { fill: '#fff', stroke: fx.grey, 'stroke-width': 2.5 });
        });
      }
      rejected.forEach(function (p) {
        var q = S.xy(p);
        fx.el('path', { d: 'M' + (q[0] - 8) + ' ' + (q[1] - 8) + ' L' + (q[0] + 8) + ' ' + (q[1] + 8) + ' M' + (q[0] - 8) + ' ' + (q[1] + 8) + ' L' + (q[0] + 8) + ' ' + (q[1] - 8), stroke: '#D22B2B', 'stroke-width': 3 }, layer);
      });
      // Each vertex to its nearest prototype; a red ring where it is not its own.
      var bad = [];
      if (full) {
        [0, 1, 2].forEach(function (k) {
          var j = nearest(mus, E[k]), a0 = S.xy(E[k]), a1 = S.xy(mus[j]);
          fx.el('line', { x1: a0[0], y1: a0[1], x2: a1[0], y2: a1[1], stroke: fx.cls[j], 'stroke-width': 3 }, layer);
          if (j !== k) { bad.push([k, j]); fx.el('circle', { cx: a0[0], cy: a0[1], r: 15, fill: 'none', stroke: '#D22B2B', 'stroke-width': 4 }, layer); }
        });
      }
      mus.forEach(function (m, k) {
        dot(m, 11, { fill: fx.cls[k], stroke: '#fff', 'stroke-width': 3 });
        var q = S.xy(m);
        fx.tex(layer, q[0] + 24, q[1] - 18, '\\boldsymbol{\\mu}_' + k, { size: 22, color: fx.cls[k] });
      });
      var lines;
      if (mode === 'sequential') {
        lines = full
          ? ['every vertex is nearest to its own prototype', '(Proposition 7.5.1)', lastRejected + ' rejected for μ2; rejections are rare at K = 3']
          : ['placing μ' + nextI + ': a draw is kept in the shaded area,', 'outside the dashed disc, inside the dotted one', nextI > 1 ? lastRejected + ' rejected for μ' + (nextI - 1) : 'μ0 is kept at once: nothing to compare with'];
      } else {
        var t = tally[mode];
        lines = [bad.length ? 'the DLC fails at ' + bad.map(function (b) { return 'e' + b[0]; }).join(', ') : 'the DLC holds: each vertex to its own class',
          NAMES[mode] + ': ' + t[1] + ' of ' + t[0] + ' draws break the DLC'];
      }
      readout.innerHTML = lines.join('<br>');
    }
    reset();
  };
})();
