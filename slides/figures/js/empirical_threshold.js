// The analytical and the empirical threshold on a miscalibrated estimate
// (K = 2), with a growing validation sample.
//
// A worked example, not data from the benchmark. The estimate p_hat(X) is
// uniform on [0, 1] and miscalibrated: among the points with p_hat = q the
// frequency of the positive class is
//     r(q) = A0 + A1 q = 0.42 + 0.5 q,
// an overconfident estimate (slope below one) with an upward bias. Costs
// have a zero diagonal, C01 = 4 (false positive) and C10 = 2.
//   The analytical threshold is p* = C01 / (C01 + C10) = 2/3: right for a
//     calibrated estimate, misplaced here. It needs no data.
//   The best threshold on p_hat solves r(t) = p*: t* = (p* - A0) / A1.
//   The risk of the threshold rule is, in closed form,
//     R(t) = C10 (A0 t + A1 t^2 / 2) + C01 ((1 - A0)(1 - t) - A1 (1 - t^2) / 2),
//     a parabola with its minimum at t*.
// The empirical risk is minimized over the grid of G = 301 thresholds of
// the binary study, the first minimizer being kept (np.argmin, as in the
// library).
//
// Top: the risk and the empirical risk of one sample, with the two
// thresholds. The sample is nested (size m is the first m points of one
// stream), so the slider changes the picture gradually. Bottom, on the same
// axis: the histogram of the empirical threshold over 200 other samples of
// size m. The readout gives the share of those samples in which the
// empirical threshold has lower risk than the analytical one.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C01 = 4, C10 = 2, A0 = 0.42, A1 = 0.5, G = 301;
  var PSTAR = C01 / (C01 + C10), TSTAR = (PSTAR - A0) / A1;
  var M_MIN = 16, M_SLIDER = 8192, M_MAX = 65536, STEPS = 96, STREAMS = 200;

  function freq(q) { return A0 + A1 * q; }

  function risk(t) {
    return C10 * (A0 * t + A1 * t * t / 2) + C01 * ((1 - A0) * (1 - t) - A1 * (1 - t * t) / 2);
  }
  var R_ANALYTICAL = risk(PSTAR);

  // Sample sizes on a geometric grid from M_MIN to M_MAX, without repeats.
  var MS = (function () {
    var out = [];
    for (var j = 0; j <= STEPS; j++) {
      var m = Math.round(M_MIN * Math.pow(M_MAX / M_MIN, j / STEPS));
      if (!out.length || m > out[out.length - 1]) out.push(m);
    }
    return out;
  })();
  var J_SLIDER = MS.filter(function (m) { return m <= M_SLIDER; }).length - 1;

  function nearest(m) {
    var best = 0;
    MS.forEach(function (mm, k) { if (Math.abs(Math.log(mm / m)) < Math.abs(Math.log(MS[best] / m))) best = k; });
    return best;
  }

  // Points are binned on the threshold grid: a point with score z is
  // predicted 1 by every threshold tau_g <= z. one[b] and zero[b] add the
  // cost of predicting 1 and 0 over the points with b = floor(z (G - 1)).
  function emptyBins() { return { one: new Float64Array(G), zero: new Float64Array(G) }; }

  function add(bins, z, y) {
    var b = Math.min(G - 1, Math.floor(z * (G - 1)));
    if (y) bins.zero[b] += C10; else bins.one[b] += C01;
  }

  // Empirical risk on the grid and its first minimizer.
  function fit(bins, m) {
    var curve = new Float64Array(G), above = 0, below = 0, g;
    for (g = G - 1; g >= 0; g--) { above += bins.one[g]; curve[g] = above; }
    for (g = 0; g < G; g++) { curve[g] = (curve[g] + below) / m; below += bins.zero[g]; }
    var best = 0;
    for (g = 1; g < G; g++) if (curve[g] < curve[best] - 1e-12) best = g;
    return { curve: curve, tau: best / (G - 1) };
  }

  function stream(seed, n) {
    var rng = fx.rng(seed), pts = [];
    for (var i = 0; i < n; i++) { var z = rng(); pts.push([z, rng() < freq(z) ? 1 : 0]); }
    return pts;
  }

  // The learned threshold at every size of MS, for one stream drawn on the fly.
  function path(seed) {
    var rng = fx.rng(seed), bins = emptyBins(), out = [], i = 0;
    MS.forEach(function (m) {
      for (; i < m; i++) { var z = rng(); add(bins, z, rng() < freq(z) ? 1 : 0); }
      out.push(fit(bins, m).tau);
    });
    return out;
  }

  // Learned thresholds of independent streams per size; computed once.
  var learned = null;
  function summarize() {
    if (learned) return learned;
    var all = [];
    for (var s = 0; s < STREAMS; s++) all.push(path(50000 + s));
    learned = MS.map(function (_, j) { return all.map(function (a) { return a[j]; }); });
    return learned;
  }

  // Share of the samples in which the learned threshold beats the analytical one.
  function beats(col) {
    var n = 0;
    col.forEach(function (t) { if (risk(t) < R_ANALYTICAL - 1e-12) n += 1; });
    return Math.round(100 * n / col.length);
  }

  Deck.figures.empirical_threshold = function (host, state) {
    var j = nearest(64), seed = 11, timer = null;
    if (state.preset === 'small') j = nearest(24);
    if (state.preset === 'large') j = J_SLIDER;
    var taus = summarize();
    var pts = stream(seed * 7919, M_SLIDER);

    fx.flex(host, 'column', 6);
    var W = 1320, H = 400, L = 110, R = 1280, T = 14, B = 232, HT = 272, HB = 346;
    var svg = fx.svg(host, W, H);
    var clip = fx.el('clipPath', { id: 'emp-thr-clip' }, fx.el('defs', {}, svg));
    fx.el('rect', { x: L, y: T, width: R - L, height: B - T }, clip);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl', host);
    var slider = fx.slider(bar, { label: 'validation size m', min: 0, max: J_SLIDER, step: 1, value: j, width: 420,
      format: function (v) { return String(MS[v]); } }, function (v) { j = v; draw(); });
    var play = fx.button(bar, 'grow', function () {
      if (timer) { stop(); return; }
      if (j >= J_SLIDER) { j = 0; }
      play.textContent = 'pause';
      timer = setInterval(function () {
        if (j >= J_SLIDER) { stop(); return; }
        j += 1; slider.set(j); draw();
      }, 110);
    });
    fx.button(bar, 'another sample', function () { seed += 1; pts = stream(seed * 7919, M_SLIDER); draw(); });
    var readout = fx.html('div', 'readout', host);
    function stop() { clearInterval(timer); timer = null; play.textContent = 'grow'; }

    // The risk axis always contains the bowl (0.88 to 1.42) and widens to
    // hold the empirical curve of the current sample, so it tightens onto
    // the bowl as the sample grows.
    var YMIN = 0.88, YMAX = 1.42;
    var X = function (t) { return L + t * (R - L); };
    var Y = function (v) { return B - (v - YMIN) / (YMAX - YMIN) * (B - T); };

    var LX = X(0.36);                                   // legend, above the bottom of the bowl

    function draw() {
      fx.clear(layer);
      var m = MS[j], bins = emptyBins(), i;
      for (i = 0; i < m; i++) add(bins, pts[i][0], pts[i][1]);
      var one = fit(bins, m);
      var lo = 0.88, hi = 1.42, g0;
      for (g0 = 0; g0 < G; g0++) { lo = Math.min(lo, one.curve[g0] - 0.04); hi = Math.max(hi, one.curve[g0] + 0.04); }
      YMIN = Math.max(0, lo);
      YMAX = hi;
      var step = YMAX - YMIN > 1.6 ? 0.5 : YMAX - YMIN > 0.75 ? 0.2 : 0.1, ticks = [];
      for (var tk = Math.ceil((YMIN + 0.02) / step) * step; tk < YMAX - 0.02; tk += step) ticks.push(tk);

      // ---- top: the risk and the empirical risk of this sample ----
      ticks.forEach(function (t) {
        fx.el('line', { x1: L, y1: Y(t), x2: R, y2: Y(t), stroke: '#e3e3e3', 'stroke-width': 1.5 }, layer);
        fx.text(layer, L - 12, Y(t) + 9, fx.fmt(t, 1), { 'font-size': 26, 'text-anchor': 'end' });
      });
      fx.el('line', { x1: L, y1: B, x2: R, y2: B, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.el('line', { x1: L, y1: B, x2: L, y2: T, stroke: fx.grey, 'stroke-width': 2 }, layer);
      fx.text(layer, 34, (T + B) / 2, 'risk', { 'font-size': 28, transform: 'rotate(-90 34 ' + (T + B) / 2 + ')' });
      var curves = fx.el('g', { 'clip-path': 'url(#emp-thr-clip)' }, layer);
      var truth = '', emp = '';
      for (var g = 0; g < G; g++) {
        var tau = g / (G - 1);
        truth += (g ? 'L' : 'M') + X(tau).toFixed(1) + ' ' + Y(risk(tau)).toFixed(1) + ' ';
        // A step function: constant between consecutive grid points.
        emp += (g ? 'L' : 'M') + X(tau).toFixed(1) + ' ' + Y(one.curve[g]).toFixed(1) + ' ';
        if (g < G - 1) emp += 'L' + X((g + 1) / (G - 1)).toFixed(1) + ' ' + Y(one.curve[g]).toFixed(1) + ' ';
      }
      fx.el('path', { d: emp, fill: 'none', stroke: fx.softblue, 'stroke-width': 3 }, curves);
      fx.el('path', { d: truth, fill: 'none', stroke: fx.ink, 'stroke-width': 4 }, curves);
      // The analytical threshold, fixed, and the best one.
      fx.el('line', { x1: X(PSTAR), y1: HB, x2: X(PSTAR), y2: T, stroke: '#B8860B', 'stroke-width': 3, 'stroke-dasharray': '9 6' }, layer);
      fx.el('circle', { cx: X(PSTAR), cy: Y(R_ANALYTICAL), r: 8, fill: '#B8860B' }, layer);
      fx.tex(layer, X(PSTAR) + 10, T + 20, '\\text{analytical } p^*', { size: 26, anchor: 'start', color: '#8a6508' });
      fx.el('circle', { cx: X(TSTAR), cy: Y(risk(TSTAR)), r: 6, fill: fx.ink }, layer);
      fx.text(layer, X(TSTAR), Y(risk(TSTAR)) - 16, 'best threshold', { 'font-size': 23 });
      fx.el('line', { x1: X(one.tau), y1: HB, x2: X(one.tau), y2: T, stroke: fx.red, 'stroke-width': 3 }, layer);
      fx.el('circle', { cx: X(one.tau), cy: Y(risk(one.tau)), r: 8, fill: fx.red }, layer);
      fx.tex(layer, X(one.tau) + 10, B + 20, '\\tau_{emp}', { size: 28, anchor: 'start', color: fx.red });
      fx.el('line', { x1: LX, y1: T + 22, x2: LX + 54, y2: T + 22, stroke: fx.ink, 'stroke-width': 4 }, layer);
      fx.tex(layer, LX + 66, T + 22, 'R(\\tau)', { size: 26, anchor: 'start' });
      fx.el('line', { x1: LX, y1: T + 58, x2: LX + 54, y2: T + 58, stroke: fx.softblue, 'stroke-width': 3 }, layer);
      fx.tex(layer, LX + 66, T + 58, '\\hat{R}(\\tau)', { size: 26, anchor: 'start', color: fx.softblue });

      // ---- bottom: where the learned threshold lands over 200 samples ----
      var col = taus[j], NB = 50, counts = [], peak = 1, k;
      for (k = 0; k < NB; k++) counts.push(0);
      col.forEach(function (t) { counts[Math.min(NB - 1, Math.floor(t * NB))] += 1; });
      counts.forEach(function (c) { peak = Math.max(peak, c); });
      for (k = 0; k < NB; k++) {
        if (!counts[k]) continue;
        var h = counts[k] / peak * (HB - HT);
        fx.el('rect', { x: X(k / NB) + 1, y: HB - h, width: (R - L) / NB - 2, height: h, fill: fx.red, opacity: 0.55 }, layer);
      }
      fx.el('line', { x1: L, y1: HB, x2: R, y2: HB, stroke: fx.grey, 'stroke-width': 2 }, layer);
      [0, 0.25, 0.5, 0.75, 1].forEach(function (t) { fx.text(layer, X(t), HB + 30, String(t), { 'font-size': 26 }); });
      fx.tex(layer, X(0.875), HB + 26, '\\text{threshold } \\tau', { size: 28 });
      fx.text(layer, R - 8, HT + 4, 'repeated ' + STREAMS + ' times:', { 'font-size': 25, 'text-anchor': 'end', fill: fx.red });
      fx.tex(layer, R - 8, HT + 36, '\\text{where } \\tau_{emp} \\text{ lands}', { size: 26, anchor: 'end', color: fx.red });

      readout.innerHTML = 'm = ' + m + ': the empirical threshold beats the analytical one in <strong>' + beats(col) + '%</strong> of the samples; with 8 m, in <strong>' + beats(taus[nearest(8 * m)]) + '%</strong>';
    }
    draw();
  };
})();
