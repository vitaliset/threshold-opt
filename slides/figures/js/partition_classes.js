// The three hypothesis classes of "Learning the partition", one panel each,
// with their free parameters as sliders (K = 3). The decision rule is
// argmin_i a_i' p with a_i the column i of the partition matrix A:
//   H_PS    A_ij = C_ij + alpha_j, sum alpha = 0      K - 1 = 2 parameters
//   H_Sc    A_ij = C_ij beta_i, beta_0 = 1            K - 1 = 2 parameters
//   H_Full  A any 3 x 3 matrix                        K^2 = 9 parameters
// C is the matrix of the earlier example (C_20 = 5, C_21 = 3); its Bayes
// partition is dashed in every panel, and every panel starts at A = C.
(function () {
  'use strict';
  var fx = Deck.fx;
  var C = [[0, 1, 1], [1, 0, 1], [5, 3, 0]];

  function scoresOf(A, a) {
    return [0, 1, 2].map(function (i) {
      return function (p) { return A[0][i] * p[0] + A[1][i] * p[1] + A[2][i] * p[2] + a[i]; };
    });
  }
  var BASE = fx.regions(scoresOf(C, [0, 0, 0]));

  // One panel: title, simplex, a slot for the controls. Returns the redraw function.
  function panel(host, title, count, rule) {
    var box = fx.html('div', '', host);
    box.style.width = '480px';
    fx.math(fx.html('div', 'readout', box), title).style.marginTop = '0';
    var W = 340, H = 222;
    var svg = fx.svg(box, W, H);
    var S = fx.simplex(W / 2, 34, 204);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 22);
    var slot = fx.html('div', '', box);
    slot.style.minHeight = '132px';
    var cnt = fx.html('div', 'readout', box);
    cnt.innerHTML = count;
    cnt.style.fontSize = '22px';
    cnt.style.whiteSpace = 'nowrap';
    var draw = function () {
      fx.clear(layer);
      var r = rule(), regions = fx.regions(scoresOf(r.A, r.a));
      fx.drawRegions(layer, S, regions);
      BASE.forEach(function (poly) {
        if (poly.length > 2) fx.el('path', { d: S.path(poly), fill: 'none', stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: 0.55 }, layer);
      });
      S.outline(layer);
      regions.forEach(function (poly, k) {
        if (fx.area(poly) < 0.02) return;
        var c = S.xy(fx.centroid(poly));
        fx.tex(layer, c[0], c[1], '\\mathcal{P}_' + k, { color: fx.cls[k], size: 24 });
      });
    };
    return { slot: slot, draw: draw };
  }

  Deck.figures.partition_classes = function (host, state) {
    var alpha = [0, 0], logb = [0, 0], A = C.map(function (r) { return r.slice(); });
    if (state.preset === 'moved') { alpha = [0.5, -0.4]; logb = [1.2, -1]; A = [[0.5, 2, 1], [3, 0, 1.5], [2, 4, 0.5]]; }
    fx.flex(host, 'row', 16);
    host.style.alignItems = 'flex-start';

    // Parallel shift.
    var ps = panel(host, '$\\mathcal{H}_{PS}$: $A_{ij} = C_{ij} + \\alpha_j$', 'K − 1 = 2 parameters: boundaries translate', function () {
      return { A: C, a: [-(alpha[0] + alpha[1]), alpha[0], alpha[1]] };
    });
    var bar = fx.html('div', 'ctl col', ps.slot);
    bar.style.alignItems = 'center';
    [0, 1].forEach(function (k) {
      fx.slider(bar, { label: 'α<sub>' + (k + 1) + '</sub>', min: -1, max: 1, step: 0.05, value: alpha[k], width: 220 }, function (v) { alpha[k] = v; ps.draw(); });
    });

    // Linear scaling.
    var sc = panel(host, '$\\mathcal{H}_{Sc}$: $A_{ij} = C_{ij}\\,\\beta_i$', 'K − 1 = 2 parameters: boundaries rotate', function () {
      var b = [1, Math.pow(2, logb[0]), Math.pow(2, logb[1])];
      return { A: C.map(function (row, k) { return row.map(function (v) { return v * b[k]; }); }), a: [0, 0, 0] };
    });
    bar = fx.html('div', 'ctl col', sc.slot);
    bar.style.alignItems = 'center';
    [0, 1].forEach(function (k) {
      fx.slider(bar, { label: 'β<sub>' + (k + 1) + '</sub>', min: -3, max: 3, step: 0.1, value: logb[k], width: 220,
        format: function (v) { return fx.fmt(Math.pow(2, v), 2); } }, function (v) { logb[k] = v; sc.draw(); });
    });

    // Full matrix: one slider per entry, row = true class, column = prediction.
    var full = panel(host, '$\\mathcal{H}_{Full}$: any $\\mathbf{A} \\in \\mathbb{R}^{K \\times K}$', 'K² = 9 parameters: any convex polyhedral partition', function () {
      return { A: A, a: [0, 0, 0] };
    });
    // Labelled like the matrix C of "From a threshold to a partition": the
    // columns are the predictions, in the colours of their regions.
    var grid = fx.html('div', '', full.slot);
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'auto repeat(3, 128px)';
    grid.style.gap = '0 6px';
    grid.style.justifyContent = 'center';
    grid.style.alignItems = 'center';
    grid.style.marginTop = '6px';
    fx.html('div', '', grid);
    [0, 1, 2].forEach(function (j) {
      var head = fx.html('div', '', grid);
      head.textContent = 'predict ' + j;
      head.style.color = fx.cls[j];
      head.style.textAlign = 'center';
      head.style.fontSize = '22px';
    });
    var inputs = A.map(function (row, i) {
      var lab = fx.html('div', '', grid);
      lab.textContent = 'true ' + i;
      lab.style.color = '#555';
      lab.style.fontSize = '22px';
      return row.map(function (_, j) {
        return fx.slider(fx.html('div', '', grid), { label: '', min: 0, max: 6, step: 0.1, value: A[i][j], width: 70, digits: 1 }, function (v) { A[i][j] = v; full.draw(); });
      });
    });
    var seed = 1;
    var buttons = fx.html('div', 'ctl', full.slot);
    buttons.style.marginTop = '4px';
    fx.button(buttons, 'random A', function () {
      var rng = fx.rng(seed++ * 977);
      A = A.map(function (row) { return row.map(function () { return Math.round(60 * rng()) / 10; }); });
      inputs.forEach(function (row, i) { row.forEach(function (input, j) { input.set(A[i][j]); }); });
      full.draw();
    });
    fx.button(buttons, 'A = C', function () {
      A = C.map(function (r) { return r.slice(); });
      inputs.forEach(function (row, i) { row.forEach(function (input, j) { input.set(A[i][j]); }); });
      full.draw();
    });

    ps.draw(); sc.draw(); full.draw();
  };
})();
