// The two threshold-based rules of "Four methods, two classes", under 0/1
// loss (K = 3), with one threshold per class as a slider.
//   Multidimensional thresholding  argmax_k (p_k - tau_k): the boundary
//     between i and j is p_i - p_j = tau_i - tau_j, parallel to the argmax
//     boundary. It is the parallel shift with alpha_k = tau_k.
//   Probability-ratio thresholding argmax_k p_k / tau_k: the boundary
//     between i and j is p_i / tau_i = p_j / tau_j, a line through the
//     vertex of the third class. It is linear scaling with beta_k = 1 / tau_k.
// The argmax boundaries are dashed. Adding a constant to every threshold of
// the first rule, or multiplying every threshold of the second by one, does
// not change the partition: each rule has K - 1 free parameters.
(function () {
  'use strict';
  var fx = Deck.fx;
  var ARGMAX = fx.regions([0, 1, 2].map(function (k) { return function (p) { return -p[k]; }; }));

  function panel(host, title, effect, scoresOf, tau, range) {
    var box = fx.html('div', '', host);
    box.style.width = '640px';
    fx.math(fx.html('div', 'readout', box), title).style.marginTop = '0';
    var W = 520, H = 330;
    var svg = fx.svg(box, W, H);
    var S = fx.simplex(W / 2, 40, 318);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 24);
    var bar = fx.html('div', 'ctl', box);
    bar.style.gap = '2px 22px';
    var draw = function () {
      fx.clear(layer);
      var regions = fx.regions(scoresOf(tau));
      fx.drawRegions(layer, S, regions);
      ARGMAX.forEach(function (poly) {
        fx.el('path', { d: S.path(poly), fill: 'none', stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: 0.55 }, layer);
      });
      S.outline(layer);
      regions.forEach(function (poly, k) {
        if (fx.area(poly) < 0.02) return;
        var c = S.xy(fx.centroid(poly));
        fx.tex(layer, c[0], c[1], '\\mathcal{P}_' + k, { color: fx.cls[k], size: 26 });
      });
    };
    tau.forEach(function (_, k) {
      fx.slider(bar, { label: 'τ<sub>' + k + '</sub>', min: range[0], max: range[1], step: 0.01, value: tau[k], width: 96 }, function (v) { tau[k] = v; draw(); });
    });
    fx.math(fx.html('div', 'readout', box), effect);
    draw();
  }

  Deck.figures.threshold_rules = function (host, state) {
    var shift = [0, 0.14, -0.12], ratio = [1, 1.5, 0.7];
    if (state.preset === 'argmax') { shift = [0, 0, 0]; ratio = [1, 1, 1]; }
    if (state.preset === 'strong') { shift = [-0.2, 0.25, 0.05]; ratio = [0.5, 1.8, 1]; }
    fx.flex(host, 'row', 60);
    host.style.alignItems = 'flex-start';
    panel(host, 'multidimensional thresholding: $\\arg\\max_k \\, (p_k - \\tau_k)$',
      'in $\\mathcal{H}_{PS}$: the boundaries translate',
      function (t) { return [0, 1, 2].map(function (k) { return function (p) { return -(p[k] - t[k]); }; }); },
      shift, [-0.4, 0.4]);
    panel(host, 'probability-ratio thresholding: $\\arg\\max_k \\, p_k / \\tau_k$',
      'in $\\mathcal{H}_{Sc}$: each boundary turns about a vertex',
      function (t) { return [0, 1, 2].map(function (k) { return function (p) { return -p[k] / t[k]; }; }); },
      ratio, [0.3, 2.5]);
  };
})();
