// Geometry of the variance-scaling check: K Gaussian classes with means on
// a regular K-gon of fixed radius. The reference P_1 has standard deviation
// sigma (solid circles); P_lambda multiplies the variance by lambda, so its
// circles have radius sqrt(lambda) sigma (dashed). Proportions follow the
// static figure: radius 2, sigma 0.8; its dashed circles are lambda = 3.06.
(function () {
  'use strict';
  var fx = Deck.fx;

  Deck.figures.scaling_setup_geometry = function (host, state) {
    var K = 5, lambda = 3.06;
    if (state.preset === 'binary') K = 2;
    if (state.preset === 'many') { K = 20; lambda = 16; }
    fx.flex(host, 'column', 4);
    var W = 520, H = 440;
    var svg = fx.svg(host, W, H);
    var layer = fx.el('g', {}, svg);
    var bar = fx.html('div', 'ctl col', host);
    bar.style.gap = '2px';
    fx.slider(bar, { label: 'K', min: 2, max: 20, step: 1, value: K, digits: 0, width: 210 }, function (v) { K = v; draw(); });
    fx.slider(bar, { label: 'λ', min: 1, max: 16, step: 0.02, value: lambda, digits: 2, width: 210 }, function (v) { lambda = v; draw(); });

    function draw() {
      fx.clear(layer);
      // The largest dashed circle (lambda = 16) must fit: radius 2 + 4 * 0.8.
      var unit = (Math.min(W, H) / 2 - 6) / (2 + 4 * 0.8);
      var cx = W / 2, cy = H / 2;
      var means = [];
      for (var k = 0; k < K; k++) {
        var a = -Math.PI / 2 + 2 * Math.PI * k / K;
        means.push([cx + 2 * unit * Math.cos(a), cy + 2 * unit * Math.sin(a)]);
      }
      means.forEach(function (m) {
        fx.el('circle', { cx: m[0], cy: m[1], r: 0.8 * Math.sqrt(lambda) * unit, fill: '#D22B2B', 'fill-opacity': Math.min(0.07, 0.35 / K), stroke: '#A32020', 'stroke-width': 2, 'stroke-dasharray': '7 6' }, layer);
      });
      means.forEach(function (m, k) {
        fx.el('circle', { cx: m[0], cy: m[1], r: 0.8 * unit, fill: 'none', stroke: '#2C3FC7', 'stroke-width': 3 }, layer);
        fx.el('circle', { cx: m[0], cy: m[1], r: 4.5, fill: fx.ink }, layer);
        if (K <= 8) fx.tex(layer, m[0] + 20, m[1] - 16, '\\mathbf{m}_' + k, { size: 20 });
      });
    }
    draw();
  };
})();
