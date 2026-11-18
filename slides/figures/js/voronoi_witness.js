// The witness of Theorem 7.3.1 (K = 3, 0/1 loss): mass 1/3 on each of
// p_C = (0.8, 0.1, 0.1), p_A = (0.4, 0.3, 0.3), p_B = (0, 0.5, 0.5), with
// labels 0, 1, 2. A parallel-shift rule argmin_k (1 - p_k + alpha_k) cannot
// label p_A as 1 and p_B as 2, both on the altitude p_1 = p_2; the Voronoi
// prototypes of the slide classify all three. The risk is shown live.
(function () {
  'use strict';
  var fx = Deck.fx;
  var POINTS = [
    { p: [.8, .1, .1], y: 0, tex: '\\mathbf{p}_C', dx: 30, dy: -6 },
    { p: [.4, .3, .3], y: 1, tex: '\\mathbf{p}_A', dx: -44, dy: 0 },
    { p: [0, .5, .5], y: 2, tex: '\\mathbf{p}_B', dx: 0, dy: 30 }
  ];
  var WITNESS = [[1, 0, 0], [.4, .5, .1], [.1, .4, .5]];

  Deck.figures.voronoi_witness = function (host, state) {
    var mode = state.preset === 'shift' ? 'shift' : 'voronoi';
    var alpha = [0, 0];                       // alpha_1, alpha_2; alpha_0 = -(alpha_1 + alpha_2)
    var mus = WITNESS.map(function (m) { return m.slice(); });
    fx.flex(host, 'column', 6);
    var bar = fx.html('div', 'ctl', host);
    fx.choice(bar, [['shift', 'parallel shift'], ['voronoi', 'Voronoi']], mode, function (k) { mode = k; sync(); draw(); });
    var W = 560, H = 430;
    var svg = fx.svg(host, W, H);
    var S = fx.simplex(W / 2, 36, 420);
    var layer = fx.el('g', {}, svg);
    S.labels(svg, 24);
    var handles = fx.el('g', {}, svg);
    var sliders = fx.html('div', 'ctl', host);
    fx.slider(sliders, { label: 'α<sub>1</sub>', min: -0.6, max: 0.6, step: 0.02, value: 0, width: 150 }, function (v) { alpha[0] = v; draw(); });
    fx.slider(sliders, { label: 'α<sub>2</sub>', min: -0.6, max: 0.6, step: 0.02, value: 0, width: 150 }, function (v) { alpha[1] = v; draw(); });
    var readout = fx.html('div', 'readout', host);

    var dots = mus.map(function (_, k) {
      var g = fx.el('g', {}, handles);
      fx.el('circle', { r: 24, fill: 'transparent' }, g);
      fx.el('rect', { x: -10, y: -10, width: 20, height: 20, fill: fx.cls[k], stroke: '#fff', 'stroke-width': 3, transform: 'rotate(45)' }, g);
      fx.drag(g, svg, function (x, y) { mus[k] = S.clamp(S.bary(x, y)); draw(); });
      return g;
    });

    function sync() {
      handles.style.display = mode === 'voronoi' ? '' : 'none';
      sliders.style.visibility = mode === 'shift' ? 'visible' : 'hidden';
    }

    function scores() {
      if (mode === 'voronoi') return fx.voronoi(mus);
      var a = [-(alpha[0] + alpha[1]), alpha[0], alpha[1]];
      return [0, 1, 2].map(function (k) { return function (p) { return 1 - p[k] + a[k]; }; });
    }

    function draw() {
      fx.clear(layer);
      var sc = scores();
      fx.drawRegions(layer, S, fx.regions(sc));
      // The altitude p_1 = p_2, on which the three points lie.
      var a = S.xy([1, 0, 0]), b = S.xy([0, .5, .5]);
      fx.el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: fx.ink, 'stroke-width': 1.5, 'stroke-dasharray': '6 6', opacity: .5 }, layer);
      S.outline(layer);
      var wrong = 0;
      POINTS.forEach(function (pt) {
        var pred = fx.argmin(sc.map(function (s) { return s(pt.p); }));
        var ok = pred === pt.y;
        if (!ok) wrong += 1;
        var q = S.xy(pt.p);
        fx.el('circle', { cx: q[0], cy: q[1], r: 11, fill: fx.cls[pt.y], stroke: ok ? '#fff' : fx.ink, 'stroke-width': ok ? 3 : 4 }, layer);
        if (!ok) {
          fx.el('path', { d: 'M' + (q[0] - 17) + ' ' + (q[1] - 17) + ' L' + (q[0] + 17) + ' ' + (q[1] + 17) + ' M' + (q[0] - 17) + ' ' + (q[1] + 17) + ' L' + (q[0] + 17) + ' ' + (q[1] - 17), stroke: fx.ink, 'stroke-width': 4 }, layer);
        }
        fx.tex(layer, q[0] + pt.dx, q[1] + pt.dy, pt.tex, { size: 24, color: fx.cls[pt.y] });
      });
      dots.forEach(function (g, k) {
        var q = S.xy(mus[k]);
        g.setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')');
      });
      var risk = wrong === 0 ? '0' : wrong + '/3';
      var extra = '';
      if (mode === 'voronoi') {
        var ok = fx.dlc(mus);
        extra = ok.every(Boolean) ? ', DLC holds' : ', <span class="bad">DLC violated</span>';
      }
      readout.innerHTML = 'risk = <strong>' + risk + '</strong>' + extra;
    }
    sync(); draw();
  };
})();
