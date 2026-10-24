// What the dominant label constraint (DLC) allows (K = 3). With mu_1 and
// mu_2 fixed, the shaded area is the set of positions of mu_0 in the
// simplex for which the configuration satisfies the DLC: mu_0 must be
// nearer to e_0 than mu_1 and mu_2 are, and no nearer to e_1 (to e_2) than
// mu_1 (mu_2) is. In the simplex these are a disc around e_0 minus discs
// around e_1 and e_2. All three prototypes can be dragged; if mu_1 and
// mu_2 already break the constraint between themselves, nothing is shaded.
(function () {
  'use strict';
  var fx = Deck.fx;

  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]); }
  var count = 0;   // the figure appears twice; each copy needs its own clip and mask ids

  Deck.figures.dlc_region = function (host, state) {
    var E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    var uid = ++count;
    // state.plain: a readout that does not name the Voronoi classes, for the
    // main slide where the DLC is defined before them.
    var plain = !!state.plain;
    var mus = [[.7, .2, .1], [.35, .45, .2], [.2, .25, .55]];
    if (state.preset === 'outside') mus[0] = [.3, .5, .2];
    // The plain copy sits in a column of the main slide: smaller, readout below.
    fx.flex(host, plain ? 'column' : 'row', plain ? 4 : 60);
    var W = plain ? 560 : 660, H = plain ? 470 : 540;
    var svg = fx.svg(fx.html('div', '', host), W, H);
    var S = fx.simplex(W / 2, 40, plain ? 470 : 540);
    var defs = fx.el('defs', {}, svg);
    var tri = fx.el('clipPath', { id: 'dlc-tri-' + uid }, defs);
    fx.el('path', { d: S.path(E) }, tri);
    var mask = fx.el('mask', { id: 'dlc-mask-' + uid, maskUnits: 'userSpaceOnUse', x: 0, y: 0, width: W, height: H }, defs);
    var layer = fx.el('g', {}, svg);
    S.labels(svg);
    var handles = fx.el('g', {}, svg);
    var side = fx.html('div', '', host);
    side.style.width = plain ? 'auto' : '560px';
    var readout = fx.html('div', 'readout', side);
    readout.style.textAlign = 'left';
    readout.style.lineHeight = '1.45';
    // Distances in the simplex and on screen differ by the factor side / sqrt 2.
    var px = S.side / Math.SQRT2;

    var dots = mus.map(function (_, k) {
      var g = fx.el('g', {}, handles);
      fx.el('circle', { r: 26, fill: 'transparent' }, g);
      fx.el('circle', { r: 13, fill: fx.cls[k], stroke: '#fff', 'stroke-width': 3 }, g);
      fx.drag(g, svg, function (x, y) { mus[k] = S.clamp(S.bary(x, y)); draw(); });
      return g;
    });

    function draw() {
      fx.clear(layer);
      fx.clear(mask);
      fx.drawRegions(layer, S, fx.regions(fx.voronoi(mus)), { fill: ['#EEF8EE', '#EEEEFF', '#FFEEEE'] });
      // mu_1 and mu_2 must respect the constraint between themselves.
      var pairOk = dist(E[1], mus[1]) < dist(E[1], mus[2]) && dist(E[2], mus[2]) < dist(E[2], mus[1]);
      var r0 = Math.min(dist(E[0], mus[1]), dist(E[0], mus[2]));
      if (pairOk) {
        var c0 = S.xy(E[0]), c1 = S.xy(E[1]), c2 = S.xy(E[2]);
        fx.el('circle', { cx: c0[0], cy: c0[1], r: r0 * px, fill: '#fff' }, mask);
        fx.el('circle', { cx: c1[0], cy: c1[1], r: dist(E[1], mus[1]) * px, fill: '#000' }, mask);
        fx.el('circle', { cx: c2[0], cy: c2[1], r: dist(E[2], mus[2]) * px, fill: '#000' }, mask);
        var g = fx.el('g', { 'clip-path': 'url(#dlc-tri-' + uid + ')' }, layer);
        fx.el('rect', { x: 0, y: 0, width: W, height: H, fill: fx.yellow, opacity: 0.55, mask: 'url(#dlc-mask-' + uid + ')' }, g);
      }
      S.outline(layer);
      dots.forEach(function (gk, k) {
        var q = S.xy(mus[k]);
        gk.setAttribute('transform', 'translate(' + q[0] + ' ' + q[1] + ')');
        fx.tex(layer, q[0] + 30, q[1] - 22, '\\boldsymbol{\\mu}_' + k, { size: 24, color: fx.cls[k] });
      });
      var ok = fx.dlc(mus).every(Boolean);
      fx.math(readout, '<span class="swatch" style="height:18px;width:26px;background:' + fx.yellow + ';opacity:.7"></span>positions of $\\boldsymbol{\\mu}_0$ allowed by the DLC,<br>given $\\boldsymbol{\\mu}_1$ and $\\boldsymbol{\\mu}_2$<br>' +
        (plain ? (ok ? 'this configuration satisfies the DLC' : '<span class="bad">DLC violated</span>: a vertex is nearer to another prototype')
          : (ok ? 'this configuration is in $\\mathcal{H}_{Vor}^{\\mathrm{DLC}}$' : '<span class="bad">DLC violated</span>: in $\\mathcal{H}_{Vor}^{\\Delta}$, not in $\\mathcal{H}_{Vor}^{\\mathrm{DLC}}$')) +
        '<br>drag any prototype');
    }
    draw();
  };
})();
