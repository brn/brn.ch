/*
 * Two Louvre busts rendered as a drifting, rotating particle cloud that morphs
 * from one into the next and loops through the Louvre scans in MODELS.
 *
 * The points are sampled from the scanned meshes offline (see assets/*.pcld -
 * 'PCLD', a uint32 count, int16 positions, then int8 normals). Carrying the
 * surface normal is what makes these solid: the light is evaluated per frame
 * against the rotated normal, so the form is genuinely lit rather than painted.
 *
 * Plain WebGL - no dependencies. Degrades silently to a black background.
 */
(function () {
  'use strict';

  var MODELS = [
    'assets/annibal.pcld',
    'assets/diane.pcld',
    'assets/cesar.pcld',
    'assets/enee.pcld'
  ];

  var FOV = 48 * Math.PI / 180;
  var PARTICLE_RADIUS = 0.0042; // model-space radius, model height == 1
  var HOLD = 5.5;               // seconds a bust stays settled
  var MORPH = 2.8;              // seconds spent crossing to the next one
  var DOLLY = 0.09;             // camera z swing, as a fraction of its distance
  // Both scans are modelled facing -Z, so the sway is centred half a turn round
  // and offset to sit near a three-quarter view rather than dead frontal.
  var YAW_CENTRE = Math.PI + 0.30;
  var YAW_SWING = 0.38;         // how far either side of that it sways
  var EXPOSURE = 1.15;          // additive exposure for the busts
  var SHED_FRACTION = 0.20;     // share of points caught up in the shedding
  var SHED_DIST = 0.62;         // how far a shed point drifts before it is gone
  var FOG_RADIUS = 3.4;         // fog cylinder radius, camera sits inside it
  var STRIDE = 8;               // per point: x, y, z, wander, nx, ny, nz, phase
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var canvas = document.getElementById('bg');
  if (!canvas) return;

  var gl = canvas.getContext('webgl', {
    alpha: false,
    antialias: false,
    depth: true,
    premultipliedAlpha: false,
    powerPreference: 'high-performance'
  });
  if (!gl) return;

  /* ---------------------------------------------------------------- shaders */

  var VERT = [
    'precision highp float;',
    'attribute vec4 a_pos;',   // position being left, plus its wander radius
    'attribute vec4 a_nrm;',   // surface normal, plus the drift phase
    'attribute vec4 b_pos;',   // position being entered
    'attribute vec4 b_nrm;',
    'uniform mat4 u_mvp;',
    'uniform mat3 u_normal;',  // the model rotation, for turning the normals
    'uniform vec3 u_light;',
    'uniform float u_time;',
    'uniform float u_mix;',    // 0 = fully a, 1 = fully b
    'uniform float u_burst;',  // outward scatter, peaks mid-morph
    'uniform float u_sizeK;',  // pxHeight / (2 * tan(fov/2))
    'uniform float u_near;',
    'uniform float u_far;',
    'uniform float u_drift;',
    'uniform float u_twinkle;', // 0 for the busts, 1 for the fog
    'varying float v_bright;',
    'varying float v_alpha;',
    'void main() {',
    '  float ph = a_nrm.w;',
    '  vec3 p = mix(a_pos.xyz, b_pos.xyz, u_mix);',
    '  p += u_burst * vec3(sin(ph * 1.7), cos(ph * 2.3), sin(ph * 3.1));',
    '',
    // The bust barely moves as a body - the turn and the dolly carry it. Each
    // grain instead wanders within its own small radius, so the surface
    // shimmers without the silhouette softening.
    '  vec3 q = p * 1.9;',
    '  p.x += u_drift * 0.006 * sin(u_time * 0.31 + q.y);',
    '  p.y += u_drift * 0.005 * cos(u_time * 0.27 + q.x);',
    '  float wr = mix(a_pos.w, b_pos.w, u_mix);',
    '  p += u_drift * wr * vec3(sin(u_time * 0.85 + ph * 1.7),',
    '    cos(u_time * 0.71 + ph * 2.3), sin(u_time * 0.58 + ph * 3.1));',
    '',
    // The fog is a body of mist in the same volume: it advects, swirls and rises.
    '  float fg = u_twinkle * u_drift;',
    '  p.x += fg * 0.075 * sin(u_time * 0.13 + q.z * 0.8 + ph * 0.3);',
    '  p.y += fg * 0.048 * cos(u_time * 0.11 + q.x * 0.7 + ph * 0.2);',
    '  p.z += fg * 0.085 * sin(u_time * 0.09 + q.y * 0.9 + ph * 0.25);',
    '',
    '',
    // Shedding. A share of the points is always in the middle of tearing loose:
    // each drifts off along its own surface normal, biased upward and swirled,
    // accelerating and fading as it goes, then starts over. The state is a pure
    // function of time and the point's phase, so nothing has to be simulated.
    '  float sel = fract(sin(ph * 37.13) * 1234.567);',
    '  float shedding = (1.0 - u_twinkle) * step(sel, SHED_FRACTION);',
    '  float life = fract(u_time * (0.028 + 0.042 * sel) + ph * 0.159);',
    '  vec3 away = normalize(a_nrm.xyz + vec3(0.0, 0.9, 0.0)',
    '    + 0.5 * vec3(sin(ph * 5.1), 0.0, cos(ph * 4.3)));',
    '  p += shedding * life * life * SHED_DIST * away;',
    '  float shedFade = 1.0 - shedding * life * life * (3.0 - 2.0 * life);',
    '',
    '  gl_Position = u_mvp * vec4(p, 1.0);',
    '  float depth = max(gl_Position.w, 0.001);',
    '  float size = 0.6 + 0.9 * fract(sin(ph * 91.7) * 4371.3);',
    '  gl_PointSize = clamp(u_sizeK * PARTICLE_RADIUS * size / depth, 0.8, 8.0);',
    '',
    // Lighting, against the normal as the bust turns. The rim term picks out
    // the silhouette; the facing term dims the far wall of what is really a
    // hollow shell of points, so the two surfaces do not read as one sheet.
    '  vec3 n = normalize(u_normal * mix(a_nrm.xyz, b_nrm.xyz, u_mix));',
    '  float lam = max(dot(n, u_light), 0.0);',
    '  float rim = pow(1.0 - abs(n.z), 3.0);',
    '  float facing = 0.55 + 0.45 * smoothstep(-0.25, 0.35, n.z);',
    '  float lit = (0.04 + 1.00 * lam * lam + 0.22 * rim) * facing;',
    '',
    '  float tw = mix(1.0, 0.45 + 0.55 * (0.5 + 0.5 * sin(u_time * 0.35 + ph * 2.1)), u_twinkle);',
    '  v_bright = mix(lit, a_nrm.x, u_twinkle) * tw * shedFade;',
    '  float f = clamp((u_far - depth) / (u_far - u_near), 0.0, 1.0);',
    // The fog also fades as it comes at the camera, so a mote passing close by
    // dissolves rather than flaring into a white blob.
    '  float nf = mix(1.0, smoothstep(0.12, 1.10, depth), u_twinkle);',
    '  v_alpha = mix(0.34, 1.0, f * f) * nf;',
    '}'
  ].join('\n')
    .replace(/PARTICLE_RADIUS/g, PARTICLE_RADIUS.toFixed(5))
    .replace(/SHED_FRACTION/g, SHED_FRACTION.toFixed(3))
    .replace(/SHED_DIST/g, SHED_DIST.toFixed(3));

  var FRAG = [
    'precision mediump float;',
    'varying float v_bright;',
    'varying float v_alpha;',
    'uniform float u_exposure;',
    'void main() {',
    '  float d = length(gl_PointCoord - vec2(0.5));',
    '  float a = smoothstep(0.5, 0.08, d);',
    '  gl_FragColor = vec4(vec3(v_bright), a * v_alpha * u_exposure);',
    '}'
  ].join('\n');

  function compile(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  var vs = compile(gl.VERTEX_SHADER, VERT);
  var fs = compile(gl.FRAGMENT_SHADER, FRAG);
  if (!vs || !fs) return;

  var prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return;
  gl.useProgram(prog);

  var loc = {
    aPos: gl.getAttribLocation(prog, 'a_pos'),
    aNrm: gl.getAttribLocation(prog, 'a_nrm'),
    bPos: gl.getAttribLocation(prog, 'b_pos'),
    bNrm: gl.getAttribLocation(prog, 'b_nrm'),
    mvp: gl.getUniformLocation(prog, 'u_mvp'),
    normal: gl.getUniformLocation(prog, 'u_normal'),
    light: gl.getUniformLocation(prog, 'u_light'),
    time: gl.getUniformLocation(prog, 'u_time'),
    mix: gl.getUniformLocation(prog, 'u_mix'),
    burst: gl.getUniformLocation(prog, 'u_burst'),
    sizeK: gl.getUniformLocation(prog, 'u_sizeK'),
    near: gl.getUniformLocation(prog, 'u_near'),
    far: gl.getUniformLocation(prog, 'u_far'),
    drift: gl.getUniformLocation(prog, 'u_drift'),
    twinkle: gl.getUniformLocation(prog, 'u_twinkle'),
    exposure: gl.getUniformLocation(prog, 'u_exposure')
  };

  /* ------------------------------------------------------------- mat4 utils */

  function mat4() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function multiply(out, a, b) {
    for (var c = 0; c < 4; c++) {
      var b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
      out[c * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
  }

  function perspective(out, fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0;
    return out;
  }

  function rotationY(out, r) {
    var s = Math.sin(r), c = Math.cos(r);
    out[0] = c; out[1] = 0; out[2] = -s; out[3] = 0;
    out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
    out[8] = s; out[9] = 0; out[10] = c; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
  }

  function rotationX(out, r) {
    var s = Math.sin(r), c = Math.cos(r);
    out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = c; out[6] = s; out[7] = 0;
    out[8] = 0; out[9] = -s; out[10] = c; out[11] = 0;
    out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
    return out;
  }

  // The rotation part of a mat4, for turning normals with the model.
  function mat3From(out, m) {
    out[0] = m[0]; out[1] = m[1]; out[2] = m[2];
    out[3] = m[4]; out[4] = m[5]; out[5] = m[6];
    out[6] = m[8]; out[7] = m[9]; out[8] = m[10];
    return out;
  }

  /* ------------------------------------------------------------ point cloud */

  function targetCount() {
    var area = window.innerWidth * window.innerHeight;
    if (area < 380000) return 30000;  // phones
    if (area < 900000) return 44000;  // tablets / small laptops
    return 62000;
  }

  // Unpack a .pcld into the interleaved buffer the shader wants. The offline
  // sampler wrote the points in random order, so a prefix is an even subsample.
  function decode(buffer, count) {
    var head = new DataView(buffer);
    if (String.fromCharCode(head.getUint8(0), head.getUint8(1),
        head.getUint8(2), head.getUint8(3)) !== 'PCLD') return null;

    var stored = head.getUint32(4, true);
    var n = Math.min(count, stored);
    var pos = new Int16Array(buffer, 8, stored * 3);
    var nrm = new Int8Array(buffer, 8 + stored * 6, stored * 3);

    var data = new Float32Array(n * STRIDE);
    var loX = 1e9, hiX = -1e9, loY = 1e9, hiY = -1e9, loZ = 1e9, hiZ = -1e9;
    for (var i = 0; i < n; i++) {
      var k = i * STRIDE, s = i * 3;
      var x = pos[s] / 32767, y = pos[s + 1] / 32767, z = pos[s + 2] / 32767;
      data[k] = x;
      data[k + 1] = y;
      data[k + 2] = z;
      // A point may roam a little over the surface it sits on. Keep it well
      // under the size of any feature, or the scan turns to fuzz.
      data[k + 3] = 0.0012 + Math.random() * 0.0026;
      data[k + 4] = nrm[s] / 127;
      data[k + 5] = nrm[s + 1] / 127;
      data[k + 6] = nrm[s + 2] / 127;
      data[k + 7] = Math.random() * 6.283;
      if (x < loX) loX = x; if (x > hiX) hiX = x;
      if (y < loY) loY = y; if (y > hiY) hiY = y;
      if (z < loZ) loZ = z; if (z > hiZ) hiZ = z;
    }

    // Centre on the bounding box, not the centroid: most of a bust's mass is in
    // the shoulders, so centring on the mean drops the head off the top.
    var mid = [(loX + hiX) / 2, (loY + hiY) / 2, (loZ + hiZ) / 2];
    var width = 0, height = hiY - loY;
    for (var j = 0; j < n; j++) {
      var b = j * STRIDE;
      data[b] = (data[b] - mid[0]) / height;
      data[b + 1] = (data[b + 1] - mid[1]) / height;
      data[b + 2] = (data[b + 2] - mid[2]) / height;
      var span = Math.max(Math.abs(data[b]), Math.abs(data[b + 2]));
      if (span > width) width = span;
    }

    // The widest silhouette a bust can turn into is its diagonal, so the fit
    // has to allow for depth as well as width.
    return { data: data, count: n, modelW: width * 2 };
  }

  // The morph pairs points by array index, so both clouds are sorted into the
  // same spatial order: top to bottom in bands, left to right within a band.
  // Points then slide short distances instead of scattering at random.
  function sortSpatially(cloud) {
    var n = cloud.count, src = cloud.data;
    var key = new Float32Array(n);
    var idx = new Array(n);
    for (var i = 0; i < n; i++) {
      idx[i] = i;
      key[i] = Math.floor((0.5 - src[i * STRIDE + 1]) * 120)
        + (src[i * STRIDE] + 0.6) / 1.2;
    }
    idx.sort(function (a, b) { return key[a] - key[b]; });

    var out = new Float32Array(n * STRIDE);
    for (var j = 0; j < n; j++) {
      var s = idx[j] * STRIDE, d = j * STRIDE;
      for (var f = 0; f < STRIDE; f++) out[d + f] = src[s + f];
    }
    cloud.data = out;
  }

  // Volumetric fog. The motes fill a cylinder around the camera rather than a
  // slab in front of it, so there is depth on every side and the field stays
  // evenly filled as it turns. Uniform-in-disc sampling keeps the density even
  // through the volume. The cylinder reaches past the camera, so motes pass
  // close by: those are the thickness cue.
  function buildAmbient() {
    var count = Math.round(targetCount() * 1.15);
    var data = new Float32Array(count * STRIDE);

    for (var i = 0; i < count; i++) {
      var k = i * STRIDE;
      var ang = Math.random() * 6.283;
      var rad = Math.sqrt(Math.random()) * FOG_RADIUS;
      data[k] = Math.cos(ang) * rad;
      data[k + 1] = (Math.random() - 0.5) * 3.2;
      data[k + 2] = Math.sin(ang) * rad;
      data[k + 3] = 0.02 + Math.random() * 0.05; // nothing to preserve out here
      // The fog is unlit; the shader reads its brightness straight out of .x.
      data[k + 4] = 0.06 + Math.pow(Math.random(), 2.2) * 0.70;
      data[k + 5] = 0;
      data[k + 6] = 1;
      data[k + 7] = Math.random() * 6.283;
    }
    return { data: data, count: count };
  }

  /* ------------------------------------------------------------------ render */

  function start(clouds, dust) {
    function upload(c) {
      var b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, c.data, gl.STATIC_DRAW);
      return b;
    }

    // Both busts draw the same number of points, so index i in one maps to
    // index i in the other.
    var shared = clouds[0].count;
    var maxModelW = 0;
    var buffers = [];
    for (var i = 0; i < clouds.length; i++) {
      shared = Math.min(shared, clouds[i].count);
      maxModelW = Math.max(maxModelW, clouds[i].modelW);
      buffers.push(upload(clouds[i]));
    }

    var dustBuf = upload(dust);

    gl.enableVertexAttribArray(loc.aPos);
    gl.enableVertexAttribArray(loc.aNrm);
    gl.enableVertexAttribArray(loc.bPos);
    gl.enableVertexAttribArray(loc.bNrm);

    function draw(from, to, count, m, burst, mvp, near, far, exposure, drift, twinkle) {
      gl.bindBuffer(gl.ARRAY_BUFFER, from);
      gl.vertexAttribPointer(loc.aPos, 4, gl.FLOAT, false, STRIDE * 4, 0);
      gl.vertexAttribPointer(loc.aNrm, 4, gl.FLOAT, false, STRIDE * 4, 16);
      gl.bindBuffer(gl.ARRAY_BUFFER, to);
      gl.vertexAttribPointer(loc.bPos, 4, gl.FLOAT, false, STRIDE * 4, 0);
      gl.vertexAttribPointer(loc.bNrm, 4, gl.FLOAT, false, STRIDE * 4, 16);
      gl.uniformMatrix4fv(loc.mvp, false, mvp);
      gl.uniform1f(loc.mix, m);
      gl.uniform1f(loc.burst, burst);
      gl.uniform1f(loc.near, near);
      gl.uniform1f(loc.far, far);
      gl.uniform1f(loc.exposure, exposure);
      gl.uniform1f(loc.drift, drift);
      gl.uniform1f(loc.twinkle, twinkle);
      gl.drawArrays(gl.POINTS, 0, count);
    }

    // A point cloud off a scan is a hollow shell: without depth testing the
    // far wall adds straight through the near one and the bust flattens into an
    // even haze. Depth-testing the bust throws the back surface away and leaves
    // a lit, solid front. The fog draws first and writes no depth, so it never
    // punches holes in what follows.
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // additive: overlapping points glow
    gl.clearColor(0, 0, 0, 1);
    gl.clearDepth(1);
    gl.uniform3f(loc.light, -0.46, 0.58, 0.67); // key light, upper left, front

    var proj = mat4(), view = mat4(), model = mat4(), tmp = mat4(), mvp = mat4();
    var dustModel = mat4(), dustMvp = mat4();
    var rotY = mat4(), rotX = mat4();
    var nrm3 = new Float32Array(9), dustNrm3 = new Float32Array(9);
    var camDist = 2;

    function resize() {
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      var cw = Math.round(window.innerWidth * dpr);
      var ch = Math.round(window.innerHeight * dpr);
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }
      gl.viewport(0, 0, cw, ch);

      // Pull the camera back far enough that the bust fits both axes.
      var aspect = cw / ch;
      var visH = 1 / 0.64;                                // fill 64% of height,
      // leaving room for the shed points to drift clear of the frame
      visH = Math.max(visH, maxModelW / (0.92 * aspect)); // ...and stay inside the width
      camDist = visH / (2 * Math.tan(FOV / 2));

      perspective(proj, FOV, aspect, 0.05, 20);
      gl.uniform1f(loc.sizeK, ch / (2 * Math.tan(FOV / 2)));
    }

    window.addEventListener('resize', resize);
    resize();

    var cycle = HOLD + MORPH;
    var total = cycle * clouds.length;
    var t = 0, last = 0;

    function frame(now) {
      requestAnimationFrame(frame);
      if (document.hidden) { last = now; return; }

      var dt = last ? Math.min((now - last) / 1000, 0.05) : 0;
      last = now;
      t += REDUCED ? dt * 0.08 : dt;

      // A very slow sway around the vertical axis, plus a gentle nod.
      rotationY(rotY, YAW_CENTRE + YAW_SWING * Math.sin(t * 0.04));
      rotationX(rotX, Math.sin(t * 0.019) * 0.075);
      multiply(model, rotX, rotY);
      // Bob vertically only, so the bust stays anchored at screen centre.
      model[13] = Math.sin(t * 0.23) * 0.028;

      // Breathe the camera along z, drifting closer and further (~105s period).
      var dist = camDist + Math.sin(t * 0.06) * DOLLY * camDist;
      view[14] = -dist;

      multiply(tmp, view, model);
      multiply(mvp, proj, tmp);
      mat3From(nrm3, model);

      // The fog turns on its own, so near motes overtake far ones.
      rotationY(dustModel, t * 0.020);
      multiply(tmp, view, dustModel);
      multiply(dustMvp, proj, tmp);
      mat3From(dustNrm3, dustModel);

      // Hold on one bust, then cross to the other; wrap round to the first.
      var phase = t % total;
      var index = Math.floor(phase / cycle);
      var local = phase - index * cycle;
      var m = local <= HOLD ? 0 : (local - HOLD) / MORPH;
      m = m * m * (3 - 2 * m);
      var burst = Math.sin(Math.PI * m) * 0.055;

      var drift = REDUCED ? 0.25 : 1;
      gl.uniform1f(loc.time, t);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.depthMask(false);
      gl.uniformMatrix3fv(loc.normal, false, dustNrm3);
      draw(dustBuf, dustBuf, dust.count, 0, 0,
        dustMvp, dist - 2.6, dist + 3.4, 0.48, drift * 3, 1);

      gl.depthMask(true);
      gl.uniformMatrix3fv(loc.normal, false, nrm3);
      draw(buffers[index], buffers[(index + 1) % buffers.length], shared, m, burst,
        mvp, dist - 0.75, dist + 0.75, EXPOSURE, drift, 0);
    }

    requestAnimationFrame(frame);
    canvas.classList.add('is-ready');
  }

  /* -------------------------------------------------------------- bootstrap */

  function load(url) {
    return fetch(url)
      .then(function (r) { return r.ok ? r.arrayBuffer() : null; })
      .catch(function () { return null; });
  }

  Promise.all(MODELS.map(load)).then(function (buffers) {
    var count = targetCount();
    var clouds = [];
    for (var i = 0; i < buffers.length; i++) {
      if (!buffers[i]) continue;
      var cloud = decode(buffers[i], count);
      if (!cloud || cloud.count === 0) continue;
      sortSpatially(cloud);
      clouds.push(cloud);
    }
    if (clouds.length) start(clouds, buildAmbient());
  });
})();
