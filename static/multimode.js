
var sfs = -1;
var fronts = [];
var ranges = [];
var locations = [];
var viewOption = 1;
var zoomFactor = 1.0; //45000;
var vecA = [];
var vecB = [];
var vecC = [];
var vecD = [];
var vecEt = [];
var vecW = [];

function getInitMultyMode() {
    const sel = document.getElementById("incomingFront");
    let vf = [];
    let n = 512;
    console.log("Sel ", sel.value)
    switch (sel.value) {
        case "Gaussian Beam":
            z = n / 2 - 0.5;
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(1 * Math.exp(-(i - z) * (i - z) / 12500)))
            }
            break;
        case "Two Slit":
            z1 = 2 * n / 5
            z2 = 3 * n / 5
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(1 * (Math.exp(-(i - z1) * (i - z1) / 50) + Math.exp(-(i - z2) * (i - z2) / 50))))
            }
            break;
        case "Gaussian shift":
            z1 = 2 * n / 5
            for (let i = 0; i < n; i++) {
                vf.push(math.complex((Math.exp(-(i - z1) * (i - z1) / 150))))
            }
            break;    
        case "Mode He5":
            z = n / 2 - 0.5;
            let f = 50 / Math.sqrt(Math.sqrt(Math.PI) * 32 * 120);
            for (let i = 0; i < n; i++) {
                let x = (i - z) / 15;
                vf.push(math.complex(f * (x * x * x * x * x - 10 * x * x * x + 15 * x) * Math.exp(- x * x / 2)))
            }
            break;    
        case "Delta":
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(0))
            }
            vf[n / 2] = math.complex(1);
            break;
        case "Zero":
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(0))
            }
            break;    
    }

    // for (let i = 0; i < 20; i++) {
    //     vf[i] = math.complex(0);
    //     vf[n - 1 - i] = math.complex(0);
    // }

    console.log(vf);
    return vf;
}

function zoomMultiMode(z) {
    if (z > 0) {
        zoomFactor = zoomFactor * 1.5
    } else {
        zoomFactor = zoomFactor / 1.5
    }
    drawMultiMode();
}
const drawSx = 20;
const drawW = 3;
const drawMid = 400;

function drawMultiMode() {
    const canvas = document.getElementById("funCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `red`;
    ctx.fillRect(0, 0, 1000, 1000);

    for (let f = 0; f < fronts.length; f++) {
        let fi = fronts[f];
        let r = ranges[f];
        let l = fi.length;       
        let h = Math.abs(r) / l * zoomFactor * 45000;
        if (f == 0) {
            ctx.fillStyle = `orange`;
            ctx.fillRect(0, drawMid - l / 2, 1000, l);          
        }
        for (let i = 0; i < l; i++) {
            ii = r > 0.0 ? i : l - 1 - i;
            if (viewOption == 1) {
                c = Math.floor(fi[ii].toPolar().r * 255.0);
            } else {
                c = Math.floor((fi[ii].toPolar().phi / (2 * Math.PI) + 0.5) * 255.0);
            }
            ctx.fillStyle = `rgba(${c}, ${c}, ${c}, 255)`;
            ctx.fillRect(drawSx + f * drawW, (i - (l / 2)) * (h) + drawMid, drawW, h + 1);
        }
    }

    drawLenses();
    drawGraph();
}

function drawLenses() {
    const canvas = document.getElementById("funCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `yellow`;

    for (let iLens = 0; iLens < lenses.length; iLens++) {
        let px = drawSx + lenses[iLens][0] / distStep * drawW;
        ctx.fillRect(px, drawMid - 20 * zoomFactor, 2, 40 * zoomFactor);          
    }
}

function drawVector(v) {
    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `white`;
    ctx.fillRect(0, 0, 1000, 200);     
   
    let l = v.length;
    ctx.strokeStyle = `black`;
    ctx.beginPath();
    ctx.moveTo(drawSx, 100);
    ctx.lineTo(drawSx + l * drawW, 100);
    ctx.stroke(); 
    let fac = Math.max(Math.abs(Math.max(...v)), Math.abs(Math.min(...v)));
    console.log(`fac = ${fac}, M = ${Math.max(...v)}, m = ${Math.min(...v)}`)
    if (fac > 0) {
        fac = 90 / fac
    }
    ctx.strokeStyle = `red`;
    ctx.beginPath();
    ctx.moveTo(drawSx, 100 - Math.floor(fac * v[0]));
    for (let i = 1; i < l; i++) {
        ctx.lineTo(drawSx + i * drawW, 100 - Math.floor(fac * v[i]));
    }
    ctx.stroke();
}
function drawGraph() {
    const sel = document.getElementById("displayOption");

    let v = null;
    switch (sel.value) {
        case "A": v = vecA; break;
        case "B": v = vecB; break;
        case "C": v = vecC; break;
        case "D": v = vecD; break;
        case "E(x)":
            break;
        case "Width(x)": v = vecW; break;
    }

    if (v != null) {
        drawVector(v);
    }
}

function initMultiMode() {
    let iLens = 0;
    lenses = [];
    do {
        let lensDist = document.getElementById(`lens${iLens}dist`);
        let lensFocal = document.getElementById(`lens${iLens}focal`);
        if (lensDist == null || lensFocal == null) {
            console.log(`Error ${lensDist}, ${lensFocal}`)
            break;
        }
        valDist = parseFloat(lensDist.value);
        valFocal = parseFloat(lensFocal.value);
        console.log(`${lensDist.value} ${valDist}, ${lensFocal.value} ${valFocal}`)

        if (isNaN(valDist) || isNaN(valFocal)) {
            break;
        }
        lenses.push([valDist, valFocal]);
        console.log(lenses);
        iLens++;
    } while(true);

    fronts = [getInitMultyMode()];
    ranges = [0.001];
    sfs = 0;
    locations = [0];
    drawMultiMode();
}

/**
Discrete Fourier transform (DFT).
(the slowest possible implementation)
Assumes `inpReal` and `inpImag` arrays have the same size.
*/
function dft(inp, ss) {
    const out = [];
    const sin = [];
    const cos = [];
    let inpReal = [];
    let inpImag = [];
    let s = ss * 1;
  
    const N = inp.length;
    const twoPiByN = 2 * Math.PI / N;
  
    /* initialize Sin / Cos tables */
    for (let k = 0; k < N; k++) {
      inpReal.push(math.re(inp[k]));
      inpImag.push(math.im(inp[k]));
      const angle = twoPiByN * k;
      sin.push(Math.sin(angle));
      cos.push(Math.cos(angle));
    }
  
    for (let k = 0; k < N; k++) {
      let sumReal = 0;
      let sumImag = 0;
      let nn = 0;
      for (let n = 0; n < N; n++) {
        nm = (n + N / 2) % N;
        sumReal +=  inpReal[nm] * cos[nn] + inpImag[nm] * sin[nn];
        sumImag += -inpReal[nm] * sin[nn] + inpImag[nm] * cos[nn];
        nn = (nn + k) % N;
      }
      out.push(math.complex(sumReal * s, sumImag * s));
    }
    let o = [];
    for (let k = 0; k < N; k++) {
        o.push(out[(k + N / 2) % N]);
    }
    return o;
}

function propogateMultiMode() {
    if (fronts.length <= 0) {
        return;
    }
    let distS = 0.002;
    let lambda = 0.00000051;
    lfs = fronts.length;
    let dist = distS * (lfs - sfs);
    fi = math.clone(fronts[sfs]);
    let r = ranges[sfs];
    let L = fi.length;
    let dxi = r / L;
    let dxf = lambda * dist / r;
     let factor = math.divide(math.exp(math.complex(0, dist * Math.PI * 2 / lambda)), math.complex(dist));
    let ff = Math.sqrt(1 / (dist * lambda * 2));
    factor = math.complex(- ff, ff);
    let coi = Math.PI * dxi * dxi / (dist * lambda);
    console.log(`factor = ${factor}, lambda = ${lambda}`)

    let cof = Math.PI * dxf * dxf / (dist * lambda);
    console.log(`dxi = ${dxi}, dxf = ${dxf}, coi = ${coi}, cof = ${cof}, r = ${r}, dist = ${dist}`)

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        fi[i] = math.multiply(fi[i], math.exp(math.complex(0, coi * ii * ii)))
    }
    ff = dft(fi, dxi);

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        ff[i] = math.multiply(math.multiply(ff[i], factor), math.exp(math.complex(0, cof * ii * ii)))
    }

    fronts.push(ff);
    ranges.push(L * dxf);

    drawMultiMode();
}

function lensMultiMode() {
    if (fronts.length <= 0) {
        return;
    }

    fl = fronts.length
    ff = math.clone(fronts[fl - 1]);
    let r = ranges[fl - 1];
    let L = fi.length;
    let dx = r / L;

    let z = L / 2.0;
    for (let i = 0; i < L; i++) {
        let factor = (i - z) * (i - z) * dx * dx
        console.log(`i = ${i}, factor = ${factor}`);
        ff[i] = math.multiply(ff[i], math.exp(math.complex(- factor * 10000,  - factor * 100000000)))
    }

    sfs = fl;
    fronts.push(ff);
    ranges.push(ranges[sfs - 1]);
    console.log(`LENS range = ${ranges[sfs]}`)
    drawMultiMode();
}

function switchViewMultiMode() {
    viewOption = 1 - viewOption;
    drawMultiMode();
}

function MDist(d) {
    return [[1, d], [0, 1]];
}
function MLens(f) {
    return [[1, 0], [- 1/ f, 1]];
}
function MMult(m, m2) {
    a = m[0][0] * m2[0][0] + m[0][1] * m2[1][0];
    b = m[0][0] * m2[0][1] + m[0][1] * m2[1][1];
    c = m[1][0] * m2[0][0] + m[1][1] * m2[1][0];
    d = m[1][0] * m2[0][1] + m[1][1] * m2[1][1];

    return  [[a, b], [c, d]];
}

let lenses = [[0.1, 0.1], [0.25, 0.1]];
let distStep = 0.002;

function getMatOnStep(dStep) {

    let iLens = 0;
    let prevLensPos = 0.0;
    let M = [[1, 0], [0, 1]];
    let rdStep = dStep;
    while (iLens < lenses.length && rdStep > lenses[iLens][0] - prevLensPos) {
        M = MMult(MDist(lenses[iLens][0] - prevLensPos), M);
        M = MMult(MLens(lenses[iLens][1]), M);
        rdStep -= lenses[iLens][0] - prevLensPos;
        prevLensPos = lenses[iLens][0];
        iLens++;
    }
    M = MMult(MDist(rdStep), M);
    return M;
}

function calcWidth(v) {
    let l = v.length;
    let N = 0; sumX = 0; sumX2 = 0;
    for (let i = 0; i < l; i++) {
        let val = v[i].toPolar().r;
        N += val;
        sumX += i * val;
        sumX2 += i * i * val;
    }
    if (isNaN(N)) {
        return 0.0;
    }
    sumX /= N;
    sumX2 /= N;

    let w = Math.sqrt(sumX2 - sumX * sumX);

    console.log(`N = ${N} sumX = ${sumX} sumX2 = ${sumX2} w = ${w}`)

    return w;
}
function fullCavityMultiMode() {
    if (fronts.length <= 0) {
        return;
    }
    let lambda = 0.00000051;
    
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0];
    for (let iStep = 1; iStep < 300; iStep++) {
        let f0 = math.clone(fronts[0]);
        let r0 = ranges[0];
        let L = f0.length;
        let dx0 = r0/ L;
        let dStep = iStep * distStep;
        let M  = getMatOnStep(dStep);
        let A = M[0][0];
        let B = M[0][1];
        let C = M[1][0];
        let D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        console.log(`A = ${A}, B = ${B}, C = ${C}, D = ${D}`)
        let dxf = lambda * B / r0;
        let factor = math.sqrt(math.complex(0, - 1 / (B * lambda)));
        let co0 = Math.PI * dx0 * dx0 * A / (B * lambda);
        console.log(`factor = ${factor}, lambda = ${lambda}`)

        let cof = Math.PI * dxf * dxf * D / (B * lambda);
        console.log(`dx0 = ${dx0}, dxf = ${dxf}, co0 = ${co0}, cof = ${cof}, r0 = ${r0}, dStep = ${dStep}`)

        for (let i = 0; i < L; i++) {
            let ii = i - L / 2;
            f0[i] = math.multiply(f0[i], math.exp(math.complex(0, co0 * ii * ii)))
        }
        let ff = dft(f0, dx0);

        for (let i = 0; i < L; i++) {
            let ii = i - L / 2;
            ff[i] = math.multiply(math.multiply(ff[i], factor), math.exp(math.complex(0, cof * ii * ii)))
        }

        vecW.push(calcWidth(ff) * Math.abs(dxf));
        fronts.push(ff);
        ranges.push(L * dxf);
    }
    drawMultiMode();
}

function mainCanvasMouseMove(e) {
    console.log(e);
    const sel = document.getElementById("displayOption");
    if (sel.value == "E(x)") {
        var bounds = e.target.getBoundingClientRect();
        var x = e.clientX - bounds.left;
        var y = e.clientY - bounds.top;

        let ix = (x - drawSx) / drawW;
        if (ix >= 0 && ix < fronts.length) {
            fi = fronts[ix];
            fr = [];
            for (let i = 0; i < fi.length; i++) {
                fr.push(fi[i].toPolar().r);
            }
            drawVector(fr);
        }
    }

    console.log(x, y);
}