
var sfs = -1;
var fronts = [];
const lambda = 0.000000780;
var initialRange = 0.01047;
var ranges = [];
var nSamples = 256;
var locations = [];
var viewOption = 1;
var zoomFactor = 1.0; 45000;
var basicZoomFactor = 50000.0; // pixels per meter
var vecA = [];
var vecB = [];
var vecC = [];
var vecD = [];
var vecEt = [];
var vecW = [];
var vecWaist = [];
var vecQ = [];
var RayleighRange;
var graphData = [];
var isMouseDownOnGraph = false;
var mouseOnGraphStart = 0;
var mouseOnGraphEnd = 0;

function getInitMultyMode() {
    const sel = document.getElementById("incomingFront");
    const par = document.getElementById("beamParam");
    const rng = document.getElementById("initialRange");
    initialRange = parseFloat(rng.value);
    let vf = [];
    console.log("Sel ", sel.value)
    switch (sel.value) {
        case "Gaussian Beam":
            let waist = parseFloat(par.value);
            let dx = initialRange / nSamples;
            x0 = nSamples / 2 * dx;
            for (let i = 0; i < nSamples; i++) {
                px = i * dx;
                x = (px - x0) / waist;
                vf.push(math.complex(1 * Math.exp(- x * x)))
            }
            RayleighRange = Math.PI * waist * waist * 1.0 / lambda;

            break;
        case "Two Slit":
            z1 = 2 * nSamples / 5
            z2 = 3 * nSamples / 5
            for (let i = 0; i < nSamples; i++) {
                vf.push(math.complex(1 * (Math.exp(-(i - z1) * (i - z1) / 50) + Math.exp(-(i - z2) * (i - z2) / 50))))
            }
            break;
        case "Gaussian shift":
            z1 = 2 * nSamples / 5
            for (let i = 0; i < nSamples; i++) {
                vf.push(math.complex((Math.exp(-(i - z1) * (i - z1) / 150))))
            }
            break;    
        case "Mode He5":
            z = nSamples / 2 - 0.5;
            let f = 50 / Math.sqrt(Math.sqrt(Math.PI) * 32 * 120);
            for (let i = 0; i < nSamples; i++) {
                let x = (i - z) / 15;
                vf.push(math.complex(f * (x * x * x * x * x - 10 * x * x * x + 15 * x) * Math.exp(- x * x / 2)))
            }
            break;    
        case "Delta":
            for (let i = 0; i < nSamples; i++) {
                vf.push(math.complex(0))
            }
            vf[nSamples / 2] = math.complex(1);
            break;
        case "Zero":
            for (let i = 0; i < nSamples; i++) {
                vf.push(math.complex(0))
            }
            break;    
    }

    // for (let i = 0; i < 20; i++) {
    //     vf[i] = math.complex(0);
    //     vf[nSamples - 1 - i] = math.complex(0);
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
const drawSx = 50;
const drawW = 3;
const drawMid = 400;

function drawMultiMode() {
    const canvas = document.getElementById("funCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffb0e0";
    ctx.fillRect(0, 0, 1000, 1000);

    const canvasHeight = canvas.height;
    const heightM = (canvasHeight / 2) / (zoomFactor * basicZoomFactor);
    const powHeight = Math.log10(heightM);
    const markHeightFixed = Math.max(0, - Math.floor(powHeight));
    const markHeight = Math.pow(10, Math.floor(powHeight));
    const markHeightPixel = markHeight * (zoomFactor * basicZoomFactor);

    const canvasWidth = canvas.width;
    const widthM = (canvasWidth - 200) * (distStep / drawW);
    const powWidth = Math.log10(widthM);
    const markWidthFixed = Math.max(0, - Math.floor(powWidth));
    const markWidth = Math.pow(10, Math.floor(powWidth));
    const markWidthPixel = markWidth / (distStep / drawW);

    for (let f = 0; f < fronts.length; f++) {
        let fi = fronts[f];
        let r = ranges[f];
        let l = fi.length;       
        let h = Math.abs(r) / l * (zoomFactor * basicZoomFactor);
        if (f == 0) {
            ctx.fillStyle = `#ffddaa`;
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
    drawElements();

    ctx.beginPath();
    ctx.strokeStyle = "white";
    ctx.globalCompositeOperation = "difference";
    ctx.fillStyle = "white";
    //ctx.font = "bold 10pt Courier";

    for (it = 0; it * markHeightPixel < canvasHeight / 2; it++) {
        let t = canvasHeight / 2 - it * markHeightPixel;
        ctx.moveTo(0, t);
        ctx.lineTo(8, t);
        t = canvasHeight / 2 + it * markHeightPixel;
        ctx.moveTo(0,  t);
        ctx.lineTo(8, t);
    }
    ctx.stroke();
    for (it = 0; it * markHeightPixel < canvasHeight / 2; it++) {
        t = canvasHeight / 2 - it * markHeightPixel;
        ctx.fillText((it * markHeight).toFixed(markHeightFixed), 10, t + 4);
        t = canvasHeight / 2 + it * markHeightPixel;
        ctx.fillText((- it * markHeight).toFixed(markHeightFixed), 10, t + 4);
    }

    ctx.beginPath();
    for (it = 0; it * markWidthPixel < canvasWidth; it++) {
        let t = it * markWidthPixel + drawSx;
        ctx.moveTo(t, canvasHeight);
        ctx.lineTo(t,canvasHeight - 10);
    }
    ctx.stroke();
    for (it = 0; it * markWidthPixel < canvasWidth; it++) {
        let t = it * markWidthPixel + drawSx;
        ctx.fillText((it * markWidth).toFixed(markWidthFixed), t, canvasHeight - 20);
    }
    ctx.globalCompositeOperation = "source-over";

    drawGraph();

}

function drawElements() {
    const canvas = document.getElementById("funCanvas");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `yellow`;
    for (let iEl = 0; iEl < elements.length; iEl++) {
        switch (elements[iEl].t) {
            case "L":
                ctx.fillStyle = `yellow`;
                break;
            case "X":
                ctx.fillStyle = `blue`;
                break;
    
        }
        let px = drawSx + elements[iEl].par[0] / distStep * drawW;
        ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
    }
}

function vecDeriv(v, dx = 1) {
    let vd = math.clone(v);
    vd[0] = 0;
    for (let i = 1; i < v.length; i++) {
        vd[i] = (v[i] - v[i - 1]) / dx;
    }
    return vd;
}

function vecWaistFromQ(v) {
    let vw = math.clone(v);
    for (let i = 0; i < v.length; i++) {
        console.log(`i = ${i}`);
        console.log(v[i]);
        console.log(math.divide(1, v[i]));
        console.log(math.divide(1, v[i]).im);
        vw[i] = Math.sqrt(- lambda / (Math.PI * (math.divide(1, v[i]).im)));
        console.log(- lambda / (Math.PI * (math.divide(1, v[i]).im)));
        console.log(math.divide(1, v[i]).im);

    }
    return vw;
}

function drawVector(v, clear = true, color = "red") {
    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    let l = v.length;
    if (clear) {
        ctx.fillStyle = `white`;
        ctx.fillRect(0, 0, 1000, 200);     
        if (isMouseDownOnGraph) {
            const canvas = document.getElementById("graphCanvas");
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ddd";
            ctx.fillRect(drawSx + mouseOnGraphStart * drawW, 0, (mouseOnGraphEnd - mouseOnGraphStart) * drawW, 200)
        }
        ctx.strokeStyle = `black`;
        ctx.beginPath();
        ctx.moveTo(drawSx, 100);
        ctx.lineTo(drawSx + l * drawW, 100);
        ctx.stroke();
    }

    let fac = Math.max(Math.abs(Math.max(...v)), Math.abs(Math.min(...v)));
    if (fac > 0) {
        fac = 90 / fac
    }
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(drawSx, 100 - Math.floor(fac * v[0]));
    for (let i = 1; i < l; i++) {
        ctx.lineTo(drawSx + i * drawW, 100 - Math.floor(fac * v[i]));
    }
    ctx.stroke();
}
function drawGraph() {
    const sel = document.getElementById("displayOption");

    graphData = [];
    switch (sel.value) {
        case "A": graphData.push(vecA); break;
        case "B": graphData.push(vecB); break;
        case "C": graphData.push(vecC); break;
        case "D": graphData.push(vecD); break;
        case "E(x)":
            break;
        case "Width(x)": 
            graphData.push(vecW);
            graphData.push(vecDeriv(vecW, distStep));
            break
        case "Waist(x)": 
            graphData.push(vecWaist);
            graphData.push(vecDeriv(vecWaist, distStep));
            break;
        case "QWaist(x)": 
            graphData.push(vecWaistFromQ(vecQ));
            break;
    }

    if (graphData.length > 0) {
        drawVector(graphData[0]);
        if (graphData.length > 1) {
            drawVector(graphData[1], false, "purple");
        }
    }
}

function initMultiMode() {
    let iEl = 0;
    elements = [];
    do {
        let elementTypeControl = document.getElementById(`type${iEl}`);
        if (elementTypeControl == null) {
            break;
        }
        let elementType = elementTypeControl.innerHTML[0];
        let lensDist = document.getElementById(`el${iEl}dist`);
        if (lensDist == null) {
            console.log(`Error ${lensDist}`)
            break;
        }
        valDist = parseFloat(lensDist.value);
        if (elementType == "L") {
            let lensFocal = document.getElementById(`el${iEl}focal`);
            if (lensFocal == null) {
                console.log(`Error ${lensFocal}`)
                break;
            }
            valFocal = parseFloat(lensFocal.value);
            console.log(`${lensDist.value} ${valDist}, ${lensFocal.value} ${valFocal}`)
            if (isNaN(valDist) || isNaN(valFocal)) {
                break;
            }
        } else {
            valFocal = -1;
        }


        elements.push({t: elementType, par: [valDist, valFocal]});
        console.log(elements);
        iEl++;
    } while(true);

    fronts = [getInitMultyMode()];
    ranges = [initialRange];
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
      for (let iN = 0; iN < N; iN++) {
        nm = (iN + N / 2) % N;
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

let elements = [];
let distStep = 0.003;

function getMatOnStep(dStep) {

    let iEl = 0;
    let prevLensPos = 0.0;
    let M = [[1, 0], [0, 1]];
    let rdStep = dStep;
    while (iEl < elements.length && rdStep > elements[iEl].par[0] - prevLensPos) {
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(elements[iEl].par[0] - prevLensPos), M);
            M = MMult(MLens(elements[iEl].par[1]), M);
            rdStep -= elements[iEl].par[0] - prevLensPos;
            prevLensPos = elements[iEl].par[0];
        }
        iEl++;
    }
    M = MMult(MDist(rdStep), M);
    return M;
}

function getMatOnRoundTrip(c) {
    let iEl = 0;
    let prevLensPos = 0.0;
    let M = [[1, 0], [0, 1]];
    while (iEl < elements.length) {
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(elements[iEl].par[0] - prevLensPos), M);
            M = MMult(MLens(elements[iEl].par[1]), M);
            prevLensPos = elements[iEl].par[0];
            break;
        case "X": // end wall
            M = MMult(MDist(elements[iEl].par[0] - prevLensPos), M);
            prevLensPos = elements[iEl].par[0];
            break;
        }
        if (elements[iEl].t == "X") {
            iEl--;
            break;
        }
        iEl++;
    }
    while (iEl >= 0) {
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(prevLensPos - elements[iEl].par[0]), M);
            M = MMult(MLens(elements[iEl].par[1]), M);
            prevLensPos = elements[iEl].par[0];
            break;
        }
        iEl--;
    }    
    M = MMult(MDist(prevLensPos), M);

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
    
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0], vecQ[0] = math.complex(0, RayleighRange);
    for (let iStep = 1; iStep < 300; iStep++) {
        let f0 = math.clone(fronts[0]);
        let r0 = ranges[0];
        let L = f0.length;
        let dx0 = r0/ L;
        let dStep = iStep * distStep;
        let M  = getMatOnStep(dStep);
        let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        console.log(`A = ${A}, B = ${B}, C = ${C}, D = ${D}`);
        let newQ = math.chain(vecQ[0]).multiply(A).add(B).divide(math.chain(vecQ[0]).multiply(C).add(D).done()).done();
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

        vecQ.push(newQ);
        vecW.push(calcWidth(ff) * Math.abs(dxf));
        vecWaist.push(calcWidth(ff) * Math.abs(dxf) * 1.41421356237);
        fronts.push(ff);
        ranges.push(L * dxf);
    }
    drawMultiMode();
}

function roundtripMultiMode() {
    if (fronts.length <= 0) {
        return;
    }
    let M  = getMatOnRoundTrip(1);
    let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
    let v = (A + D) / 2;
    console.log(`A0 = ${A}, B0 = ${B}, C0 = ${C}, D0 = ${D}, v = ${v}`)
    if (v * v <= 1) {
        disc = Math.sqrt(1 - v * v);
        l1 = v + disc;
        l2 = v - disc;
        qi = B / disc;
        w = qi * lambda / Math.PI;
        console.log(`disc = ${disc}, w = ${w}, l1 = ${l1}, l2 = ${l2}`)
    }
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0], vecQ[0] = math.complex(0, RayleighRange);
    for (let iStep = 1; iStep < 300; iStep++) {
        let f0 = math.clone(fronts[iStep - 1]);
        let r0 = ranges[iStep - 1];
        let L = f0.length;
        let dx0 = r0 / L;
        let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        console.log(`A = ${A}, B = ${B}, C = ${C}, D = ${D}`)
        let dxf = lambda * B / r0;
        let factor = math.sqrt(math.complex(0, - 1 / (B * lambda)));
        console.log(`factor = ${factor}, lambda = ${lambda}`)
        console.log(`dxf = lambda * B / r0   ==> ${dxf} = ${lambda} * ${B} / ${r0}`);

        let co0 = Math.PI * dx0 * dx0 * A / (B * lambda);
        let cof = Math.PI * dxf * dxf * D / (B * lambda);
        console.log(`dx0 = ${dx0}, dxf = ${dxf}, co0 = ${co0}, cof = ${cof}, r0 = ${r0}`)

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
        vecWaist.push(calcWidth(ff) * Math.abs(dxf) * 1.41421356237);

        fronts.push(ff);
        ranges.push(L * dxf);
    }
    drawMultiMode();
}

function mainCanvasMouseMove(e) {
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

}

function graphCanvasMouseMove(e) {
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;

    let ix = Math.floor((x - drawSx) / drawW);

    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    if (isMouseDownOnGraph) {
        mouseOnGraphEnd = ix;
        drawGraph();
        if (graphData.length >= 1 &&
            mouseOnGraphEnd > 0 && mouseOnGraphEnd < graphData[0].length &&
            mouseOnGraphStart > 0 && mouseOnGraphStart < graphData[0].length &&
            mouseOnGraphStart < mouseOnGraphEnd) {
            let count = mouseOnGraphEnd - mouseOnGraphStart + 1
            let part = graphData[0].slice(mouseOnGraphStart, mouseOnGraphEnd + 1);
            let logPart  = math.map(part, Math.log)
            let partMax = Math.max(...part);
            let partMin = Math.min(...part);
            drawTextBG(ctx, `Count ${count}`, 160, 120 + 0 * 16, "orange");
            drawTextBG(ctx, `Max ${partMax}`, 160, 120 + 1 * 16, "orange");
            drawTextBG(ctx, `Min ${partMin}`, 160, 120 + 2 * 16, "orange");
            drawTextBG(ctx, `Avg ${part.reduce((a, b) => a + b, 0) / count} [${0.5 * (partMax + partMin)}]`, 160, 120 + 3 * 16, "orange");
            drawTextBG(ctx, `AvgGeo ${Math.exp(logPart.reduce((a, b) => a + b, 0) / count)} [${Math.sqrt(partMax * partMin)}]`, 160, 120 + 4 * 16, "orange");
        }
    }

    if (ix >= 0) {
        ctx.fillStyle = "black";
        
        drawTextBG(ctx, (ix * distStep).toFixed(6), 20, 120);
        for (let iVec = 0; iVec < graphData.length; iVec++) {
            if (graphData[iVec].length > ix) {
                let val = graphData[iVec][ix];
                drawTextBG(ctx, val.toFixed(8), 20, 120 + (iVec + 1) * 16, iVec == 0 ? "red" : "purple");
            }
        }
    }

}

function graphCanvasMouseDown(e) {
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;

    let ix = Math.floor((x - drawSx) / drawW);
    isMouseDownOnGraph = true;
    mouseOnGraphStart = ix;

}

function graphCanvasMouseUp(e) {
    console.log("up ", e);
    isMouseDownOnGraph = false;
}

function drawTextBG(ctx, txt, x, y, color = '#000', font = "10pt Courier") {
    ctx.save();
    ctx.font = font;
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#fff';
    var width = ctx.measureText(txt).width;
    ctx.fillRect(x, y, width, parseInt(font, 10));
    ctx.fillStyle = color;
    ctx.fillText(txt, x, y);
    ctx.restore();
}