
var workingTab = 1;
var sfs = -1;
var multiFronts = [[], []];
const lambda = 0.000000780;
var initialRange = 0.01047;
var multiRanges = [[], []];
var nSamples = 256;
var viewOption = 1;
var zoomFactor = 1.0;
var basicZoomFactor = 50000.0; // pixels per meter

var vecA = [];
var vecB = [];
var vecC = [];
var vecD = [];
var vecEt = [];
var vecW = [];
var vecWaist = [];
var vecQ = [];
var vecMats = [];
var RayleighRange;
var graphData = [];
var vecAperture = [];
var isMouseDownOnMain = false;
var isMouseDownOnGraph = false;
var mouseOnGraphStart = 0;
var mouseOnGraphEnd = 0;
var drawOption = true;
var drawMode = 1;
var deltaGraphX, deltaGraphY,  deltaGraphYHalf, deltaGraphYCalc;
var displayTemp = [[], []];
var displayTempPrevious = [[], []];


function getFieldFloat(id, defaultValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        const value = cont.value;
        const numVal = parseFloat(value);
        if (!isNaN(numVal)) {
            return numVal;
        }
    }

    return defaultValue;
}

function setFieldFloat(id, newValue) {
    const cont = document.getElementById(id);
    if (cont != null) {
        cont.value = `${newValue}`;
    }
}
var saveBeamParam, saveBeamDist;
function getInitFront(pPar = - 1) {
    const sel = document.getElementById("incomingFront");
    //const par = document.getElementById("beamParam");
    const rng = document.getElementById("initialRange");
    initialRange = parseFloat(rng.value);
    let vf = [];
    RayleighRange = 0.0;
    switch (sel.value) {
        case "Gaussian Beam":
            let waist = pPar > 0 ? pPar : getFieldFloat("beamParam", 0.0005);
            let beamDist = getFieldFloat("beamDist", 0.0);
            saveBeamParam = waist;
            saveBeamDist = beamDist;
            let dx = initialRange / nSamples;
            x0 = nSamples / 2 * dx;
            for (let i = 0; i < nSamples; i++) {
                let px = i * dx;
                let x = (px - x0);
                xw = x / waist;
                //vf.push(math.complex(1 * Math.exp(- xw * xw)))
                let theta = Math.abs(beamDist) < 0.000001 ? 0 : - x * x  / (Math.PI * lambda * beamDist);
                let fVal = math.exp(math.complex(- xw * xw, theta))
                vf.push(fVal);
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

    apertureChanged();

    return vf;
}

function apertureChanged() {
    vecAperture  = [];
    let aperture = getFieldFloat('apreture', 0.000056)
    let dx = initialRange / nSamples;
    x0 = nSamples / 2 * dx;
    for (let i = 0; i < nSamples; i++) {
        px = i * dx;
        x = (px - x0) / aperture;
        vecAperture.push(math.complex(1 * Math.exp(- x * x)))
    }
}

function nSamplesChanged() {
    const val = document.getElementById("nSamples").value;
    nSamples = parseInt(val);
    let step = 0.0003;
    initialRange =  math.sqrt(lambda * step * nSamples);
    setFieldFloat("initialRange", initialRange);
}


function zoomMultiMode(z) {
    if (z > 0) {
        zoomFactor = zoomFactor * 1.5
    } else {
        zoomFactor = zoomFactor / 1.5
    }
    drawMultiMode();
}
let drawSx = 50;
let drawW = 3;
let drawMid = 400;

function drowShenets(ctx, dType, valPerPixel, startVal = 0) {
    let canvasSize;
    let canvasHeight = ctx.canvas.height;

    switch (dType) {
        case "V":
            canvasSize = ctx.canvas.height;
            break;
        case "V0":
            canvasSize = ctx.canvas.height / 2;
            break;
        case "H":
            canvasSize = ctx.canvas.width - drawSx - 10;
            break;
    }
    const sizeM = canvasSize / valPerPixel + startVal;
    const powSize = Math.log10(sizeM);
    const markSizeFixed = Math.max(0, - Math.floor(powSize));
    const markSize = Math.pow(10, Math.floor(powSize));
    const markSizePixel = markSize * valPerPixel;

    ctx.beginPath();
    ctx.strokeStyle = "white";
    ctx.globalCompositeOperation = "difference";
    ctx.fillStyle = "white";

    if (dType == "V0") {
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            let t = canvasSize - it * markSizePixel;
            ctx.moveTo(0, t);
            ctx.lineTo(8, t);
            t = canvasSize + it * markSizePixel;
            ctx.moveTo(0,  t);
            ctx.lineTo(8, t);
        }
        ctx.stroke();
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            t = canvasSize - it * markSizePixel;
            ctx.fillText((it * markSize).toFixed(markSizeFixed), 10, t + 4);
            if (it > 0) {
                t = canvasSize + it * markSizePixel;
                ctx.fillText((- it * markSize).toFixed(markSizeFixed), 10, t + 4);
            }
        }
    }

    if (dType == "H") {
        ctx.beginPath();
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            let t = (it - startVal / markSize) * markSizePixel + drawSx;
            ctx.moveTo(t, canvasHeight);
            ctx.lineTo(t, canvasHeight - 10);
        }
        ctx.stroke();
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            let t = (it - startVal / markSize) * markSizePixel + drawSx;
            ctx.fillText((it * markSize + distStart).toFixed(markSizeFixed), t, canvasHeight - 20);
        }
    }

    if (dType == "V") {
        ctx.beginPath();
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            let t = canvasHeight - 1 - (it - startVal / markSize) * markSizePixel;
            ctx.moveTo(0, t);
            ctx.lineTo(8, t);
        }
        ctx.stroke();
        for (it = 0; it * markSizePixel < canvasSize; it++) {
            let t = canvasHeight - 1 - (it - startVal / markSize) * markSizePixel;
            ctx.fillText((it * markSize).toFixed(markSizeFixed), 10, t + 4);
        }
    }

    ctx.globalCompositeOperation = "source-over";
}

function drawDeltaGraph(ctx) {
    if (deltaGraphX.length < 5) {
        return;
    }
    const deltaMinX = Math.min(...deltaGraphX);
    const deltaMaxX = Math.max(...deltaGraphX);
    const deltaMinY = Math.min(...deltaGraphY, ...deltaGraphYHalf);
    const deltaMaxY = Math.max(...deltaGraphY, ...deltaGraphYHalf);

    const height = ctx.canvas.height;
    const gWidth = ctx.canvas.width - 200;
    const gHeight = ctx.canvas.height - 200;

    const zoomY = gHeight / (deltaMaxY - deltaMinY);
    const zoomX = gWidth / (deltaMaxX - deltaMinX);

    ctx.fillStyle = "red";
    for (let i = 0; i < deltaGraphX.length; i++) {
        ctx.fillRect((deltaGraphX[i] - deltaMinX) * zoomX + drawSx - 3, - (deltaGraphY[i] - deltaMinY) * zoomY + height - 4, 6, 6);
    }
    ctx.fillStyle = "blue";
    for (let i = 0; i < deltaGraphX.length; i++) {
        ctx.fillRect((deltaGraphX[i] - deltaMinX) * zoomX + drawSx - 3, - (deltaGraphYHalf[i] - deltaMinY) * zoomY + height - 4, 6, 6);
    }

    drowShenets(ctx, "V", zoomY, deltaMinY);
    drowShenets(ctx, "H", zoomX, deltaMinX);

}

function drawMultiMode() {
    if (!drawOption) {
        return;
    }
    if (workingTab == 3) {
        drawMultiTime();
        return;
    }
    const canvasList = document.querySelectorAll('[id^="funCanvas"]');
    canvasList.forEach((canvas, index) => {
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffb0e0";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawMid = canvas.height / 2;

        if (drawMode == 3) {
            drawDeltaGraph(ctx);
            return;
        }
        let fronts = multiFronts[index];
        let ranges = multiRanges[index];

        for (let f = 0; f < fronts.length; f++) {
            let fi = fronts[f];
            let r = ranges[f];
            let l = fi.length;       
            let h = Math.abs(r) / l * (zoomFactor * basicZoomFactor);
            if (f == 0 && drawMode <= 2) {
                ctx.fillStyle = `#ffddaa`;
                ctx.fillRect(0, drawMid - l / 2, canvas.width, l);          
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
        if (drawMode == 1) {
            drawElements(index + 1);
            displayTemp[index].forEach((el, i, a) => {
                drawTextBG(ctx, el.toFixed(7), canvas.width - 80, canvas.height - 28 - 16 * ((a.length - i)));
            });
            displayTempPrevious[index].forEach((el, i, a) => {
                drawTextBG(ctx, el.toFixed(7), canvas.width - 160, canvas.height - 28 - 16 * ((a.length - i)));
            });
        }

        drowShenets(ctx, "V0", zoomFactor * basicZoomFactor);
        switch (workingTab) {
            case 1:
                if (drawMode == 2) {
                    drowShenets(ctx, "H",  drawW);
                } else {
                    drowShenets(ctx, "H",  drawW / distStep);
                }
                break;
            case 2:
                drowShenets(ctx, "H",  drawW / distStep, 0, distStart);
                break;

        }
    });    

    drawGraph();
}

function drawElements(index) {
    if (!drawOption) {
        return
    }
    const canvas = document.getElementById(`funCanvas${index}`);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `yellow`;
    for (let iEl = 0; iEl < elements.length; iEl++) {
        switch (elements[iEl].t) {
            case "L":
                ctx.fillStyle = `yellow`;
                px = drawSx + (elements[iEl].par[0] - distStart) / distStep * drawW ;
                ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                break;
            case "C":
                ctx.fillStyle = `purple`;
                px = drawSx + (elements[iEl].par[0] - distStart) / distStep * drawW ;
                ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                px = drawSx + (elements[iEl].par[0] + elements[iEl].par[1] - distStart) / distStep * drawW ;
                ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);
                let spx = (elements[iEl].par[1] / 10);
                ctx.strokeStyle = `purple`;
                ctx.setLineDash([5, 3]);
                ctx.beginPath();
                for (let iL = 0; iL < 5; iL++) {
                    px = drawSx + (elements[iEl].par[0] + (1 + 2 * iL) * spx - distStart) / distStep * drawW ;
                    ctx.moveTo(px, drawMid - 80 * zoomFactor);
                    ctx.lineTo(px, drawMid + 80 * zoomFactor);
                }
                ctx.stroke();
                break;
            case "X":
                ctx.fillStyle = `blue`;
                px = drawSx + (elements[iEl].par[0] - distStart) / distStep * drawW ;
                ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                break;
                
        }
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

var drawVectorComparePrevious = [];

function drawVector(v, clear = true, color = "red", pixelWidth = drawW) {
    if (!drawOption) {
        return
    }
    const prevCompare = document.getElementById('cbxPrevCompare').checked;

    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    let l = v.length;
    if (pixelWidth > canvas.width / l) {
        pixelWidth = canvas.width / l;
    }
    if (clear) {
        ctx.fillStyle = `white`;
        ctx.fillRect(0, 0, 1000, 200);     
        if (isMouseDownOnGraph) {
            const canvas = document.getElementById("graphCanvas");
            const ctx = canvas.getContext("2d");
            ctx.fillStyle = "#ddd";
            ctx.fillRect(drawSx + mouseOnGraphStart * drawW, 0, (mouseOnGraphEnd - mouseOnGraphStart) * drawW, 200)
        }
        ctx.strokeStyle = `gray`;
        ctx.beginPath();
        ctx.moveTo(drawSx, 100);
        ctx.lineTo(drawSx + l * pixelWidth, 100);
        ctx.stroke();
    }

    let fac = Math.max(Math.abs(Math.max(...v)), Math.abs(Math.min(...v)));
    if (prevCompare) {
        let facPrev = Math.max(Math.abs(Math.max(...drawVectorComparePrevious)), Math.abs(Math.min(...drawVectorComparePrevious)));
        if (fac < facPrev) {
            fac = facPrev;
        }
        let diffPrev = math.subtract(v, drawVectorComparePrevious)
        facPrev = Math.max(Math.abs(Math.max(...diffPrev)), Math.abs(Math.min(...diffPrev)));
        if (fac < facPrev) {
            fac = facPrev;
        }
    }
    if (fac > 0) {
        fac = 90 / fac
    }
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(drawSx, 100 - Math.floor(fac * v[0]));
    for (let i = 1; i < l; i++) {
        ctx.lineTo(drawSx + i * pixelWidth, 100 - Math.floor(fac * v[i]));
    }
    ctx.stroke();
    if (prevCompare) {
        ctx.strokeStyle = 'green';
        ctx.beginPath();
        ctx.moveTo(drawSx, 100 - Math.floor(fac * drawVectorComparePrevious[0]));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(drawSx + i * pixelWidth, 100 - Math.floor(fac * drawVectorComparePrevious[i]));
        }
        ctx.stroke();
        ctx.strokeStyle = 'blue';
        ctx.beginPath();
        ctx.moveTo(drawSx, 100 - Math.floor(fac * (v[0] - drawVectorComparePrevious[0])));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(drawSx + i * pixelWidth, 100 - Math.floor(fac * (v[i] - drawVectorComparePrevious[i])));
        }
        ctx.stroke();
    } else {
        drawVectorComparePrevious = math.clone(v);
    }
}

function calcNewRange(B, lambda, L, r0) {
    return (B * lambda / r0) * L;
}
function drawMatDecomposition(ix, clear = true, color = "red") {
    if (!drawOption) {
        return
    }
    const canvas = document.getElementById("graphCanvas");
    const ctx = canvas.getContext("2d");
    let ranges = multiRanges[0];

    if (clear) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, 1000, 200);    

        ctx.fillStyle = 'black';
        ctx.fillText(ix.toFixed(0), 10, 120);

        ctx.fillStyle = color;
 
        let mats = vecMats[ix];
        let px = 10, dx = 60;

        let mMat = [[1, 0], [0, 1]];
        let newRange = ranges[0];
        for (let iMat = 0; iMat < mats.length; iMat++) {
            let [A, B, C, D] = [mats[iMat][0][0], mats[iMat][0][1], mats[iMat][1][0], mats[iMat][1][1]];
            ctx.fillText(math.re(A).toFixed(6), px, 20);
            ctx.fillText(math.re(B).toFixed(6), px + dx, 20);
            ctx.fillText(math.re(C).toFixed(6), px, 40);
            ctx.fillText(math.re(D).toFixed(6), px + dx, 40);
            newRange = calcNewRange(B, lambda, nSamples, newRange);
            ctx.fillText(newRange.toFixed(6), px, 70);
            if (iMat == 0) {
                ctx.fillStyle = 'blue';
                newRange = ranges[0];
                px += 200 * (mats.length - 1)
            } else {
                mMat = MMult(mats[iMat], mMat);
                let [A, B, C, D] = [mMat[0][0], mMat[0][1], mMat[1][0], mMat[1][1]];
                ctx.fillText(math.re(A).toFixed(6), px, 120);
                ctx.fillText(math.re(B).toFixed(6), px + dx, 120);
                ctx.fillText(math.re(C).toFixed(6), px, 140);
                ctx.fillText(math.re(D).toFixed(6), px + dx, 140);
                px -= 200;
            }
        }

    }
}
function drawGraph() {
    if (!drawOption) {
        return
    }
    const sel = document.getElementById("displayOption");

    graphData = [];
    switch (sel.value) {
        case "A": graphData.push(vecA); break;
        case "B": graphData.push(vecB); break;
        case "C": graphData.push(vecC); break;
        case "D": graphData.push(vecD); break;
        case "AbsE(x)":
            break;
        case "ArgE(x)":
            break;
        case "M(x)":
            break;
        case "Width(x)": 
            graphData.push(vecW);
            graphData.push(vecDeriv(vecW, distStep));
            break
        case "Waist(x)": 
            graphData.push(math.abs(vecWaist));
            graphData.push(vecDeriv(math.abs(vecWaist), distStep));
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

function initElementsMultiMode() {
    let iEl = 0;
    elements = [];
    do {
        let valOther = 0;
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
            valOther = parseFloat(lensFocal.value);
            //console.log(`${lensDist.value} ${valDist}, ${lensFocal.value} ${valFocal}`)
            if (isNaN(valDist) || isNaN(valOther)) {
                break;
            }
        } else if (elementType == "C") {
            let crystalLength = document.getElementById(`el${iEl}length`);
            if (crystalLength == null) {
                console.log(`Error ${crystalLength}`)
                break;
            }
            valOther = parseFloat(crystalLength.value);
            if (isNaN(valDist) || isNaN(valOther)) {
                break;
            }
        } else {
            valOther = -1;
        }


        elements.push({t: elementType, par: [valDist, valOther]});
        //console.log(elements);
        iEl++;
    } while(true);
}
function focusOnCrystal() {
    elements.forEach((el, index) => {
        if (el.t == "C") {
            distStep = el.par[1] / 10.0;
            distStart = el.par[0] - distStep;
        }
    });

}

function initMultiMode(setWorkingTab = - 1, beamParam = - 1) {
    if (setWorkingTab > 0) {
        workingTab = setWorkingTab
    }
    switch (workingTab) {
        case 1:
            drawW = 3;
            distStart = 0.0;
            distStep = 0.003;

            break
        case 2:
            drawW = 30;
            focusOnCrystal();
            break
    }

    displayTemp = [[], []];
    displayTempPrevious = [[], []];

    multiFronts[0] = [getInitFront(beamParam)];
    multiRanges[0] = [initialRange];
    sfs = 0;
    drawMode = 1;
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
  
    console.log("minus");
    /* initialize Sin / Cos tables */
    for (let k = 0; k < N; k++) {
      inpReal.push(math.re(inp[k]));
      inpImag.push(math.im(inp[k]));
      const angle = - twoPiByN * k;
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
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

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
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

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
var distStart = 0.0;


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

function getMatOnRoundTrip(oneWay = false) {
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

    if (oneWay) {
        return M;
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
        console.log(`calcWidth error l = ${l} `);
        return 0.0;
    }
    sumX /= N;
    sumX2 /= N;

    let w = Math.sqrt(sumX2 - sumX * sumX);

    return w;
}

// function decomposeMat(M, dStep, r0, L) {
//     let [A, B, C, D] = [M[0][0], M[0][1], M[1][0], M[1][1]];
//     let M1, M2, M3, M4;

//     if (Math.abs(A + 1) > 0.1) {
//         console.log(`AA dStep = ${dStep} A = ${A}, B = ${B}, C = ${C}, D = ${D} B1 = ${B / (A + 1)} Rdxf = ${lambda * B / (A + 1) / r0} RR${L * lambda * B / (A + 1) / r0}`);
//         M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]];
//         dxMid = lambda * B / (A + 1) / r0;
//         M1 = [[1, B / (A + 1)], [0, 1]];
//     } else {

//         console.log(`BB dStep = ${dStep} A = ${A}, B = ${B}, C = ${C}, D = ${D} B1 = ${B / (- A + 1)} Rdxf = ${lambda * B / (- A + 1) / r0} RR${L * lambda * B / (- A + 1) / r0}`);
//         M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]];
//         dxMid = lambda * B / (- A + 1) / r0;
//         M1 = [[-1, B / (-A + 1)], [0, -1]];
//     }

//     return [M1, M2];
// }

function getMatricesAtDistFromStart(M, dStep, r0) {
    let L = nSamples;
    let mats = [], isBack = [];
    let spDist = r0 * r0 / (L * lambda);

    mats.push(math.clone(M));
    isBack.push(false);

    let MS, MD, useDistFix = 0;
    if (Math.abs(M[0][1]) < 1.8 * spDist) {
        if (dStep > 2 * spDist) {
            MD = [[1.0, spDist], [0, 1]];
            useDistFix = 1;
            let mPush = [[1, - spDist], [0, 1]];
            MS = MMult(mPush, M);
            while (Math.abs(MS[0][1]) < spDist) {
                MS = MMult(mPush, MS);
                useDistFix++;
            }
        } else {
            MD = [[1.0, - spDist], [0, 1]];
            useDistFix = 2;
            let mPush = [[1, 2 * spDist], [0, 1]];
            MS = MMult(mPush, M);
            while (Math.abs(MS[0][1]) < spDist) {
                MS = MMult(mPush, MS);
                useDistFix++;
            }
        }
    } else {
        MS = math.clone(M);
    }
    let A = MS[0][0], B = MS[0][1], C = MS[1][0], D = MS[1][1];

    let M1, M2;//, dxMid, ff;
    //console.log(`===== StartM XNew ${M[0][0]},${M[0][1]},${M[1][0]},${M[1][1]},`);

    //[M1, M2] = decomposeMat(M, dStep, r0, L);

    // decompose into two matrices
    //if (Math.abs(A + 1) > 0.1) {
    if (A > 0) {
        //console.log(`AAN dStep = ${dStep} A = ${A}, B = ${B}, C = ${C}, D = ${D} B1 = ${B / (A + 1)} Rdxf = ${lambda * B / (A + 1) / r0} RR${L * lambda * B / (A + 1) / r0}`);
        M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]];
        //dxMid = lambda * B / (A + 1) / r0;
        M1 = [[1, B / (A + 1)], [0, 1]];
    } else {
        // negate matrix and then decompose
        //console.log(`BBN dStep = ${dStep} A = ${A}, B = ${B}, C = ${C}, D = ${D} B1 = ${B / (- A + 1)} Rdxf = ${lambda * B / (- A + 1) / r0} RR${L * lambda * B / (- A + 1) / r0}`);
        M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]];
        //dxMid = lambda * B / (- A + 1) / r0;
        M1 = [[-1, B / (-A + 1)], [0, -1]];
    }

    // dxMid = lambda * M2[0][1] / r0;

    mats.push(math.clone(M1));
    isBack.push(A < 0);
    mats.push(math.clone(M2));
    isBack.push(A < 0);

    for (let iDistFix = 0; iDistFix < useDistFix; iDistFix++) {
        mats.push(math.clone(MD));
        isBack.push(MD[0][1] < 0.0);
    }

    return [mats, isBack];
}

function fullCavityMultiMode() {
    drawMode = 1;
    
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

    if (fronts.length <= 0) {
        return;
    }
    
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0], vecQ[0] = math.complex(0, RayleighRange), vecMats = [];
    
    for (let iStep = 1; iStep < 300; iStep++) {
        let f0 = math.clone(fronts[0]);
        let r0 = ranges[0];
        let L = f0.length;
        let dx0 = r0 / L;
        let dStep = iStep * distStep;

        let MS = getMatOnStep(dStep);

        let [mats, isBack] = getMatricesAtDistFromStart(MS, dStep, r0);

        let M  = mats[0];
        let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        let newQ = math.chain(vecQ[0]).multiply(A).add(B).divide(math.chain(vecQ[0]).multiply(C).add(D).done()).done();

        let dx = dx0;
        let fx = math.clone(fronts[0]);
        for (let iMat = 1; iMat < mats.length; iMat++) {
            //console.log(`===== MM ${mats[iMat][0][0]},${mats[iMat][0][1]},${mats[iMat][1][0]},${mats[iMat][1][1]}, dx = ${dx}`);
            [ff, dx] = CalcNextFrontOfM(fx, L, mats[iMat], dx, isBack[iMat]);
            fx = math.clone(ff);
        }
        ff = math.clone(fx);    
        vecMats.push(mats);
        let dxf = dx;

        vecQ.push(newQ);
        let width = calcWidth(ff);
        if (width < 0.0000001) {
            break;
        }
        vecW.push(width * Math.abs(dxf));
        vecWaist.push(width * Math.abs(dxf) * 1.41421356237);
        fronts.push(ff);
        ranges.push(L * dxf);
    }

    drawMultiMode();
}

function calcPower(f) {
    return math.sum(math.dotMultiply(f, math.conj(f)))
}

var fullCavityCrystalPrevFocal = []
function fullCavityCrystal(modePrev = 1) {
    drawMode = 1;
    let fullCavityCrystalPrevFocalIndex = 0;

    if (modePrev == 1) {
        fullCavityCrystalPrevFocal = [];
    }

    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [], vecWaist = [], vecQ[0] = math.complex(0, RayleighRange), vecMats = [];

    let power = getFieldFloat('power', 30000000);
    let lens_aperture = 56e-6;
    //let fa = ((2 * Math.PI * lens_aperture ** 2) / lambda);
    let fa = ((2 * lens_aperture ** 2));
    let n2 = 3e-20; // n2 of sapphire m^2/W
    let crystalLength = 3e-3;
    let kerrPar =  4 * crystalLength * n2;
    let Ikl = kerrPar / 5 / 50;
    let M, imagA;
    let MatSide = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  // right
                 [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]];   // left
    let MatsSide = [];
    let MatTotal = [[1, 0], [0, 1]];
    let focalAper = ((2 * Math.PI * lens_aperture ** 2) / lambda)
    let matAperture = [[1, 0], [math.complex(0, -1/ focalAper), 1]];

    for (let tt = 0; tt < 1; tt++) {
    MatSide.forEach((m, index) => {
        displayTempPrevious[index] = displayTemp[index];
        displayTemp[index] = [];
        let [[A, B], [C, D]] = m;

        let M1, M2;
        if (A > 0) {
            // A not close to -1
            M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]];
            M1 = [[1, B / (A + 1)], [0, 1]];
        } else {
            // A close to -1, so negate matrix and then decompose
            M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]];
            M1 = [[-1, B / (-A + 1)], [0, -1]];
        }
        console.log(`M1 ${M1}`);
        console.log(`M2 ${M2}`);

        MatsSide.push([M1, M2]);
    });

    let gain = 1.030;
    let L = nSamples;

    for (let iDir = 0; iDir < 2; iDir++) {
        let fronts = multiFronts[iDir];
        let ranges = multiRanges[iDir];
        let frontsO = multiFronts[1 - iDir];
        let rangesO = multiRanges[1 - iDir];

        if (fronts.length <= 0) {
            return;
        }
    
        for (let iStep = 1; iStep < 12; iStep++) {
            let fx = math.clone(fronts[iStep - 1]);
            // for (let ii = 0; ii < fx.length; ii++) {
            //     fx[ii] = math.multiply(fx[ii], 1.031);
            // }

            let rx = ranges[iStep - 1];

            let dxf, dx0 = rx / L;

            if (iStep <= 1) {
                let width = calcWidth(fx);
                let waist = width * dx0 * 1.41421356237;
                console.log(`------ waist ${waist}`);
                vecW.push(width);
                vecWaist.push(waist);
    
                let pi = calcPower(fx);
                for (let ii = 0; ii < fx.length; ii++) {
                    fx[ii] = math.multiply(fx[ii], vecAperture[ii]);
                }
                let pf = calcPower(fx);
                let fff = Math.sqrt(pi / pf);
                for (let ii = 0; ii < fx.length; ii++) {
                    fx[ii] = math.multiply(fx[ii], fff);
                }
                //let pt = calcPower(fx);

                //MatTotal = math.multiply(matAperture, MatTotal);

                //console.log(`pi = ${pi}, pf = ${pf}, pt = ${pt}, `)
            }

            imagA = - dx0 * dx0 / fa;
            let width = calcWidth(fx);
            let waist = width * dx0 * 1.41421356237;
            vecW.push(width);
            vecWaist.push(waist);
            //waist = 0.000030;
            if (iStep % 2 == 0) {
                let focal = Math.pow(waist, 4) / (Ikl * power);
                if (modePrev == 1) {
                    fullCavityCrystalPrevFocal.push(focal);
                } else {
                    focal = fullCavityCrystalPrevFocal[fullCavityCrystalPrevFocalIndex];
                    fullCavityCrystalPrevFocalIndex++;
                }

                console.log(`waist ${waist}, focal ${focal}`);
                displayTemp[iDir].push(focal);
                M = [[1 - distStep / focal, distStep], [- 1 / focal, 1]];
            } else {
                //console.log(`waist ${waist} power ${math.sum(math.dotMultiply(fx, math.conj(fx)))}`);
                M = [[1, distStep], [0, 1]];
            }
            let gainFactor = Math.min(1.0, fa / (2 * waist * waist));
            [ff, dxf] = CalcNextFrontOfM(fx, L, M, dx0, false, imagA, gain * gainFactor);
            MatTotal = math.multiply(M, MatTotal);

            // console.log(`lambda ${lambda}, B = ${M[0][1]}, L = ${L}, dx = ${dx0}, dxf = ${dxf}`)

            if (fronts.length <= iStep) {
                fronts.push(ff);
                ranges.push(L * dxf);
            } else {
                fronts[iStep] = ff;                
                ranges[iStep] = L * dxf;                
            }
        }


        [M1, M2] = MatsSide[iDir];
        let fx = math.clone(fronts[fronts.length - 1]);
        let dxf, dx0 = ranges[ranges.length - 1] / L;
        let waist = calcWidth(fx) * dx0 * 1.41421356237;
        console.log(`waist before M2 ${waist} dx = ${dx0}`);

        [fx, dxf] = CalcNextFrontOfM(fx, L, M2, dx0, iDir == 1);
        MatTotal = math.multiply(M2, MatTotal);
        console.log(`after 2  waist = ${calcWidth(fx) * dxf * 1.41421356237} dx = ${dxf}`);
        console.log(fx);

        waist = calcWidth(fx) * dxf * 1.41421356237;
        console.log(`waist before M1 ${waist} power ${math.sum(math.dotMultiply(fx, math.conj(fx)))}`);
        [fx, dxf] = CalcNextFrontOfM(fx, L, M1, dxf, iDir == 1);
        console.log(`after 1  waist = ${calcWidth(fx) * dxf * 1.41421356237} dx = ${dxf}`);
        MatTotal = math.multiply(M1, MatTotal);

        waist = calcWidth(fx) * dxf * 1.41421356237;
        console.log(`======= waist after M1 ${waist} power ${math.sum(math.dotMultiply(fx, math.conj(fx)))}`);
        
        console.log(`Before ${power} `)
        // power = normalizePower(fx, power, dxf);
        // console.log(`After ${power} `, fx)

        frontsO[0] = fx;
        rangesO[0] = L * dxf;

        if (iDir == 1) {
            fronts.reverse();
            ranges.reverse();
        }
    }
    }

    let [[A, B], [C, D]] = MatTotal;
    let [lambda1, lambda2] = calculateEigenvalues(A, B, C, D);

    console.log(`eigen values = [${lambda1}, ${lambda2}] mult=${math.multiply(lambda1, lambda2)}`);
    console.log(`abs = [${math.abs(lambda1)}, ${math.abs(lambda2)}] `);
    console.log(`|A+D| = ${math.abs(math.add(MatTotal[0][0], MatTotal[1][1]))}, det = ${math.det(MatTotal)}`)
    console.log(`ABCD = ${MatTotal[0][0]} ${MatTotal[0][1]} ${MatTotal[1][0]} ${MatTotal[1][1]}`);
    if (math.abs(math.add(MatTotal[0][0], MatTotal[1][1])) < 2.0) {
        // let [[A, B], [C, D]] = MatTotal;
        // let oneOverQ = math.complex(math.divide((math.subtract(D,  A)), (2 * B)), 
        //         math.divide(math.sqrt(math.subtract(1, math.multiply(0.25, math.multiply(math.add(A, D), math.add(A, D))),)), B));
        // let actualQ = math.divide(1, oneOverQ);
        // let nextQ = math.divide(math.add(math.multiply(actualQ, A), B), math.add(math.multiply(actualQ, C), D));
        // let diffQ = math.subtract(actualQ, nextQ);
        // console.log(`A = ${A}, B = ${B}, C = ${C}, D = ${D}`);
        // console.log(`SOL1 => Q = ${actualQ} 1/Q = ${oneOverQ} nextQ = ${nextQ} diff = ${math.abs(diffQ)}`);
        
        // beamDist = 2.0 * B / (D - A);
        // beamWaist = Math.sqrt(lambda * Math.abs(B) / (Math.PI * Math.sqrt(1 - 0.25 * (A + D) * (A + D))));
        // console.log(`ORIG beamWaist = ${saveBeamParam}, beamDist = ${saveBeamDist}`);


        // console.log(`beamWaist = ${beamWaist}, beamDist = ${beamDist}, z = ${actualQ.re}`);
        // setFieldFloat('beamDist', beamDist);
        // setFieldFloat('beamParam', beamWaist);
    }
    //setFieldFloat('power', power);

    drawMultiMode();
}

function calculateEigenvalues(A, B, C, D) { // thank you chatgpt
    // Calculate trace and determinant
    const trace = math.add(A, D); // A + D
    const determinant = math.subtract(math.multiply(A, D), math.multiply(B, C)); // AD - BC

    // Discriminant: (trace / 2)^2 - determinant
    const halfTrace = math.divide(trace, 2); // trace / 2
    const discriminant = math.subtract(math.pow(halfTrace, 2), determinant);

    // Square root of the discriminant
    const sqrtDiscriminant = math.sqrt(discriminant);

    // Eigenvalues: λ1 = trace/2 + sqrt(discriminant), λ2 = trace/2 - sqrt(discriminant)
    const lambda1 = math.add(halfTrace, sqrtDiscriminant);
    const lambda2 = math.subtract(halfTrace, sqrtDiscriminant);

    return [lambda1, lambda2];
}

function  normalizePower(fx, power, dxf) {
    const p = math.sum(math.dotMultiply(fx, math.conj(fx))) / nSamples;
    console.log(`dot ${p}`)
    fx = math.multiply(fx, 1 /  Math.sqrt(p));
    return power * p; 
}

function CalcNextFrontOfM(f0, L, M, dx0, isBack = false, imagA = 0, gain = 1) {
    let A = M[0][0];
    let B = M[0][1];
    let D = M[1][1];

    let dxf = B * lambda / (L * dx0); 
    gain = 1;
    let factor = math.multiply(math.complex(0, gain), math.sqrt(math.complex(0, - 1 / (B * lambda))));
    if (isBack) {
        factor = math.multiply(factor, math.complex(0, gain));
    }

    let co0 = - Math.PI * dx0 * dx0 * A / (B * lambda);
    let cof = - Math.PI * dxf * dxf * D / (B * lambda);

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        f0[i] = math.multiply(f0[i], math.exp(math.complex(0, co0 * ii * ii)))
    }
    let ff = fft(f0, dx0);

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        ff[i] = math.multiply(math.multiply(ff[i], factor), math.exp(math.complex(0/*imagA * ii * ii*/, cof * ii * ii)))
    }

    return [ff, dxf];
}

function roundtripMultiMode(waist = - 1) {
    drawMode = 2;
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

    if (fronts.length <= 0) {
        console.log(`Fronts length ${fronts.length}`);
        return;
    }

    let M  = getMatOnRoundTrip(false);
    let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
    let v = (A + D) / 2;
    //console.log(`A0 = ${A}, B0 = ${B}, C0 = ${C}, D0 = ${D}, v = ${v}`)
    if (v * v <= 1) {
        disc = Math.sqrt(1 - v * v);
        l1 = v + disc;
        l2 = v - disc;
        qi = B / disc;
        w = qi * lambda / Math.PI;
        //console.log(`disc = ${disc}, w = ${w}, l1 = ${l1}, l2 = ${l2}`)
    }
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0], vecQ[0] = math.complex(0, RayleighRange), vecMats = [];
    for (let iStep = 1; iStep < 50; iStep++) {

        let f0 = math.clone(fronts[iStep - 1]);
        let r0 = ranges[iStep - 1];
        let L = f0.length;
        let dx0 = r0 / L;
        let B = M[0][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);

        let [ff, dxf] = CalcNextFrontOfM(f0, L, M, dx0);

        let width = calcWidth(ff);
        if (width < 0.0000001) {
            break;
        }
        vecW.push(width * Math.abs(dxf));
        vecWaist.push(width * Math.abs(dxf) * 1.41421356237);

        fronts.push(ff);
        ranges.push(L * dxf);
    }
    drawMultiMode();
}

function autoRangeMultiMode(M = null) {
    //test();
    if (M == null) {
        initMultiMode(1);
        M = getMatOnRoundTrip(false);
    }
    let B = M[0][1];
    let dxSquare = lambda * Math.abs(B) / nSamples;

    let newRange = Math.sqrt(dxSquare) * nSamples;

    const rng = document.getElementById("initialRange");
    rng.value = newRange.toFixed(6);
}

function doDeltaStep(delta, waist) {
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

    const origValue = elements[1].par[0];
    elements[1].par[0] += delta;
    
    initMultiMode(1, waist);

    let M = getMatOnRoundTrip(false);
    let A = M[0][0];
    let B = M[0][1];
    let D = M[1][1];

    autoRangeMultiMode(M);

    if (Math.abs(A + D) > 2.0) {
        elements[1].par[0] = origValue;
        return (waist);
    }
    let ad = 0.5 * (A + D);
    let imQ = Math.sqrt(1.0 - ad * ad) / Math.abs(B);
    let waistCalc = Math.sqrt(lambda / imQ / Math.PI);

    for (let iter = 0; iter < 10; iter++) {
        initMultiMode(1, waist);

        roundtripMultiMode(waist);

        let lvw = vecWaist.length;
        let part = vecWaist.slice(10, lvw - 10);
        let partMax = Math.max(...part);
        let partMin = Math.min(...part);

        let avg = Math.sqrt(partMax * partMin);
        let rel = (partMax - partMin) / avg;
        waist = avg;
        //console.log(`I = ${iter}, (${partMin} - ${partMax}), rel = ${rel}, waist = ${waist}`);

        if (rel < 0.0001) {
            console.log(`===== delta = ${delta.toFixed(4)} waist = ${waist} iter = ${iter} rel = ${Math.abs(waist - waistCalc) / waist}`)
            deltaGraphX.push(delta);
            deltaGraphY.push(waist);
            deltaGraphYCalc.push(waistCalc);

            let M = getMatOnRoundTrip(true);
            let f0 = math.clone(fronts[0]);
            let r0 = ranges[0];
            let L = f0.length;
            let dx0 = r0 / L;
            let B = M[0][1];
            let [ff, dxf] = CalcNextFrontOfM(f0, L, M, dx0);
            let width = calcWidth(ff);
            deltaGraphYHalf.push(width * Math.abs(dxf) * 1.41421356237);

            elements[1].par[0] = origValue;
            return (waist);
        }

    }

    elements[1].par[0] = origValue;
    return (waist);
}

var coverWaist;
function doDeltaStepCover(delta) {
    drawOption = false;
    coverWaist = doDeltaStep(delta, coverWaist);

    delta += 0.0002;
    if (delta < 0.045) {
        setTimeout(doDeltaStepCover, 1, delta);
    } else {
        drawOption = true;
        return;
    }
    drawMode = 3;
    drawOption = true;
    drawMultiMode();
}

function deltaGraphMultiMode() {
    drawOption = false;

    coverWaist = 0.0005;
    deltaGraphX = [];
    deltaGraphY = [];
    deltaGraphYHalf = [];
    deltaGraphYCalc = [];

    initMultiMode(1, coverWaist);

    let delta = 0.001;

    doDeltaStepCover(delta);

    drawMode = 3;
    drawOption = true;
    drawMultiMode();
}

function mainCanvasMouseMove(e, id) {
    if (!isMouseDownOnMain) {
        return;
    }
    const sel = document.getElementById("displayOption");
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;

    let fronts = multiFronts[id - 1];

    let ix = Math.floor((x - drawSx) / drawW);
    if (ix >= 0 && ix < fronts.length) {
        if (sel.value == "AbsE(x)") {
            fi = fronts[ix];
            fr = [];
            for (let i = 0; i < fi.length; i++) {
                fr.push(fi[i].toPolar().r);
            }
            drawVector(fr, true, "red", 1);
        } else if (sel.value == "ArgE(x)") {
            fi = fronts[ix];
            fr = [];
            for (let i = 0; i < fi.length; i++) {
                fr.push(fi[i].toPolar().phi);
            }
            drawVector(fr, true, "purple", 1);
        } else if (sel.value == "M(x)") {
            drawMatDecomposition(ix);
        }
    }

}
function mainCanvasMouseDown(e, id) {
    isMouseDownOnMain = true;
    mainCanvasMouseMove(e, id);
}

function mainCanvasMouseUp(e, id) {
    isMouseDownOnMain = false;
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
    isMouseDownOnGraph = false;
}

function drawTextBG(ctx, txt, x, y, color = '#000', font = "10pt Courier") {
    ctx.save();
    ctx.font = font;
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#fff';
    var width = ctx.measureText(txt).width;
    ctx.beginPath();
    ctx.roundRect(x, y + 1, width, parseInt(font, 10) + 3, 3);
    ctx.fill();
    //ctx.fillRect(x, y + 1, width, parseInt(font, 10) + 3);
    ctx.fillStyle = color;
    ctx.fillText(txt, x, y);
    ctx.restore();
}