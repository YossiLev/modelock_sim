
var workingTab = 1;
var sfs = -1;
var multiFronts = [[], []];
var gaussianFronts = [[], []];
const lambda = 0.000000780;
var initialRange = 0.01047;
var multiRanges = [[], []];
var nSamples = 256;
var nMaxMatrices = 1000;
var nRounds = 0;
var viewOption = 1;
var zoomFactor = 1.0;
var basicZoomFactor = 50000.0; // pixels per meter
var zoomHorizontalCenter = 0.5;
var zoomHorizontalShift = 0.0;
var zoomHorizontalAmount = 1.0;

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
var drawMode = 1; //
var deltaGraphX, deltaGraphY,  deltaGraphYHalf, deltaGraphYCalc;
var displayTemp = [[], []];
var displayTempPrevious = [[], []];
var presentedVectors = new Map();
var crystalNumberOfLenses = 5;
var crystalLength = 3e-3;
var stabilityGraph;

let totalRightSide;
let totalLeftSide;

var saveBeamParam, saveBeamDist;
function getInitFront(pPar = - 1) {
    let waist0, beamDist, theta, waist, dx;
    const sel = document.getElementById("incomingFront");
    initialRange = getFieldFloat("initialRange");
    let vf = [];
    RayleighRange = 0.0;
    switch (sel.value) {
        case "Gaussian Beam":
            waist0 = pPar > 0.0 ? pPar : getFieldFloat("beamParam", 0.0005);
            beamDist = getFieldFloat("beamDist", 0.0);
            RayleighRange = Math.PI * waist0 * waist0 / lambda;
            theta = Math.abs(beamDist) < 0.000001 ? 0 : Math.PI  / (lambda * beamDist);
            waist = waist0 * Math.sqrt(1 + beamDist / RayleighRange);
            saveBeamParam = waist0;
            saveBeamDist = beamDist;
            dx = initialRange / nSamples;
            console.log(`on creeate range=${initialRange} dx = ${dx}`)
            x0 = (nSamples - 1) / 2 * dx;
            for (let i = 0; i < nSamples; i++) {
                let px = i * dx;
                let x = (px - x0);
                xw = x / waist;
                let fVal = math.exp(math.complex(- xw * xw, - theta * x * x))
                vf.push(fVal);
            }
            break;
        case "Gaussian Noise":
            waist0 = getFieldFloat("beamParam", 0.0005);
            beamDist = getFieldFloat("beamDist", 0.0);
            RayleighRange = Math.PI * waist0 * waist0 / lambda;
            theta = Math.abs(beamDist) < 0.000001 ? 0 : Math.PI  / (lambda * beamDist);
            waist = waist0 * Math.sqrt(1 + beamDist / RayleighRange);
            saveBeamParam = waist0;
            saveBeamDist = beamDist;
            dx = initialRange / nSamples;

            x0 = (nSamples / 2 - 1) * dx;
            for (let i = 0; i < nSamples; i++) {
                let px = i * dx;
                let x = (px - x0);
                xw = x / waist;
                let fVal = math.add(math.exp(math.complex(- xw * xw, - theta * x * x)), 
                    math.complex((Math.random() * 2 - 1) * 0.05, (Math.random() * 2 + 1) * 0.05));
                vf.push(fVal);
            }
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
    nSamples = getFieldInt("nSamples");
    let step = 0.0003;
    initialRange =  math.sqrt(lambda * step * nSamples);
    setFieldFloat("initialRange", initialRange);
}

function nLensesChanged() {
    crystalNumberOfLenses = getFieldInt("nLenses");
}

function nMaxMatricesChanged() {
    const val = document.getElementById("nMaxMatrices").value;
    if (val == "All") {
        nMaxMatrices = 1000;
    } else {
        nMaxMatrices = parseInt(val);
    }
}

function nRoundsChanged() {
    nRounds = getFieldInt("nRounds");
}

function zoomMultiMode(z) {
    switch (z) {
        case   1: zoomFactor           = zoomFactor           * 1.5; break;
        case - 1: zoomFactor           = zoomFactor           / 1.5; break;
        case - 2: 
            zoomHorizontalShift = zoomHorizontalCenter - (zoomHorizontalCenter - zoomHorizontalShift) / 1.5;
            zoomHorizontalAmount = zoomHorizontalAmount * 1.5; 
            racalcHorizontalZoom();
            break;
        case   2: 
            zoomHorizontalShift = zoomHorizontalCenter - (zoomHorizontalCenter - zoomHorizontalShift) * 1.5;
            zoomHorizontalAmount = zoomHorizontalAmount / 1.5;
            racalcHorizontalZoom(); 
            break;
        case   20: 
            zoomHorizontalShift = 0;
            zoomHorizontalAmount = 1;
            racalcHorizontalZoom(); 
            break;
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
    const sizeM = (canvasSize / valPerPixel + startVal) / 1.5;
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

function toCtxCoords(ctx, xy, zoomXY, shiftXY, pushXY) {
    const h = ctx.canvas.height;
    return [(xy[0] - shiftXY[0]) * zoomXY[0] + pushXY[0], h - 1 - (xy[1] - shiftXY[1]) * zoomXY[1] + pushXY[1]];
}
function fromCtxCoords(ctx, xy, zoomXY, shiftXY, pushXY) {
    const h = ctx.canvas.height;
    return [(xy[0] - pushXY[0]) / zoomXY[0] + shiftXY[0], (h - 1 - (xy[1] - pushXY[1])) / zoomXY[1] + shiftXY[1]];
}

let deltaMinX = 0;
let deltaMaxX = 1;
let deltaMinY = 0;
let deltaMaxY = 1;

function drawDeltaGraph(ctx) {
    if (deltaGraphX.length < 5) {
        return;
    }
    deltaMinX = Math.min(...deltaGraphX);
    deltaMaxX = Math.max(...deltaGraphX);
    deltaMinY = Math.min(...deltaGraphY, ...deltaGraphYHalf);
    deltaMaxY = Math.max(...deltaGraphY, ...deltaGraphYHalf);

    const height = ctx.canvas.height;
    const gWidth = ctx.canvas.width - 200;
    const gHeight = ctx.canvas.height - 200;

    const zoomY = gHeight / (deltaMaxY - deltaMinY);
    const zoomX = gWidth / (deltaMaxX - deltaMinX);

    ctx.fillStyle = "red";
    for (let i = 0; i < deltaGraphX.length; i++) {
        let [x, y] = [deltaGraphX[i], deltaGraphY[i]];
        let [cx, cy] = toCtxCoords(ctx, [x, y], [zoomX, zoomY], [deltaMinX, deltaMinY], [drawSx, 0]);
        ctx.fillRect(cx - 3, cy - 3, 6, 6);
    }
    ctx.fillStyle = "blue";
    for (let i = 0; i < deltaGraphX.length; i++) {
        let [x, y] = [deltaGraphX[i], deltaGraphYHalf[i]];
        let [cx, cy] = toCtxCoords(ctx, [x, y], [zoomX, zoomY], [deltaMinX, deltaMinY], [drawSx, 0]);
        ctx.fillRect(cx - 3, cy - 3, 6, 6);
    }
    drowShenets(ctx, "V", zoomY, deltaMinY);
    drowShenets(ctx, "H", zoomX, deltaMinX);
}

function drawFronts(canvas, ctx, fronts, ranges, gaussian) {
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

    let waists = vecWaistFromQ(gaussian);
    let hFactor = zoomFactor * basicZoomFactor;
    h = 0.0;
    ctx.strokeStyle = `rgba(255, 128, 0, 255)`;
    ctx.beginPath();
    ctx.moveTo(drawSx, drawMid);
    for (let f = 0; f < waists.length; f++) {
        wi = waists[f];
        ctx.lineTo(drawSx + f * drawW, drawMid + wi * hFactor);
    }
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(drawSx, drawMid);
    for (let f = 0; f < waists.length; f++) {
        wi = waists[f];
        ctx.lineTo(drawSx + f * drawW, drawMid - wi * hFactor);
    }
    ctx.stroke();


}

function drawMultiMode(startDraw = 0.0) {
    if (!drawOption) {
        return;
    }
    let canvasList;
    if (workingTab == 3) {
        drawMultiTime();
        canvasList = [document.getElementById("funCanvasTest")];
    } else {
        canvasList = document.querySelectorAll('[id^="funCanvas"]');
    }
    canvasList.forEach((canvas, index) => {
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffb0e0";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        drawMid = canvas.height / 2;

        if (drawMode == 3) {
            drawDeltaGraph(ctx);
            return;
        }
        if (drawMode == 4) {
            stabilityGraph.plot(ctx, stabilityColor);
            return;
        }

        drawFronts(canvas, ctx, multiFronts[index], multiRanges[index], gaussianFronts[index]);

        if (drawMode == 1) {
            drawElements(index + 1, startDraw);
            displayTemp[index].forEach((el, i, a) => {
                drawTextBG(ctx, el.toFixed(7), canvas.width - 80, canvas.height - 28 - 16 * ((a.length - i)));
            });
            let totalDisplayTemp = 1.0 / displayTemp[index].reduce((acc, v) => acc + 1.0 / v, 0);
            drawTextBG(ctx, totalDisplayTemp.toFixed(7), canvas.width - 80, canvas.height - 28 - 16 * ((displayTemp[index].length + 1)), "blue");

            displayTempPrevious[index].forEach((el, i, a) => {
                drawTextBG(ctx, el.toFixed(7), canvas.width - 160, canvas.height - 28 - 16 * ((a.length - i)));
            });
            let totalDisplayTempPrevious = 1.0 / displayTempPrevious[index].reduce((acc, v) => acc + 1.0 / v, 0);
            drawTextBG(ctx, totalDisplayTempPrevious.toFixed(7), canvas.width - 160, canvas.height - 28 - 16 * ((displayTempPrevious[index].length + 1)), "blue");
        }

        if (drawMode == 2) {  // roundtrip
            let M = getMatOnRoundTrip(false);
            let [stable, lambda1, lambda2, beamWaist, beamDist] = analyzeStability(M);
            drawTextBG(ctx, M[0][0].toFixed(7), canvas.width - 180, 10);
            drawTextBG(ctx, M[0][1].toFixed(7), canvas.width - 80, 10);
            drawTextBG(ctx, M[1][0].toFixed(7), canvas.width - 180, 30);
            drawTextBG(ctx, M[1][1].toFixed(7), canvas.width - 80, 30);
            if (stable) {
                drawTextBG(ctx, lambda1.re.toFixed(7), canvas.width - 180, 60);
                drawTextBG(ctx, lambda1.im.toFixed(7), canvas.width - 80, 60);
                drawTextBG(ctx, lambda2.re.toFixed(7), canvas.width - 180, 80);
                drawTextBG(ctx, lambda2.im.toFixed(7), canvas.width - 80, 80);
                drawTextBG(ctx, beamWaist.toFixed(7), canvas.width - 180, 120);
                drawTextBG(ctx, beamDist.toFixed(7), canvas.width - 80, 120);
            }

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
            case 3:
            case 4:
                drowShenets(ctx, "H",  drawW / distStep * zoomHorizontalAmount, 0, distStart);
                break;

        }
        if (drawMode == 1) {
            if (zoomHorizontalCenter > 0.0) {
                ctx.fillStyle = 'red';
                ctx.beginPath();
                let x = zoomHorizontalCenter * drawW / distStep + drawSx;
                ctx.moveTo(x, 10);
                ctx.lineTo(x + 5, 0);
                ctx.lineTo(x - 5, 0);
                
                ctx.closePath();
                ctx.fill();
            }
    
        }
    });


    if (workingTab != 3) {
        drawGraph();
    }
}

function drawElements(index, startDraw) {
    if (!drawOption) {
        return
    }
    let canvas;
    if (workingTab == 3) {
        drawMultiTime();
        canvas = document.getElementById("funCanvasTest");
    } else {
        canvas = document.getElementById(`funCanvas${index}`);
    }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = `yellow`;
    let globalDelta = elements.find((el) => el.t == "X").delta;
    let cavityLength = elements.find((el) => el.t == "X").par[0];

    for (let r = 0; r < 7; r += 2) {
        ctx.fillStyle = `red`;
        px = drawSx + (0 - distStart - zoomHorizontalShift - startDraw + cavityLength * r) / distStep * zoomHorizontalAmount * drawW ;
        ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
        for (let iEl = 0; iEl < elements.length; iEl++) {
            let deltaFactor = elements[iEl].delta;
            let pos = elements[iEl].par[0] + deltaFactor * globalDelta
            switch (elements[iEl].t) {
                case "L":
                    ctx.fillStyle = `yellow`;
                    px = drawSx + (pos - distStart - zoomHorizontalShift - startDraw + cavityLength * r) / distStep * zoomHorizontalAmount * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    break;
                case "C":
                    ctx.fillStyle = `purple`;
                    px = drawSx + (pos - distStart- startDraw + cavityLength * r) / distStep * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    px = drawSx + (pos + elements[iEl].par[1] - distStart - startDraw + cavityLength * r) / distStep * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);
                    let spx = (elements[iEl].par[1] / (2 * crystalNumberOfLenses));
                    ctx.strokeStyle = `purple`;
                    ctx.setLineDash([5, 3]);
                    ctx.beginPath();
                    for (let iL = 0; iL < crystalNumberOfLenses; iL++) {
                        px = drawSx + (pos + (1 + 2 * iL) * spx - distStart- startDraw + cavityLength * r) / distStep * drawW ;
                        ctx.moveTo(px, drawMid - 80 * zoomFactor);
                        ctx.lineTo(px, drawMid + 80 * zoomFactor);
                    }
                    ctx.stroke();
                    break;
                case "X":
                    ctx.fillStyle = `blue`;
                    //px = drawSx + (elements[iEl].par[0] - distStart) / distStep * drawW ;
                    px = drawSx + (elements[iEl].par[0] - distStart - zoomHorizontalShift - startDraw + cavityLength * r) / distStep * zoomHorizontalAmount * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    break;
            }
        }
        for (let iEl = elements.length - 1; iEl >= 0; iEl--) {
            let deltaFactor = elements[iEl].delta;
            let pos = elements[iEl].par[0] + deltaFactor * globalDelta
            switch (elements[iEl].t) {
                case "L":
                    ctx.fillStyle = `yellow`;
                    px = drawSx + (- pos - distStart - zoomHorizontalShift - startDraw + cavityLength * (r + 2)) / distStep * zoomHorizontalAmount * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    break;
                case "C":
                    ctx.fillStyle = `purple`;
                    px = drawSx + (- pos - distStart- startDraw + cavityLength * (r + 2)) / distStep * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    px = drawSx + (- pos + elements[iEl].par[1] - distStart - startDraw + cavityLength * (r + 2)) / distStep * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);
                    let spx = (elements[iEl].par[1] / (2 * crystalNumberOfLenses));
                    ctx.strokeStyle = `purple`;
                    ctx.setLineDash([5, 3]);
                    ctx.beginPath();
                    for (let iL = 0; iL < crystalNumberOfLenses; iL++) {
                        px = drawSx + (- pos + (1 + 2 * iL) * spx - distStart- startDraw + cavityLength * (r + 2)) / distStep * drawW ;
                        ctx.moveTo(px, drawMid - 80 * zoomFactor);
                        ctx.lineTo(px, drawMid + 80 * zoomFactor);
                    }
                    ctx.stroke();
                    break;
                case "X":
                    ctx.fillStyle = `blue`;
                    //px = drawSx + (elements[iEl].par[0] - distStart) / distStep * drawW ;
                    px = drawSx + (- elements[iEl].par[0] - distStart - zoomHorizontalShift - startDraw + cavityLength * (r + 2)) / distStep * zoomHorizontalAmount * drawW ;
                    ctx.fillRect(px, drawMid - 80 * zoomFactor, 2, 160 * zoomFactor);          
                    break;
            }
        }
    }
}

function vecDeriv(v, dx = 1) {
    let vd = math.clone(v);
    vd[0] = 0;
    vd[1] = 0;
    for (let i = 2; i < v.length; i++) {
        vd[i] = (v[i] - v[i - 1]) / dx;
    }
    return vd;
}

function vecDeriv2(v, dx = 1, n = 5) {
    let vd = math.clone(v);
    let l = v.length;

    for (let i = 2; i < l - 2; i++) {
        vd[i] = (- (v[i - 2] + v[i + 2])
                + 16.0 * (v[i - 1] + v[i + 1])
                - 30.0 * v[i]) / (12.0 * dx * dx);
    }
    vd[0] = vd[1] = v[2];
    vd[l - 1] = vd[l - 2] = v[l - 3];
    
    return vd;
}

function vecWaistFromQ(v) {
    let vw = math.clone(v);
    for (let i = 0; i < v.length; i++) {
        vw[i] = Math.sqrt(- lambda / (Math.PI * (math.divide(1, v[i]).im)));
    }
    return vw;
}

var drawVectorComparePrevious = [];

function drawVectorPar(v, id, params) {
    if (!drawOption) {
        return
    }
    // let clear = params.params.clear || true;
    // let color = params.color || "red";
    // let pixelWidth = params.pixelWidth || drawW;
    // let allowChange = params.allowChange || false;
    // let name = params.name || "";
    // let start = params.start || drawSx;
    // let message = params.message || "";
    // let zoomX = params.zoomX || 1;
    // let backColor = params.backColor || "white";


    let clear = params.hasOwnProperty("clear") ? params.clear : true;
    let color = params.hasOwnProperty("color") ? params.color : "red";
    let pixelWidth = params.hasOwnProperty("pixelWidth") ? params.pixelWidth : drawW;
    let allowChange = params.hasOwnProperty("allowChange") ? params.allowChange : false;
    let name = params.hasOwnProperty("name") ? params.name : "";
    let start = params.hasOwnProperty("start") ? params.start : drawSx;
    let message = params.hasOwnProperty("message") ? params.message : "";
    let zoomX = params.hasOwnProperty("zoomX") ? params.zoomX : 1;
    let backColor = params.hasOwnProperty("backColor") ? params.backColor : "white";

    drawVector(v, clear, color, pixelWidth, allowChange, id, name, start, message, zoomX, backColor);
}

function calcDegauss(vec) {
    let mx = Math.max(...vec);
    return vec.map(v => {
        if (v > 0) {
            return Math.sqrt(- Math.log(v / mx));
        } else {
            return 0.0;
        }
    });
}
function calcDeSech(vec) {
    let mx = Math.max(...vec);
    return vec.map(v => {
        if (v > 0) {
            return - Math.log(v / mx);
        } else {
            return 0.0;
        }
    });
}
function calcDeLorentz(vec) {
    let mx = Math.max(...vec);
    return vec.map(v => {
        if (v > 0) {
            return Math.sqrt(mx / v - 1);
        } else {
            return 0.0;
        }
    });
}
function drawVector(v, clear = true, color = "red", pixelWidth = drawW, allowChange = false, 
    id = "graphCanvas", name = "", start = drawSx, message = "", zoomX = 1, backColor = "white") {
    if (!drawOption) {
        return
    }

    let degaussVal = parseInt(document.getElementById(`${id}-degaussVal`).innerHTML);
    let deLorentzVal = parseInt(document.getElementById(`${id}-delorentzVal`).innerHTML);
    let deSechVal = parseInt(document.getElementById(`${id}-desechVal`).innerHTML);

    let vectors = presentedVectors.get(id);
    if (clear || vectors == null) {
        vectors = [];
        presentedVectors.set(id, vectors);
    }

    let l = v.length;
    let fac;
    const prevCompare = document.getElementById('cbxPrevCompare')?.checked;
    if (l > 0) {
        fac = Math.max(Math.abs(Math.max(...v)), Math.abs(Math.min(...v)));
        let vecObj = {vecOrig: math.clone(v), w: pixelWidth, ch:allowChange,  s: start, c: color, n: name, f: fac, m: message, z: zoomX};
        vectors.push(vecObj);
        presentedVectors.set(id, vectors);
    } else {
        fac = 0;
    }
    vectors.forEach((vv) => {
        if (degaussVal == 1) {
            vv.vec = calcDegauss(vv.vecOrig);
        } else if (deLorentzVal == 1) {
            vv.vec = calcDeLorentz(vv.vecOrig);
        } else if (deSechVal == 1) {
            vv.vec = calcDeSech(vv.vecOrig);
        } else {
            vv.vec = vv.vecOrig;
        }
        v = vv.vec;
        vv.f = Math.max(Math.abs(Math.max(...v)), Math.abs(Math.min(...v)));
    });

    l = vectors.reduce((p, c) => Math.max(p, c.vec.length), 0);
    start = vectors.reduce((p, c) => Math.max(p, c.s), 0);
    let change = vectors.reduce((p, c) => p || c.ch, false);
    pixelWidth = vectors.reduce((p, c) => Math.max(p, c.w), 0);

    const canvas = document.getElementById(id);
    const ctx = canvas.getContext("2d");
    if (change && pixelWidth > (canvas.width - start) / l) {
        pixelWidth = (canvas.width - start) / l;
    }

    ctx.fillStyle = backColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);     
    if (isMouseDownOnGraph) {
        const canvas = document.getElementById(id);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ddd";
        ctx.fillRect(start + mouseOnGraphStart * drawW, 0, (mouseOnGraphEnd - mouseOnGraphStart) * drawW, 200)
    }
    ctx.strokeStyle = `gray`;
    ctx.beginPath();
    ctx.moveTo(start, 100);
    ctx.lineTo(start + l * pixelWidth, 100);
    ctx.stroke();

    let selectVal = parseInt(document.getElementById(`${id}-selectVal`).innerHTML);
    fac = vectors.filter((v, iv) => (selectVal == 0 || iv < selectVal)).reduce((p, c) => Math.max(p, c.f), 0.001);
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
    let zoomVal = parseFloat(document.getElementById(`${id}-zoomVal`).innerHTML);
    fac *= zoomVal;
    vectors.forEach((vo, iVec) => {
        if (selectVal == 0 || iVec < selectVal) {
            let sx = vo.s - pixelWidth * vo.z * vo.vec.length / 2 + canvas.width / 2;
            let dx = pixelWidth * vo.z;
            ctx.strokeStyle = vo.c;
            ctx.beginPath();
            ctx.moveTo(sx, 100 - Math.floor(fac * vo.vec[0]));
            for (let i = 1; i < l; i++) {
                ctx.lineTo(sx + i * dx, 100 - Math.floor(fac * vo.vec[i]));
            }
            ctx.stroke();

            drawTextBG(ctx, `${vo.n}`, 20, 120 + (iVec + 1) * 16, vo.c);
        }
    });
    document.getElementById(`${id}-message`).innerHTML = 
            vectors.map((c) => `<span style="color:${c.c}">${c.m}</span>`).filter((c) => c.length > 0).join("</br>");
    if (prevCompare) {
        ctx.strokeStyle = 'green';
        ctx.beginPath();
        ctx.moveTo(start, 100 - Math.floor(fac * drawVectorComparePrevious[0]));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(start + i * pixelWidth, 100 - Math.floor(fac * drawVectorComparePrevious[i]));
        }
        ctx.stroke();
        ctx.strokeStyle = 'blue';
        ctx.beginPath();
        ctx.moveTo(start, 100 - Math.floor(fac * (v[0] - drawVectorComparePrevious[0])));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(start + i * pixelWidth, 100 - Math.floor(fac * (v[i] - drawVectorComparePrevious[i])));
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
            ctx.fillText(math.re(A).toFixed(6), px, 40);
            ctx.fillText(math.re(B).toFixed(6), px + dx, 40);
            ctx.fillText(math.re(C).toFixed(6), px, 60);
            ctx.fillText(math.re(D).toFixed(6), px + dx, 60);
            newRange = calcNewRange(B, lambda, nSamples, newRange);
            ctx.fillText(newRange.toFixed(6), px, 90);
            if (iMat == 0) {
                ctx.fillStyle = 'blue';
                newRange = ranges[0];
                px += 160 * (mats.length - 1)
            } else {
                mMat = MMult(mats[iMat], mMat);
                let [A, B, C, D] = [mMat[0][0], mMat[0][1], mMat[1][0], mMat[1][1]];
                ctx.fillText(math.re(A).toFixed(6), px, 120);
                ctx.fillText(math.re(B).toFixed(6), px + dx, 120);
                ctx.fillText(math.re(C).toFixed(6), px, 140);
                ctx.fillText(math.re(D).toFixed(6), px + dx, 140);
                px -= 160;
            }
        }

    }
}

function zoomGraph(id, change) {
    val = parseFloat(document.getElementById(`${id}-zoomVal`).innerHTML);
    if (change > 0) {
        val *= 2;
    } else {
        val *= 0.5;
    }
    document.getElementById(`${id}-zoomVal`).innerHTML = val.toFixed(4);
    drawVectorPar([], id, {clear: false, allowChange: true, start: 0});

}

function selectGraph(id) {
    val = parseInt(document.getElementById(`${id}-selectVal`).innerHTML);
    document.getElementById(`${id}-selectVal`).innerHTML = `${1 - val}`;
    drawVectorPar([], id, {clear: false, allowChange: true, start: 0});
}
function degaussGraph(id) {
    val = parseInt(document.getElementById(`${id}-degaussVal`).innerHTML);
    document.getElementById(`${id}-degaussVal`).innerHTML = `${1 - val}`;
    document.getElementById(`${id}-delorentzVal`).innerHTML = `0`;
    document.getElementById(`${id}-desechVal`).innerHTML = `0`;
    drawVectorPar([], id, {clear: false, allowChange: true, start: 0});
}
function delorentzGraph(id) {
    val = parseInt(document.getElementById(`${id}-delorentzVal`).innerHTML);
    document.getElementById(`${id}-delorentzVal`).innerHTML = `${1 - val}`;
    document.getElementById(`${id}-degaussVal`).innerHTML = `0`;
    document.getElementById(`${id}-desechVal`).innerHTML = `0`;
    drawVectorPar([], id, {clear: false, allowChange: true, start: 0});
}
function desechGraph(id) {
    val = parseInt(document.getElementById(`${id}-desechVal`).innerHTML);
    document.getElementById(`${id}-desechVal`).innerHTML = `${1 - val}`;
    document.getElementById(`${id}-degaussVal`).innerHTML = `0`;
    document.getElementById(`${id}-delorentzVal`).innerHTML = `0`;
    drawVectorPar([], id, {clear: false, allowChange: true, start: 0});
}

function drawGraph() {
    if (!drawOption) {
        return
    }
    const sel = document.getElementById("displayOption");
    if (sel == null) {
        return;
    }

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
    if (document.getElementById("pickEl_text") != null) {
        elements = initElementsFromCavityText(document.getElementById("pickEl_text").value);
        return;
    }
    do {
        let valOther = 0;
        let elementTypeControl = document.getElementById(`type${iEl}`);
        if (elementTypeControl == null) {
            break;
        }
        let elementType = elementTypeControl.innerHTML[0];
        let elDelta = getFieldFloat(`el${iEl}delta`);
        let valDist = getFieldFloat(`el${iEl}dist`);
        if (elementType == "L") {
            valOther = getFieldFloat(`el${iEl}focal`);
            if (isNaN(valDist) || isNaN(valOther)) {
                break;
            }
        } else if (elementType == "C") {
            valOther = getFieldFloat(`el${iEl}length`);
            if (isNaN(valDist) || isNaN(valOther)) {
                break;
            }
        } else {
            valOther = -1;
        }

        elements.push({t: elementType, par: [valDist, valOther], delta: elDelta});
        iEl++;
    } while(true);
}

function extractLength(str) {
    if (str.toUpperCase().endsWith("MM")) {
        return {val: parseFloat(str.slice(0, -2)) / 1000, units: str.slice(-2)};
    } else if (str.toUpperCase().endsWith("CM")) {
        return  {val: parseFloat(str.slice(0, -2)) / 100, units: str.slice(-2)};
    } else if (str.toUpperCase().endsWith("M")) {
        return  {val: parseFloat(str.slice(0, -1)), units: str.slice(-1)};
    } else {
        throw new Error(`Unknown unit in '${str}'`);
    }
}

function buildLength(lengthObj) {
    let val = 0.0;
    switch (lengthObj.units.toUpperCase()) {
        case "MM":
            val = lengthObj.val * 1000;
            break;
        case "CM":
            val = lengthObj.val * 100;
            break;
        case "M":
            val = lengthObj.val;
            break;
        default:
            throw new Error(`Unknown unit in '${lengthObj.units}'`);
    }
    valStr = val.toFixed(5);
    while (valStr.endsWith("0")) {
        valStr = valStr.slice(0, -1);
    }
    if (valStr.endsWith(".")) {
        valStr = valStr.slice(0, -1);
    }

    return `${valStr}${lengthObj.units}`;
}

function initElementsFromCavityText(text) {
    let elementsT = [];
    let lines = text.split("\n");
    let position = 0.0;
    lines.forEach((line) => {
        let el = line.toUpperCase().split(" ");
        if (el[0].startsWith(">")) {
            el[0] = el[0].slice(1);
        }

        if (el.length >= 1) {
            try {
            switch (el[0]) {
                case "P":
                    position += extractLength(el[1]).val;
                    break;
                case "L":
                    elementsT.push({t: "L", par: [position, extractLength(el[1]).val], delta: 0.0});
                    break;
                case "LC":
                    elementsT.push({t: "LC", par: [position, extractLength(el[1].val), extractLength(el[2].val)], delta: 0.0});
                    break;
                case "C":
                    elementsT.push({t: "C", par: [position, extractLength(el[1].val)], delta: 0.0});
                    break;
                case "E":
                    elementsT.push({t: "X", par: [position], delta: 0.0});
                    break;
                case "S":
                    break;
                case "D":
                    break;
            }
            } catch (e) {
                console.error(`Error parsing element in line '${line}': ${e.message}`);
            }
        }
    });

    return elementsT;
}

function focusOnCrystal() {
    elements.forEach((el, index) => {
        if (el.t == "C") {
            crystalLength = el.par[1];
            distStep = crystalLength / (2.0 * crystalNumberOfLenses);
            distStart = el.par[0] - distStep;
        }
    });
}

function initMultiMode(setWorkingTab = - 1, beamParam = - 1) {
    if (setWorkingTab > 0) {
        workingTab = setWorkingTab
    }
    switch (workingTab) {
        case 1: // multimode
            drawW = 3;
            distStart = 0.0;
            distStep = 0.003;
            break
        case 2: // crysyal
            drawW = 30;
            focusOnCrystal();
            break;
        case 3: // multitime
            drawW = 3;
            distStep = 0.006;
            break;
        case 4: //test
            drawW = 3;
            distStep = 0.006;
            break
    }

    displayTemp = [[], []];
    displayTempPrevious = [[], []];

    multiFronts[0] = [getInitFront(beamParam)];
    multiRanges[0] = [initialRange];
    waist0 = getFieldFloat("beamParam", 0.0005);
    gaussianFronts[0] = [math.complex(0.0, Math.pow(waist0, 2) * Math.PI / lambda)];
    multiFronts[1] = [];
    multiRanges[1] = [];
    sfs = 0;
    drawMode = 1;
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

function MMultV(...Ms) {
    return Ms.reduce((acc, v) => MMult(v, acc), [[1.0, 0.0], [0.0, 1.0]]);
}

function MMult(m, m2) {
    a = m[0][0] * m2[0][0] + m[0][1] * m2[1][0];
    b = m[0][0] * m2[0][1] + m[0][1] * m2[1][1];
    c = m[1][0] * m2[0][0] + m[1][1] * m2[1][0];
    d = m[1][0] * m2[0][1] + m[1][1] * m2[1][1];

    return  [[a, b], [c, d]];
}
function MMultInv(m, m2) {
    a =   m[0][0] * m2[1][1] - m[0][1] * m2[1][0];
    b = - m[0][0] * m2[0][1] + m[0][1] * m2[0][0];
    c =   m[1][0] * m2[1][1] - m[1][1] * m2[1][0];
    d = - m[1][0] * m2[0][1] + m[1][1] * m2[0][0];

    return  [[a, b], [c, d]];
}
function MInv(m) {
    return [[m[1][1], - m[0][1]], [- m[1][0], m[0][0]]];
}

let elements = [];
let distStep = 0.003;
var distStart = 0.0;


function getMatOnStep(dStep) {

    let iEl = 0;
    let prevLensPos = 0.0;
    let M = [[1, 0], [0, 1]];
    let rdStep = dStep;
    let globalDelta = elements.find((el) => el.t == "X").delta;

    while (iEl < elements.length) {
        let deltaFactor = elements[iEl].delta;
        let pos = elements[iEl].par[0] + deltaFactor * globalDelta
        if (rdStep < pos - prevLensPos) {
            break;
        }
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(pos - prevLensPos), M);
            M = MMult(MLens(elements[iEl].par[1]), M);
            rdStep -= pos - prevLensPos;
            prevLensPos = pos;
        }
        iEl++;
    }
    M = MMult(MDist(rdStep), M);
    return M;
}

let MLeft1, MLeft2, MRight1, MRight2;

function getMatOnRoundTrip(oneWay = false) {
    let iEl = 0;
    let prevLensPos = 0.0;
    let M = [[1, 0], [0, 1]];
    let globalDelta = elements.find((el) => el.t == "X").delta;
    let MMid; 

    while (iEl < elements.length) {
        let deltaFactor = elements[iEl].delta;
        let pos = elements[iEl].par[0] + deltaFactor * globalDelta
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(pos - prevLensPos), M);
            if (iEl == 1) {
                MLeft2 = math.clone(M);
            }
            M = MMult(MLens(elements[iEl].par[1]), M);
            if (iEl == 1) {
                MRight1 = math.clone(M);
            }
            prevLensPos = pos;
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

    MRight1 = MMultInv(M, MRight1);
    MMid = math.clone(M);

    if (oneWay) {
        return M;
    }

    while (iEl >= 0) {
        let deltaFactor = elements[iEl].delta;
        let pos = elements[iEl].par[0] + deltaFactor * globalDelta
        switch (elements[iEl].t) {
        case "L":
            M = MMult(MDist(prevLensPos - pos), M);
            if (iEl == 1) {
                MRight2 = MMultInv(M, MMid);
            }
            M = MMult(MLens(elements[iEl].par[1]), M);
            if (iEl == 1) {
                MMid = math.clone(M);
            }
            prevLensPos = pos;
            break;
        }
        iEl--;
    }    
    M = MMult(MDist(prevLensPos), M);
    MLeft1 = MMultInv(M, MMid);

    return M;
}

function getMatDistanceForever(dist) {
    let M = [[1, 0], [0, 1]];
    let globalDelta = elements.find((el) => el.t == "X").delta;

    while (true) {
        let iEl = 0;
        let prevLensPos = 0.0, pos;
            while (iEl < elements.length) {
            switch (elements[iEl].t) {
            case "L":
                pos = elements[iEl].par[0] + elements[iEl].delta * globalDelta
                if (dist < pos - prevLensPos) {
                    M = MMult(MDist(dist), M);
                    return M;
                }
                M = MMult(MDist(pos - prevLensPos), M);
                M = MMult(MLens(elements[iEl].par[1]), M);
                dist -= pos - prevLensPos;
                prevLensPos = pos;
                break;
            case "X": // end wall
                pos = elements[iEl].par[0];
                if (dist < pos - prevLensPos) {
                    M = MMult(MDist(dist), M);
                    AbcdMatStore(M[0][0], M[0][1], M[1][0], M[1][1]);
                    return M;
                }
                M = MMult(MDist(pos - prevLensPos), M);
                dist -= pos - prevLensPos;
                prevLensPos = pos;
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
                pos = elements[iEl].par[0] + elements[iEl].delta * globalDelta
                if (dist < prevLensPos - pos) {
                    M = MMult(MDist(dist), M);
                    return M;
                }
                M = MMult(MDist(prevLensPos - pos), M);
                M = MMult(MLens(elements[iEl].par[1]), M);
                dist -= prevLensPos - pos;
                prevLensPos = pos;
                break;
            }
            iEl--;
        }
        if (dist < prevLensPos) {
            M = MMult(MDist(dist), M);
            return M;
        }
        dist -= prevLensPos;
        M = MMult(MDist(prevLensPos), M);
    }
}

function calcWidth(v) {
    let l = v.length;
    let N = 0; sumX = 0; sumX2 = 0;
    let va = math.abs(v)
    for (let i = 0; i < l; i++) {
        let val = va[i];
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

function getMatricesAtDistFromStart(M, dStep, r0, mode) {
    let L = nSamples;
    let mats = [], isBack = [];
    let spDist = r0 * r0 / (L * lambda);

    mats.push(math.clone(M));
    isBack.push(false);

    if (mode == 1) {
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
                mPush = [[1, spDist], [0, 1]];
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
    } else {
        mats.push(math.clone(M));
        isBack.push(false);
    }

    return [mats, isBack];
}

function getDistanceOfStep(iStep) {

    return iStep * distStep / zoomHorizontalAmount + zoomHorizontalShift;
    
}

function racalcHorizontalZoom() {
    if (drawMode == 1) {
        fullCavityMultiMode();
    }
}

function fullCavityNewParams(delta, focal, waist) {
    console.log(` ======== delta = ${delta}, focal = ${focal}, waist = ${waist}`)
    setFieldFloat("el1focal", focal);
    setFieldFloat("el3delta", delta);
    if (waist > 0) {
        setFieldFloat("beamParam", waist);
    }

    initElementsMultiMode();
    initMultiMode(4);

    fullCavityMultiMode();
}

function fullCavityMultiMode(mode = 1, startDist = 0.0) {
    drawMode = 1;
    
    let fronts = multiFronts[0];
    let ranges = multiRanges[0];

    if (fronts.length <= 0) {
        return;
    }
    
    let MS0, MStartDistInv = [[1, 0], [0, 1]];
    if (startDist > 0.000001) {
        MStartDistInv = MInv(getMatDistanceForever(startDist));
        MS0 = MMult(getMatDistanceForever(startDist + 2 * 0.982318181), MStartDistInv);
        console.log("MS0 ", MS0)
    }
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0.0005], vecQ[0] = math.complex(0, RayleighRange), vecMats = [];
    
    for (let iStep = 1; iStep < 400; iStep++) {
        let f0 = math.clone(fronts[0]);
        let r0 = ranges[0];
        let L = f0.length;
        let dx0 = r0 / L;
        let dStep = getDistanceOfStep(iStep) + startDist;
        if (dStep < 0) {
            fronts.push(fronts[0].map((x) => math.complex(0.0)));
            ranges.push(r0);
            continue;
        }

        let MS; 
        // if (iStep < 15) {
        //     MS = MMult(getMatDistanceForever(startDist), MStartDistInv);
        // } else if (iStep < 30) {
        //     MS = math.clone(MS0);
        // } else {
            MS = MMult(getMatDistanceForever(dStep), MStartDistInv);
        // }

        let [mats, isBack] = getMatricesAtDistFromStart(MS, dStep, r0, mode);

        let M  = mats[0];
        let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        let newQ = math.chain(vecQ[0]).multiply(A).add(B).divide(math.chain(vecQ[0]).multiply(C).add(D).done()).done();

        let dx = dx0;
        let fx = math.clone(fronts[0]);

        if (mode == 1) {
            for (let iMat = 1; iMat < Math.min(mats.length, nMaxMatrices + 1); iMat++) {
                //console.log(`===== MM ${mats[iMat][0][0]},${mats[iMat][0][1]},${mats[iMat][1][0]},${mats[iMat][1][1]}, dx = ${dx}`);
                [ff, dx] = CalcNextFrontOfM(fx, L, mats[iMat], dx, isBack[iMat]);
                fx = math.clone(ff);
            }
            let width = calcWidth(fx);
            if (width < 0.0000001) {
                break;
            }
            vecQ.push(newQ);
            vecW.push(width * Math.abs(dx));
            vecWaist.push(width * Math.abs(dx) * 1.41421356237);
        } else {
            vecQ.push(newQ);
            let waist = Math.sqrt(- lambda / (Math.PI * (math.divide(1, newQ).im)));
            dx = ranges[0] / L / vecWaist[0] * waist
            vecW.push(waist / 1.41421356237);
            vecWaist.push(waist);
        }
        ff = math.clone(fx);    
        vecMats.push(mats);
        let dxf = dx;

        fronts.push(ff);
        ranges.push(L * dxf);
    }

    drawMultiMode();
}

function fullCavityGaussian(startDist = 0.0) {
    drawMode = 1;
    
    let fronts = gaussianFronts[0];

    if (fronts.length <= 0) {
        return;
    }
    
    let MS0, MStartDistInv = [[1, 0], [0, 1]];
    if (startDist > 0.000001) {
        MStartDistInv = MInv(getMatDistanceForever(startDist));
        MS0 = MMult(getMatDistanceForever(startDist + 2 * 0.982318181), MStartDistInv);
        console.log("MS0 ", MS0)
    }
    vecA = [0]; vecB = [0]; vecC = [0]; vecD = [0]; vecW = [0], vecWaist = [0], vecQ[0] = math.complex(0, RayleighRange), vecMats = [];
    
    for (let iStep = 1; iStep < 400; iStep++) {
        let f0 = fronts[0];

        let dStep = getDistanceOfStep(iStep) + startDist;
        if (dStep < 0) {
            fronts.push(math.complex(0.0));
            continue;
        }

        let M = MMult(getMatDistanceForever(dStep), MStartDistInv);

        //let [mats, isBack] = getMatricesAtDistFromStart(MS, dStep, r0);

        let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
        vecA.push(M[0][0]);
        vecB.push(M[0][1]);
        vecC.push(M[1][0]);
        vecD.push(M[1][1]);
        let newQ = math.chain(f0).multiply(A).add(B).divide(math.chain(f0).multiply(C).add(D).done()).done();
        fronts.push(newQ);

        vecMats.push([M]);

        vecQ.push(newQ);
        let waist = Math.sqrt(- lambda / (Math.PI * (math.divide(1, newQ).im)));
        if (waist < 0.0000001) {
            break;
        }
        vecW.push(waist / 1.41421356237);
        vecWaist.push(waist);
    }

    AbcdMatPaste("Total");
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
    let kerrPar =  4 * crystalLength * n2;
    let Ikl = kerrPar / 5 / 50;
    let M, imagA;
    let MatSide = calcOriginalSimMatricesWithoutCrystal(crystalLength);
                //[[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  // right
                // [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]];   // left
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
    
        for (let iStep = 1; iStep < 2 + 2 * crystalNumberOfLenses; iStep++) {
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

    let [stable, lambda1, lambda2, beamWaist, beamDist] = analyzeStability(MatTotal);

    if (stable) {
        setFieldFloat('beamDist', beamDist);
        setFieldFloat('beamParam', beamWaist);
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

function analyzeStability(MatTotal) {
    let beamWaist, beamDist;
    let [[A, B], [C, D]] = MatTotal;
    let [lambda1, lambda2] = calculateEigenvalues(A, B, C, D);

    try {
        // console.log(`eigen values = [${lambda1}, ${lambda2}] mult=${math.multiply(lambda1, lambda2)}`);
        // console.log(`abs = [${math.abs(lambda1)}, ${math.abs(lambda2)}] `);
        // console.log(`|A+D| = ${math.abs(math.add(MatTotal[0][0], MatTotal[1][1]))}, det = ${math.det(MatTotal)}`)
        // console.log(`ABCD = ${MatTotal[0][0]} ${MatTotal[0][1]} ${MatTotal[1][0]} ${MatTotal[1][1]}`);
        if (math.abs(math.add(MatTotal[0][0], MatTotal[1][1])) > 2.0) {
            return [false, lambda1, lambda2];
        }
        let oneOverQ = math.complex(math.divide((math.subtract(D,  A)), (2 * B)), 
                math.divide(math.sqrt(math.subtract(1, math.multiply(0.25, math.multiply(math.add(A, D), math.add(A, D))),)), B));
        let actualQ = math.divide(1, oneOverQ);
        let nextQ = math.divide(math.add(math.multiply(actualQ, A), B), math.add(math.multiply(actualQ, C), D));
        let diffQ = math.subtract(actualQ, nextQ);
        // console.log(`A = ${A}, B = ${B}, C = ${C}, D = ${D}`);
        // console.log(`SOL1 => Q = ${actualQ} 1/Q = ${oneOverQ} nextQ = ${nextQ} diff = ${math.abs(diffQ)}`);
        
        beamDist = 2.0 * B / (D - A);
        beamWaist = Math.sqrt(lambda * Math.abs(B) / (Math.PI * Math.sqrt(1 - 0.25 * (A + D) * (A + D))));
        // console.log(`ORIG beamWaist = ${saveBeamParam}, beamDist = ${saveBeamDist}`);
        // console.log(`beamWaist = ${beamWaist}, beamDist = ${beamDist}, z = ${actualQ.re}`);

        return [true, lambda1, lambda2, beamWaist, beamDist];
    } catch(er) {
        return [false, lambda1, lambda2];
    }
}

function  normalizePower(fx, power, dxf) {
    const p = math.sum(math.dotMultiply(fx, math.conj(fx))) / nSamples;
    console.log(`dot ${p}`)
    fx = math.multiply(fx, 1 /  Math.sqrt(p));
    return power * p; 
}

function vectorsForFresnel(M, N, dx0, gain, isBack) {
    let [[A, B], [C, D]] = M;
    let dxf = B * lambda / (N * dx0);
    let factor = math.multiply(math.complex(0, gain), math.sqrt(math.complex(0, - 1 / (B * lambda))));
    if (isBack) {
        factor = math.multiply(factor, math.complex(0, gain));
    }    
    let co0 = - Math.PI * dx0 * dx0 * A / (B * lambda);
    let cof = - Math.PI * dxf * dxf * D / (B * lambda);

    let vec0 = [], vecF = []
    for (let i = 0; i < N; i++) {
        let ii = i - N / 2;
        vec0.push(math.exp(math.complex(0, co0 * ii * ii)));
        vecF.push(math.dotMultiply(factor, math.exp(math.complex(0, cof * ii * ii))));
    }

    return {dx: dx0, vecs: [vec0, vecF]};
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
    totalRightSide = MMult(MRight2, MRight1);
    totalLeftSide = MMult(MLeft2, MLeft1);
    console.log(`RIGHT [[${totalRightSide[0][0]}, ${totalRightSide[0][1]} ], [${totalRightSide[1][0]}, ${totalRightSide[1][1]} ]]`);
    console.log(`LEFT [[${totalLeftSide[0][0]}, ${totalLeftSide[0][1]} ], [${totalLeftSide[1][0]}, ${totalLeftSide[1][1]} ]]`);
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

    let xEl = elements.find((el) => el.t == "X");
    let origValue = xEl.delta;

    xEl.delta = delta;
    
    initMultiMode(1, waist);

    let M = getMatOnRoundTrip(false);
    let A = M[0][0];
    let B = M[0][1];
    let D = M[1][1];

    autoRangeMultiMode(M);
    console.log(`delta ${delta} A+D=${A+D}`);

    if (Math.abs(A + D) > 2.0 - 0.00001) {
        xEl.delta = origValue;
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

            xEl.delta = origValue;
            return (waist);
        }

    }

    xEl.delta = origValue;
    return (waist);
}

var coverWaist;
function doDeltaStepCover(delta) {
    drawOption = false;
    coverWaist = doDeltaStep(delta, coverWaist);

    delta += 0.0004;
    if (delta < 0.085) {
        setTimeout(doDeltaStepCover, 1, delta);
    } else {
        drawOption = true;
        drawMode = 3;
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
    deltaMinX = 0;
    deltaMaxX = 1;
    deltaMinY = 0;
    deltaMaxY = 1;

    initMultiMode(1, coverWaist);

    let delta = 0.001;

    doDeltaStepCover(delta);

    drawMode = 3;
    drawOption = true;
    drawMultiMode();
}

function deltaCanvasMouseMove(e) {
    if (deltaGraphX.length < 5) {
        return;
    }
    let canvas = document.getElementById(e.target.id);
    let ctx = canvas.getContext("2d");

    const gWidth = ctx.canvas.width - 200;
    const gHeight = ctx.canvas.height - 200;

    let [x, y] = getClientCoordinates(e);

    const zoomY = gHeight / (deltaMaxY - deltaMinY);
    const zoomX = gWidth / (deltaMaxX - deltaMinX);

    let [gx, gy] = fromCtxCoords(ctx, [x, y], [zoomX, zoomY], [deltaMinX, deltaMinY], [drawSx, 0]);

    drawTextBG(ctx, gx.toFixed(6), ctx.canvas.width - 100, 10);
    drawTextBG(ctx, gy.toFixed(6), ctx.canvas.width - 100, 30);
}

function fetchGraphData(sample, x, y) {
    fetch(`/mmGraph/${sample}/${x}/${y}`, {
        method: 'POST',
        headers: {
           'Accept': 'application/json',
           'Content-Type': 'application/json'
        },
        body: JSON.stringify({localId: getLocalId()})
     })
     .then(resp => resp.json()) // or, resp.text(), etc.
     .then(data => {
        spreadUpdatedData(data);
     })
     .catch(error => {
        console.error(error);
     });
}

function mainCanvasMouseMove(e) {
    if (drawMode == 3) {
        return deltaCanvasMouseMove(e)
    }
    if (drawMode == 4) {
        return stabilityCanvasMouseMove(e)
    }
    if (!isMouseDownOnMain) {
        return;
    }
    const id = e.target.id;
    let fronts = []
    if (id == "funCanvas1" || id == "funCanvasTest") {
        fronts = multiFronts[0];
    } else if (id == "funCanvas2") {
        fronts = multiFronts[1];
    } else if (id == "funCanvasTime" || id == "funCanvasFrequency") {
        multiTimeCanvasMouseMove(e);
        return;
    } else if (id == "funCanvasSample1top") {
        let [x, y] = getClientCoordinates(e);
        fetchGraphData(0, x, y);
        return;
    } else if (id == "funCanvasSample2top") {
        let [x, y] = getClientCoordinates(e);
        fetchGraphData(1, x, y);
        return;
    } else {
        return;
    }
    const sel = document.getElementById("displayOption");
    let [x, y] = getClientCoordinates(e);


    //fronts = multiFronts[id - 1];

    let ix = Math.floor((x - drawSx) / drawW);
    if (ix >= 0 && ix < fronts.length) {
        if (sel.value == "AbsE(x)") {
            fi = fronts[ix];
            fr = [];
            for (let i = 0; i < fi.length; i++) {
                fr.push(fi[i].toPolar().r);
            }
            drawVectorPar(fr, "graphCanvas", {pixelWidth: 1, allowChange: true});
 
        } else if (sel.value == "ArgE(x)") {
            fi = fronts[ix];
            fr = [];
            for (let i = 0; i < fi.length; i++) {
                fr.push(fi[i].toPolar().phi);
            }
            drawVectorPar(fr, "graphCanvas", {color: "purple", pixelWidth: 1, allowChange: true});

        } else if (sel.value == "M(x)") {
            drawMatDecomposition(ix);
        }
    }

}
function mainCanvasMouseDown(e) {
    isMouseDownOnMain = true;
    mainCanvasMouseMove(e);
}

function mainCanvasMouseUp(e) {
    isMouseDownOnMain = false;
    const id = e.target.id;

    if (id != "funCanvasSample1top" && id != "funCanvasSample2top") {
        moverHide();

        let [x, y] = getClientCoordinates(e);
        if (zoomHorizontalCenter != (x - drawSx) / ( drawW / distStep)) {
            zoomHorizontalCenter = (x - drawSx) / ( drawW / distStep);
            drawMultiMode();
        }
    }

    if (id == "funCanvasTime" || id == "funCanvasFrequency") {
        multiTimeCanvasMouseMove(e, true);
    }

}

function graphCanvasMouseMove(e) {
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;

    let vectors = presentedVectors.get(e.target.id);
    if (vectors == null || vectors.lenght < 1) {
        return;
    }
    const canvas = document.getElementById(e.target.id);
    const ctx = canvas.getContext("2d");
    let vecObj = vectors[0];
    let ix = Math.floor((x - (vecObj.s - vecObj.vec.length / 2 * vecObj.w * vecObj.z + canvas.width / 2)) / (vecObj.w * vecObj.z));

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
        
        drawTextBG(ctx, (ix * vecObj.w).toFixed(2), 20, 120);
        for (let iVec = 0; iVec < vectors.length; iVec++) {
            let vecObj = vectors[iVec];
            if (vecObj.vec.length > ix) {
                let val = vecObj.vec[ix];
                drawTextBG(ctx, `${vecObj.n} ${val.toFixed(8)}`, 20, 120 + (iVec + 1) * 16, vecObj.c);
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

function saveMultiTimeParametersProcess() {
    document.getElementById("saveParametersDialog").style.visibility = "visible";
}
function restoreMultiTimeParametersProcess() {
    let list = document.getElementById("restoreParametersList");
    [...list.children].forEach(c => c.remove());
    let namedObjects = loadSafeNamedObjects();
    exportElement = document.getElementById("exportParametersListArea");
    if (exportElement) {
        exportElement.parentElement.removeChild(exportElement);
    }
    for (let iObj in namedObjects) {
        let child = document.createElement("div");
        let text = document.createElement("span");
        text.innerText = `-> ${namedObjects[iObj].name}`;
        text.setAttribute("onclick",`restoreMultiTimeParameters(${iObj})`);
        child.appendChild(text);
        let img = document.createElement("img");
        img.src = "static/delete.png";
        img.style.paddingLeft = "20px";
        img.style.verticalAlign = "middle";
        img.setAttribute("onclick",`deleteMultiTimeParameters(${iObj})`);
        child.appendChild(img);
        list.appendChild(child);
    }
    document.getElementById("restoreParametersDialog").style.visibility = "visible";
}

function saveMultiTimeParameters(isSave) {
    if (isSave) {
        let obj = { name: document.getElementById("parametersName").value};
        let form = document.getElementById("multiTimeOptionsForm");
        for (let item of form.children) {
            if (item instanceof HTMLInputElement) {
                obj[item.id] = item.value;
            } else if (item instanceof HTMLSelectElement) {
                obj[item.id] = item.selectedIndex;
            }  else if (item instanceof HTMLTextAreaElement) {
                console.log(item.id, "TEXTAREA")
            }  else {
                console.log(item.id, "ERROR")
            }
        }
        addNamedObject(obj);
    }
    document.getElementById("saveParametersDialog").style.visibility = "hidden";
}

function restoreMultiTimeParameters(index) {
    if (index >= 0) { 
        let obj = getNamedObjectByIndex(index);
        console.log(obj)
        let form = document.getElementById("multiTimeOptionsForm");
        for (let item of form.children) {
            if (obj.hasOwnProperty(item.id)) {
                if (item instanceof HTMLInputElement) {
                    item.value = obj[item.id];
                } else if (item instanceof HTMLSelectElement) {
                    item.selectedIndex = obj[item.id];
                }  else if (item instanceof HTMLTextAreaElement) {
                    console.log(item.id, "TEXTAREA")
                }  else {
                    console.log(item.id, "ERROR")
                }
            }
        }
    }
    document.getElementById("restoreParametersDialog").style.visibility = "hidden";
}

function deleteMultiTimeParameters(index) {
    deleteNamedObjectByIndex(index);
    document.getElementById("restoreParametersDialog").style.visibility = "hidden";
}

function exportMultiTimeParameters() {
    const ser = localStorage.getItem("namedObjects");
    let copy = document.getElementById("copyParametersList");
    [...copy.children].forEach(c => c.remove());
    let area = document.createElement("textarea");
    area.value = ser;
    area.id = "exportParametersListArea";
    area.style.width = "500px";
    area.style.height = "400px";

    //area.setAttribute("onclick",`restoreMultiTimeParameters(${iName})`);
    copy.appendChild(area);
}

function importMultiTimeParameters() {
    let copy = document.getElementById("copyParametersList");
    [...copy.children].forEach(c => {
        objs = JSON.parse(c.value);
        mergeToNamedObject(objs);
    });

    document.getElementById("restoreParametersDialog").style.visibility = "hidden";
}
