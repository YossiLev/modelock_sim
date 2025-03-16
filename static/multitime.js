/*
v counter for number of steps
- side report on progress record power, lensing, A+D, etc.
v automatic cosideration of design data on screen.
- go back to calculation of full gaussian rather than pixels
- add prints for minimum maximum of graphs
- update all data including xVec yVec on every update
- fixing in the the frequancy domain
v remove phases that carry no intensity
- working in 2D (square the fine on width deviation)
- Fresnel in rings (Bessel)
- original sim with one lens only
- present focal length from 2nd derivative
-  
*/
var stepsCounter = 0;
var nTimeSamples = 1024;
var multiTimeFronts = [];
var multiTimeFrontsSaves = [[], [], [], [], [], []];
var multiFrequencyFronts = [];
var factorGain = [];
var IntensitySaturationLevel = 400000000000000.0;
var intensityTotalByIx = [];
var factorGainByIx = [];
var Ikl = 0.02;
let IklTimesI = math.complex(0, Ikl * 160 * 0.000000006);
var rangeW = [];
var spectralGain = [];
var dispersion = [];
var sumPowerIx = [];
var gainReduction = [];
var gainReductionWithOrigin = [];
var gainReductionAfterAperture = [];
var gainReductionAfterDiffraction = [];
var gainFactor = 0.50;
var dispersionFactor = 1.0;
var lensingFactor = 1.0;
var IsFactor = 200 * 352000;
var pumpGain0 = [];
var multiTimeAperture  = [];
var multiTimeApertureVal = 0.000056;
var multiTimeDiffraction  = [];
var multiTimeDiffractionVal = 0.000030;
var frequencyTotalMultFactor = [];
var mirrorLoss = 0.95;
var fresnelData = [];
var totalRange = 0.001000;
var dx0 = totalRange / nSamples;
var scalarOne = math.complex(1);
var nTimeSamplesOnes = Array.from({length: nTimeSamples}, (v) => scalarOne)
var nSamplesOnes = Array.from({length: nSamples}, (v) => scalarOne)
var nSamplesOnesR = Array.from({length: nSamples}, (v) => 1.0)
var kerrFocalLength = 0.0075;
var ps1 = [];
var ps2 = [];
var timeContentOption = 0;
var freqContentOption = 0;
var timeContentView = 0;
var freqContentView = 0;
var contentOptionVals = ["F", "1", "2", "3", "4", "5", "6", "M"];
var contentViewVals = ["Am", "ph"];
var matrices = [];

let MatSide = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  // right
                 [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]];   // left

// delta 0.0095 focal 0.0075
//let MatSide = [[[-0.2982666667, -0.006166766667], [147.7333333, -0.2982666667]],  // right
//                [[0.3933333333, -0.002881666667], [293.3333333, 0.3933333333]]];   // left

// delta 0.005  focal 20.00
//let MatSide = [[[-0.6266666667, -0.0040666666677], [149.3333333,-0.6266666667]],  // right
//                [[-0.2666666667, -0.003166666667], [293.3333333, -0.2666666667]]];   // left

function fixMat(M) {
    return MMult(MDist(0.00120), MMult(M, MDist(0.00150)))
}

function calcCurrentCavityMatrices() {
    initElementsMultiMode();
    getMatOnRoundTrip();
    totalRightSide = MMult(MRight2, MRight1);
    totalLeftSide = MMult(MLeft2, MLeft1);
    refreshCavityMatrices();
}

function calcOriginalSimMatrices() {
    //let positionLens = 0.0005;
    let positionLens = -0.00015;
    let MLong = MMultV(MDist(positionLens), MDist(0.081818181), MLens(0.075), MDist(0.9), 
                    MDist(0.9), MLens(0.075), MDist(0.081818181), MDist(positionLens));
    let MShort = MMultV(MDist(0.001 - positionLens), MDist(0.075), MLens(0.075), MDist(0.5), 
                    MDist(0.5), MLens(0.075), MDist(0.075), MDist(0.001 - positionLens));

    MatSide = [MShort, MLong];

    return MatSide
}

function calcOriginalSimMatricesWithoutCrystal(crystalLength) {
    let cMat = MDist(- 0.5 * crystalLength);
    let MS = calcOriginalSimMatrices()

    MatSide = MS.map((M) => MMultV(cMat, M, cMat));

    return MatSide
}

function refreshCavityMatrices() {
    MatSide = [math.clone(totalRightSide), math.clone(totalLeftSide)];
    console.log(MatSide);
}

function updateContentOptions() {
    let options = contentOptionVals.map((v, i) => {
        style=`"margin: 2px; padding: 2px; border: 1px solid black; border-radius: 2px; background:${i == timeContentOption ? "yellow": "white"};"`
        return `<div onclick="timeContentOption = ${i}; updateContentOptions(); drawMultiTime();" style=${style}>${v}</div>`
    }).join("");
    document.getElementById("TimeCanvasOptions").innerHTML = options;
    options = contentOptionVals.map((v, i) => {
        style=`"margin: 2px; padding: 2px; border: 1px solid black; border-radius: 2px; background:${i == freqContentOption ? "yellow": "white"};"`
        return `<div onclick="freqContentOption = ${i}; updateContentOptions(); drawMultiTime();" style=${style}>${v}</div>`
    }).join("");
    document.getElementById("FrequencyCanvasOptions").innerHTML = options;
    options = contentViewVals.map((v, i) => {
        style=`"margin: 2px; padding: 2px; border: 1px solid black; border-radius: 2px; background:${i == timeContentView ? "yellow": "white"};"`
        return `<div onclick="timeContentView = ${i}; updateContentOptions(); drawMultiTime();" style=${style}>${v}</div>`
    }).join("");
    document.getElementById("TimeCanvasViews").innerHTML = options;
    options = contentViewVals.map((v, i) => {
        style=`"margin: 2px; padding: 2px; border: 1px solid black; border-radius: 2px; background:${i == freqContentView ? "yellow": "white"};"`
        return `<div onclick="freqContentView = ${i}; updateContentOptions(); drawMultiTime();" style=${style}>${v}</div>`
    }).join("");
    document.getElementById("FrequencyCanvasViews").innerHTML = options;    
}

function initMultiTime() {
    workingTab = 3
    // 256 slices of 1024 length each slice
    multiTimeFronts = [];
    multiFrequencyFronts = [];
    multiTimeFrontsSaves = [[], [], [], [], [], []];
    updateStepsCounter(0);
    for (let i = 0; i < nSamples; i++) {
        multiTimeFronts.push([]);
        multiFrequencyFronts.push([]);
    }
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let rnd = math.complex((Math.random() * 2 - 1), (Math.random() * 2 - 1));
        let fr = math.multiply(rnd, getInitFront(beamParam));
        for (let i = 0; i < nSamples; i++) {
            multiTimeFronts[i].push(fr[i]);
            multiFrequencyFronts[i].push(math.complex(0));
        }
    }

    gainFactorChanged();
    isFactorChanged();
    nRoundsChanged();

    updateContentOptions();
 
    prepareLinearFresnelHelpData();

    prepareGainPump();
    initGainByFrequency();

    multiTimeApertureChanged();

    fftToFrequency();
    ifftToTime();

    drawMultiMode();
}

function initGainByFrequency() {
    // all the contribution to the frequency domain are collected together
    //
    // spectralGain[wi] = 1 / (1 + ((wi - n/2) / 200 )^2)
    //
    // dispersion[wi] = exp(-i * disp_par * wi^2)
    //
    //  
    let specGain = math.complex(200);
    let disp_par = dispersionFactor * 0.5e-3 * 2 * Math.PI / specGain;    
    rangeW = math.range(- nTimeSamples / 2 , nTimeSamples / 2).toArray().map((v) => math.complex(v + 0.0));
    let ones = rangeW.map((_) => math.complex(1.0));
    let mid = math.dotDivide(rangeW, rangeW.map((_) => specGain));
    spectralGain = math.dotDivide(ones, math.add(math.square(mid), 1));
    dispersion = math.exp(math.multiply(math.complex(0, - disp_par), math.square(rangeW)));
    let expW = math.exp(math.multiply(math.complex(0, - 2 * Math.PI), rangeW));
    frequencyTotalMultFactor = math.multiply(0.5, math.add(1.0, math.dotMultiply(expW, math.dotMultiply(spectralGain, dispersion))));
}

function multiTimeRoundTrip(iCount) {
    if (iCount % 10 == 0) {
        const endTime = performance.now()

        let fs = multiTimeFronts;
        let meanV, meanMean;
        fs = math.abs(fs);
        fs = math.dotMultiply(fs, fs);
        meanV = math.mean(fs, 0);
        meanMean = math.mean(meanV);

        console.log(`${iCount} - ${((endTime - startTime) * 0.001).toFixed(3)} mean=${meanMean}`);
    }

    [0, 1].forEach((side) => {
        phaseChangeDuringKerr(side);

        spectralGainDispersion();
        // gainByfrequency (V)
        // dispersionByFrequency (V)

        linearCavityOneSide(side);
        // gainCorrectionDueToSaturation
        // oneSideCavity
        // mirrorLoss (only on left side)
    });
    updateStepsCounter(1);
}
function updateStepsCounter(p) {
    if (p == 0) {
        stepsCounter = 0;
    } else {
        stepsCounter += p;
    }
    setFieldInt("stepsCounter", stepsCounter);
}
function coverRound(params) {
    if (params[0] <= 0) {
        drawMultiMode();
        const endTime = performance.now();
        console.log(`Call to full took ${((endTime - startTime)/1000).toFixed(3)} s`)
        return null;
    }
    multiTimeRoundTrip(1);
    if (params[0] % 2 == 0) {
        drawMultiMode();
    }

    if (params[0] <= 1) {
        drawMultiMode();
        const endTime = performance.now()
        console.log(`Call to full took ${((endTime - startTime)/1000).toFixed(3)} s`)
        return null;
    }

    return [params[0] - 1];
}

var startTime;
var startTimeFull;
function timeCavityStep(step, redraw) {
    startTime = performance.now()

    switch (step) {
        case 1: phaseChangeDuringKerr(); fftToFrequency(); break;
        case 2: spectralGainDispersion(); break;
        case 3: linearCavityOneSide(0); break;
        case 4: linearCavityOneSide(1); break;
        case 6: math.range(0, Math.pow(10, nRounds)).forEach((x)=> multiTimeRoundTrip(x)); break;
        case 5:
            startTimeFull = performance.now()
            doCover(coverRound, [Math.pow(10, nRounds)]);
            break;
    }
    const endTime = performance.now()

    if (redraw) {
        drawMultiMode();
    }
}
function fftToFrequency() {
    for (let ix = 0; ix < nSamples; ix++) {
        if (sumPowerIx[ix] > 0.000001) {
            multiFrequencyFronts[ix] = fft(multiTimeFronts[ix], 1.0);
        }
    }
}

function ifftToTime() {
    for (let ix = 0; ix < nSamples; ix++) {
        if (sumPowerIx[ix] > 0.000001) {
            multiTimeFronts[ix] = ifft(multiFrequencyFronts[ix], 1.0);
        }
    }
}

function phaseChangeDuringKerr(side) {

    sumPowerIx = [];
    ps1 = [];
    ps2 = [];
    // artificial factor override on Kerr lensing by power
    let totalKerrLensing = math.multiply(lensingFactor, IklTimesI);

    // kerr phase shift according to local power
    for (let ix = 0; ix < nSamples; ix++) {
        let bin = multiTimeFronts[ix];
        let bin2 = math.abs(math.dotMultiply(bin, math.conj(bin)));
        // total power of bin is set into sumPowerIx
        sumPowerIx.push(math.sum(bin2));
        let phaseShift1 = math.multiply(totalKerrLensing, bin2);
        multiTimeFronts[ix] = math.dotMultiply(bin, math.exp(phaseShift1));
        ps1.push(phaseShift1[0].im); // record for presentation
        // let x = (ix - nSamples / 2) * dx0;
        // let phaseShift = math.complex(0.0, - Math.PI / lambda / 0.0075 * x * x);
        // ps2.push(- Math.PI / lambda / kerrFocalLength * x * x);
    }
    multiTimeFrontsSaves[side * 3 + 0] = math.clone(multiTimeFronts);

    // shrink front using a gaussian virtual aperture
    let multiTimeFrontsTrans = math.transpose(multiTimeFronts) 
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = multiTimeFrontsTrans[iTime];
        let pFrBefore = math.sum(math.dotMultiply(fr, math.conj(fr)));
        let frAfter = math.dotMultiply(fr, multiTimeAperture);
        let pFrAfter = math.sum(math.dotMultiply(frAfter, math.conj(frAfter)));
        fr = math.multiply(frAfter, Math.sqrt(pFrBefore / pFrAfter));
        multiTimeFrontsTrans[iTime] = fr;
    }
    multiTimeFronts = math.transpose(multiTimeFrontsTrans) 

}

function spectralGainDispersion() {


    // let multiFrequencyFrontsT = fft(multiTimeFronts[nSamples / 2], 1.0);
    // multiFrequencyFrontsT = math.dotMultiply(multiFrequencyFrontsT, frequencyTotalMultFactor);  // rrrrrrrrrrr

    // let multiTimeFrontsT = ifft(multiFrequencyFrontsT, 1.0);

    // let div = math.dotDivide(multiTimeFrontsT, multiTimeFronts[nSamples / 2]);
    // for (let ix = 0; ix < nSamples; ix++) {
    //     multiTimeFronts[ix] = math.dotMultiply(div, multiTimeFronts[ix]);
    // }
    
    fftToFrequency();
    for (let ix = 0; ix < nSamples; ix++) {
        if (sumPowerIx[ix] > 0.000001) {
            multiFrequencyFronts[ix] = math.dotMultiply(multiFrequencyFronts[ix], frequencyTotalMultFactor);
        }
    }
    ifftToTime();
}

function linearCavityOneSide(side) {
    multiTimeFrontsSaves[side * 3 + 1] = math.clone(multiTimeFronts);

    let Is = IsFactor; // 200 * 352000 / 2;
    gainReduction = math.dotMultiply(pumpGain0, math.dotDivide(nSamplesOnes, math.add(1, math.divide(sumPowerIx, Is * nTimeSamples)))).map((v) => v.re);
    gainReductionWithOrigin = math.multiply(gainFactor, math.add(1, gainReduction));
    gainReductionAfterDiffraction = math.dotMultiply(gainReductionWithOrigin, multiTimeDiffraction);

    // tranpose to 1024 slices of 256 so that multiTimeFrontsTrans[i] will be the frint number i
    let multiTimeFrontsTrans = math.transpose(multiTimeFronts) 

    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = multiTimeFrontsTrans[iTime];
        // let pFrBefore = math.sum(math.dotMultiply(fr, math.conj(fr)));
        // let frAfter = math.dotMultiply(fr, multiTimeAperture);
        // let pFrAfter = math.sum(math.dotMultiply(frAfter, math.conj(frAfter)));
        // fr = math.multiply(frAfter, Math.sqrt(pFrBefore / pFrAfter));
        // pFrAfter = math.sum(math.dotMultiply(frAfter, math.conj(frAfter)))

        fr = math.dotMultiply(fr, gainReductionAfterDiffraction);
        fresnelData[side].forEach((fresnelSideData) => {
            fr = math.dotMultiply(fr, fresnelSideData.vecs[0]);
            fr = fft(fr, fresnelSideData.dx);
            fr = math.dotMultiply(fr, fresnelSideData.vecs[1]);
        });

        multiTimeFrontsTrans[iTime] = fr;
    }
    multiTimeFronts = math.transpose(multiTimeFrontsTrans) 
    multiTimeFrontsSaves[side * 3 + 2] = math.clone(multiTimeFronts);
}

function prepareGainPump() {
    let epsilon = 0.55;//0.2;   // rrrrrr
    let pumpWidth = 0.000030 * 0.5;
    let g0 = 1 / mirrorLoss + epsilon;
    pumpGain0 = [];
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / pumpWidth;
        pumpGain0.push(g0 * math.exp(- xw * xw));
    }
}

function prepareAperture() {
    multiTimeAperture  = [];
    let apertureWidth = multiTimeApertureVal * 0.5;
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / apertureWidth;
        multiTimeAperture.push(math.exp(- xw * xw));
    }

    multiTimeDiffraction  = [];
    let diffractionWidth = multiTimeDiffractionVal;
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / diffractionWidth;
        multiTimeDiffraction.push(math.exp(- xw * xw));
    }
    
}

function prepareLinearFresnelHelpData() {
    //let matProg = [[1, 0.003], [0, 1]];
    fresnelData = [];

    console.log(`lambda = ${lambda} range = ${totalRange} nSamples = ${nSamples} dx = ${totalRange / nSamples}`);

    MatSide.forEach((sideM, indexSide) => {
        //if (indexSide == 1) {
        //    sideM = math.multiply(matProg, math.multiply(sideM, matProg));
        //}
        let [[A, B], [C, D]] = sideM;
        let M1, M2;
        console.log(`Side ${indexSide + 1} mat A=${A}, B=${B}, C=${C}, D=${D}`);
        if (A > 0) {
            // A not close to -1
            M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]];
            M1 = [[1, B / (A + 1)], [0, 1]];
        } else {
            // A close to -1, so negate matrix and then decompose
            M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]];
            M1 = [[-1, B / (-A + 1)], [0, -1]];
        }
        console.log(`M1 A=${M1[0][0]}, B=${M1[0][1]}, C=${M1[1][0]}, D=${M1[1][1]}`);
        console.log(`M2 A=${M2[0][0]}, B=${M2[0][1]}, C=${M2[1][0]}, D=${M2[1][1]}`);

        fresnelSideData =[];
        let dx = dx0;
        //console.log(`orig dx = ${dx}`);
        [M1, M2].forEach((M, index) => {
            let loss = (index == 0 && indexSide == 1) ? mirrorLoss : 1; // left side mirror loss
            fresnelSideData.push(vectorsForFresnel(M, nSamples, dx, loss, M[0][0] < 0));
            dx = M[0][1] * lambda / (nSamples * dx);
            //console.log(`After step ${index + 1} dx = ${dx}`);
        })
        //console.log(fresnelSideData);

        fresnelData.push(fresnelSideData);
    });
}

function totalIxPower() {
    intensityTotalByIx = [];

    for (let ix = 0; ix < nSamples; ix++) {
        intensityTotalByIx.push(math.sum(math.dotMultiply(multiTimeFronts[ix], math.conj(multiTimeFronts[ix]))));
    }
}

function SatGain() {
    factorGainByIx = [];

    for (let ix = 0; ix < nSamples; ix++) {
        if (ix == nSamples / 2) {
            console.log(`ix=${ix}, int=${intensityTotalByIx[ix]}`)
        }
        factorGainByIx.push(g0 / (1 + intensityTotalByIx[ix] / IntensitySaturationLevel))
    }
}

function drawTimeFrontsWithOptions(opt, view, canvas) {
    if (opt == 0) {
        drawTimeFronts(multiFrequencyFronts, view, canvas);
    } else if (opt == 7) {
        calcMatrices();
        drawMatrices(canvas);
    } else {
        if (multiTimeFrontsSaves[opt - 1].length > 0) {
            drawTimeFronts(multiTimeFrontsSaves[opt - 1], view, canvas);
        } else {
            drawTimeFronts(multiTimeFronts, view, canvas);
        }
    }    
}

function drawMultiTime() {
    if (!drawOption) {
        return
    }
    drawTimeFrontsWithOptions(timeContentOption, timeContentView, document.getElementById("funCanvasTime"));
    drawTimeFrontsWithOptions(freqContentOption, freqContentView, document.getElementById("funCanvasFrequency"));
    drawVector(pumpGain0, true, "green", 1,  false,"gainSat", "Pump", 0, "", 8);
    drawVector(gainReduction, false, "red", 1, false, "gainSat", "PumpSat", 0, "", 8);
    drawVector(gainReductionWithOrigin, false, "blue", 1,  false, "gainSat", "Pump + 1", 0, "", 8);
    drawVector(gainReductionAfterDiffraction, false, "black", 1,  false,"gainSat", "with Diff", 0, "", 8);
    //drawVector(multiTimeAperture, false, "gray", 1,  false,"gainSat", "aperture", 0, "", 8);
    drawVector(sumPowerIx, true, "blue", 1,  false,"meanPower", "Power", 0);
    drawVector(ps1, true, "red", 1, false, "kerrPhase", "Kerr", 0, "hello");
    if (ps1.length > 0) {
        drawVector(focalFromPhase(ps1), false, "green", 1,  false,"kerrPhase", "KerrD2", 0);
    }
    //drawVector(ps2, false, "green", 1,  false,"kerrPhase", "Lens", 0, `f=${kerrFocalLength}`);
}

function drawTimeFronts(fs, view, canvas) {

    if (fs == null) {
        return;
    }
    const ctx = canvas.getContext("2d");
    drawMid = canvas.height / 2;

    var canvasWidth = canvas.width;
    var canvasHeight = canvas.height;
    var id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    var pixels = id.data;
    
    let maxV, maxS, meanV, meanS, meanMean, totalSumPower, meanH, maxH;

    if (view == 0) {
        fs = math.abs(fs);
        fs = math.dotMultiply(fs, fs);
        totalSumPower = math.sum(fs);
        maxV = math.max(fs, 0);
        maxS = math.max(maxV);
        meanV = math.mean(fs, 0);
        meanS = math.max(meanV) * nSamples;
        meanMean = math.mean(meanV);
        //maxH = math.max(meanH) * nSamples;
    }

    let sum0;
    for (let i = 0; i < nSamples; i++) {
        if (view == 0) {
            if (i == 0) {
                sum0 = math.clone(fs[i]);
            } else {
                sum0 = math.add(sum0, fs[i]);
            }
        }
        let off = i * nTimeSamples * 4;
        let line = (view == 0) ? math.dotDivide(fs[i], maxS): fs[i];
        for (let iTime = 0; iTime < nTimeSamples; iTime++) {
            if (view == 0) {
                c = Math.floor(line[iTime] * 255.0);
            } else {
                c = Math.floor((line[iTime].toPolar().phi / (2 * Math.PI) + 0.5) * 255.0);
            }
            pixels[off++] = c;
            pixels[off++] = c;
            pixels[off++] = c;
            pixels[off++] = 255;
        }
    }

    ctx.putImageData(id, 0, 0);
    if (view == 0) {
        let maxSum0 = math.max(sum0);
        factor = (canvas.height - 10) / maxSum0;
        sum0 = math.multiply(sum0, factor);
    
        let maxVN = math.floor(math.multiply(maxV, (canvas.height - 10) / (maxS + 0.0001)));
        let y = canvas.height - 5 - sum0[0];
        ctx.strokeStyle = 'red';
        ctx.beginPath();
        ctx.moveTo(0, y);
        for (let iTime = 1; iTime < nTimeSamples; iTime++) {
            y = canvas.height - 0 - sum0[iTime];
            ctx.lineTo(iTime, y);
        }
        ctx.stroke();
        // let meanVN = math.floor(math.multiply(meanV, (canvas.height - 10) / (meanS + 0.0001)));
        // y = canvas.height - 1 - meanVN[0];
        // ctx.strokeStyle = 'green';
        // ctx.fillStyle = 'green';
        // ctx.beginPath();
        // ctx.moveTo(0, y);
        // for (let iTime = 1; iTime < nTimeSamples; iTime++) {
        //     y = canvas.height - 1 - meanVN[iTime];
        //     ctx.lineTo(iTime, y);
        // }
        // ctx.stroke();
        drawTextBG(ctx, (meanS).toFixed(1), 10, 10);
        drawTextBG(ctx, (meanMean).toFixed(1), 10, 30);
        drawTextBG(ctx, (totalSumPower).toFixed(1), 10, 50);
    }
}
function focalFromPhase(phase) {
    let deriv2NoZero = vecDeriv2(phase, dx0).map((v) => Math.abs(v) < 0.000000001 ? 0.000000001 : v)
    let focalVec = math.dotDivide(nSamplesOnesR, math.multiply(-  lambda / (2 * Math.PI), deriv2NoZero));
    return focalVec.map((v) => v < -10.0 ? 0.0 : (v > 10.0 ? 0.0 : v) );
}
function multiTimeCanvasMouseMove(e, updateTest = false) {
    id = e.target.id;

    let opt = id == "funCanvasTime" ? timeContentOption: freqContentOption;
    let view  = id == "funCanvasTime" ? timeContentView: freqContentView;
    if (opt == 7) {
        return;
    }
    let [x, y] = getClientCoordinates(e);
    
    if (opt == 7) {
        return;
    }
    let fs = (opt == 0 ? multiFrequencyFronts : multiTimeFrontsSaves[opt - 1]);

    let front = math.transpose(fs)[x];
    let xVec, yVec;
    if (view == 0) {
        xVec = math.abs(front);
        yVec = math.abs(fs[y]);
    } else {
        xVec = front.map((v) => v.toPolar().phi);
        yVec = fs[y].map((v) => v.toPolar().phi);
    }
    if (opt % 7 != 0) {
        multiFronts[0] = [math.clone(front)];
        multiRanges[0] = [totalRange];
    }

    let front2 = math.abs(math.dotMultiply(front, math.conj(front)));
    let totalKerrLensing = math.multiply(lensingFactor, IklTimesI)
    ps1 = math.multiply(totalKerrLensing.im, front2);

    let pw = math.sum(math.dotMultiply(xVec, xVec));
    let width = calcWidth(xVec)
    let waist = width * dx0 * 2.0 * 1.414;
    //console.log(`total range ${totalRange} dx0 = ${dx0} widthunit = ${width}`);
    let IklLocal = 1.44E-24;
    let focal = (waist ** 4) / (IklLocal * pw);
    ps3 = xVec.map((dumx, ix) => (- Math.PI / lambda / focal * ((ix - nSamples / 2) * dx0) * ((ix - nSamples / 2) * dx0)));
    let message = `t=${x}</br>Wa=${(waist*1000000).toFixed(0)}mic</br>p=${pw.toExponential(4)}</br>f=${focal.toFixed(4)}`;

    drawVector(xVec, true, "red", 1, false, "sampleX", "by-X", 0, message);
    drawVector(yVec, true, "red", 1, false, "sampleY", "by-Y", 0);
    drawVector(ps1, true, "red", 1, false, "kerrPhase", "Kerr", 0);
    drawVector(focalFromPhase(ps1), false, "green", 1,  false,"kerrPhase", "KerrD2", 0);
    //drawVector(ps2, false, "green", 1,  false,"kerrPhase", "Lens", 0, `f=${kerrFocalLength}`);
    drawVector(ps3, false, "blue", 1,  false,"kerrPhase", "LensW", 0, `f=${focal.toFixed(4)}`);

    if (updateTest) {
        const canvas = document.getElementById(`funCanvasTest`);
        const ctx = canvas.getContext("2d");
        drawFronts(canvas, ctx, multiFronts[0], multiRanges[0]);
    }
}

function shiftFronts(s) {
    s += nTimeSamples;
    let multiTimeFrontsTrans = math.transpose(multiTimeFronts) 
    let newMultiTimeFrontsTrans = []
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        newMultiTimeFrontsTrans.push(multiTimeFrontsTrans[(iTime + s) % nTimeSamples])
    }
    multiTimeFronts = math.transpose(newMultiTimeFrontsTrans) 
}

function calcMatrices() {
    let IklLocal = 1.44E-24;

    let frontsForLens = [math.transpose(multiTimeFrontsSaves[0]), 
                    math.transpose(multiTimeFrontsSaves[3])];
    let lenses = frontsForLens.map((frs) => frs.map((fr) => {
        let absfr = math.abs(fr);
        let pw = math.sum(math.dotMultiply(absfr, absfr));
        let width = calcWidth(absfr);
        let waist = width * dx0 * 2.0 * 1.414;
        let focal = (waist ** 4) / (IklLocal * pw);
        return focal;
    }));
    
    matrices = lenses[0].map((dum, i) => {
        let mat = MMult(MatSide[0], MLens(lenses[0][i]));
        mat = MMult(MLens(lenses[1][i]), mat);
        mat = MMult(MatSide[1], mat);
        return mat;
    });
}

function drawMatrices(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let fac = 80;
    ctx.strokeStyle = 'red';
    ctx.beginPath();
    ctx.moveTo(0, canvas.height - 1 - Math.floor(fac * 2.0));
    ctx.lineTo(canvas.width, canvas.height - 1 - Math.floor(fac * 2.0));
    ctx.stroke();

    ctx.strokeStyle = 'blue';
    ctx.beginPath();
    matrices.forEach((m, i) => {
        let val = Math.abs(m[0][0] + m[1][1]);
        if (i == 0) {
            ctx.moveTo(i, canvas.height - 1 - Math.floor(fac * val));
        } else {
            ctx.lineTo(i, canvas.height - 1 - Math.floor(fac * val));
        }
    });
    ctx.stroke();

}
function progressMultiTime(direction) {
    drawOption = false;

    initElementsMultiMode();

    let globalDelta = elements.find((el) => el.t == "X").delta;
    let rightLength = elements.find((el) => el.t == "X").par[0];

    let crsitalPosition = elements[1].par[0] + elements[1].delta * globalDelta;

    initMultiMode(3);

    let startCalc = direction == 2 ? crsitalPosition + 0.000001 : rightLength + rightLength - crsitalPosition + 0.000001;
    
    fullCavityMultiMode(startCalc);

    drawOption = true;
    drawMultiMode(startCalc);
}

function gainFactorChanged() {
    gainFactor = getFieldFloat('gainFactor', gainFactor);
}

function dispersionFactorChanged() {
    dispersionFactor = getFieldFloat('dispersionFactor', dispersionFactor);
    initGainByFrequency();
}
function lensingFactorChanged() {
    lensingFactor = getFieldFloat('lensingFactor', lensingFactor);
}
function isFactorChanged() {
    IsFactor = getFieldFloat('isFactor', IsFactor);
}

function multiTimeApertureChanged() {
    multiTimeApertureVal = getFieldFloat('aperture', multiTimeApertureVal);
    prepareAperture();
}
