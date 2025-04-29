/*
v counter for number of steps
- side report on progress record power, lensing, A+D, etc.
v automatic cosideration of design data on screen.
- go back to calculation of full gaussian rather than pixels
- add prints for minimum maximum of graphs
v update all data including xVec yVec on every update
- fixing in the frequancy domain
v remove phases that carry no intensity
- working in 2D (square the fine on width deviation)
- Fresnel in rings (Bessel)
v original sim with one lens only
- present focal length from 2nd derivative
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
var modulationGainFactor = 0.1;
var modulatorGain = [];
var dispersion = [];
var sumPowerIx = [];
var gainReduction = [];
var gainReductionWithOrigin = [];
var gainReductionAfterAperture = [];
var gainReductionAfterDiffraction = [];
var gainFactor = 0.50;
var epsilon = 0.2;
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
    console.log(`MShort = ${MShort}`);
    console.log(`MLong = ${MLong}`);

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
    randomLCGSetSeed(123);
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let rnd = math.complex((randomLCG() * 2 - 1), (randomLCG() * 2 - 1));
        let fr = math.multiply(rnd, getInitFront(beamParam));
        for (let i = 0; i < nSamples; i++) {
            multiTimeFronts[i].push(fr[i]);
            multiFrequencyFronts[i].push(math.complex(0));
        }
    }

    gainFactorChanged(false);
    isFactorChanged(false);
    nRoundsChanged();

    updateContentOptions();
 
    prepareLinearFresnelHelpData();

    prepareGainPump();
    initGainByFrequency();

    multiTimeApertureChanged(false);

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
    modulatorGain = rangeW.map((w) => 1.0 + modulationGainFactor * math.cos(2 * Math.PI * w / nTimeSamples));
    console.log(`modulatorGain = ${modulatorGain[80]} ${modulatorGain[180]}`);
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
        console.log(`Side ${side}`);
        phaseChangeDuringKerr(side);
        printSamples();


        spectralGainDispersion();
        printSamples();
        if (side == 1) {
            modulatorGainMultiply();
            printSamples();
        }
        // gainByfrequency (V)
        // dispersionByFrequency (V)

        linearCavityOneSide(side);
        printSamples();

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

function printSamples() {
    console.log('-----------------------------------')
    console.log(`${multiTimeFronts[128][63]}`)
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

function modulatorGainMultiply() {
    for (let ix = 0; ix < nSamples; ix++) {
        multiTimeFronts[ix] = math.dotMultiply(multiTimeFronts[ix], modulatorGain);
    }  
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

        fresnelData.push(fresnelSideData);
    });
}

function totalIxPower() {
    intensityTotalByIx = [];

    for (let ix = 0; ix < nSamples; ix++) {
        intensityTotalByIx.push(math.sum(math.dotMultiply(multiTimeFronts[ix], math.conj(multiTimeFronts[ix]))));
    }
}

// function SatGain() {
//     factorGainByIx = [];

//     for (let ix = 0; ix < nSamples; ix++) {
//         if (ix == nSamples / 2) {
//             console.log(`ix=${ix}, int=${intensityTotalByIx[ix]}`)
//         }
//         factorGainByIx.push(g0 / (1 + intensityTotalByIx[ix] / IntensitySaturationLevel))
//     }
// }

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
        //fs = math.abs(fs);
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
function drawTimeNumData(fs, view, canvas) {
    if (fs == null || fs.length == 0 || canvas == null) {
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
        meanS = math.max(meanV) * fs.length;
        meanMean = math.mean(meanV);
        //maxH = math.max(meanH) * fs.length;
    }

    let sum0;
    for (let i = 0; i < fs.length; i++) {
        if (view == 0) {
            if (i == 0) {
                sum0 = math.clone(fs[i]);
            } else {
                sum0 = math.add(sum0, fs[i]);
            }
        }
        let off = i * fs[0].length * 4;
        let line = (view == 0) ? math.dotDivide(fs[i], maxS): fs[i];
        for (let iTime = 0; iTime < fs[0].length; iTime++) {
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


    // if (view == 0) {
    //     let maxSum0 = math.max(sum0);
    //     factor = (canvas.height - 10) / maxSum0;
    //     sum0 = math.multiply(sum0, factor);
    
    //     let maxVN = math.floor(math.multiply(maxV, (canvas.height - 10) / (maxS + 0.0001)));
    //     let y = canvas.height - 5 - sum0[0];
    //     ctx.strokeStyle = 'red';
    //     ctx.beginPath();
    //     ctx.moveTo(0, y);
    //     for (let iTime = 1; iTime < nTimeSamples; iTime++) {
    //         y = canvas.height - 0 - sum0[iTime];
    //         ctx.lineTo(iTime, y);
    //     }
    //     ctx.stroke();
    //     // let meanVN = math.floor(math.multiply(meanV, (canvas.height - 10) / (meanS + 0.0001)));
    //     // y = canvas.height - 1 - meanVN[0];
    //     // ctx.strokeStyle = 'green';
    //     // ctx.fillStyle = 'green';
    //     // ctx.beginPath();
    //     // ctx.moveTo(0, y);
    //     // for (let iTime = 1; iTime < nTimeSamples; iTime++) {
    //     //     y = canvas.height - 1 - meanVN[iTime];
    //     //     ctx.lineTo(iTime, y);
    //     // }
    //     // ctx.stroke();
    //     drawTextBG(ctx, (meanS).toFixed(1), 10, 10);
    //     drawTextBG(ctx, (meanMean).toFixed(1), 10, 30);
    //     drawTextBG(ctx, (totalSumPower).toFixed(1), 10, 50);
    // }
}

function drawGraphNumData(lines, canvas) {
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

function epsilonChanged() {
    epsilon = getFieldFloat('epsilon', epsilon);
    prepareGainPump();
    snackBar(`Change epsilon to ${epsilon}`);
}

function gainFactorChanged(showSnack = true) {
    gainFactor = getFieldFloat('gainFactor', gainFactor);
    if (showSnack) {
        snackBar(`Change gain factor to ${gainFactor}`);
    }
}

function dispersionFactorChanged() {
    dispersionFactor = getFieldFloat('dispersionFactor', dispersionFactor);
    initGainByFrequency();
    snackBar(`Change dispersion factor to ${dispersionFactor}`);
}
function lensingFactorChanged() {
    lensingFactor = getFieldFloat('lensingFactor', lensingFactor);
    snackBar(`Change lensing factor to ${lensingFactor}`);
}
function modulationGainFactorChanged() {
    modulationGainFactor = getFieldFloat('modulationGainFactor', modulationGainFactor);
    initGainByFrequency();
    snackBar(`Change modulation factor to ${modulationGainFactor}`);
}

function isFactorChanged(showSnack = true) {
    IsFactor = getFieldFloat('isFactor', IsFactor);
    if (showSnack) {
        snackBar(`Change Intensity saturation factor to ${IsFactor}`);
    }
}

function multiTimeApertureChanged(showSnack = true) {
    multiTimeApertureVal = getFieldFloat('aperture', multiTimeApertureVal);
    prepareAperture();
    if (showSnack) {
        snackBar(`Change aperture`);
    }
}

function openVec(fs) {
    if (fs == null) {
        return null
    }

    return fs.map(l => l.map((v) => {
        if (v.length == 0) {
            return 0;
        }
        return parseFloat(v);
    }));    
}
function modifyPointer(pointer) {
    let ctx = [document.getElementById("funCanvasSample1top").getContext("2d"),
            document.getElementById("funCanvasSample2top").getContext("2d")];
    ctx[0].reset();
    ctx[1].reset();
    ctx[0].fillStyle = "rgba(255, 0, 0, 0.4)";
    ctx[0].arc(pointer[1], pointer[2], 10, 0, 2 * Math.PI);
    ctx[0].fill();
    ctx[1].fillStyle = "rgba(0, 0, 255, 0.4)";
    ctx[1].arc(pointer[1], pointer[2], 10, 0, 2 * Math.PI);
    ctx[1].fill();
}
var currentPlot3dValues = [];
var plots3dObject = [];

function drawPlot3d() {
    Plotly.newPlot('plotData1', plots3dObject, {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        scene: {xaxis: {title: 'X'}, yaxis: {title: 'Y'}, zaxis: {title: 'Z'}}
    });
}

function ClearPlot3D() {
    plots3dObject = [];
    drawPlot3d();
}

function AddPlot3D() { 
    let values = currentPlot3dValues;
    let count = parseInt(document.getElementById("stepsCounter").value)
    if (values != null && values.length > 0) {
        let x = values.map((l, i) => i);
        let c = values.map((l) => count);
        plots3dObject.push({
                "type": "scatter3d",
                "mode": "lines",
                "x": c,
                "y": x,
                "z": values,
                "xaxis": {title: 'Round'},
                "yaxis": {title: 'X'},
                "zaxis": {title: 'Power'},
                "line": {
                    "width": 4
                },
                "name": `Round ${count}`,
        });
        drawPlot3d();
    }
}

function spreadUpdatedData(data) {
    if (data.rounds) {
        document.getElementById("stepsCounter").value = `${data.rounds}`;
    }
    if (data.more) {
        document.getElementById("stepsCounter").style.color = "red";
        document.getElementById("stepsCounter").style.animation = "blink 1s infinite";
    } else {
        document.getElementById("stepsCounter").style.color = "black";
        document.getElementById("stepsCounter").style.animation = "";
    }
    if (data.samples) {
        for (sample of data.samples) {
            drawTimeNumData(openVec(sample.samples), 0, document.getElementById(sample.name));
        }
    }
    if (data.pointer) {
        modifyPointer(data.pointer)
    }
    if (data.graphs) {
        let backColor = data.more ? "#ffeedd": "white";
        for (graph of data.graphs) {
            let clear = true;
            for (line of graph.lines) {
                drawVector(line.values, clear, line.color, 1, true, graph.name, "",  0, line.text, 1, backColor);
                clear = false;
            }
            if (graph.name == "gr5") {
                if (graph.lines.length > 0) {
                    currentPlot3dValues = graph.lines[0].values;
                }
            }
        }

    }
    if (data.view_buttons) {
        for (let part in [0, 1]) {
            for (let i = 0; i < 14; i++) {
                but = document.getElementById(`view_button-${part}-${i + 1}`);  
                if (data.view_buttons.view_on_stage[part] == `${i + 1}`) {
                    but.classList.add("buttonH");
                } else {
                    but.classList.remove("buttonH");
                }
            }
            for (let i of ["Frq", "Amp"]) {
                but = document.getElementById(`view_button-${part}-${i}`);  
                if (data.view_buttons.view_on_amp_freq[part] == `${i}`) {
                    but.classList.add("buttonH");
                } else {
                    but.classList.remove("buttonH");
                }
            }
            for (let i of ["Phs", "Abs"]) {
                but = document.getElementById(`view_button-${part}-${i}`);  
                if (data.view_buttons.view_on_abs_phase[part] == `${i}`) {
                    but.classList.add("buttonH");
                } else {
                    but.classList.remove("buttonH");
                }
            }
        }
    }
}
function numDataMutated() {
    numData = document.getElementById("numData");
    s = numData.innerText;
    if (s.length > 0) {
        data = JSON.parse(s);
        numData.innerText = "";
        spreadUpdatedData(data)

    }
}