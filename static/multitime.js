var nTimeSamples = 1024;
var multiTimeFronts = [];
var multiTimeFrontsSaves = [[], [], [], []];
var multiFrequencyFronts = [];
var factorGain = [];
var IntensitySaturationLevel = 400000000000000.0;
var intensityTotalByIx = [];
var factorGainByIx = [];
var Ikl = 0.02;
let IklTimesI = math.complex(0, Ikl * 80 * 0.001);
var rangeW = [];
var spectralGain = [];
var dispersion = [];
var sumPowerIx = [];
var gainReduction = [];
var gainReductionWithOrigin = [];
var gainReductionAfterAperture = [];
var pumpGain0 = [];
var multiTimeAperture  = [];
var frequencyTotalMultFactor = [];
var mirrorLoss = 0.95;
var fresnelData = [];
var totalRange = 0.001000;
var dx0 = totalRange / nSamples;
var scalarOne = math.complex(1);
var nTimeSamplesOnes = Array.from({length: nTimeSamples}, (v) => scalarOne)
var nSamplesOnes = Array.from({length: nSamples}, (v) => scalarOne)
var kerrFocalLength = 0.0075;
var ps1 = [];
var ps2 = [];
var contentOption = 0;
var contentOptionVals = ["F", "1", "2", "3", "4"];


// let MatSide = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  // right
//                 [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]];   // left

// delta 0.0095 focal 0.0075
//let MatSide = [[[-0.2982666667, -0.006166766667], [147.7333333, -0.2982666667]],  // right
//                [[0.3933333333, -0.002881666667], [293.3333333, 0.3933333333]]];   // left

// delta 0.005  focal 20.00
let MatSide = [[[-0.6266666667, -0.0040666666677], [149.3333333,-0.6266666667]],  // right
                [[-0.2666666667, -0.003166666667], [293.3333333, -0.2666666667]]];   // left


function refreshCacityMatrices() {
    MatSide = [math.clone(totalRightSide), math.clone(totalLeftSide)];
    console.log(MatSide);
}

function updateContentOptions() {
    let options = contentOptionVals.map((v, i) => {
        style=`"margin: 3px; padding: 2px; border: 1px solid black; border-radius: 2px; background:${i == contentOption ? "yellow": "white"};"`
        return `<div onclick="contentOption = ${i}; updateContentOptions(); drawMultiTime();" style=${style}>${v}</div>`
    }).join("");
    document.getElementById("FrequencyCanvasOptions").innerHTML = options;
}

function initMultiTime() {
    workingTab = 3
    multiTimeFronts = [];
    for (let i = 0; i < nSamples; i++) {
        multiTimeFronts.push([]);
    }
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = getInitFront(beamParam);
        for (let i = 0; i < nSamples; i++) {
            multiTimeFronts[i].push(fr[i]);
        }
    }

    updateContentOptions();
 
    prepareLinearFresnelHelpData();

    prepareGainPump();
    initGainByFrequency();

    prepareAperture()

    fftToFrequency();
    ifftToTime();

    drawMultiMode();
}

function initGainByFrequency() {
    let specGain = math.complex(200);
    let disp_par = 0.5e-3 * 2 * Math.PI / specGain;    
    rangeW = math.range(- nTimeSamples / 2 , nTimeSamples / 2).toArray().map((v) => math.complex(v + 0.0));
    let ones = rangeW.map((v) => math.complex(1.0));
    let mid = math.dotDivide(rangeW, rangeW.map((v) => specGain));
    spectralGain = math.dotDivide(ones, math.add(math.square(mid), 1));
    dispersion = math.exp(math.multiply(math.complex(0, - disp_par), math.square(rangeW)));
    let expW = math.exp(math.multiply(math.complex(0, - 2 * Math.PI), rangeW));
    frequencyTotalMultFactor = math.dotMultiply(expW, math.dotMultiply(spectralGain, dispersion));
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
        phaseChangeDuringKerr();
        // phaseChangeDuringKerr (V)

        spectralGainDispersion();
        // gainByfrequency (V)
        // dispersionByFrequency (V)

        linearCavityOneSide(side);
        // gainCorrectionDueToSaturation
        // oneSideCavity
        // mirrorLoss (only on left side)
    });
}

function coverRound(params) {
    if (params[0] <= 0) {
        drawMultiMode();
        const endTime = performance.now()
        console.log(`Call to full took ${endTime - startTime} milliseconds`)
        return null;
    }
    multiTimeRoundTrip(1);
    if (params[0] % 10 == 0) {
        drawMultiMode();
    }

    if (params[0] <= 1) {
        drawMultiMode();
        const endTime = performance.now()
        console.log(`Call to full took ${endTime - startTime} milliseconds`)
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
    //console.log(`Call to timeCavityStep took ${endTime - startTime} milliseconds`)

    if (redraw) {
        drawMultiMode();
        drawVector(pumpGain0, true, "green", 1,  false,"gainSat", "Pump", 0);
        drawVector(gainReduction, false, "red", 1, false, "gainSat", "PumpSat", 0);
        drawVector(gainReductionWithOrigin, false, "blue", 1,  false,"gainSat", "Pump + 1", 0);
        drawVector(gainReductionAfterAperture, false, "black", 1,  false,"gainSat", "with aper", 0);
        drawVector(multiTimeAperture, false, "gray", 1,  false,"gainSat", "aperture", 0);
        drawVector(sumPowerIx, true, "blue", 1,  false,"meanPower", "Power", 0);
        drawVector(ps1, true, "red", 1, false, "kerrPhase", "Kerr", 0, "hello");
        drawVector(ps2, false, "green", 1,  false,"kerrPhase", "Lens", 0, `f=${kerrFocalLength}`);
    }
}
function fftToFrequency() {
    multiFrequencyFronts = [];
    for (let ix = 0; ix < nSamples; ix++) {
        multiFrequencyFronts.push(fft(multiTimeFronts[ix], 1.0));
    }
}

function ifftToTime() {
    multiTimeFronts = [];
    for (let ix = 0; ix < nSamples; ix++) {
        multiTimeFronts.push(ifft(multiFrequencyFronts[ix], 1.0));
    }
}

function phaseChangeDuringKerr() {
    sumPowerIx = [];
    ps1 = [];
    ps2 = [];
    for (let ix = 0; ix < nSamples; ix++) {
        let bin = multiTimeFronts[ix];
        let bin2 = math.abs(math.dotMultiply(bin, math.conj(bin)));
        sumPowerIx.push(math.sum(bin2));
        let phaseShift1 = math.multiply(IklTimesI, bin2);
        let x = (ix - nSamples / 2) * dx0;
        let phaseShift = math.complex(0.0, - Math.PI / lambda / 0.0075 * x * x);
        ps1.push(phaseShift1[0].im);
        ps2.push(- Math.PI / lambda / kerrFocalLength * x * x);
        multiTimeFronts[ix] = math.dotMultiply(bin, math.exp(phaseShift1));
    }
}

function spectralGainDispersion() {
    fftToFrequency();
    for (let ix = 0; ix < nSamples; ix++) {
        multiFrequencyFronts[ix] = math.dotMultiply(multiFrequencyFronts[ix], frequencyTotalMultFactor);
    }
    ifftToTime();
}

function linearCavityOneSide(side) {
    multiTimeFrontsSaves[side * 2] = math.clone(multiTimeFronts);

    let Is = 200;
    gainReduction = math.dotMultiply(pumpGain0, math.dotDivide(nSamplesOnes, math.add(1, math.divide(sumPowerIx, Is * nTimeSamples)))).map((v) => v.re);
    gainReductionWithOrigin = math.add(1, gainReduction);
    gainReductionAfterAperture = math.dotMultiply(gainReductionWithOrigin, multiTimeAperture);

    let multiTimeFrontsTrans = math.transpose(multiTimeFronts) 
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = multiTimeFrontsTrans[iTime];
        fr = math.dotMultiply(fr, gainReductionAfterAperture);
        fresnelData[side].forEach((fresnelSideData) => {
            fr = math.dotMultiply(fr, fresnelSideData.vecs[0]);
            fr = fft(fr, fresnelSideData.dx);
            fr = math.dotMultiply(fr, fresnelSideData.vecs[1]);
        });

        multiTimeFrontsTrans[iTime] = fr;
    }
    multiTimeFronts = math.transpose(multiTimeFrontsTrans) 
    multiTimeFrontsSaves[side * 2 + 1] = math.clone(multiTimeFronts);
}

function prepareGainPump() {
    let epsilon = 0.2;
    let pumpWidth = 0.000030;
    let g0 = 1 / mirrorLoss + epsilon;
    pumpGain0 = [];
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / pumpWidth;
        pumpGain0.push(g0 * math.exp(- 0.5 * xw * xw));
    }
}

function prepareAperture() {
    multiTimeAperture  = [];
    let apertureWidth = 0.000056;
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / apertureWidth;
        multiTimeAperture.push(math.exp(- 0.5 * xw * xw));
    }
}

function prepareLinearFresnelHelpData() {


    let matProg = [[1, 0.003], [0, 1]];
    fresnelData = [];

    console.log(`lambda = ${lambda} range = ${totalRange} nSamples = ${nSamples} dx = ${totalRange / nSamples}`);

    MatSide.forEach((sideM, indexSide) => {
        if (indexSide == 1) {
            sideM = math.multiply(matProg, math.multiply(sideM, matProg));
        }
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

function drawMultiTime() {
    if (!drawOption) {
        return
    }

    drawTimeFronts(multiTimeFronts, document.getElementById("funCanvasTime"));
    if (contentOption == 0) {
        drawTimeFronts(multiFrequencyFronts, document.getElementById("funCanvasFrequency"));
    } else {
        drawTimeFronts(multiTimeFrontsSaves[contentOption - 1], document.getElementById("funCanvasFrequency"));
    }
}

function drawTimeFronts(fs, canvas) {

    const ctx = canvas.getContext("2d");
    drawMid = canvas.height / 2;

    var canvasWidth = canvas.width;
    var canvasHeight = canvas.height;
    var id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    var pixels = id.data;
    
    let maxV, maxS, meanV, meanS, meanMean, totalSumPower;

    if (viewOption == 1) {
        fs = math.abs(fs);
        fs = math.dotMultiply(fs, fs);
        totalSumPower = math.sum(fs);
        maxV = math.max(fs, 0);
        maxS = math.max(maxV);
        meanV = math.mean(fs, 0);
        meanS = math.max(meanV);
        meanMean = math.mean(meanV);
    }

    let sum0;
    for (let i = 0; i < nSamples; i++) {
        if (viewOption == 1) {
            if (i == 0) {
                sum0 = math.clone(fs[i]);
            } else {
                sum0 = math.add(sum0, fs[i]);
            }
        }
        let off = i * nTimeSamples * 4;
        let line = (viewOption == 1) ? math.dotDivide(fs[i], maxS): fs[i];
        for (let iTime = 0; iTime < nTimeSamples; iTime++) {
            if (viewOption == 1) {
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
    if (viewOption == 1) {
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
        drawTextBG(ctx, (meanMean).toFixed(6), 10, 10);
        drawTextBG(ctx, (totalSumPower).toFixed(1), 10, 30);
    }
}

function multiTimeCanvasMouseMove(e, updateTest = false) {
    id = e.target.id;
    let fs = (id == "funCanvasTime") ? multiTimeFronts : 
        (contentOption == 0 ? multiFrequencyFronts : multiTimeFrontsSaves[contentOption - 1]);
    let [x, y] = getClientCoordinates(e);

    let front = math.transpose(fs)[x];
    let xVec, yVec;
    if (viewOption == 1) {
        xVec = math.abs(front);
        yVec = math.abs(fs[y]);
    } else {
        xVec = front.map((v) => v.toPolar().phi);
        yVec = fs[y].map((v) => v.toPolar().phi);
    }
    multiFronts[0] = [front];
    multiRanges[0] = [totalRange];

    let front2 = math.abs(math.dotMultiply(front, math.conj(front)));
    ps1 = math.multiply(IklTimesI.im, front2);

    drawVector(xVec, true, "red", 1, false, "sampleX", "by-X", 0, `w=${calcWidth(xVec).toFixed(2)}`);
    drawVector(yVec, true, "red", 1, false, "sampleY", "by-Y", 0);
    drawVector(ps1, true, "red", 1, false, "kerrPhase", "Kerr", 0);
    drawVector(ps2, false, "green", 1,  false,"kerrPhase", "Lens", 0, `f=${kerrFocalLength}`);

    if (updateTest) {
        const canvas = document.getElementById(`funCanvasTest`);
        const ctx = canvas.getContext("2d");
        drawFronts(canvas, ctx, multiFronts[0], multiRanges[0]);
    }
}

function progressMultiTime() {
    drawOption = false;

    initElementsMultiMode();
    initMultiMode(3);
    fullCavityMultiMode();
    drawOption = true;
    drawMultiMode();
}
