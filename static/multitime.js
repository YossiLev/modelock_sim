var nTimeSamples = 1024;
var multiTimeFronts = [];
var multiFrequencyFronts = [];
var factorGain = [];
var IntensitySaturationLevel = 4;
var intensityTotalByIx = [];
var factorGainByIx = [];
var Ikl = 0.02;
var rangeW = [];
var spectralGain = [];
var dispersion = [];
var sumPowerIx = [];
var gainReduction = [];
var pumpGain0 = [];
var frequencyTotalMultFactor = [];
var mirrorLoss = 0.95;
var fresnelData = [];
var totalRange = 0.001000;
var dx0 = totalRange / nSamples;
var scalarOne = math.complex(1);
var nTimeSamplesOnes = Array.from({length: nTimeSamples}, (v) => scalarOne)
var nSamplesOnes = Array.from({length: nSamples}, (v) => scalarOne)

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

    prepareLinearFresnelHelpData();

    prepareGainPump();
    initGainByFrequency();

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
    if (iCount % 50 == 0) {
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

var startTime;
function timeCavityStep(step, redraw) {
    startTime = performance.now()

    switch (step) {
        case 1: phaseChangeDuringKerr(); fftToFrequency(); break;
        case 2: spectralGainDispersion(); break;
        case 3: linearCavityOneSide(0); break;
        case 4: linearCavityOneSide(1); break;
        case 5: math.range(0, 1).forEach((x)=> multiTimeRoundTrip(x)); break;
    }
    const endTime = performance.now()
    console.log(`Call to timeCavityStep took ${endTime - startTime} milliseconds`)

    if (redraw) {
        drawMultiMode();
        drawVector(gainReduction, true, "red", 1, "gainSat", "GainSat", 0);
        drawVector(pumpGain0, false, "green", 1, "gainSat", "Pump", 0);
        drawVector(sumPowerIx, true, "blue", 1, "meanPower", "Power", 0);
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
    let IklTimesI = math.complex(0, Ikl * 100);
    sumPowerIx = [];
    for (let ix = 0; ix < nSamples; ix++) {
        let bin = multiTimeFronts[ix];
        let bin2 = math.abs(math.dotMultiply(bin, math.conj(bin)));
        sumPowerIx.push(math.sum(bin2));
        let phaseShift = math.multiply(IklTimesI, bin2);
        multiTimeFronts[ix] = math.dotMultiply(bin, math.exp(phaseShift));
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
    let Is = 20;
    gainReduction = math.dotMultiply(pumpGain0, math.dotDivide(nSamplesOnes, math.add(1, math.divide(sumPowerIx, Is * nTimeSamples)))).map((v) => v.re);

    let multiTimeFrontsTrans = math.transpose(multiTimeFronts) 
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = multiTimeFrontsTrans[iTime];
        fr = math.dotMultiply(fr, gainReduction);
        fresnelData[side].forEach((fresnelSideData) => {
            fr = math.dotMultiply(fr, fresnelSideData.vecs[0]);
            fr = fft(fr, fresnelSideData.dx);
            fr = math.dotMultiply(fr, fresnelSideData.vecs[1]);
        });

        multiTimeFrontsTrans[iTime] = fr;
    }
    multiTimeFronts = math.transpose(multiTimeFrontsTrans) 
}

function prepareGainPump() {
    let epsilon = 0.4;
    let pumpWidth = 0.000056;
    let g0 = 1 / 0.05/*mirrorLoss*/ + epsilon;
    pumpGain0 = [];
    for (let ix = 0; ix < nSamples; ix++) {
        let x = (ix - nSamples / 2) * dx0;
        let xw = x / pumpWidth;
        pumpGain0.push(g0 * math.exp(- 0.5 * xw * xw));
    }
    
}
function prepareLinearFresnelHelpData() {
    let MatSide = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  // right
                    [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]];   // left
    let matProg = [[1, 0.003], [0, 1]];
    fresnelData = [];

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
        console.log(`orig dx = ${dx}`);
        [M1, M2].forEach((M, index) => {
            let loss = (index == 0 && indexSide == 1) ? mirrorLoss : 1; // left side mirror loss
            fresnelSideData.push(vectorsForFresnel(M, nSamples, dx, loss, M[0][0] < 0));
            dx = M[0][1] * lambda / (nSamples * dx);
            console.log(`After step ${index + 1} dx = ${dx}`);
        })
        console.log(fresnelSideData);

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
        factorGainByIx.push(g0 / (1 + intensityTotalByIx[ix] / IntensitySaturationLevel))
    }
}

function drawMultiTime() {
    if (!drawOption) {
        return
    }

    drawTimeFronts(1, document.getElementById("funCanvasTime"));
    drawTimeFronts(2, document.getElementById("funCanvasFrequency"));
}

function drawTimeFronts(domainOption, canvas) {

    const ctx = canvas.getContext("2d");
    drawMid = canvas.height / 2;

    var canvasWidth = canvas.width;
    var canvasHeight = canvas.height;
    var id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    var pixels = id.data;
    
    let fs = (domainOption == 1) ? multiTimeFronts : multiFrequencyFronts;
    let maxV, maxS, meanV, meanS, meanMean;

    if (viewOption == 1) {
        fs = math.abs(fs);
        fs = math.dotMultiply(fs, fs);
        maxV = math.max(fs, 0);
        maxS = math.max(maxV);
        meanV = math.mean(fs, 0);
        meanS = math.max(meanV);
        meanMean = math.mean(meanV);
    }

    for (let i = 0; i < nSamples; i++) {
        let off = i * nTimeSamples * 4;
        let line = (viewOption == 1) ? math.dotDivide(fs[i], maxV): fs[i];
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
        let maxVN = math.floor(math.multiply(maxV, (canvas.height - 10) / (maxS + 0.0001)));
        let y = canvas.height - 1 - maxVN[0];
        ctx.strokeStyle = 'red';
        ctx.beginPath();
        ctx.moveTo(0, y);
        for (let iTime = 1; iTime < nTimeSamples; iTime++) {
            y = canvas.height - 1 - maxVN[iTime];
            ctx.lineTo(iTime, y);
        }
        ctx.stroke();
        let meanVN = math.floor(math.multiply(meanV, (canvas.height - 10) / (meanS + 0.0001)));
        y = canvas.height - 1 - meanVN[0];
        ctx.strokeStyle = 'green';
        ctx.fillStyle = 'green';
        ctx.beginPath();
        ctx.moveTo(0, y);
        for (let iTime = 1; iTime < nTimeSamples; iTime++) {
            y = canvas.height - 1 - meanVN[iTime];
            ctx.lineTo(iTime, y);
        }
        ctx.stroke();
        drawTextBG(ctx, (meanMean).toFixed(6), 10, 10);
    }
}

function multiTimeCanvasMouseMove(e) {
    id = e.target.id;
    let fs = (id == "funCanvasTime") ? multiTimeFronts : multiFrequencyFronts;
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;

    let xVec, yVec;
    if (viewOption == 1) {
        xVec = math.abs(math.transpose(fs)[x]);
        yVec = math.abs(fs[y]);
    } else {
        xVec = math.transpose(fs)[x].map((v) => v.toPolar().phi);
        yVec = fs[y].map((v) => v.toPolar().phi);
    }

    drawVector(xVec, true, "red", 1, "sampleX", "X", 0);
    drawVector(yVec, true, "red", 1, "sampleY", "Y", 0);
}
