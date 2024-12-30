var nTimeSamples = 1024;
var multiTimeFronts = [];
var multiFrequencyFronts = [];
var factorGain = [];
var IntensitySaturationLevel = 4;
var intensityTotalByIx = [];
var factorGainByIx = [];
var Ikl = 0.02;

function initMultiTime() {
    workingTab = 3
    multiTimeFronts = [];
    for (let i = 0; i < nSamples; i++) {
        multiTimeFronts.push([]);
    }
    for (let iTime = 0; iTime < nTimeSamples; iTime++) {
        let fr = getInitFront(beamParam);
        multiTimeFronts.push(getInitFront(beamParam))
        for (let i = 0; i < nSamples; i++) {
            multiTimeFronts[i].push(fr[i]);
        }
    }

    fftToFrequency();

    drawMultiMode();
}

function multiTimeRoundTrip() {

    // phaseChangeDuringKerr

    // gainCorrectionDueToSaturation
    // gainByfrequency
    // dispersionByFrequency

    // oneSideCavity
    // mirrorLoss
}

function timeCavityStep(step, redraw) {
    switch (step) {
        case 1:
            phaseChangeDuringKerr();
            break;
    }
    if (redraw) {
        fftToFrequency()
        drawMultiMode();
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
    for (let ix = 0; ix < nSamples; ix++) {
        let bin = multiTimeFronts[ix];
        let bin2 = math.abs(math.dotMultiply(bin, math.conj(bin)));
        let phaseShift = math.multiply(IklTimesI, bin2);
        multiTimeFronts[ix] = math.dotMultiply(bin, math.exp(phaseShift));
    }
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
        factorGainByIx.push(g0 / (1 +  intensityTotalByIx[ix] / IntensitySaturationLevel))
    }
}

function drawMultiTime() {
    if (!drawOption) {
        return
    }

    drawTimeFronts(1, document.getElementById("funCanvas1"));
    drawTimeFronts(2, document.getElementById("funCanvas2"));
}

function drawTimeFronts(domainOption, canvas) {

    const ctx = canvas.getContext("2d");
    drawMid = canvas.height / 2;

    var canvasWidth = canvas.width;
    var canvasHeight = canvas.height;
    var id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    var pixels = id.data;
    
    let fs = domainOption == 1 ? multiTimeFronts : multiFrequencyFronts
    for (let i = 0; i < nSamples; i++) {
        let off = i * nTimeSamples * 4;
        let  line = fs[i];
        for (let iTime = 0; iTime < nTimeSamples; iTime++) {
            if (viewOption == 1) {
                c = Math.floor(line[iTime].toPolar().r * 255.0);
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
}
