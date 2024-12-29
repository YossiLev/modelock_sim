var nTimeSamples = 1024;
var multiTimeFronts = [];
var multiFrequencyFronts = [];

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

    multiFrequencyFronts = [];
    for (let i = 0; i < nSamples; i++) {
        multiFrequencyFronts.push(fft(multiTimeFronts[i], 1.0));
    }

    drawMultiMode();
}

function drawMultiTime() {
    if (!drawOption) {
        return
    }

    drawTimeFronts(1, document.getElementById("funCanvas1"));
    drawTimeFronts(2, document.getElementById("funCanvas2"));
}

function drawTimeFronts(viewOption, canvas) {

    const ctx = canvas.getContext("2d");
    drawMid = canvas.height / 2;

    var canvasWidth = canvas.width;
    var canvasHeight = canvas.height;
    var id = ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    var pixels = id.data;
    
    let fs = viewOption == 1 ? multiTimeFronts : multiFrequencyFronts
    for (let i = 0; i < nSamples; i++) {
        let off = i * nTimeSamples * 4;
        let  line = fs[i];
        for (let iTime = 0; iTime < nTimeSamples; iTime++) {
            //if (viewOption == 1) {
                c = Math.floor(line[iTime].toPolar().r * 255.0);
            //} else {
            //    c = Math.floor((line[iTime].toPolar().phi / (2 * Math.PI) + 0.5) * 255.0);
            //}
            pixels[off++] = c;
            pixels[off++] = c;
            pixels[off++] = c;
            pixels[off++] = 255;
        }
    }  
    ctx.putImageData(id, 0, 0);
}
