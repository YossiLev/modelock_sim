document.addEventListener("DOMContentLoaded", function () {
    const fullPageDiv = document.getElementById('fullPage');

    function mutationCallback(mutationsList, observer) {
        if (document.getElementById('numData')) {
            setTimeout(numDataMutated, 1);
        }
    }

    if (fullPageDiv) {
        const observer = new MutationObserver(mutationCallback);
        const mutationConfig = { attributes: true, childList: true, subtree: true, characterData: true };
        observer.observe(fullPageDiv, mutationConfig);
    }
});

function drawTextBG(ctx, txt, x, y, color = '#000', font = "8pt Courier") {
    ctx.save();
    ctx.font = font;
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#fff';
    var width = ctx.measureText(txt).width;
    ctx.beginPath();
    ctx.roundRect(x, y - 1, width, parseInt(font, 10) + 3, 3);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.fillText(txt, x, y);
    ctx.restore();
}

function getClientCoordinates(e) {
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - bounds.left;
    var y = e.clientY - bounds.top;
    return [x, y];
}

var doCoverStopper;
function doCover(func, params) {
    let counter = 1;
    doCoverStopper = false;

    doCoverX(func, params, counter);
}
function doCoverX(func, params, counter) {

    let nextParams = func(params);

    if (!doCoverStopper && nextParams != null) {
        let inner = `<div>Process in progress (${counter})</div>` + 
                    `<button type="button" onclick="doCoverStopper = true;">Stop Process</button>`
        moverShow(null, inner);
        setTimeout(doCoverX, 1, func, nextParams, counter + 1);
    } else {
        moverHide(1);
    }
}

function snackBar(message) {
    var x = document.getElementById("snackbar");
    x.innerHTML = message;
    x.className = "show";
    setTimeout(function(){ x.className = x.className.replace("show", ""); }, 3000);
  }

randomSeed = 12345;
function randomLCG() {
    let a = 1664525;
    let c = 1013904223;
    let m = Math.pow(2, 32);
    randomSeed = (a * randomSeed + c) % m;
    return randomSeed / m; // Returns a float between 0 and 1
}
function randomLCGSetSeed(seed) {
    randomSeed = seed;
}
