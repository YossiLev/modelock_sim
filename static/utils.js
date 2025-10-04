


document.addEventListener("DOMContentLoaded", function () {
    initializeDragAndDrop();

    document.body.addEventListener('htmx:afterSwap', function(evt) {
        initializeDragAndDrop();
    });
    
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

function numDataMutated() {
    numData = document.getElementById("numData");
    s = numData.innerText;
    if (s.length > 0) {
        data = JSON.parse(s);
        numData.innerText = "";
        let spreadFunction = null;
        if (data.type == "diode") {
            spreadFunction = spreadDiodeUpdatedData;
        } else if (data.type == "multi_mode") {
            spreadFunction = spreadMultiModeUpdatedData;
        }
        if (spreadFunction == null) {
            console.log("Unknown data type: " + data.type);
            return;
        }
        if (data.delay > 0) {
            setTimeout(() => {
                spreadFunction(data);
            }, data.delay);
        } else {
            spreadFunction(data);
        }
    }
}

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

function toggleVisibility(el) {
    if (el.style.visibility === "hidden" || el.style.visibility === "") {
        el.style.visibility = "visible";
    } else {
        el.style.visibility = "hidden";
    }
}

function getClientCoordinates(e) {
    var bounds = e.target.getBoundingClientRect();
    var x = e.clientX - Math.round(bounds.left);
    var y = e.clientY - Math.round(bounds.top);
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


function initializeDragAndDrop() {
    let dragging = null;

    const handles = document.querySelectorAll('.handle');
    const containers = document.querySelectorAll('.container');
    handles.forEach(handle => {
        handle.addEventListener('dragstart', e => {
            dragging = handle.parentElement;
            setTimeout(() => dragging.style.display = 'none', 0);
        });

        handle.addEventListener('dragend', e => {
            dragging.style.display = '';
            dragging = null;
        });
    });

    containers.forEach(container => {
        container.addEventListener('dragover', e => {
            e.preventDefault();
        });

        container.addEventListener('dragenter', e => {
            if (container !== dragging) {
            container.classList.add('drag-over');
            }
        });

        container.addEventListener('dragleave', () => {
            container.classList.remove('drag-over');
        });

        container.addEventListener('drop', e => {
            e.preventDefault();
            container.classList.remove('drag-over');

            if (container !== dragging) {
                const list = document.getElementById('containerList');
                const items = Array.from(list.children);
                const draggedIndex = items.indexOf(dragging);
                const targetIndex = items.indexOf(container);

                if (draggedIndex < targetIndex) {
                    list.insertBefore(dragging, container.nextSibling);
                } else {
                    list.insertBefore(dragging, container);
                }
            }
        });
    });
}

function parseStrictFloat(str) {
  const floatRegex = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
  return floatRegex.test(str.trim()) ? parseFloat(str) : NaN;
}

function validateMat(event) {
    tv = event.target.id;
    matName = tv.split("_")[0];
    validateMatName(matName);
}
function validateMatName(matName) {

    chars = "ABCD".split("")
    vals = chars.map(c => document.getElementById(`${matName}_${c}`).value);
    nums = vals.map(v => parseStrictFloat(v));
    if (nums.findIndex(v => isNaN(v) >= 0)) {
        nums.forEach((v, i) => { 
            document.getElementById(`${matName}_${chars[i]}`).style.color = isNaN(v) ? "red" : "black";
        });
    } else {
        det = nums[0] * nums[3] - nums[1] * nums[2];
        el = document.getElementById(`${matName}_msg`)
        if (Math.abs(det - 1.0) > 0.000001) {
            el.innerHTML = `&#9888; det=${det}`;
            el.style.visibility = "visible";
        } else {
            el.style.visibility = "hidden";
        }
    }
}

function AbcdMatEigenValuesCalc(name) {
    let eigenText = document.getElementById(name + "_eigen"); 
    if (eigenText.style.visibility == "visible") {
        toggleVisibility(eigenText);
        eigenText.innerHTML = "";
        return;
    }
    let A = document.getElementById(name + "_A").value;
    let D = document.getElementById(name + "_D").value; 

    let a = parseFloat(A);
    let d = parseFloat(D);

    let disc = 1 - (a + d) * (a + d) * 0.25;
    if (disc >= 0) {
        // stable
        let re = (d + a) / 2.0
        let im = Math.sqrt(disc)
        eigenText.innerHTML = `&#x1F603; ${re.toFixed(3)} &#xb1; i${im.toFixed(3)}`;
    } else {
        // unstable
        let re = (d + a) / 2.0
        let re2 = Math.sqrt(-disc)
        eigenText.innerHTML = `&#x1F62D; ${(re + re2).toFixed(3)} or ${(re - re2).toFixed(3)}`;
    }
    toggleVisibility(eigenText);
}

function MatOnQvec(M, Q) {
    let [A, B, C, D] = [M[0][0], M[0][1], M[1][0], M[1][1]];
    let [x, y] = Q;
    let denom = C * x + D;
    if (Math.abs(denom) < 1e-20) {
        return [1e20, 0];
    }
    let x2 = (A * x + B) / denom;
    let y2 = y / (denom * denom);
    return [x2, y2];
}

function MatOnQvecInve(M, QInv) {
    let [A, B, C, D] = [M[0][0], M[0][1], M[1][0], M[1][1]];
    let [x, y] = QInv;
    let denom = B * x + A;
    if (Math.abs(denom) < 1e-20) {
        return [1e20, 0];
    }
    let x2 = (D * x + C) / denom;
    let y2 = y / (denom * denom);
    return [x2, y2];
}

function OneOverQ(Q) {
    let [x, y] = Q;
    let denom = x * x + y * y;
    if (Math.abs(denom) < 1e-20) {
        return [1e20, 0];
    }
    return [x / denom, -y / denom];
}

function changeVal(val, incVal) {
    let obj = extractLength(val);
    obj.val += incVal;
    return buildLength(obj);
}
function changeline(line, incVal) {
    let parts = line.split(" ");
    parts[1] = changeVal(parts[1], incVal);

    return parts.join(" ");
}

// change my value up or down
function changeCavityLines(lines, selectedVal, incVal) {
    let newLines = [];
    for (let i = 0; i < lines.length; i++) {
        if (i == selectedVal) {
            newLines.push(changeline(lines[i], incVal));
        } else {
            newLines.push(lines[i]);
        }
    }
    return newLines;
}

// change the values of my previous P (propogation) and next P one up and one down to create a left/right shift
function changeCavityLinesShift(lines, selectedVal, incVal) {
    if (selectedVal <= 0 || selectedVal >= lines.length - 1) {
        return lines;
    }
    // find the previous and next P lines
    let prevP = -1;
    let nextP = -1;
    for (let i = selectedVal - 1; i >= 0; i--) {
        if (lines[i].split(" ")[0].toUpperCase() == "P") {
            prevP = i;
            break;
        }
    }
    for (let i = selectedVal + 1; i < lines.length; i++) {
        if (lines[i].split(" ")[0].toUpperCase() == "P") {
            nextP = i;
            break;
        }
    }
    if (prevP == -1 || nextP == -1) {
        return lines;
    }
    let newLines = [];
    for (let i = 0; i < lines.length; i++) {
        if (i == prevP) {
            newLines.push(changeline(lines[i], incVal));
        } else if (i == nextP) {
            newLines.push(changeline(lines[i], - incVal));
        } else {
            newLines.push(lines[i]);
        }
    }
    return newLines;
}

function multiModeRefresh() {
    setTimeout(() => {
        initElementsMultiMode(); initMultiMode(1); fullCavityGaussian();
    }, 1);
}
function handlePickerKeyDown(event) {
    let name = event.target.id;
    let incVal = parseFloat(document.getElementById(`${name}_edit_inc`).value);
    let selectedVal = parseInt(document.getElementById(`${name}_val`).value);
    let textEl = document.getElementById(`${name}_text`);
    let textElVal = textEl.value;
    let lines = textElVal.split("\n");

    if (event.key === 'ArrowUp') {
        lines = changeCavityLines(lines, selectedVal, incVal);
        textEl.value = lines.join("\n");
        multiModeRefresh();
        event.preventDefault();
    } else if (event.key === 'ArrowDown') {
        lines = changeCavityLines(lines, selectedVal, - incVal);
        multiModeRefresh();
        textEl.value = lines.join("\n");
        event.preventDefault();
    } else if (event.key === 'ArrowLeft') {
        lines = changeCavityLinesShift(lines, selectedVal, - incVal);
        multiModeRefresh();
        textEl.value = lines.join("\n");
        event.preventDefault();
    } else if (event.key === 'ArrowRight') {
        lines = changeCavityLinesShift(lines, selectedVal, incVal);
        multiModeRefresh();
        textEl.value = lines.join("\n");
        event.preventDefault();
    }
}

function pickerDivsSelect(name, sel) {
    let divVal = document.getElementById(`${name}_val`);
    divVal.value = `${sel}`;
    for (let i = 0, divB = divVal.nextElementSibling; divB != null; i++, divB = divB.nextElementSibling) {
        if (i == sel) {
            divB.style.backgroundColor = "#00FF00";
        } else {
            divB.style.backgroundColor = "#FFFFFF";
        }
    }
    let text = document.getElementById(`${name}_text`).value;
    inpEdit.value = text.split("\n")[sel];
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

function drawPlotVector(v, id, params = {}) {
    if (!drawOption) {
        return
    }
    let {
        clear = true,      // default
        color = "red",     // default
        pixelWidth = drawW,// default
        allowChange = false, // default     
        name = "",          // default
        start = drawSx,    // default
        message = "",      // default
        zoomX = 1,         // default
        backColor = "white" // default
    } = params;
    const canvas = document.getElementById(id);
    const ctx = canvas.getContext("2d");

    const paddingLeft = 140;
    const paddingRight = 110;
    const paddingTop = 40;
    const paddingBottom = 20;
    const paddingHeight = paddingTop + paddingBottom;
    const paddingWidth = paddingLeft + paddingRight;

    const graphHeight = canvas.height - paddingHeight - 20;
    const graphTop = paddingTop + 10;
    const graphBottom = graphTop + graphHeight;
    const graphWidth = canvas.width - paddingWidth - 70;
    const graphLeft = paddingLeft + 35;
    const graphRight = graphLeft + graphWidth;

    if (typeof v === 'string') {
        v = JSON.parse(v);
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
    console.log(`draw vector length = ${l} id=${id} clear=${clear} vectors=${vectors.length}`);
    let fac;
    const prevCompare = document.getElementById('cbxPrevCompare')?.checked;
    if (l > 0) {
        vMax = Math.max(...v);
        vMin = Math.min(...v);
        fac = Math.max(Math.abs(vMax), Math.abs(vMin));
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

    if (change && pixelWidth > graphWidth / l) {
        pixelWidth = graphWidth / l;
    }

    ctx.fillStyle = backColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);     
    if (isMouseDownOnGraph) {
        const canvas = document.getElementById(id);
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ddd";
        ctx.fillRect(graphLeft + mouseOnGraphStart * drawW, 0, (mouseOnGraphEnd - mouseOnGraphStart) * drawW, 200)
    }
    ctx.strokeStyle = "black";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(paddingLeft, paddingTop, canvas.width - paddingWidth, canvas.height - paddingHeight);
    ctx.stroke();

    drawTextBG(ctx, name, 310, 2, "black", "12pt Arial");2

    ctx.strokeStyle = `gray`;
    ctx.beginPath();
    ctx.moveTo(graphLeft, graphBottom);
    ctx.lineTo(graphLeft + l * pixelWidth, graphBottom);
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
        fac = graphHeight / fac
    }
    let zoomVal = parseFloat(document.getElementById(`${id}-zoomVal`).innerHTML);
    fac *= zoomVal;
    vectors.forEach((vo, iVec) => {
        if (selectVal == 0 || iVec < selectVal) {
            let sx = vo.s;// - pixelWidth * vo.z * vo.vec.length / 2 + canvas.width / 2;
            let dx = pixelWidth * vo.z;
            ctx.strokeStyle = vo.c;
            ctx.beginPath();
            ctx.moveTo(graphLeft, graphBottom - Math.floor(fac * vo.vec[0]));
            for (let i = 1; i < l; i++) {
                ctx.lineTo(graphLeft + i * dx, graphBottom - Math.floor(fac * vo.vec[i]));
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
        ctx.moveTo(graphLeft, graphBottom - Math.floor(fac * drawVectorComparePrevious[0]));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(graphLeft + i * pixelWidth, graphBottom - Math.floor(fac * drawVectorComparePrevious[i]));
        }
        ctx.stroke();
        ctx.strokeStyle = 'blue';
        ctx.beginPath();
        ctx.moveTo(graphLeft, graphBottom - Math.floor(fac * (v[0] - drawVectorComparePrevious[0])));
        for (let i = 1; i < l; i++) {
            ctx.lineTo(graphLeft + i * pixelWidth, graphBottom - Math.floor(fac * (v[i] - drawVectorComparePrevious[i])));
        }
        ctx.stroke();
    } else {
        drawVectorComparePrevious = math.clone(v);
    }
}
