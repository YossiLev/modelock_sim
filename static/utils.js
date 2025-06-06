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

function changeCavityLinesShift(lines, selectedVal, incVal) {
    if (selectedVal <= 0 || selectedVal >= lines.length - 1) {
        return lines;
    }
    if (lines[selectedVal].split(" ")[0].toUpperCase() == "P" ||
        lines[selectedVal - 1].split(" ")[0].toUpperCase() != "P" ||
        lines[selectedVal + 1].split(" ")[0].toUpperCase() != "P"
        ) {
        return lines;
    }
    let newLines = [];
    for (let i = 0; i < lines.length; i++) {
        if (i == selectedVal - 1) {
            newLines.push(changeline(lines[i], incVal));
        } else if (i == selectedVal + 1) {
            newLines.push(changeline(lines[i], - incVal));
        } else {
            newLines.push(lines[i]);
        }
    }
    return newLines;
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
        initElementsMultiMode(); initMultiMode(1); fullCavityGaussian();
        event.preventDefault();
    } else if (event.key === 'ArrowDown') {
        lines = changeCavityLines(lines, selectedVal, - incVal);
        initElementsMultiMode(); initMultiMode(1); fullCavityGaussian();
        textEl.value = lines.join("\n");
        event.preventDefault();
    } else if (event.key === 'ArrowLeft') {
        lines = changeCavityLinesShift(lines, selectedVal, - incVal);
        initElementsMultiMode(); initMultiMode(1); fullCavityGaussian();
        textEl.value = lines.join("\n");
        event.preventDefault();
    } else if (event.key === 'ArrowRight') {
        lines = changeCavityLinesShift(lines, selectedVal, incVal);
        initElementsMultiMode(); initMultiMode(1); fullCavityGaussian();
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
