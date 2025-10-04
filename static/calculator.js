function drawPulseGraph() {
    const canvas = document.getElementById("pulsesplit");
    const ctx = canvas.getContext("2d");
    const cWidth = canvas.width;
    const cHeight = canvas.height;
    const margin = 20;
    const wy1 = margin;
    const wy2 = cHeight - margin;
    const wx1 = margin;
    const wx2 = cWidth - margin;
    const cavWidth = wx2 - wx1;

    ctx.fillStyle = "gray";
    ctx.fillRect(0, 0, cWidth, cHeight);

    ctx.strokeStyle = "red";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(wx1, wy1);
    ctx.lineTo(wx1, wy2);
    ctx.moveTo(wx2, wy1);
    ctx.lineTo(wx2, wy2);
    ctx.stroke();

    const nHar = nSamples = getFieldInt("pulseHarmony");
    const slope = 0.2;


    ctx.lineWidth = 2;
    ctx.strokeStyle = "yellow";
    ctx.beginPath();
    for (let iHar = 0; iHar < nHar; iHar++) {
        let y1 = wy1;
        let x1 = wx1 + 2 * cavWidth * iHar / nHar;
        let x2 = wx1 + cavWidth;
        let y2 = y1 + slope * (x2 - x1);
        ctx.moveTo(x1, cHeight - 1 - y1);
        ctx.lineTo(x2, cHeight - 1 - y2);
        let dir = -1;
        while (y2 < cHeight) {
            x1 = x2;
            y1 = y2;
            x2 = x1 + dir * cavWidth;
            y2 = y1 + dir * slope * (x2 - x1);
            ctx.moveTo(x1, cHeight - 1 - y1);
            ctx.lineTo(x2, cHeight - 1 - y2);
            dir = - dir;
        }

    }
    ctx.stroke();
}

function spreadDiodeUpdatedData(data) {
    // if (data.rounds) {
    //     document.getElementById("stepsCounter").value = `${data.rounds}`;
    // }
    // if (data.more) {
    //     document.getElementById("stepsCounter").style.color = "red";
    //     document.getElementById("stepsCounter").style.animation = "blink 1s infinite";
    // } else {
    //     document.getElementById("stepsCounter").style.color = "black";
    //     document.getElementById("stepsCounter").style.animation = "";
    // }
    // if (data.samples) {
    //     for (sample of data.samples) {
    //         canvas = document.getElementById(sample.name);
    //         drawTimeNumData(openVec(sample.samples), 0, canvas);
    //     }
    // }
    // if (data.pointer) {
    //     modifyPointer(data.pointer)
    // }
    if (data.graphs) {
        let backColor = data.more ? "#ffeedd": "white";
        for (graph of data.graphs) {
            let clear = true;
            for (line of graph.lines) {
                drawPlotVector(line.values, graph.id, {clear: clear, color: line.color, pixelWidth: 1, 
                    allowChange: true, name: data.title, start: 0, message: "", zoomX: Object.hasOwn(line, 'zoomx') ? line.zoomx : 1,
                    backColor: backColor});
                //
                //    drawVector(line.values, clear, line.color, 1, true, graph.id, "",  0, line.text, 
                //        Object.hasOwn(line, 'zoomx') ? line.zoomx : 1, backColor);
                clear = false;
            }
            // if (graph.name == "gr5") {
            //     if (graph.lines.length > 0) {
            //         currentPlot3dValues = graph.lines[0].values;
            //         if (document.getElementById("cbxAutoRecord").checked) {
            //             AddPlot3D();
            //         }
            //     }
            // }
        }

    }
    // if (data.view_buttons) {
    //     for (let part in [0, 1]) {
    //         for (let i = 0; i < 14; i++) {
    //             but = document.getElementById(`view_button-${part}-${i + 1}`);  
    //             if (data.view_buttons.view_on_stage[part] == `${i + 1}`) {
    //                 but.classList.add("buttonH");
    //             } else {
    //                 but.classList.remove("buttonH");
    //             }
    //         }
    //         for (let i of ["Frq", "Amp"]) {
    //             but = document.getElementById(`view_button-${part}-${i}`);  
    //             if (data.view_buttons.view_on_amp_freq[part] == `${i}`) {
    //                 but.classList.add("buttonH");
    //             } else {
    //                 but.classList.remove("buttonH");
    //             }
    //         }
    //         for (let i of ["Phs", "Abs", "Pow"]) {
    //             but = document.getElementById(`view_button-${part}-${i}`);  
    //             if (data.view_buttons.view_on_abs_phase[part] == `${i}`) {
    //                 but.classList.add("buttonH");
    //             } else {
    //                 but.classList.remove("buttonH");
    //             }
    //         }
    //     }
    // }
}

