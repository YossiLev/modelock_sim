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