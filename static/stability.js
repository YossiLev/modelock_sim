
function calculateStability() {
    drawMode = 4; // stability
    stabilityGraph = new graph2d(300, 300, stabilityCalcX, stabilityCalcY, stabilityCalcVal);
    drawMultiMode();
}
function stabilityCalcX(v) {
    return (v - 10) * 0.0005 ; // delta
}
function stabilityCalcY(v) {
    if (v == this.height - 1) return 20.0;
    return ((v - 100) / 300) * Math.abs((v - 100) / 300) ; // focal
}
function stabilityCalcVal(delta, focal) {
    let origDelta = elements[3].delta;
    let origFocal = elements[1].par[1];
    elements[3].delta = delta;
    elements[1].par[1] = focal;
    let M  = getMatOnRoundTrip(false);
    let A = M[0][0], B = M[0][1], C = M[1][0], D = M[1][1];
    let V = Math.abs(A + D);
    if (V < 2) {
        let a = 0;
    }
    let asVec = analyzeStability(M);
    elements[3].delta = origDelta;
    elements[1].par[1] = origFocal;
    return [V, asVec];
}
function stabilityColor(v) {
    if (v < 2) {
        return `rgba(${100 * v}, ${255}, ${100 * v}, 255)`;
    } else {
        return `rgba(${255}, ${Math.max(200 - 5 * v, 0)}, ${Math.max(200 - 5 * v, 0)}, 255)`;
    }
}
function stabilityCanvasMouseMove(e) {
    if (!isMouseDownOnMain) {
        return;
    }
    const canvas = document.getElementById(e.target.id);

    const o = stabilityGraph.locate(canvas, ...getClientCoordinates(e));
    if (o != null) {
        const waist = o.val[1][0] ? o.val[1][3] : -1.0;

        let inner = `<div>A + D => ${o.val[0].toFixed(3)}</div>` + 
            `<div>&delta; ${(o.x * 100.0).toFixed(2)}cm</div>` + 
            `<div>focal = ${(o.y * 1000).toFixed(2)}mm</div>` + 
            `<div>waist = ${waist.toFixed(6)}mm</div>` + 
            `<button type="button" onclick="fullCavityNewParams(${o.x}, ${o.y}, ${waist});">Show</button>`
            moverShow(e, inner);
    }
}
