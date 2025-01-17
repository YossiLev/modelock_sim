class graph2d {
    constructor(width, height, fCalcX, fCalcY, fCalcVal) {
        this.height = height;
        this.width = width;
        this.fCalcX = fCalcX;
        this.fCalcY = fCalcY;
        this.fCalcVal = fCalcVal;
        this.build();        
    }

    build() {
        this.coordX = Array.from({length: this.width}).map((x, iw) => this.fCalcX(iw));
        this.coordY = Array.from({length: this.height}).map((y, ih) => this.fCalcY(ih));
        this.vals = this.coordY.map((y) => this.coordX.map((x) => this.fCalcVal(x, y)));
    }

    plot(ctx, fColor) {
        const margin = 50;
        const sx = margin;
        const dx = (ctx.canvas.width - 2 * margin) / this.width;
        const sy = margin;
        const dy = (ctx.canvas.height - 2 * margin) / this.height;

        ctx.fillStyle = "gray";
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

        const h = ctx.canvas.height;
        this.vals.forEach((line, ih) => {
            let ssy = h - 1 - (sy + dy * ih) - dy;
            line.forEach((val, iw) => {
                ctx.fillStyle = fColor(val);
                ctx.fillRect(sx + dx * iw, ssy, dx, dy);
            });
        });
    }

    locate(canvas, x, y) {
        const margin = 50;
        const sx = margin;
        const dx = (canvas.width - 2 * margin) / this.width;
        const sy = margin;
        const dy = (canvas.height - 2 * margin) / this.height;
        const h = canvas.height;

        let ix = Math.floor((x - sx) / dx);
        let iy = Math.floor((h - 1 - y - sy) / dy);
        if (ix >= 0 && ix < this.width && iy >= 0 && iy < this.height) {
            return {x: this.coordX[ix], y: this.coordY[iy], val: this.vals[iy][ix]};
        }

        return null;
    }

}