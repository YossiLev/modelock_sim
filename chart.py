
class Chart():
    def __init__(self, x, y, name):
        self.x = x.tolist()
        self.y = y.tolist()
        self.name = name
        pass

    def draw(self, draw, markX = []):
        gWidth = draw._image.width
        gHeight = draw._image.height

        maxX = max(self.x)
        minX = min(self.x)
        maxY = max(self.y)
        minY = min(self.y)

        hScale = gWidth / (maxX - minX + 0.001)
        vScale = gHeight / (maxY - minY + 0.001)

        hShift = - minX * hScale
        vShift = - minY * vScale

        for i in range(len(self.x) - 1):
            p1 = (self.x[i] * hScale + hShift, gHeight - 1 - (self.y[i] * vScale + vShift))
            p2 = (self.x[i + 1] * hScale + hShift, gHeight - 1 - (self.y[i + 1] * vScale + vShift))

            draw.line([p1, p2], fill="red", width=1)

        for i in markX:
            p1 = (self.x[i] * hScale + hShift, 0)
            p2 = (self.x[i] * hScale + hShift, gHeight)
            draw.line([p1, p2], fill="green", width=1)
