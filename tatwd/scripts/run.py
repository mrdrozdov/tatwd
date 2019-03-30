import json
import os
import tempfile

import turtle


def tree_to_string(parse):
    if not isinstance(parse, (list, tuple)):
        return parse
    if len(parse) == 1:
        return parse[0]
    else:
        return '( ' + tree_to_string(parse[0]) + ' ' + tree_to_string(parse[1]) + ' )'


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


class Node(object):
    def __init__(self, val=None, width=None, height=None, pos=None, mid=None,
                 lx=None, rx=None):
        self.val = val
        self.width = width
        self.height = height
        self.pos = pos
        self.mid = mid
        self.lx = lx
        self.rx = rx


class CustomTurtle(object):
    def __init__(self, turtle):
        self.turtle = turtle
        self.history = []

    def goto(self, x, y):
        self.turtle.goto(x, y)
        self.history.append((x, y))

    def position(self):
        pos = self.turtle.position()
        self.history.append(pos)
        return pos

    def bounding_box(self):
        xs = [pos[0] for pos in self.history]
        ys = [pos[1] for pos in self.history]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # -740.0 -628 0 200.0
        print(xmin, xmax, ymin, ymax)

        height = 500
        width = 1500
        pad = 10

        x0 = width/2 + xmin - pad
        x1 = width/2 + xmax + pad
        y0 = height/2 + ymin - pad
        y1 = height/2 + ymax + pad

        return dict(x0=x0, x1=x1, y0=y0, y1=y1)


class TreeFig(object):
    def __init__(self, style='arc', color='#000', size=None):
        self.style = style
        self.color = color
        self.size = size
        self.turtle = None
        self.cturtle = None

    def setup_turtle(self, widthWindow, heightWindow, scale, x0):

        self.cturtle = CustomTurtle(turtle)

        turtle.setup(widthWindow, heightWindow)
        turtle.reset()
        turtle.hideturtle()
        turtle.penup()
        self.cturtle.goto(x0, 0)


    def srdraw(self, s, ts, ws, x0=0, y0=30, yMax=200, adjust_top=True, mask=None):
        n = len(s)
        offsets = [x0] + [x0 + sum(ws[:i]) for i in range(1,n)]
        buff = [Node(val=w, height=1, width=ws[n-i-1], mid=offsets[n-i-1] + ws[n-i-1]/2, pos=n-i-1) for i, w in enumerate(s[::-1])]
        sofar = 0
        stack = []

        if mask is None:
            mask = [0] * len(ts)
        mask = [x == 1 for x in mask]

        # Draw Vars
        interval = (yMax-y0) / (n-1)

        # SR
        for t, m in zip(ts, mask):
            if t == 0: # Shift
                x = buff.pop()
                sofar += 1
            elif t == 1: # Reduce
                rx = stack.pop()
                lx = stack.pop()

                xpos = min(lx.pos, rx.pos)

                lmid = lx.mid
                rmid = rx.mid
                xmid = offsets[xpos] + lx.width / 2 + (lx.width / 2 + rx.width / 2) / 2

                x = Node(val=(lx.val, rx.val),
                         height=max(lx.height, rx.height) + 1,
                         width=lx.width + rx.width,
                         pos=xpos,
                         mid=xmid,
                         )

                if adjust_top:
                    xmid = lx.mid + (rx.mid - lx.mid) / 2

                    x = Node(val=(lx.val, rx.val),
                             height=max(lx.height, rx.height) + 1,
                             width=lx.width + rx.width,
                             pos=xpos,
                             mid=xmid,
                             lx=lx,
                             rx=rx,
                             )

                ly = y0 + (lx.height-1) * interval
                ry = y0 + (rx.height-1) * interval
                xy = y0 + (x.height-1) * interval

                def arctree():

                    ly_ = ly

                    # Get Circle Coordinates
                    ((cx, cy), radius) = define_circle((lmid, ly_), (xmid, xy), (rmid, ry))

                    # Draw arc
                    turtle.penup()
                    self.cturtle.goto(lmid, ly_)
                    ldeg = turtle.towards(cx, cy)
                    self.cturtle.goto(rmid, ry)
                    rdeg = turtle.towards(cx, cy)

                    draw_line = False

                    while True:
                        if ldeg < rdeg:
                            ly_ += 10
                            ((cx, cy), radius) = define_circle((lmid, ly_), (xmid, xy), (rmid, ry))
                            self.cturtle.goto(lmid, ly_)
                            ldeg = turtle.towards(cx, cy)
                            self.cturtle.goto(rmid, ry)
                            rdeg = turtle.towards(cx, cy)
                            draw_line = True
                            # ldeg += 360
                        else:
                            break
                    turtle.setheading(rdeg - 90)
                    turtle.pendown()
                    turtle.circle(radius, extent=ldeg-rdeg)
                    turtle.penup()

                    if draw_line:
                        self.cturtle.goto(lmid, ly_)
                        turtle.pendown()
                        self.cturtle.goto(lmid, ly)
                        turtle.penup()
                        

                def boxtree():
                    turtle.penup()
                    self.cturtle.goto(lmid, ly)
                    turtle.pendown()
                    self.cturtle.goto(lmid, xy)
                    self.cturtle.goto(xmid, xy)
                    self.cturtle.goto(rmid, xy)
                    self.cturtle.goto(rmid, ry)
                    turtle.penup()

                def standardtree():
                    turtle.penup()
                    self.cturtle.goto(lmid, ly)
                    turtle.pendown()
                    self.cturtle.goto(xmid, xy)
                    self.cturtle.goto(rmid, ry)
                    turtle.penup()

                turtle.pensize(self.size)
                turtle.pencolor(self.color)

                if m:
                    default = turtle.pencolor()
                    turtle.pencolor('#0376BA')

                if self.style == 'standard':
                    standardtree()
                elif self.style == 'box':
                    boxtree()
                elif self.style == 'arc':
                    arctree()

                if m:
                    turtle.pencolor(default)

            stack.append(x)


    def draw_tree(self, parse):

        def write_sentence(s):
            
            font = ("Arial", 20 * scale, "normal")
            prv = self.cturtle.position()[0]
            for i, w in enumerate(s):
                if i > 0:
                    turtle.write(' ', move=True, font=font)
                turtle.write(w, move=True, font=font)
                nxt = self.cturtle.position()[0]
                mid = prv + (nxt - prv) / 2
                width = nxt-prv
                prv = nxt
                yield dict(mid=mid, nxt=nxt, width=width)

        scale = 1
        # x0 = -300 * scale
        y0 = 65 * scale
        yMax = 200 * scale
        widthWindow = 1500 * scale
        x0 = -widthWindow/2 + 10

        s, ts = convert_binary_bracketing(tree_to_string(parse))

        lst = list(write_sentence(s))
        mids = [d['mid'] for d in lst]
        nxts = [d['nxt'] for d in lst]
        ws = [d['width'] for d in lst]
        # x0 = nxts[0] - ws[0]
        n = len(mids)
        xMax = max(mids)

        self.srdraw(s, ts, ws, x0=x0)

        box = self.cturtle.bounding_box()

        return box


def run_one(options, data):
    example_id = data['example_id']
    parse = data['binary_tree']

    with tempfile.NamedTemporaryFile(mode='w') as f:
        path_ps = f.name
        path_pdf = os.path.join(options.out, '{}.pdf'.format(example_id))

        turtle.speed('fastest')

        fig = TreeFig(style=options.style, color=options.color, size=options.size)

        # Setup
        scale = 1
        # x0 = -300 * scale
        y0 = 65 * scale
        yMax = 200 * scale
        widthWindow = 1500 * scale
        heightWindow = 500 * scale
        x0 = -widthWindow/2 + 10

        fig.setup_turtle(widthWindow, heightWindow, scale, x0)

        # Draw
        ts = turtle.getscreen()
        ts.tracer(0, 0) # https://stackoverflow.com/questions/16119991/how-to-speed-up-pythons-turtle-function-and-stop-it-freezing-at-the-end
        bounding_box = fig.draw_tree(parse)
        ts.update()
        ts.getcanvas().postscript(file=path_ps)

        print('writing to {}'.format(path_pdf))

        os.system('ps2pdf -dEPSCrop {} {}'.format(path_ps, path_pdf))

        # Crop the image.
        from PyPDF2 import PdfFileWriter, PdfFileReader

        # print('bounding box = {}'.format(bounding_box))

        output_filename = os.path.join(options.out, '{}-cropped.pdf'.format(example_id))
        input1 = PdfFileReader(open(path_pdf, "rb"))
        output = PdfFileWriter()

        page = input1.getPage(0)
        # print('mediaBox', page.mediaBox)
        # print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
        page.trimBox.lowerLeft = (bounding_box['x0'], bounding_box['y0'])
        page.trimBox.upperRight = (bounding_box['x1'], bounding_box['y1'])
        page.cropBox.lowerLeft = (bounding_box['x0'], bounding_box['y0'])
        page.cropBox.upperRight = (bounding_box['x1'], bounding_box['y1'])
        output.addPage(page)

        print('writing to {}'.format(output_filename))
        outputStream = open(output_filename, "wb")
        output.write(outputStream)
        outputStream.close()


def run(options):

    table = {}

    with open(options.path) as f:
        for line in f:
            data = json.loads(line)
            table[data['example_id']] = data

    for example_id in options.ids.split(','):
        data = table[example_id]
        run_one(options, data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='examples/example.json', type=str)
    parser.add_argument('--ids', default='ptb1094', type=str)
    parser.add_argument('--out', default='./tmp', type=str)
    parser.add_argument('--style', default='box', choices=('standard', 'box', 'arc'))
    parser.add_argument('--color', default='#000', type=str)
    parser.add_argument('--size', default=None, type=float)
    options = parser.parse_args()

    run(options)