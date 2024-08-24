#!/usr/bin/env python3

"""A simple splines package.

This package aims to be simple to use, but also simple to read - it's as much an exegesis on splines as it is a package. It's also developed to support somewhat more general exploratory programming with curves, so it supports slightly peripheral things like the ability to fit low-order polynomials to sequences of points.

Note that the only code in the package which knows the dimensionality of the space being worked in is the Point class. This code is two-dimensional, but a few changes to Point would make it three-dimensional.

Performance is almost certainly execrable.

(c) 2005 Tom Anderson <twic@urchin.earth.li> - all rights reserved

Redistribution and use in source and binary forms, with or without modification, are permitted.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

def tuples2points(ts):
    return list(map(lambda t: Point(*t), ts))

class Point(object):
    "A point. Actually a somewhat general vector, but never mind. Implements more behaviour than it strictly needs to."
    __slots__ = ("x", "y")
    
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, factor):
        return Point(self.x * factor, self.y * factor)
    
    def __truediv__(self, factor):
        return Point(self.x / factor, self.y / factor)
    
    def __neg__(self):
        return Point(-self.x, -self.y)
    
    def __abs__(self):
        "Computes the magnitude of the vector."
        return (~self) ** 0.5
    
    def __xor__(self, other):
        "Computes the dot product."
        return self.x * other.x + self.y * other.y
    
    def __invert__(self):
        "Computes the dot-square, i.e., the dot product of the point with itself."
        return self.x ** 2 + self.y ** 2
    
    def __hash__(self):
        return hash(self.x) ^ hash(self.y)
    
    def __lt__(self, other):
        "Comparison method replacing __cmp__."
        return (self.x, self.y) < (other.x, other.y)
    
    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)
    
    def __iter__(self):
        "Makes it possible to do tuple(p)."
        return iter((self.x, self.y))
    
    def __bool__(self):
        return self.x != 0.0 or self.y != 0.0
    
    def __str__(self):
        return f"({self.x},{self.y})"
    
    def __repr__(self):
        return f"splines.Point({self.x}, {self.y})"

class Curve(object):
    """A curve describes a continuous sequence of points."""
    def __call__(self, u):
        "Finds the position of the curve at coordinate u."
        raise NotImplementedError

class Line(Curve):
    "A straight line."
    
    def __init__(self, a, b):
        "a and b are the 0th and 1st order coefficients, respectively."
        self.a = a
        self.b = b
    
    def __call__(self, u):
        return self.a + (self.b * u)
    
    def __str__(self):
        return f"{self.a} + {self.b}u"
    
    def __repr__(self):
        return f"splines.Line({repr(self.a)}, {repr(self.b)})"
    
    @staticmethod
    def fit(p, q):
        "Fits a line to two points, so that at u = 0, it passes through p, and at u = 1, it passes through q."
        a = p
        b = q - p
        return Line(a, b)

class Quadratic(Curve):
    "A quadratic curve."
    
    def __init__(self, a, b, c):
        "a, b and c are the 0th, 1st and 2nd order coefficients, respectively."
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, u):
        return self.a + (self.b * u) + (self.c * (u ** 2))
    
    def __str__(self):
        return f"{self.a} + {self.b}u + {self.c}u**2"
    
    def __repr__(self):
        return f"splines.Quadratic({repr(self.a)}, {repr(self.b)}, {repr(self.c)})"
    
    @staticmethod
    def fit(o, p, q):
        "Fits a quadratic to three points, so that it passes through o, p, and q at u = -1, 0, and 1, respectively."
        a = p
        b = (q - o) / 2.0
        c = ((q + o) / 2.0) - p
        return Quadratic(a, b, c)

class Cubic(Curve):
    "A cubic curve."
    
    def __init__(self, a, b, c, d):
        "a, b, c, and d are the 0th, 1st, 2nd, and 3rd order coefficients, respectively."
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, u):
        return self.a + (self.b * u) + (self.c * (u ** 2)) + (self.d * (u ** 3))
    
    def __str__(self):
        return f"{self.a} + {self.b}u + {self.c}u**2 + {self.d}u**3"
    
    def __repr__(self):
        return f"splines.Cubic({repr(self.a)}, {repr(self.b)}, {repr(self.c)}, {repr(self.d)})"
    
    @staticmethod
    def fit(o, p, q, r):
        "Fits a cubic to four points, so that it passes through o, p, q, and r at u = -1, 0, 1, and 2, respectively."
        a = p
        c = ((o + q) / 2.0) - a
        d = (((r - (q * 2.0)) + a) - (c * 2.0)) / 6.0
        b = ((q - o) / 2.0) - d
        return Cubic(a, b, c, d)

class Spline(Curve):
    "A spline is a curve defined by a sequence of knots."
    
    def __init__(self, knots=None):
        self.knots = list(knots) if knots else []
    
    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(map(str, self.knots))}])"

class PiecewiseSpline(Spline):
    "A piecewise spline is one in which the curve is drawn by constructing 'pieces'."
    
    def __getitem__(self, i):
        "Get the ith piece of the spline."
        raise NotImplementedError
    
    def __call__(self, u):
        if u < 0.0:
            i = 0
        elif u >= len(self):
            i = len(self) - 1
        else:
            i = int(u)
        v = u - i
        return self[i](v)
    
    def __len__(self):
        "The length of a spline is the number of pieces in it."
        return len(self.knots) - (2 * getattr(self, "looseKnots", 0)) - 1

class Polyline(PiecewiseSpline):
    "The simplest possible spline!"
    
    def __getitem__(self, i):
        return Line.fit(self.knots[i], self.knots[i + 1])

class NaturalCubicSpline(PiecewiseSpline):
    "The daddy! Caches results for efficiency."
    
    def __init__(self, knots=None):
        super().__init__(knots)
        self.cachedknots = None
    
    def __getitem__(self, i):
        if self.knots != self.cachedknots:
            self.calculate()
        return self.pieces[i]
    
    def calculate(self):
        "This code is ultimately derived from some written by Tim Lambert."
        p = self.knots
        self.cachedknots = list(p)
        g = _gamma(len(p))
        e = _epsilon(g, _delta(p, g))
        self.pieces = []
        for i in range(len(p) - 1):
            a = p[i]
            b = e[i]
            c = ((p[i + 1] - p[i]) * 3.0) - ((e[i] * 2.0) + e[i + 1])
            d = ((p[i] - p[i + 1]) * 2.0) + e[i] + e[i + 1]
            self.pieces.append(Cubic(a, b, c, d))

def _gamma(n):
    g = [0.5]
    for i in range(1, n - 1):
        g.append(1.0 / (4.0 - g[i - 1]))
    g.append(1.0 / (2.0 - g[-1]))
    return g

def _delta(p, g):
    d = [(p[1] - p[0]) * (g[0] * 3.0)]
    for i in range(1, len(p) - 1):
        d.append(((p[i + 1] - p[i - 1]) * 3.0 - d[i - 1]) * g[i])
    d.append(((p[-1] - p[-2]) * 3.0 - d[-1]) * g[-1])
    return d

def _epsilon(g, d):
    "Performs backward substitution to solve for epsilon."
    e = [d[-1]]
    for i in range(len(d) - 2, -1, -1):
        e.append(d[i] - (e[-1] * g[i]))
    e.reverse()
    return e

def test_spline(splinetype=NaturalCubicSpline, trim=False):
    "Fits a spline of some sort through a simple spiral-shaped sequence of knots."
    knots = [(0, 0), (0, 1), (1, 0), (0, -2), (-3, 0)]  # a spiral
    points = []
    c = splinetype(tuples2points(knots))
    u = 0.0
    du = 0.1
    lim = (len(c) - 1) + du if trim else len(c) + du
    while u < lim:
        p = c(u)
        points.append(tuple(p))
        u += du
    return points
