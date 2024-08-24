#!/usr/bin/env python3

import random
import math
import argparse
import pickle  # Python 3 uses 'pickle' instead of 'cPickle'
import logging
import os
import sys
import re
import colorsys
from xml.dom.minidom import parse

from PIL import Image, ImageDraw  # PIL is imported directly in Python 3

#from sfgen import *

class ColorScheme:
    Name = "__ColorScheme__"

    def __init__(self, lattice):
        self.lattice = lattice
        self.cells = self.lattice.cells

    def __call__(self, cell, **kw):
        pass

class Grayscale(ColorScheme):
    Name = "grayscale"

    def __init__(self, lattice, boundary=False):
        self.boundary = boundary
        super().__init__(lattice)

    def __call__(self, cell, **kw):
        mass = cell.diffusive_mass
        if cell.attached:
            mass = cell.crystal_mass
        color = 200 * mass
        color = min(255, int(color))
        color = (color, color, color)
        return color

class BlackWhite(ColorScheme):
    Name = "blackwhite"

    def __init__(self, lattice, boundary=False):
        self.boundary = boundary
        super().__init__(lattice)

    def __call__(self, cell):
        color = (0, 0, 0)
        if self.boundary and cell.boundary or cell.attached:
            color = (0xFF, 0xFF, 0xFF)
        return color

class Colorful(ColorScheme):
    Name = "colorful"

    def __call__(self, cell, **kw):
        return tuple([int(round(x * 0xff)) for x in colorsys.hsv_to_rgb(cell.age / float(self.lattice.iteration), 1, 1)])

class LaserScheme(ColorScheme):
    Name = "laser"

    def __init__(self, lattice, layers, scheme=None):
        self.layers = layers
        self.layer = 0
        super().__init__(lattice)
        self._init_clusters()
        self.scheme = scheme
        if self.scheme is None:
            self.scheme = BlackWhite(self.lattice)

    def select_layer(self, layer):
        self.layer = layer

    def _init_clusters(self):
        import scipy.cluster.vq
        import numpy as np
        acells = [cell for cell in self.cells if cell and cell.attached]
        fm = [cell.crystal_mass for cell in acells]
        clusters = scipy.cluster.vq.kmeans2(np.array(fm), self.layers)
        cluster_map = clusters[1]
        self._layer_cache = {cell.xy: cluster for (cell, cluster) in zip(acells, cluster_map)}

    def __call__(self, cell, layer=None, **kw):
        if layer is None:
            layer = self.layer
        if cell.xy in self._layer_cache and self._layer_cache[cell.xy] == layer:
            return self.scheme(cell, **kw)
        return (0, 0, 0)

class RenderSnowflake:
    ColorSchemes = {cls.Name: cls for cls in globals().values() if isinstance(cls, type) and issubclass(cls, ColorScheme) and cls != ColorScheme}

    def __init__(self, lattice):
        self.lattice = lattice
        self.cells = self.lattice.cells

    def save_layer(self, fn, scheme, layer, **kw):
        scheme.select_layer(layer)
        self.save_image(fn, scheme=scheme, **kw)

    def save_layers(self, fn, layers, scheme=None, **kw):
        if scheme is None:
            scheme = LaserScheme(self.lattice, layers)
        fnlist = []
        for layer in range(layers):
            _fn = fn % layer
            fnlist.append(_fn)
            self.save_layer(_fn, scheme, layer, **kw)
        return fnlist

    def save_image(self, fn, scheme=None, overwrite=True, rotate=True, scale=True, crop=True, resize=None, margin=None):
        if not overwrite and os.path.exists(fn):
            return
        if scheme is None:
            scheme = Grayscale(self.lattice)
            #scheme = Colorful(self.lattice)
        msg = f"Saving {fn}..."
        content = ''.join([''.join(map(chr, scheme(cell))) for cell in self.cells])
        img = Image.new("RGB", (self.lattice.size, self.lattice.size))
        img.frombytes(content.encode('latin1'))
        del content
        X_SCALE_FACTOR = (1.0 / math.sqrt(3))
        # post-process
        if rotate:
            img = img.rotate(45)
        if scale:
            img = img.resize((int(round(self.lattice.size * X_SCALE_FACTOR)), int(self.lattice.size)))
        if crop:
            img = img.crop(self.lattice.crop_snowflake(margin=margin))
        if resize:
            y_sz = int(round((resize / float(img.size[0])) * img.size[1]))
            if y_sz != resize:
                print("WARNING: image after resize is not square.")
            img = img.resize((resize, resize))
        img.save(fn)
