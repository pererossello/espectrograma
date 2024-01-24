import json

import matplotlib.colors as mc
import colorsys

def flatten(l):
    return [item for sublist in l for item in sublist]

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def darken_color(color, amount=0.5):
    """
    Darkens the given color by reducing the luminosity by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    # Reduce the luminance by the amount
    new_luminance = max(0, c[1] - amount)  # Ensuring luminance doesn't go below 0
    return colorsys.hls_to_rgb(c[0], new_luminance, c[2])


with open('data/chromatic_scale.json') as f:
    chromatic_scale = json.load(f)

with open('data/scales.json') as f:
    scales = json.load(f)