import os
import numpy as np


def dirtodict(dirPath):
    d = {}
    for i in [os.path.join(dirPath, i) for i in os.listdir(dirPath)
              if os.path.isdir(os.path.join(dirPath, i))]:
        d[os.path.basename(i)] = dirtodict(i)
    d['.files'] = [os.path.join(dirPath, i) for i in os.listdir(dirPath)
                   if os.path.isfile(os.path.join(dirPath, i))]
    return d


def flatten(l):
    return [item for sublist in l for item in sublist]

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

"""
Dictionary of notes
"""

cr = {}

cr['C0'] = 16.35
cr['C#0'] = 17.32
cr['D0'] = 18.35
cr['D#0'] = 19.45
cr['E0'] = 20.60
cr['F0'] = 21.83
cr['F#0'] = 23.12
cr['G0'] = 24.50
cr['G#0'] = 25.96
cr['A0'] = 27.50
cr['A#0'] = 29.14
cr['B0'] = 30.87
cr['C1'] = 32.70
cr['C#1'] = 34.65
cr['D1'] = 36.71
cr['D#1'] = 38.89
cr['E1'] = 41.20
cr['F1'] = 43.65
cr['F#1'] = 46.25
cr['G1'] = 49.00
cr['G#1'] = 51.91
cr['A1'] = 55.00
cr['A#1'] = 58.27
cr['B1'] = 61.74
cr['C2'] = 65.41
cr['C#2'] = 69.30
cr['D2'] = 73.42
cr['D#2'] = 77.78
cr['E2'] = 82.41
cr['F2'] = 87.31
cr['F#2'] = 92.50
cr['G2'] = 98.00
cr['G#2'] = 103.83
cr['A2'] = 110.00
cr['A#2'] = 116.54
cr['B2'] = 123.47
cr['C3'] = 130.81
cr['C#3'] = 138.59
cr['D3'] = 146.83
cr['D#3'] = 155.56
cr['E3'] = 164.81
cr['F3'] = 174.61
cr['F#3'] = 185.00
cr['G3'] = 196.00
cr['G#3'] = 207.65
cr['A3'] = 220.00
cr['A#3'] = 233.08
cr['B3'] = 246.94
cr['C4'] = 261.63
cr['C#4'] = 277.18
cr['D4'] = 293.66
cr['D#4'] = 311.13
cr['E4'] = 329.63
cr['F4'] = 349.23
cr['F#4'] = 369.99
cr['G4'] = 392.00
cr['G#4'] = 415.30
cr['A4'] = 440.00
cr['A#4'] = 466.16
cr['B4'] = 493.88
cr['C5'] = 523.25
cr['C#5'] = 554.37
cr['D5'] = 587.33
cr['D#5'] = 622.25
cr['E5'] = 659.25
cr['F5'] = 698.46
cr['F#5'] = 739.99
cr['G5'] = 783.99
cr['G#5'] = 830.61
cr['A5'] = 880.00
cr['A#5'] = 932.33
cr['B5'] = 987.77
cr['C6'] = 1046.50
cr['C#6'] = 1108.73
cr['D6'] = 1174.66
cr['D#6'] = 1244.51
cr['E6'] = 1318.51
cr['F6'] = 1396.91
cr['F#6'] = 1479.98
cr['G6'] = 1567.98
cr['G#6'] = 1661.22
cr['A6'] = 1760.00
cr['A#6'] = 1864.66
cr['B6'] = 1975.53
cr['C7'] = 2093.00
cr['C#7'] = 2217.46
cr['D7'] = 2349.32
cr['D#7'] = 2489.02
cr['E7'] = 2637.02
cr['F7'] = 2793.83
cr['F#7'] = 2959.96
cr['G7'] = 3135.96
cr['G#7'] = 3322.44
cr['A7'] = 3520.00
cr['A#7'] = 3729.31
cr['B7'] = 3951.07
cr['C8'] = 4186.01

freqs = list(cr.values())
intervals = np.array([freqs[i+1]-freqs[i] for i in range(len(freqs)-1)])
mitj_int = intervals/2
marges = np.add(freqs[:-1],mitj_int)

def notifica(xf,pts):
    
    """
    
    Aquesta funció discretitza espectres de frequencies en caixes dodecafòniques
    
    S'ha d'aplicar al power spectra un pic ja s'ha aplicat el filtre d'amplitud
    
    Per què ho faig? 
    
        Perquè degut a la naturalesa exponencial dels intervals (creixen com una sèria geomètrica)
        , i a la resolució de freqüència que és constant, els pics en l'espectrograma són més 
        punxaguts per altes frequencies, el qu implica que siguin més tenues en l'espectrograma
        , els pixels són prims. 
    
    
    """
    
    inds = np.digitize(xf,marges)

    fill=[]

    for i in range(len(pts)):

        if pts[i] != np.float32(0):
            
            fill.append((inds[i], pts[i]))
        
    fillip = np.array(fill)
    
    if fill==[]:
        
        return np.zeros((len(marges)))

    ids = sorted(list(set(fillip[:,0])))
    fufu = fillip[:,0]
    fefe = fillip[:,1]

    yep = []
    for i in range(len(ids)):
        yep.append([])

    for j in range(len(ids)):    
        for i in range(len(fefe)):
            if fufu[i]==ids[j]:
                yep[j].append(fefe[i])
           
    final = np.array([[marges[int(ids[i])],np.max(yep[i])] for i in range(len(ids))])    

    power=np.zeros(len(marges))

    j=0
    for i in range(len(marges)):
        if marges[i] in final[:,0]: 
            power[i] = final[j,1]
            j+=1
        else:  
            power[i]=np.float32(0)

    return power





















