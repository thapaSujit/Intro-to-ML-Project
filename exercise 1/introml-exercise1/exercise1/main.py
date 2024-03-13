from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal


# TODO: Test the functions imported in lines 1 and 2 of this file.

createChirpSignal(200, 1, 1, 10, False)

createTriangleSignal(200, 2, 10000)
createSquareSignal(200, 2, 10000)
createSawtoothSignal(200, 2, 10000, 1)